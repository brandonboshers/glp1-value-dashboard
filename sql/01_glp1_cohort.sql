---------------------------------------------------------------------------------------------------
-- 01_glp1_cohort.sql
-- Identifies GLP-1 members with first fill date, drug category, and persistence flags.
-- Returns one row per member with index date + adherence metrics.
---------------------------------------------------------------------------------------------------

WITH GLP1_FILLS_RAW AS (
    SELECT
        EM.CURRENTGUID,
        EM.GUID,
        RX.DateFilled,
        RX.RefillCycleDays,
        RX.DrugName,
        CASE
            WHEN UPPER(RX.DrugName) LIKE '%WEGOVY%' OR UPPER(RX.DrugName) LIKE '%ZEPBOUND%'
            THEN 'Weight Management'
            ELSE 'Diabetes'  -- All other GLP-1 drugs (Ozempic, Mounjaro, Trulicity, Victoza, Rybelsus, generics)
        END AS DRUG_INDICATION
    FROM SC_REPORTING.DHS_RXCLAIMS RX
    JOIN MASTER_DATA.TherapeuticConcept_NDC_XREF GLP
        ON REPLACE(RX.NdcCode, '-', '') = GLP.NDC11
    JOIN ENT_WH.ELIGMEMBER EM
        ON RX.Guid = EM.GUID
        AND EM.CUSTOMERID = '{CUSTOMERID}'
        AND EM.BIEFFECTIVEDATE <= SYSDATE::DATE
        AND EM.BIENDDATE >= TIMESTAMPADD(MONTH, -36, SYSDATE::DATE)
    WHERE (GLP.TherapeuticConceptName ILIKE '%GLP-1%'
        OR GLP.TherapeuticConceptName ILIKE '%GIP and GLP%')
      AND GLP.Deleted IS NOT TRUE
      AND RX.CustomerId = '{CUSTOMERID}'
      AND RX.Guid > 0
      AND RX.RecordStatus = 'C'
      AND REGEXP_LIKE(RX.RefillCycleDays, '^\d+$')
      AND RX.RefillCycleDays::INT > 0
),
FIRST_FILL AS (
    SELECT
        CURRENTGUID,
        MIN(DateFilled) AS INDEX_DATE,
        MIN(CASE WHEN RN = 1 THEN DRUG_INDICATION END) AS PRIMARY_INDICATION,
        COUNT(*) AS TOTAL_FILLS
    FROM (
        SELECT F.*,
            ROW_NUMBER() OVER (PARTITION BY CURRENTGUID ORDER BY DateFilled) AS RN
        FROM GLP1_FILLS_RAW F
    ) X
    GROUP BY 1
    HAVING MIN(DateFilled) BETWEEN '{INDEX_START}' AND '{INDEX_END}'
),
PERSISTENCE AS (
    SELECT
        F.CURRENTGUID,
        -- PDC: proportion of days covered in first 12 months
        -- Sum of days supply capped at 365 / 365
        LEAST(SUM(CASE
            WHEN G.DateFilled >= F.INDEX_DATE
             AND G.DateFilled < TIMESTAMPADD(MONTH, 12, F.INDEX_DATE)
            THEN G.RefillCycleDays::INT ELSE 0 END), 365)::FLOAT / 365 AS PDC_12MO,
        -- Persistence: any fill in month 10-14 window
        MAX(CASE WHEN G.DateFilled BETWEEN TIMESTAMPADD(MONTH, 10, F.INDEX_DATE)
                                        AND TIMESTAMPADD(MONTH, 14, F.INDEX_DATE)
            THEN 1 ELSE 0 END) AS PERSISTENT_12MO,
        -- Any fill in month 5-7 window
        MAX(CASE WHEN G.DateFilled BETWEEN TIMESTAMPADD(MONTH, 5, F.INDEX_DATE)
                                        AND TIMESTAMPADD(MONTH, 7, F.INDEX_DATE)
            THEN 1 ELSE 0 END) AS PERSISTENT_6MO,
        MAX(G.DateFilled) AS LAST_FILL_DATE,
        DATEDIFF('day', F.INDEX_DATE, MAX(G.DateFilled)) AS DAYS_ON_THERAPY
    FROM FIRST_FILL F
    JOIN GLP1_FILLS_RAW G ON F.CURRENTGUID = G.CURRENTGUID
    GROUP BY F.CURRENTGUID, F.INDEX_DATE
)
SELECT
    F.CURRENTGUID,
    F.INDEX_DATE,
    F.PRIMARY_INDICATION,
    F.TOTAL_FILLS,
    P.PDC_12MO,
    P.PERSISTENT_12MO,
    P.PERSISTENT_6MO,
    P.LAST_FILL_DATE,
    P.DAYS_ON_THERAPY,
    CASE
        WHEN P.PERSISTENT_12MO = 1 THEN 'Persister'
        ELSE 'Discontinuer'
    END AS PERSISTENCE_COHORT,
    'GLP-1' AS COHORT_TYPE
FROM FIRST_FILL F
JOIN PERSISTENCE P ON F.CURRENTGUID = P.CURRENTGUID
