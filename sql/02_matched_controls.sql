---------------------------------------------------------------------------------------------------
-- 02_matched_controls.sql
-- Builds a matched control group: members with obesity/diabetes diagnosis who did NOT
-- start GLP-1. Matched on: baseline cost quartile, age band, gender, diagnosis category.
--
-- Assigns a pseudo-index date = median GLP-1 index date for the matching stratum,
-- enabling apples-to-apples DID comparison.
---------------------------------------------------------------------------------------------------

WITH GLP1_GUIDS AS (
    SELECT DISTINCT EM.CURRENTGUID
    FROM SC_REPORTING.DHS_RXCLAIMS RX
    JOIN MASTER_DATA.TherapeuticConcept_NDC_XREF GLP
        ON REPLACE(RX.NdcCode, '-', '') = GLP.NDC11
    JOIN ENT_WH.ELIGMEMBER EM
        ON RX.Guid = EM.GUID AND EM.CUSTOMERID = '{CUSTOMERID}'
    WHERE (GLP.TherapeuticConceptName ILIKE '%GLP-1%'
        OR GLP.TherapeuticConceptName ILIKE '%GIP and GLP%')
      AND GLP.Deleted IS NOT TRUE
      AND RX.CustomerId = '{CUSTOMERID}' AND RX.Guid > 0
),
-- Identify members with obesity or diabetes dx who never took GLP-1
CONTROL_CANDIDATES AS (
    SELECT DISTINCT
        EM.CURRENTGUID,
        EM.GUID,
        EM.DOB,
        EM.GENDER,
        MAX(CASE WHEN MC.ICDxCode1 LIKE 'E11%' OR MC.ICDxCode2 LIKE 'E11%'
                      OR MC.ICDxCode3 LIKE 'E11%' THEN 1 ELSE 0 END) AS HAS_DIABETES,
        MAX(CASE WHEN MC.ICDxCode1 LIKE 'E66%' OR MC.ICDxCode2 LIKE 'E66%'
                      OR MC.ICDxCode3 LIKE 'E66%' THEN 1 ELSE 0 END) AS HAS_OBESITY
    FROM SC_REPORTING.DHS_MEDCLAIMS MC
    JOIN ENT_WH.ELIGMEMBER EM
        ON MC.Guid = EM.GUID AND EM.CUSTOMERID = '{CUSTOMERID}'
        AND EM.BIEFFECTIVEDATE <= SYSDATE::DATE
        AND EM.BIENDDATE >= TIMESTAMPADD(MONTH, -36, SYSDATE::DATE)
    LEFT JOIN GLP1_GUIDS G ON EM.CURRENTGUID = G.CURRENTGUID
    WHERE MC.CustomerId = '{CUSTOMERID}' AND MC.Guid > 0 AND MC.RecordStatus = 'C'
      AND MC.DateServiceFrom >= '{INDEX_START}'
      AND G.CURRENTGUID IS NULL  -- NOT on GLP-1
      AND (MC.ICDxCode1 LIKE 'E11%' OR MC.ICDxCode2 LIKE 'E11%' OR MC.ICDxCode3 LIKE 'E11%'
        OR MC.ICDxCode1 LIKE 'E66%' OR MC.ICDxCode2 LIKE 'E66%' OR MC.ICDxCode3 LIKE 'E66%')
    GROUP BY 1, 2, 3, 4
),
-- Compute baseline (pre-period) medical cost for controls
-- Use the midpoint of the GLP-1 cohort index window as the pseudo-index
CONTROL_COSTS AS (
    SELECT
        CC.CURRENTGUID,
        CC.GUID,
        CC.DOB,
        CC.GENDER,
        CC.HAS_DIABETES,
        CC.HAS_OBESITY,
        '{PSEUDO_INDEX}'::DATE AS INDEX_DATE,
        DATEDIFF('year', CC.DOB::DATE, '{PSEUDO_INDEX}'::DATE) AS AGE_AT_INDEX,
        SUM(CASE WHEN REGEXP_LIKE(MC.PaidAmt, '^\-?\d+\.?\d*$')
                 THEN MC.PaidAmt::FLOAT ELSE 0 END) AS BASELINE_MED_COST
    FROM CONTROL_CANDIDATES CC
    JOIN SC_REPORTING.DHS_MEDCLAIMS MC ON CC.GUID = MC.Guid
    WHERE MC.CustomerId = '{CUSTOMERID}' AND MC.Guid > 0 AND MC.RecordStatus = 'C'
      AND MC.DateServiceFrom::DATE >= TIMESTAMPADD(MONTH, -12, '{PSEUDO_INDEX}'::DATE)
      AND MC.DateServiceFrom::DATE < '{PSEUDO_INDEX}'::DATE
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
)
SELECT
    CURRENTGUID,
    GUID,
    INDEX_DATE,
    GENDER,
    AGE_AT_INDEX,
    HAS_DIABETES,
    HAS_OBESITY,
    BASELINE_MED_COST,
    CASE
        WHEN BASELINE_MED_COST >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY BASELINE_MED_COST) OVER () THEN 'Q4'
        WHEN BASELINE_MED_COST >= PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY BASELINE_MED_COST) OVER () THEN 'Q3'
        WHEN BASELINE_MED_COST >= PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY BASELINE_MED_COST) OVER () THEN 'Q2'
        ELSE 'Q1'
    END AS COST_QUARTILE,
    CASE
        WHEN AGE_AT_INDEX < 30 THEN '18-29'
        WHEN AGE_AT_INDEX < 40 THEN '30-39'
        WHEN AGE_AT_INDEX < 50 THEN '40-49'
        WHEN AGE_AT_INDEX < 60 THEN '50-59'
        ELSE '60+'
    END AS AGE_BAND,
    'Control' AS COHORT_TYPE
FROM CONTROL_COSTS
WHERE BASELINE_MED_COST > 0
