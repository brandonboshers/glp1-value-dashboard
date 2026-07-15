---------------------------------------------------------------------------------------------------
-- 03_claims_prepost.sql
-- Pre/Post claims for BOTH GLP-1 and Control cohorts.
-- Accepts a list of (CURRENTGUID, GUID, INDEX_DATE) as input from Python via temp logic.
-- Called once for GLP-1 members, once for controls.
--
-- Parameters: {CUSTOMERID}, {GUID_LIST} (comma-separated), {PERIOD} ('PRE' or 'POST')
-- This is used inline by the Python app, not standalone.
---------------------------------------------------------------------------------------------------

-- For GLP-1 cohort: member-level pre/post costs
-- This is the same query as glp1_prepost.sql but with additional detail columns
WITH COHORT AS (
    SELECT
        EM.CURRENTGUID,
        MIN(RX.DateFilled) AS INDEX_DATE
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
      AND RX.Guid > 0 AND RX.RecordStatus = 'C'
    GROUP BY 1
    HAVING MIN(RX.DateFilled) BETWEEN '{INDEX_START}' AND '{INDEX_END}'
),
-- All GUIDs associated with cohort members (handles guid shifts)
COHORT_GUIDS AS (
    SELECT DISTINCT EM.GUID, C.CURRENTGUID, C.INDEX_DATE
    FROM COHORT C
    JOIN ENT_WH.ELIGMEMBER EM ON C.CURRENTGUID = EM.CURRENTGUID
        AND EM.CUSTOMERID = '{CUSTOMERID}'
),
MED_PRE AS (
    SELECT G.CURRENTGUID,
        SUM(CASE WHEN REGEXP_LIKE(MC.PaidAmt, '^\-?\d+\.?\d*$') THEN MC.PaidAmt::FLOAT ELSE 0 END) AS MED_PAID,
        COUNT(*) AS MED_LINES,
        COUNT(DISTINCT CASE WHEN TRIM(MC.PlaceOfService) = '21' THEN MC.ClaimId END) AS IP_ADMITS,
        COUNT(DISTINCT CASE WHEN TRIM(MC.PlaceOfService) = '23' THEN MC.ClaimId END) AS ER_VISITS,
        COUNT(DISTINCT CASE WHEN TRIM(MC.PlaceOfService) = '23'
            AND (MC.ICDxCode1 LIKE 'J06%' OR MC.ICDxCode1 LIKE 'J20%'
              OR MC.ICDxCode1 LIKE 'M54%' OR MC.ICDxCode1 LIKE 'R10%'
              OR MC.ICDxCode1 LIKE 'K21%' OR MC.ICDxCode1 LIKE 'N39%')
            THEN MC.ClaimId END) AS AVOIDABLE_ER
    FROM COHORT_GUIDS G
    JOIN SC_REPORTING.DHS_MEDCLAIMS MC ON G.GUID = MC.Guid
    WHERE MC.CustomerId = '{CUSTOMERID}' AND MC.Guid > 0 AND MC.RecordStatus = 'C'
      AND MC.DateServiceFrom::DATE >= TIMESTAMPADD(MONTH, -12, G.INDEX_DATE)
      AND MC.DateServiceFrom::DATE < G.INDEX_DATE
    GROUP BY 1
),
MED_POST AS (
    SELECT G.CURRENTGUID,
        SUM(CASE WHEN REGEXP_LIKE(MC.PaidAmt, '^\-?\d+\.?\d*$') THEN MC.PaidAmt::FLOAT ELSE 0 END) AS MED_PAID,
        COUNT(*) AS MED_LINES,
        COUNT(DISTINCT CASE WHEN TRIM(MC.PlaceOfService) = '21' THEN MC.ClaimId END) AS IP_ADMITS,
        COUNT(DISTINCT CASE WHEN TRIM(MC.PlaceOfService) = '23' THEN MC.ClaimId END) AS ER_VISITS,
        COUNT(DISTINCT CASE WHEN TRIM(MC.PlaceOfService) = '23'
            AND (MC.ICDxCode1 LIKE 'J06%' OR MC.ICDxCode1 LIKE 'J20%'
              OR MC.ICDxCode1 LIKE 'M54%' OR MC.ICDxCode1 LIKE 'R10%'
              OR MC.ICDxCode1 LIKE 'K21%' OR MC.ICDxCode1 LIKE 'N39%')
            THEN MC.ClaimId END) AS AVOIDABLE_ER
    FROM COHORT_GUIDS G
    JOIN SC_REPORTING.DHS_MEDCLAIMS MC ON G.GUID = MC.Guid
    WHERE MC.CustomerId = '{CUSTOMERID}' AND MC.Guid > 0 AND MC.RecordStatus = 'C'
      AND MC.DateServiceFrom::DATE >= G.INDEX_DATE
      AND MC.DateServiceFrom::DATE < TIMESTAMPADD(MONTH, 12, G.INDEX_DATE)
    GROUP BY 1
),
RX_PRE AS (
    SELECT G.CURRENTGUID,
        SUM(CASE WHEN REGEXP_LIKE(RX.PaidAmt, '^\-?\d+\.?\d*$') THEN RX.PaidAmt::FLOAT ELSE 0 END) AS RX_PAID,
        COUNT(*) AS RX_FILLS
    FROM COHORT_GUIDS G
    JOIN SC_REPORTING.DHS_RXCLAIMS RX ON G.GUID = RX.Guid
    WHERE RX.CustomerId = '{CUSTOMERID}' AND RX.Guid > 0 AND RX.RecordStatus = 'C'
      AND RX.DateFilled::DATE >= TIMESTAMPADD(MONTH, -12, G.INDEX_DATE)
      AND RX.DateFilled::DATE < G.INDEX_DATE
    GROUP BY 1
),
RX_POST AS (
    SELECT G.CURRENTGUID,
        SUM(CASE WHEN REGEXP_LIKE(RX.PaidAmt, '^\-?\d+\.?\d*$') THEN RX.PaidAmt::FLOAT ELSE 0 END) AS RX_PAID,
        COUNT(*) AS RX_FILLS
    FROM COHORT_GUIDS G
    JOIN SC_REPORTING.DHS_RXCLAIMS RX ON G.GUID = RX.Guid
    WHERE RX.CustomerId = '{CUSTOMERID}' AND RX.Guid > 0 AND RX.RecordStatus = 'C'
      AND RX.DateFilled::DATE >= G.INDEX_DATE
      AND RX.DateFilled::DATE < TIMESTAMPADD(MONTH, 12, G.INDEX_DATE)
    GROUP BY 1
)
SELECT
    G.CURRENTGUID,
    G.INDEX_DATE,
    NVL(MP.MED_PAID, 0)   AS PRE_MED_PAID,
    NVL(MPO.MED_PAID, 0)  AS POST_MED_PAID,
    NVL(MP.IP_ADMITS, 0)   AS PRE_IP_ADMITS,
    NVL(MPO.IP_ADMITS, 0)  AS POST_IP_ADMITS,
    NVL(MP.ER_VISITS, 0)   AS PRE_ER_VISITS,
    NVL(MPO.ER_VISITS, 0)  AS POST_ER_VISITS,
    NVL(MP.AVOIDABLE_ER, 0) AS PRE_AVOIDABLE_ER,
    NVL(MPO.AVOIDABLE_ER, 0) AS POST_AVOIDABLE_ER,
    NVL(RP.RX_PAID, 0)    AS PRE_RX_PAID,
    NVL(RPO.RX_PAID, 0)   AS POST_RX_PAID,
    NVL(RP.RX_FILLS, 0)   AS PRE_RX_FILLS,
    NVL(RPO.RX_FILLS, 0)  AS POST_RX_FILLS
FROM COHORT G
LEFT JOIN MED_PRE MP   ON G.CURRENTGUID = MP.CURRENTGUID
LEFT JOIN MED_POST MPO ON G.CURRENTGUID = MPO.CURRENTGUID
LEFT JOIN RX_PRE RP    ON G.CURRENTGUID = RP.CURRENTGUID
LEFT JOIN RX_POST RPO  ON G.CURRENTGUID = RPO.CURRENTGUID
