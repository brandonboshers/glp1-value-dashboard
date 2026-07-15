---------------------------------------------------------------------------------------------------
-- 06_engagement.sql
-- Platform engagement metrics for GLP-1 members during their 12-month post-period.
-- Primary measure: MAU (Monthly Active User) months on the Sharecare platform.
-- Enables: "GLP-1 + High Platform Engagement" vs "GLP-1 + Low Engagement" comparison.
---------------------------------------------------------------------------------------------------

WITH GLP1 AS (
    SELECT EM.CURRENTGUID, MIN(RX.DateFilled) AS INDEX_DATE
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
GUID_MAP AS (
    SELECT DISTINCT EM.GUID, G.CURRENTGUID, G.INDEX_DATE
    FROM GLP1 G
    JOIN ENT_WH.ELIGMEMBER EM ON G.CURRENTGUID = EM.CURRENTGUID
        AND EM.CUSTOMERID = '{CUSTOMERID}'
),
-- MAU: count months with platform activity in post-period
-- Uses string comparison on MONTH_VISIT (format 'YYYY-MM') to avoid expensive date casts
MAU_COUNTS AS (
    SELECT
        GM.CURRENTGUID,
        COUNT(DISTINCT M.MONTH_VISIT) AS MAU_MONTHS_POST
    FROM GUID_MAP GM
    JOIN ENT_WH.MAU_ACTIVES M ON GM.GUID = M.GUID
    WHERE M.WAS_ELIGIBLE = 'Y'
      AND M.MONTH_VISIT >= TO_CHAR(GM.INDEX_DATE, 'YYYY-MM')
      AND M.MONTH_VISIT < TO_CHAR(TIMESTAMPADD(MONTH, 12, GM.INDEX_DATE), 'YYYY-MM')
    GROUP BY 1
),
-- Coaching calls during post-period
COACHING AS (
    SELECT
        GM.CURRENTGUID,
        COUNT(DISTINCT TRUNC(MC.ENCOUNTERDATETIME)) AS COACHING_DAYS
    FROM GUID_MAP GM
    JOIN ENT_WH.MEMBER_CALL_DATA MC
        ON NVL(MC.DERIVED_GUID, MC.GUID) = GM.GUID
    WHERE MC.CUSTOMERID = '{CUSTOMERID}'
      AND UPPER(MC.CALL_STATUS) = 'SUCCESSFUL'
      AND UPPER(MC.DIRECTION) = 'OUTBOUND'
      AND MC.ENCOUNTERDATETIME::DATE >= GM.INDEX_DATE
      AND MC.ENCOUNTERDATETIME::DATE < TIMESTAMPADD(MONTH, 12, GM.INDEX_DATE)
    GROUP BY 1
)
SELECT
    G.CURRENTGUID,
    NVL(MAU.MAU_MONTHS_POST, 0) AS MAU_MONTHS_POST,
    0 AS COACHING_DAYS,
    -- Engagement tier based on MAU activity
    CASE
        WHEN NVL(MAU.MAU_MONTHS_POST, 0) >= 9 THEN 'High Engagement'
        WHEN NVL(MAU.MAU_MONTHS_POST, 0) >= 6 THEN 'Moderate Engagement'
        ELSE 'Low/No Engagement'
    END AS ENGAGEMENT_TIER
FROM GLP1 G
LEFT JOIN MAU_COUNTS MAU ON G.CURRENTGUID = MAU.CURRENTGUID
