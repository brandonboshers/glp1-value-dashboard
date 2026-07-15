---------------------------------------------------------------------------------------------------
-- 05_biometrics.sql
-- Pre/Post biometric values for GLP-1 members.
-- Returns one row per member per test with pre and post values + status.
---------------------------------------------------------------------------------------------------

WITH GLP1 AS (
    SELECT EM.CURRENTGUID, MIN(RX.DateFilled) AS INDEX_DATE
    FROM SC_REPORTING.DHS_RXCLAIMS RX
    JOIN MASTER_DATA.TherapeuticConcept_NDC_XREF GLP
        ON REPLACE(RX.NdcCode, '-', '') = GLP.NDC11
    JOIN ENT_WH.ELIGMEMBER EM ON RX.Guid = EM.GUID AND EM.CUSTOMERID = '{CUSTOMERID}'
        AND EM.BIEFFECTIVEDATE <= SYSDATE::DATE
        AND EM.BIENDDATE >= TIMESTAMPADD(MONTH, -36, SYSDATE::DATE)
    WHERE (GLP.TherapeuticConceptName ILIKE '%GLP-1%'
        OR GLP.TherapeuticConceptName ILIKE '%GIP and GLP%')
      AND GLP.Deleted IS NOT TRUE AND RX.CustomerId = '{CUSTOMERID}'
      AND RX.Guid > 0 AND RX.RecordStatus = 'C'
    GROUP BY 1
    HAVING MIN(RX.DateFilled) BETWEEN '{INDEX_START}' AND '{INDEX_END}'
),
SECURE AS (
    SELECT DISTINCT SEM.SECURE_ID, G.CURRENTGUID, G.INDEX_DATE
    FROM GLP1 G
    JOIN ENT_WH.ELIGMEMBER EM ON G.CURRENTGUID = EM.CURRENTGUID
        AND EM.CUSTOMERID = '{CUSTOMERID}'
    JOIN ENT_WH.SC_EMPLOYER_MEMBERSHIP SEM ON EM.GUID = SEM.GUID
        AND SEM.CUSTOMER = '{CUSTOMERID}'
    WHERE SEM.DELETED_FLG IS NULL AND LENGTH(SEM.SECURE_ID) > 5
),
-- Pre-period: closest reading within 12 months before index
BIO_PRE AS (
    SELECT * FROM (
        SELECT
            S.CURRENTGUID,
            B.TESTNAME,
            B.TESTRESULTVALUE AS PRE_VALUE,
            B.STATUS AS PRE_STATUS,
            B.DOS AS PRE_DOS,
            ROW_NUMBER() OVER (PARTITION BY S.CURRENTGUID, B.TESTNAME
                ORDER BY B.DOS DESC) AS RN
        FROM ENT_WH.BIOMETRIC_RANGES B
        JOIN SECURE S ON B.SECURE_ID = S.SECURE_ID
        WHERE B.CUSTOMERID = '{CUSTOMERID}'
          AND UPPER(B.BIOSOURCE) = 'LABS'
          AND B.DOS::DATE >= TIMESTAMPADD(MONTH, -12, S.INDEX_DATE)
          AND B.DOS::DATE < S.INDEX_DATE
          AND B.TESTRESULTVALUE IS NOT NULL
          AND B.TESTNAME IN ('Body Mass Index (BMI)', 'Hemoglobin A1C',
              'Systolic Blood Pressure', 'Diastolic Blood Pressure',
              'Fasting Glucose', 'Triglycerides', 'LDL Cholesterol',
              'HDL Cholesterol', 'Waist Circumference')
    ) X WHERE RN = 1
),
-- Post-period: most recent reading 6+ months after index (allows time for effect)
BIO_POST AS (
    SELECT * FROM (
        SELECT
            S.CURRENTGUID,
            B.TESTNAME,
            B.TESTRESULTVALUE AS POST_VALUE,
            B.STATUS AS POST_STATUS,
            B.DOS AS POST_DOS,
            ROW_NUMBER() OVER (PARTITION BY S.CURRENTGUID, B.TESTNAME
                ORDER BY B.DOS DESC) AS RN
        FROM ENT_WH.BIOMETRIC_RANGES B
        JOIN SECURE S ON B.SECURE_ID = S.SECURE_ID
        WHERE B.CUSTOMERID = '{CUSTOMERID}'
          AND UPPER(B.BIOSOURCE) = 'LABS'
          AND B.DOS::DATE >= TIMESTAMPADD(MONTH, 6, S.INDEX_DATE)
          AND B.TESTRESULTVALUE IS NOT NULL
          AND B.TESTNAME IN ('Body Mass Index (BMI)', 'Hemoglobin A1C',
              'Systolic Blood Pressure', 'Diastolic Blood Pressure',
              'Fasting Glucose', 'Triglycerides', 'LDL Cholesterol',
              'HDL Cholesterol', 'Waist Circumference')
    ) X WHERE RN = 1
)
SELECT
    PRE.CURRENTGUID,
    PRE.TESTNAME,
    PRE.PRE_VALUE,
    PRE.PRE_STATUS,
    PRE.PRE_DOS,
    POST.POST_VALUE,
    POST.POST_STATUS,
    POST.POST_DOS,
    POST.POST_VALUE - PRE.PRE_VALUE AS VALUE_CHANGE,
    CASE WHEN PRE.PRE_VALUE <> 0
         THEN (POST.POST_VALUE - PRE.PRE_VALUE) / ABS(PRE.PRE_VALUE) * 100
         ELSE NULL END AS PCT_CHANGE,
    CASE
        WHEN UPPER(POST.POST_STATUS) = 'GREEN' AND UPPER(PRE.PRE_STATUS) IN ('RED','YELLOW') THEN 'Improved to Goal'
        WHEN UPPER(POST.POST_STATUS) = 'YELLOW' AND UPPER(PRE.PRE_STATUS) = 'RED' THEN 'Improved'
        WHEN UPPER(POST.POST_STATUS) = UPPER(PRE.PRE_STATUS) THEN 'No Change'
        WHEN UPPER(POST.POST_STATUS) = 'RED' AND UPPER(PRE.PRE_STATUS) IN ('GREEN','YELLOW') THEN 'Worsened'
        WHEN UPPER(POST.POST_STATUS) = 'YELLOW' AND UPPER(PRE.PRE_STATUS) = 'GREEN' THEN 'Worsened'
        ELSE 'No Change'
    END AS STATUS_DIRECTION,
    DATEDIFF('month', PRE.PRE_DOS::DATE, POST.POST_DOS::DATE) AS MONTHS_BETWEEN
FROM BIO_PRE PRE
JOIN BIO_POST POST ON PRE.CURRENTGUID = POST.CURRENTGUID AND PRE.TESTNAME = POST.TESTNAME
