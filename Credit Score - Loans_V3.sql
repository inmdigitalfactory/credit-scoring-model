WITH DETAIL AS
(
SELECT CS_DETAIL.CIF_ID, CS_DETAIL.ACID, CUST_AGE, AGE_WITH_BANK, SEGMENTATION_CLASS, CUSTOMERINCOME, SECTOR, SUBSECTOR, SECURED_FLG, PHYSICALADDRESS, MARITALSTATUS, 
OTG_TRANSACTING, 
CS_DETAIL.SCHM_CODE, CS_DETAIL.SCHM_DESC, CS_DETAIL.DIS_AMT, DISB_DATE_6, CS_DETAIL.APPLICABLE_DATE, REP_PERD_MTHS, DPD_CNTR, FLOW_AMT, NO_OF_INSTALLMENTS, 
COUNT(DISTINCT CS_NON_LOAN_ACC.ACID)ACTIVE_ACCT, 
COUNT(DISTINCT CS_LOAN_ACC.ACID)LOAN_ACCT_PRIOR, ROUND(AVG(TRAN_DATE_BAL), 2)AVERAGE_EOD_BALANCE, CEIL((CS_DETAIL.APPLICABLE_DATE - MIN(CS_OTG.TRAN_DATE))/28)NO_OF_MONTHS_PRIOR_DISB_OTG
FROM CS_DETAIL
LEFT JOIN CS_NON_LOAN_ACC ON CS_NON_LOAN_ACC.CIF_ID = CS_DETAIL.CIF_ID AND CS_NON_LOAN_ACC.ACCT_OPN_DATE < CS_DETAIL.APPLICABLE_DATE 
LEFT JOIN CS_LOAN_ACC ON CS_LOAN_ACC.CIF_ID = CS_DETAIL.CIF_ID AND CS_LOAN_ACC.ACCT_OPN_DATE < APPLICABLE_DATE 
--LEFT JOIN TRANS ON TRANS.CIF_ID = CS_DETAIL.CIF_ID
LEFT JOIN CS_EOD_BAL ON CS_EOD_BAL.CIF_ID = CS_DETAIL.CIF_ID 
LEFT JOIN CS_OTG ON CS_OTG.CIF_ID = CS_DETAIL.CIF_ID AND CS_OTG.TRAN_DATE < CS_DETAIL.APPLICABLE_DATE
WHERE CS_DETAIL.CIF_ID IN ('0174414', '0002237')
GROUP BY CS_DETAIL.CIF_ID, CS_DETAIL.ACID, CUST_AGE, AGE_WITH_BANK, SEGMENTATION_CLASS, CUSTOMERINCOME, SECTOR, SUBSECTOR, SECURED_FLG, PHYSICALADDRESS, MARITALSTATUS, 
OTG_TRANSACTING, CS_DETAIL.SCHM_CODE, CS_DETAIL.SCHM_DESC, CS_DETAIL.DIS_AMT, CS_DETAIL.DISB_DATE_6, CS_DETAIL.APPLICABLE_DATE, REP_PERD_MTHS, DPD_CNTR, FLOW_AMT, 
NO_OF_INSTALLMENTS
),
OTG AS
(
SELECT A.CIF_ID,TO_CHAR(A.TRAN_DATE,'MON-YY') TRAN_MONTH, SUM(A.TRAN_AMOUNT)TRAN_AMOUNT, COUNT(TRAN_ID) TRAN_ID  FROM CS_OTG A
GROUP BY A.CIF_ID, TO_CHAR(A.TRAN_DATE,'MON-YY')
),
TRANS_CR AS
(
SELECT H.CIF_ID, TRAN_DATE, CASE WHEN H.TRAN_CRNCY_CODE != 'KES' THEN (H.TRAN_AMT * RTH.VAR_CRNCY_UNITS) ELSE H.TRAN_AMT END CONVERTED_CR_TRAN_AMT 
FROM CS_TRANS_DETAIL H
LEFT JOIN TBAADM.RTH@FINACLE RTH ON RTH.FXD_CRNCY_CODE = H.TRAN_CRNCY_CODE AND H.TRAN_DATE = RTH.RTLIST_DATE AND RTH.VAR_CRNCY_CODE = 'KES' AND RTH.RTLIST_NUM = 1 AND RTH.RATECODE = 'MID'
WHERE H.PART_TRAN_TYPE = 'C'
),
TRANS_DR AS
(SELECT H.CIF_ID, TRAN_DATE, CASE WHEN H.TRAN_CRNCY_CODE != 'KES' THEN (H.TRAN_AMT * RTH.VAR_CRNCY_UNITS) ELSE H.TRAN_AMT END CONVERTED_DR_TRAN_AMT 
FROM CS_TRANS_DETAIL H
LEFT JOIN TBAADM.RTH@FINACLE RTH ON RTH.FXD_CRNCY_CODE = H.TRAN_CRNCY_CODE AND H.TRAN_DATE = RTH.RTLIST_DATE AND RTH.VAR_CRNCY_CODE = 'KES' AND RTH.RTLIST_NUM = 1 AND RTH.RATECODE = 'MID'
WHERE H.PART_TRAN_TYPE = 'D'
)
SELECT D.CIF_ID, D.ACID, CUST_AGE, AGE_WITH_BANK, SEGMENTATION_CLASS, CUSTOMERINCOME, SECTOR, SUBSECTOR, SECURED_FLG, PHYSICALADDRESS, MARITALSTATUS, 
OTG_TRANSACTING, D.SCHM_CODE, D.SCHM_DESC, D.DIS_AMT, D.DISB_DATE_6, D.APPLICABLE_DATE, REP_PERD_MTHS, DPD_CNTR, FLOW_AMT, NO_OF_INSTALLMENTS,
D.ACTIVE_ACCT, D.LOAN_ACCT_PRIOR, AVG(CONVERTED_CR_TRAN_AMT) AVG_CR_AMOUNT_IN_6_MONTHS, AVG(CONVERTED_DR_TRAN_AMT)AVG_DR_AMOUNT_IN_6_MONTHS, 
MEDIAN(CONVERTED_CR_TRAN_AMT )MEDIAN_CR_AMOUNT_IN_6_MONTHS , MEDIAN(CONVERTED_DR_TRAN_AMT) MEDIAN_DR_AMOUNT_IN_6_MONTHS, MAX(CONVERTED_CR_TRAN_AMT) MAX_CR_AMOUNT_IN_6_MONTHS, 
MAX(CONVERTED_DR_TRAN_AMT) MAX_DR_AMOUNT_IN_6_MONTHS, MIN(CONVERTED_CR_TRAN_AMT) MIN_CR_AMOUNT_IN_6_MONTHS, 
MIN(CONVERTED_DR_TRAN_AMT) MIN_DR_AMOUNT_IN_6_MONTHS, D.AVERAGE_EOD_BALANCE, D.NO_OF_MONTHS_PRIOR_DISB_OTG,
COUNT(DISTINCT OTG.TRAN_MONTH)NO_OF_ACTIVE_MONTHS_OTG, AVG(OTG.TRAN_AMOUNT)AVG_TRAN_AMNT_OTG, AVG(OTG.TRAN_ID)AVG_TRAN_COUNT_OTG
FROM DETAIL D
LEFT JOIN OTG ON OTG.CIF_ID = D.CIF_ID
LEFT JOIN TRANS_CR ON TRANS_CR.CIF_ID = D.CIF_ID AND TRANS_CR.TRAN_DATE BETWEEN D.DISB_DATE_6 AND D.APPLICABLE_DATE
LEFT JOIN TRANS_DR ON TRANS_DR.CIF_ID = D.CIF_ID AND TRANS_DR.TRAN_DATE BETWEEN D.DISB_DATE_6 AND D.APPLICABLE_DATE
GROUP BY D.CIF_ID, D.ACID, CUST_AGE, AGE_WITH_BANK, SEGMENTATION_CLASS, CUSTOMERINCOME, SECTOR, SUBSECTOR, SECURED_FLG, PHYSICALADDRESS, MARITALSTATUS, 
OTG_TRANSACTING, D.SCHM_CODE, D.SCHM_DESC, D.DIS_AMT, D.DISB_DATE_6, D.APPLICABLE_DATE, REP_PERD_MTHS, DPD_CNTR, FLOW_AMT, NO_OF_INSTALLMENTS,
D.ACTIVE_ACCT, D.LOAN_ACCT_PRIOR, 
D.AVERAGE_EOD_BALANCE, D.NO_OF_MONTHS_PRIOR_DISB_OTG

