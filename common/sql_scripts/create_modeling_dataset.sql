CREATE TABLE IF NOT EXISTS {} (
                                CUSTOMER_ID                                 VARCHAR(20),
                                MRN                                         VARCHAR(20),
                                MONTH                                       VARCHAR(7),
                                NUM_AGE                                     NUMBER(3,0),
                                NUM_YEARS_ON_INSULIN                        NUMBER(4,0),
                                IS_AUTO_REORDER_FLAG                        NUMBER(1,0),
                                IS_ATTRITION_FLAG_C                         NUMBER(1,0),
                                LENGTH_OF_WARRANTY_IN_MONTHS                FLOAT,
                                IS_WARRANTY_ACTIVE                          NUMBER,
                                IS_SR                                       NUMBER,
                                NUM_CUM_ACTIVE_MONTH                        NUMBER,
                                NUM_LAST_ORDER_CUM_ACTIVE_MONTHS            NUMBER,
                                NUM_IS_RETURNED_SUMLOOKBACK_6M_SCORE        FLOAT,
                                NUM_REGULAR_COMPLAINT_LOOKBACK_18M_SCORE    FLOAT,
                                NUM_HARM_FLAG_SUM_LOOKBACK_18M_SCORE        FLOAT,
                                NUM_TOTAL_NOTE_COUNT_LOOKBACK_12M_SCORE     FLOAT,
                                NUM_INFUSION_SET_RESIDUAL_ADJ               FLOAT,
                                PAYOR_TYPE_COMMERCIAL                       NUMBER(1,0),
                                PAYOR_TYPE_GENERIC_OTHER                    NUMBER(1,0),
                                PAYOR_TYPE_MANAGED_MEDICAID                 NUMBER(1,0),
                                PAYOR_TYPE_MANAGED_MEDICARE                 NUMBER(1,0),
                                PAYOR_TYPE_MEDICARE                         NUMBER(1,0),
                                PAYOR_TYPE_PHARMACY                         NUMBER(1,0),
                                PAYOR_TYPE_STATE_MEDICAID                   NUMBER(1,0),
                                PAYOR_TYPE_VA_MILITARY                      NUMBER(1,0),
                                DIAGNOSIS_TYPE_TYPE_1                       NUMBER(1,0),
                                DIAGNOSIS_TYPE_TYPE_2                       NUMBER(1,0),
                                INGESTED_DATE                               TIMESTAMP,
                                UPDATED_DATE                                TIMESTAMP,
                                STAGE                                       VARCHAR(20)
                            );