CREATE TABLE IF NOT EXISTS {} (
                                CUSTOMER_ID            VARCHAR(20),
                                MRN                    VARCHAR(20), 
                                NUM_AGE                NUMBER(3,0), 
                                IS_PAYOR_TYPE          VARCHAR(100),
                                NUM_YEARS_ON_INSULIN   NUMBER(4,0),
                                IS_DIAGNOSIS_TYPE      VARCHAR(50),
                                IS_AUTO_REORDER_FLAG   NUMBER(1,0),
                                IS_ATTRITION_FLAG_C    NUMBER(1,0),
                                INGESTED_DATE          TIMESTAMP,
                                UPDATED_DATE           TIMESTAMP,
                                STAGE                  VARCHAR(20)
                            );