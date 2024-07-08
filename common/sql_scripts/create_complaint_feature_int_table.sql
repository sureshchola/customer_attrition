CREATE TABLE IF NOT EXISTS {} (
                                CUSTOMER_ID                                 VARCHAR(20),
                                MRN                                         VARCHAR(20), 
                                MONTH                                       VARCHAR(7), 
                                NUM_REGULAR_COMPLAINT_LOOKBACK_18M_SCORE    FLOAT,
                                NUM_HARM_FLAG_SUM_LOOKBACK_18M_SCORE        FLOAT,
                                INGESTED_DATE                               TIMESTAMP,
                                UPDATED_DATE                                TIMESTAMP,
                                STAGE                                       VARCHAR(20)
                            );