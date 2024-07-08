CREATE TABLE IF NOT EXISTS {} (
                                CUSTOMER_ID                               VARCHAR(20),
                                MRN                                       VARCHAR(20), 
                                MONTH                                     VARCHAR(7), 
                                NUM_INFUSION_SET_RESIDUAL_ADJ             FLOAT,
                                NUM_IS_RETURNED_SUMLOOKBACK_6M_SCORE      FLOAT,
                                NUM_LAST_ORDER_CUM_ACTIVE_MONTHS          NUMBER,
                                NUM_CUM_ACTIVE_MONTH                      NUMBER,
                                INGESTED_DATE                             TIMESTAMP,
                                UPDATED_DATE                              TIMESTAMP,
                                STAGE                                     VARCHAR(20)
                            );