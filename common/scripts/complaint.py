import common.utils.database as sc
import pandas as pd
from common.utils.time_bucket import time_bucket_indexing_weighted

def complaint_features(db_secrets_name,stage):

    """
    args:
    db_secrets_name - database secrets name
    stage - stage of pipeline either train or inference
    
    returns:
    write - dictionary of table stats
    """

    db = sc.Database(db_secrets_name)
    db.connect()
    df,fetched_time = db.execute('common/sql_scripts/fetch_complaint_feature_data.sql')

    # check if empty dataframe
    if df.shape[0]==0:
        raise Exception("Returned dataframe is empty")

    # Perform data cleaning and FE
    print('Executing data cleaning and feature engineering...')

    if df['CREATED_DATE'].dtypes == 'object':
        df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], format='%Y%m%d').dt.strftime('%m/%d/%Y')

    harm_df = df[df['REASON_CODE'] == 'Z101']
    complaint_raw = df[df['REASON_CODE'] != 'Z101']

    harm_df = harm_df.groupby(['CUSTOMER_ID','MRN','SVN_NUMBER'],as_index=False)['REASON_CODE'].count()
    harm_df['HARM_FLAG'] = 1

    comp = complaint_raw.groupby(['CUSTOMER_ID','MRN','SVN_NUMBER','CREATED_DATE'],as_index=False)['REASON_CODE'].count()

    comp_df = comp.merge(harm_df, on=['CUSTOMER_ID','MRN','SVN_NUMBER'], how='left')
    del comp_df['REASON_CODE_x']
    del comp_df['REASON_CODE_y']

    comp_df['HARM_FLAG'] = comp_df['HARM_FLAG'].fillna(0)

    comp_df['MONTH'] = pd.to_datetime(comp_df['CREATED_DATE']).dt.to_period('M')

    comp_agg = comp_df.groupby(['CUSTOMER_ID','MRN','MONTH'],as_index=False).agg({'SVN_NUMBER':'count','HARM_FLAG':'sum'}).rename({'SVN_NUMBER':'NUM_TOTAL_COMPLANT_MONTH','HARM_FLAG':'NUM_HARM_FLAG_SUM'},axis=1)
    comp_agg['NUM_REGULAR_COMPLAINT'] = comp_agg['NUM_TOTAL_COMPLANT_MONTH'] - comp_agg['NUM_HARM_FLAG_SUM']

    comp_agg = comp_agg.sort_values(['CUSTOMER_ID','MRN','MONTH']).reset_index(drop=True)

    comp_agg1 = time_bucket_indexing_weighted(original_df=comp_agg, col='NUM_HARM_FLAG_SUM', lookback_mnt=18)
    comp_agg1 = time_bucket_indexing_weighted(original_df=comp_agg1, col='NUM_REGULAR_COMPLAINT', lookback_mnt=18)

    # filter data to between 2018 to Apr, 2020
    # Again, don't need to do given we are left joining back to sales
    #comp_agg1 = comp_agg1[(comp_agg1['MONTH']>='2018-01')&(comp_agg1['MONTH']<='2021-04')]

    del comp_agg1['DATE']

    comp_agg1.fillna(0, inplace=True)

    comp_agg1 = comp_agg1.drop(['NUM_TOTAL_COMPLANT_MONTH',
       'NUM_HARM_FLAG_SUM', 'NUM_REGULAR_COMPLAINT'], axis=1)

    comp_agg1['MONTH'] = comp_agg1['MONTH'].astype(str)
    comp_agg1['STAGE'] = stage

    # Create table to write to if it doesn't already exist
    tableName = "PA_COMPLAINTS_INTERMEDIATE_FEATURES"
    
    try:
        db.execute('common/sql_scripts/create_complaint_feature_int_table.sql',False,tableName)
        db.execute('common/sql_scripts/create_hist_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/insert_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/truncate_table_generic.sql',False,tableName)
        write = db.write(comp_agg1,tableName,fetched_time)
        db.conn.commit()
        db.cursor.close()
        db.conn.close() 
        print("finished writing complaints table")
        return write
    except Exception as e:
        db.conn.rollback()
        db.conn.close() 
        raise e