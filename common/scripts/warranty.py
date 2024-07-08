import common.utils.database as sc
import pandas as pd
from datetime import datetime

def warranty_features(db_secrets_name,stage):

    """
    args:
    db_secrets_name - database secrets name
    stage - stage of pipeline either train or inference
    
    returns:
    write - dictionary of table stats
    """

    db = sc.Database(db_secrets_name)
    db.connect()
    df, fetched_time = db.execute('common/sql_scripts/fetch_warranty_feature_data.sql')

    # check if empty dataframe
    if df.shape[0]==0:
        raise Exception("Returned dataframe is empty")

    # Perform data cleaning and FE
    print('Executing data cleaning and feature engineering...')

    x = df[['CUSTOMER_ID','MRN','WARRANTY_START_DATE','WARRANTY_END_DATE','TRAINING_STATUS']].copy(deep=True)

    x['WARRANTY_END_DATE'] = pd.to_datetime(x['WARRANTY_END_DATE'])
    x['WARRANTY_START_DATE'] = pd.to_datetime(x['WARRANTY_START_DATE'])
    x['LENGTH_OF_WARRANTY_IN_MONTHS'] = (x['WARRANTY_END_DATE'] - x['WARRANTY_START_DATE']).astype('timedelta64[M]')

    x['WARRANTY_START_DATE'] = x['WARRANTY_START_DATE'].dt.to_period('M')
    x['WARRANTY_END_DATE'] = x['WARRANTY_END_DATE'].dt.to_period('M')
    x = x.groupby(['CUSTOMER_ID','MRN','WARRANTY_START_DATE','WARRANTY_END_DATE'], as_index=False).mean()

    df['WARRANTY_START_DATE'] = pd.to_datetime(df['WARRANTY_START_DATE']).dt.to_period('M')
    df['WARRANTY_END_DATE'] = pd.to_datetime(df['WARRANTY_END_DATE']).dt.to_period('M')

    mapp = {
        'E0001': 'IN TRANSIT to MiniMed',
        'E0002': 'Active',
        'E0003': 'Inactive',
        'E0004': 'Loaner',
        'E0005': 'Malfunctioning',
        'E0006': 'Pending Training',
        'E0007': 'Returned',
        'MAL':'Malfunctioning',
        'PEND': 'Pending Training',
        'INTR': 'IN TRANSIT to MiniMed',
        'ACT': 'Active',
        'RETU': 'Returned',
        'LOAN': 'Loaner',
        'INAC': 'Inactive'}

    for i in mapp.keys():
        df.loc[df['TRAINING_STATUS'] == i, 'TRAINING_STATUS'] = mapp[i]

    cols = ['CUSTOMER_ID','MRN','WARRANTY_START_DATE','WARRANTY_END_DATE','TRAINING_STATUS']

    # take sum of warranties
    temp = pd.get_dummies(df[cols], columns=['TRAINING_STATUS'], prefix='num_of')

    temp = temp.groupby(['CUSTOMER_ID','MRN','WARRANTY_START_DATE','WARRANTY_END_DATE'], as_index=False).sum()

    # merge average length of warranty with device status
    out = temp.merge(x, how='left', on=['CUSTOMER_ID','MRN','WARRANTY_START_DATE','WARRANTY_END_DATE'])

    # create a date range list and then explode it
    today = pd.to_datetime(datetime.strptime(str(pd.to_datetime("today"))[:7],'%Y-%m'))

    out['MONTH'] = out.apply(lambda row: pd.date_range(row['WARRANTY_START_DATE'].to_timestamp(),today,freq='MS',closed=None)
                                         if today <= row['WARRANTY_END_DATE'].to_timestamp() 
                                         else pd.date_range(row['WARRANTY_START_DATE'].to_timestamp(),row['WARRANTY_END_DATE'].to_timestamp(),freq='MS',closed=None),axis=1)
    out=out.dropna(subset=['MONTH'])       
    out = out.explode('MONTH')

    # people = list(out[['CUSTOMER_ID','MRN']].drop_duplicates().values)
    # months = {}  # Creating a dictionary with unique ID as key and the value is a list with start date and end date. 
    # for i in people:
    #     months[(i[0],i[1])] = [list(out.loc[(out['CUSTOMER_ID']==i[0]) & (out['MRN'] == i[1]), 'WARRANTY_START_DATE'].values), 
    #                            list(out.loc[(out['CUSTOMER_ID']==i[0]) & (out['MRN'] == i[1]), 'WARRANTY_END_DATE'].values)]

    # # Iterate and get start and end months of warranty 
    # # create a list of timestamps from start month to end month.
    # # create a list of lists where index 0 is the deidentified_id and index 1 is the timestamp. 
    # # These months are all months with active warranty. 
    # q = []
    # for i in people:
    #     start_month,end_month = months[(i[0],i[1])]
    #     for d_range in zip(start_month, end_month):  # for date range in iterator
    #         start = d_range[0].to_timestamp()
    #         end = d_range[1].to_timestamp()
    #         today = pd.to_datetime(datetime.strptime(str(pd.to_datetime("today"))[:7],'%Y-%m'))
    #         if today <= end:
    #             w = pd.date_range(start, today, freq='MS', closed=None)
    #         else:
    #             w = pd.date_range(start, end, freq='MS', closed=None)
    #         for mon in w:
    #             q.append([i[0],i[1],mon,start,end])

    # active = pd.DataFrame(q)
    # active.columns = ['CUSTOMER_ID','MRN','MONTH','WARRANTY_START_DATE','WARRANTY_END_DATE']

    # active['WARRANTY_START_DATE'] = active['WARRANTY_START_DATE'].dt.to_period('M')
    # active['WARRANTY_END_DATE'] = active['WARRANTY_END_DATE'].dt.to_period('M')
    # active = active.merge(out, how='left', on=['CUSTOMER_ID','MRN','WARRANTY_START_DATE','WARRANTY_END_DATE'])
    
    # active = active[['CUSTOMER_ID','MRN','MONTH','num_of_Active','LENGTH_OF_WARRANTY_IN_MONTHS']]

    # active = active.groupby(['CUSTOMER_ID','MRN','MONTH'], as_index=False).mean()
    # active = active.loc[active['num_of_Active'] > 0, :]
    # active['IS_WARRANTY_ACTIVE'] = 1
    # active.drop('num_of_Active', axis=1, inplace=True)
    # active['MONTH'] = active['MONTH'].dt.to_period('M')
    # active['MONTH'] = active['MONTH'].astype(str)

    # active['STAGE'] = stage

    out.drop(['WARRANTY_START_DATE', 'WARRANTY_END_DATE'], inplace=True, axis=1)
    
    out = out[['CUSTOMER_ID','MRN','MONTH','num_of_Active','LENGTH_OF_WARRANTY_IN_MONTHS']]

    out = out.groupby(['CUSTOMER_ID','MRN','MONTH'], as_index=False).mean()

    out = out.loc[out ['num_of_Active'] > 0, :]
    out['IS_WARRANTY_ACTIVE'] = 1
    out.drop('num_of_Active', axis=1, inplace=True)
    out['MONTH'] = out['MONTH'].dt.to_period('M')
    out['MONTH'] = out['MONTH'].astype(str)

    out['STAGE'] = stage

    # Create table to write to if it doesn't already exist
    tableName = "PA_WARRANTY_INTERMEDIATE_FEATURES"
    
    try:
        db.execute('common/sql_scripts/create_warranty_feature_int_table.sql',False,tableName)
        db.execute('common/sql_scripts/create_hist_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/insert_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/truncate_table_generic.sql',False,tableName)
        # write = db.write(active, tableName, fetched_time)
        write = db.write(out, tableName, fetched_time)
        db.conn.commit()
        db.cursor.close()
        db.conn.close() 
        print("finished writing warranty table")
        return write
    except Exception as e:
        db.conn.rollback()
        db.conn.close() 
        raise e