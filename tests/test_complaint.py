import common.scripts.complaint as complaint
import common.utils.database as sc
import boto3

ssm_client = boto3.client('ssm', region_name='us-east-1')
secret_name = ssm_client.get_parameter(Name='secret_name')['Parameter']['Value']

db = sc.Database(secret_name)

try:
    db.connect()
    pre_df,pre_time = db.execute("select * from {}",True,"PA_COMPLAINTS_INTERMEDIATE_FEATURES")
    min_time = pre_df['INGESTED_DATE'].min()
    max_time = pre_df['INGESTED_DATE'].max()
    complaint.complaint_features(secret_name,"TEST")
    curr_df,curr_time = db.execute("select * from {}",True,"PA_COMPLAINTS_INTERMEDIATE_FEATURES")
    post_df,post_time = db.execute(f"select count(*) as COUNT from PA_COMPLAINTS_INTERMEDIATE_FEATURES_HIST where INGESTED_DATE between '{min_time}' and '{max_time}'",True)
    db.conn.commit()
    db.cursor.close()
    db.conn.close() 
except Exception as e:
    db.conn.rollback()
    db.conn.close() 
    raise e

def test_nonull():
    assert curr_df.isnull().values.any() == False

def test_nonempty():
    assert curr_df.shape[0]>0

def test_unique():
    assert curr_df.groupby(['CUSTOMER_ID','MRN','MONTH']).size().shape[0] == curr_df.shape[0]

def test_hist():
    assert pre_df.shape[0] == post_df['COUNT'].iloc[0]

def test_reg_lookback_min():
    assert curr_df['NUM_REGULAR_COMPLAINT_LOOKBACK_18M_SCORE'].min()>=0

def test_harm_lookback_min():
    assert curr_df['NUM_HARM_FLAG_SUM_LOOKBACK_18M_SCORE'].min()>=0




