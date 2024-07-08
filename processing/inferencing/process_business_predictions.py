import os
import yaml
import common.scripts.fulldata as fd
import common.scripts.score as score
import common.utils.notifications as ntf
import common.utils.config_checker as cc
import traceback

def handler():

    """
    main inferencing script
    
    args:
    None
    
    returns:
    dictionary of all table stats
    """

    config = cc.check("common/config.yaml")

#     db_secrets_name = os.environ['secret_name']
#     region = os.environ['region']
#     account = os.environ['account']
#     endpoint_name = os.environ['endpoint_name']
#     env = os.environ['ENVIRONMENT']
    
    db = None

    db_secrets_name = 'general'
    region_name = 'us-east-1'
    account = '740984199597'
    env = 'dev'
    endpoint_name = 'attrition-model'
    bucket='inference-outputs'

    profile = config['inference']['profiling']
    drop_cols = config['train']['drop']
    attrit_months = config['inference']['min_history_months']
    forecast_months = config['inference']['num_month_forecast']
    target = config['train']['target']
    stage='inference'

    try:
        # FULL_FEATURES = fd.full_features(db_secrets_name,dummy_cols,features,rename_cols,forecast_months,start_month,stage)
        SCORES = score.score_individuals(db_secrets_name,endpoint_name,profile,drop_cols,target,attrit_months,forecast_months,stage,region_name,bucket,env)

#         ntf.job_status('inference business predictions','succeeded',account,region,env)

        # Return a dict of dicts that contains write output information from each individual function
        # return {
        #         'Demo': FULL_FEATURES['DEMO'],
        #         'Touchpoint': FULL_FEATURES['TOUCHPOINT'],
        #         'Startright': FULL_FEATURES['STARTRIGHT'],
        #         'Complaint': FULL_FEATURES['COMPLAINT'],
        #         'Sales': FULL_FEATURES['SALES'],
        #         'Warranty': FULL_FEATURES['WARRANTY'],
        #         'Modeling Dataset': FULL_FEATURES['MERGE'],
        #         'Scored': SCORES
        #         }
    except Exception as e:
#         ntf.job_status('inference business predictions','failed',account,region,env,traceback.format_exc())
        raise e    
    
if __name__=='__main__':

    handler()