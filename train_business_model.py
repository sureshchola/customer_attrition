import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_auc_score
from dateutil.rrule import rrule, MONTHLY
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
import math
import argparse
import pandas_profiling
from hyperopt import fmin, tpe, STATUS_OK, hp, Trials, space_eval
import json
import sys
from sagemaker_containers.beta.framework import worker
import common.utils.database as sc
import common.scripts.fulldata as fd
import common.utils.notifications as ntf
import common.utils.config_checker as cc
import traceback

def main(args):

    """
    main training script
    
    args:
    args - Sagemaker args
    
    returns:
    write - None
    """

    db = None

    print('Starting the training.')

    try:
        print("reading in config")
        
        config = cc.check("common/config.yaml")
        print(config)

        env_vars = json.loads(args.hyperparameters)

        db_secrets_name = env_vars['secret_name']
        region = env_vars['region']
        account = env_vars['account']
        env = env_vars['env']

        target = config['train']['target']
        split_mode = config['train']['split']['mode']
        sampling = config['train']['sampling']
        profiling = config['train']['profiling']
        target = config['train']['target']
        drop = config['train']['drop']
        start_month = config['train']['start_month']

        forecast_months = config['inference']['num_month_forecast']
        target = config['train']['target']
        dummy_cols = config['train']['dummy_cols']
        features = config['train']['features']
        rename_cols = config['train']['rename_cols']

        eta_min = config['train']['hyperparameters']['eta']['min']                   
        depth_min = config['train']['hyperparameters']['max_depth']['min']
        depth_max = config['train']['hyperparameters']['max_depth']['max']                   
        gamma_min = config['train']['hyperparameters']['gamma']['min']
        gamma_max = config['train']['hyperparameters']['gamma']['max']
        alpha_min = config['train']['hyperparameters']['reg_alpha']['min']
        alpha_max = config['train']['hyperparameters']['reg_alpha']['max']
        weight_min = config['train']['hyperparameters']['min_child_weight']['min']
        weight_max = config['train']['hyperparameters']['min_child_weight']['max']
        subsample_min = config['train']['hyperparameters']['subsample']['min'] 
        subsample_max = config['train']['hyperparameters']['subsample']['max']

        optimize = config['train']['hyperparameters']['optimize']

        num_tuning_rounds = config['train']['hyperparameters']['num_tuning_rounds']

        stage = 'train'

        db = sc.Database(db_secrets_name)
        db.connect()

        fd.full_features(db_secrets_name,dummy_cols,features,rename_cols,forecast_months,start_month,stage)

        data, fetched_time = db.execute('common/sql_scripts/fetch_scoring_data.sql', read=True)
        data.drop(columns=['INGESTED_DATE','UPDATED_DATE'],inplace=True)

        if data.shape[0] == 0:
            raise Exception("empty data for model")

        print("splitting data")
        if split_mode=='last_n_months':

            max_month = config['train']['max_month']
            split_months = config['train']['split']['months']
            
            until=datetime.strptime(max_month, '%Y-%m')
            dtstart = until+relativedelta(months=-(split_months-1))

            test_months = [str(dt)[:7] for dt in rrule(MONTHLY, dtstart=dtstart, until=until)]  

        elif split_mode=='randomized':

            perc_test = config['train']['split']['perc_test']

            months = data['MONTH'].unique().tolist()

            test_months = random.choice(months,math.floor(len(months)*perc_test))
            
        print("test months")
        print(test_months)

        test = data[data['MONTH'].isin(test_months)].reset_index(drop=True)
        train = data[data['MONTH']<test_months[0]].reset_index(drop=True)

        # profile train and test datasets
        if profiling:

            print("profiling train dataset")
            train_profile = pandas_profiling.ProfileReport(train, correlations = None, pool_size = 1,interactions=None)
            print("profiling test dataset")
            test_profile = pandas_profiling.ProfileReport(test, correlations = None, pool_size = 1,interactions=None)

            print("writing data profiling results")
            train_profile.to_file(os.path.join(args.model_dir, f"train_profile_{datetime.today().strftime('%Y%m%d')}"))
            test_profile.to_file(os.path.join(args.model_dir, f"test_profile_{datetime.today().strftime('%Y%m%d')}"))

        # sampling train set 
        pos = train[train[target]==1]
        print("positive num rows")
        print(pos.shape[0])
        neg = train[train[target]==0]
        print("negative num rows")
        print(neg.shape[0])

        if sampling=='down':
            if pos.shape[0]<neg.shape[0]:
                train = pd.concat([pos,neg.sample(pos.shape[0],replace=False)])
            elif pos.shape[0]>neg.shape[0]:
                train = pd.concat([neg,pos.sample(neg.shape[0],replace=False)])
        elif sampling=='up':
            if pos.shape[0]<neg.shape[0]:
                train = pd.concat([neg,pos.sample(neg.shape[0],replace=True)])
            elif pos.shape[0]>neg.shape[0]:
                train = pd.concat([pos,neg.sample(pos.shape[0],replace=True)])

        # write train and test set to snowflake
        tableName = "PA_TRAIN"
        db.execute('common/sql_scripts/create_traintest.sql',False,tableName)
        db.execute('common/sql_scripts/create_hist_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/insert_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/truncate_table_generic.sql',False,tableName)
        print("finished writing train table")
        db.write(train,tableName,fetched_time)
        tableName = "PA_TEST"
        db.execute('common/sql_scripts/create_traintest.sql',False,tableName)
        db.execute('common/sql_scripts/create_hist_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/insert_table_generic.sql',False,tableName)
        db.execute('common/sql_scripts/truncate_table_generic.sql',False,tableName)
        print("finished writing test table")
        db.write(test,tableName,fetched_time)
        db.conn.commit()
        db.cursor.close()
        db.conn.close() 

        # dropping the identifier columns
        print("dropping identifier columns")
        print(drop)
        test = test.drop(columns=drop)
        train = train.drop(columns=drop)

        # target and features
        X_train = train.drop([target],axis=1)
        X_test = test.drop([target],axis=1)
        y_train = train[target]
        y_test = test[target]

        # convert to dmatrix format
        dtrain = xgb.DMatrix(X_train.values, y_train.values)
        dval = xgb.DMatrix(X_test.values, y_test.values)

        # hyperparameter tuning
        print("starting hyperparameter tuning")
        space={
                'objective':'binary:logistic',
                'num_round': 100,    
                'eta': hp.choice('eta', np.array([10**i for i in range(eta_min,1)])),                      
                'max_depth': hp.choice('max_depth', np.arange(depth_min,depth_max+1,1)),                
                'gamma': hp.uniform('gamma', gamma_min,gamma_max),  
                'reg_alpha': hp.uniform('reg_alpha', alpha_min,alpha_max),
                'min_child_weight': hp.uniform('min_child_weight', weight_min,weight_max),
                'subsample':hp.uniform('subsample', subsample_min,subsample_max)}

        print(space)

        def metrics(preds, dtrain):
            y_true = dtrain.get_label()
            y_preds = np.where(preds >= 0, 1, 0)
            f1 = float(f1_score(y_true,y_preds,zero_division=0))
            auc = float(roc_auc_score(y_true,y_preds))
            accuracy = float(accuracy_score(y_true,y_preds))
            precision = float(precision_score(y_true,y_preds,zero_division=0))
            recall = float(recall_score(y_true,y_preds,zero_division=0))
            return [('auc',auc),('f1',f1),('accuracy',accuracy),('precision',precision),('recall',recall)]

        def score(params):
            print("Training with params: ")
            print(params)
            
            evals_result = {}
            model = xgb.train(params=params,
                            dtrain=dtrain,
                            feval=metrics,
                            evals=[(dtrain, 'train'), (dval, 'validation')],
                            num_boost_round=params['num_round'],
                            verbose_eval = False,
                            evals_result = evals_result)

            if optimize=='auc':
                score = evals_result['validation']['auc'][-1]
            elif optimize=='f1':
                score = evals_result['validation']['f1'][-1]
            elif optimize=='accuracy':
                score = evals_result['validation']['accuracy'][-1]

            val_results = {k:v[-1] for k,v in evals_result['validation'].items()}

            return {'loss': -score, 'status': STATUS_OK,'model':model,'params':params,'metrics': val_results}

        trials = Trials()
        best_hp = fmin(score, space, algo=tpe.suggest, max_evals=num_tuning_rounds,trials=trials)
        print("best hyperparameters")
        print(space_eval(space, best_hp))

        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        
        # write metrics and params for all trials
        print("writing trial results")
        with open(os.path.join(args.model_dir, "trials.json"), 'w') as f:
            for trial in trials.trials:
                result_dict = {}
                result_dict['trial_id'] = trial['tid']
                result_dict['objective_value'] = -trial['result']['loss']
                result_dict['params'] = {k:(v.item() if isinstance(v,np.generic) else v) for k,v in trial['result']['params'].items()}
                result_dict['metrics'] = trial['result']['metrics']
                json.dump(result_dict, f, indent=4)

        # save best model
        print("saving model")
        trained_model_path = os.path.join(args.model_dir, 'attrition.model')
        best_model.save_model(trained_model_path)

        # save confusion matrix for best model
        print("saving confusion matrix")
        predictions = best_model.predict(dval)

        conf_matrix = pd.crosstab(index=y_test, columns=np.round(predictions), rownames=['actuals'], colnames=['predictions'])
        print(conf_matrix)
        conf_matrix.to_html(os.path.join(args.model_dir, "conf_matrix.html"))

        ntf.job_status('train business model','succeeded',account,region,env)

    except Exception as e:
        if db:
            db.conn.rollback()
            db.conn.close() 

        ntf.job_status('train business model','failed',account,region,env,traceback.format_exc())
        raise e

def model_fn(model_dir):

    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "attrition.model"))

    return model

def input_fn(input, content_type):

    print(f"memory size of payload is {sys.getsizeof(input)}")

    data = np.asarray(json.loads(input))

    print("succesfully loaded json to numpy")
    print(f"Invoked with {data.shape[0]} records")

    return data

def predict_fn(input, model):

    predictions = model.predict(xgb.DMatrix(input))

    return predictions

def output_fn(prediction, accept):

    return worker.Response(json.dumps(prediction.tolist()), accept, mimetype= "application/json")

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    parser.add_argument('--hyperparameters', type=str, default=os.environ.get('SM_HPS'))

    args, _ = parser.parse_known_args()

    main(args)