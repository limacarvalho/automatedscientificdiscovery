import ray
import pandas as pd
from utils import helper, config, rayer, kaggle_dataset_helper

from ml.models import Ensemble, list_base_models
from ml.xai.model import Explainable, list_xai_algos

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import traceback
import sys

# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
# from ml.models import common
from utils.asd_logging import logger as  customlogger
from pprint import pprint


from ml.xai.non_model import simulate_knockoffs, KnockoffSetting, list_fstats



def relevance(df, input_columns, target, options) -> None:
        
    # return
    ## base_model_scores
    
    threshold = None
    base_models = None
    n_trials = None
    max_depth = None
    boosted_round = None
    epochs = None
    max_n_estimators = None
    n_estimators = None
    n_neighbors = None
    
    
    attr_algos = ['IG', 'SHAP', 'GradientSHAP']
    knockoff_runs = 1000
    fdr=0.1
    fstats = ['lasso', 'ridge', 'randomforest']


    ensemble_n_estimators = None
    ensemble_n_trials = None    
    
    attr_algos = None    
    
    # list_ensemble_base_models = ['briskxgboost', 'slugxgboost', 'slugann', 'slugrf', 'slugknn', 'briskbagging']

#     list_fstats = ['lasso', 'ridge', 'randomforest']

    if 'threshold' in options:
        if options['threshold']:
            threshold = options['threshold']
            if (threshold < 0 ) or (threshold>1):
                raise ValueError("0 < threshold < 1")


    if 'base_models' in options:
        base_models = options['base_models']
        base_models = [x.lower() for x in base_models]
        result =  all(elem in list_base_models for elem in base_models)
        if not result:
            raise ValueError("Pls select base_models from: " + str(list_base_models))

    if 'n_trials' in options:
        n_trials = options['n_trials']
        if n_trials < 0:
            raise ValueError("n_trials > 0 required, you specified " + str(n_trials))

    if 'max_depth' in options:
        max_depth = options['max_depth']
        if max_depth < 0:
            raise ValueError("max_depth > 0 required, you specified " + str(max_depth))

    if 'boosted_round' in options:
        boosted_round = options['boosted_round']
        if boosted_round < 0:
            raise ValueError("boosted_round > 0 required, you specified " + str(boosted_round))


    if 'epochs' in options:
        epochs = options['epochs']
        if epochs < 0:
            raise ValueError("epochs > 0 required, you specified " + str(epochs))


    if 'max_n_estimators' in options:
        max_n_estimators = options['max_n_estimators']
        if max_n_estimators < 0:
            raise ValueError("max_n_estimators > 0 required, you specified " + str(max_n_estimators))            


    if 'n_estimators' in options:
        n_estimators = options['n_estimators']
        if n_estimators < 0:
            raise ValueError("n_estimators > 0 required, you specified " + str(n_estimators))



    if 'n_neighbors' in options:
        n_neighbors = options['n_neighbors']
        if n_neighbors < 0:
            raise ValueError("n_neighbors > 0 required, you specified " + str(n_neighbors))
            

            
    if 'ensemble_n_estimators' in options:
        ensemble_n_estimators = options['ensemble_n_estimators']
        if ensemble_n_estimators < 0:
            raise ValueError("ensemble_n_estimators > 0 required, you specified " + str(ensemble_n_estimators))            
            


    if 'ensemble_n_trials' in options:
        ensemble_n_trials = options['ensemble_n_trials']
        if ensemble_n_trials < 0:
            raise ValueError("ensemble_n_trials > 0 required, you specified " + str(ensemble_n_trials))            



    if 'attr_algos' in options:
        attr_algos = options['attr_algos']
        attr_algos = [x.lower() for x in attr_algos]
        result =  all(elem in list_xai_algos for elem in attr_algos)
        if not result:
            raise ValueError("Pls select attr_algos from: " + str(list_xai_algos))

        if 'knockoffs' in attr_algos:
            if 'fdr' in options:
                fdr = options['fdr']
                if (fdr < 0) or (fdr>1.0):
                    raise ValueError(" 0 < fdr < 1 is required")

            if 'fstats' in options:
                fstats = options['fstats']
                fstats = [x.lower() for x in fstats]
                result =  all(elem in list_fstats for elem in fstats)                
                if not result:
                    raise ValueError("Pls select fstats from: " + str(list_fstats))
    
            if 'knockoff_runs' in options:
                knockoff_runs = options['knockoff_runs']
                if knockoff_runs < 0:
                    raise ValueError("knockoff_runs > 0 required, you specified, we recommend knockoff_runs=20000 " + str(knockoff_runs))            



    if attr_algos is None:
        attr_algos = ['IG', 'SHAP', 'GradientSHAP']
        

    if base_models is None:
        base_models = list_base_models
        
    if n_trials is None:
        n_trials = 100

    if epochs is None:
        epochs = 150

    if max_depth is None:
        max_depth = 30

    if boosted_round is None:
        boosted_round = 100
                
    if max_n_estimators is None:
        max_n_estimators = 50
                        
    if n_estimators is None:
        n_estimators = 50

    if n_neighbors is None:
        n_neighbors = 50
                
        
    ### Ensemble param        
    if ensemble_n_estimators is None:
        ensemble_n_estimators = 50

    if ensemble_n_trials is None:
        ensemble_n_trials = 50
            

    
    customlogger.info('Params:')
    customlogger.info('threshold:' + str(threshold))
    customlogger.info('n_trials:' + str(n_trials))
    customlogger.info('max_depth:' + str(max_depth))
    customlogger.info('boosted_round:' + str(boosted_round))
    customlogger.info('epochs:' + str(epochs))
    customlogger.info('base_models:' + str(base_models))
    customlogger.info('attr_algos:' + str(attr_algos))
    
    
    try:                
        
        rayer.get_global_cluster(num_cpus=config.num_cpus)
        
        df_X = df[input_columns]
        df_y = df[target]
        df_knockoffs = None
        
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=config.rand_state)

        ss = StandardScaler()
        X_train_scalar = pd.DataFrame(ss.fit_transform(X_train), columns = X_train.columns)
        X_test_scalar = pd.DataFrame(ss.fit_transform(X_test), columns = X_test.columns)
                    

        if 'knockoffs' in attr_algos:
            df_knockoffs = simulate_knockoffs(fdr, fstats, knockoff_runs, df_X, df_y)
        

        # list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']

        ensemble_set = Ensemble(   
                                        xgb_objective='count:poisson',  # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]
                                        lgbm_objective='poisson',    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
                                        pred_class='regression',
                                        score_func=None,
                                        metric_func=None,
                                        list_base_models= base_models, #['sluglgbm', 'briskxgboost'],
                                        n_trials=10,          
                                        epochs=15,            
                                        boosted_round=10,     
                                        max_depth=30,         
                                        max_n_estimators=1500,
                                        n_estimators=30,      
                                        n_neighbors=30,

                                        ensemble_n_estimators=30,  ###  must be > 10
                                        ensemble_n_trials=10,
                                        timeout=None
                        )            
            

        customlogger.info('starting ray cluster')
        
        #!ray status --address='raycluster-autoscaler-head-svc.dev.svc.cluster.local:6379'

        
        customlogger.info('base model search started')
        ensemble_set.fetch_models(X_train_scalar, X_test_scalar, y_train, y_test, threshold=None)

        customlogger.info('base model scores' + str(ensemble_set.base_model_scores))
                
        
        customlogger.info('running xai on trained models')
        ex = Explainable(ensemble_set, df_X)   
        ex.get_attr(attr_algos)
        # json_scores = ex.df_scores.to_json(orient = 'columns')
        
        ret = {
            'base_model_scores': ensemble_set.base_model_scores,
            'score': ensemble_set.scores,
            'xai_model': ex.df_scores,
            'xai_non_model': df_knockoffs,
#            'raw': ex.raw
        }
        
    except Exception:
        err_trace = traceback.format_exc()
        customlogger.error(err_trace)
        
        ret = {
            'error':err_trace
        }
    
    finally:
        return ret
        
        