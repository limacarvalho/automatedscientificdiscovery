
import pandas as pd

from asd.relevance.utils import config
from asd.relevance.ml.models import Ensemble, list_base_models
from asd.relevance.ml.xai.model import Explainable, list_xai_algos
from asd.relevance.ml.xai.non_model import simulate_knockoffs, list_fstats
from asd.relevance.utils.asd_logging import logger as  customlogger


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import traceback


from pprint import pprint



def relevance(df, input_columns, target, options) -> None:
    '''
    The goal of this function to measure measure variable relevance using knockoffs and explainable AI methods such as SHAP.
    :param df: pandas dataframe
    :param input_columns: data columns to consider for variable relevance 
    :param target: target column 
    :param options: dict
        'xgb_objective': str
             objective function if xgboost model is given in list_base_model. I.e., default='binary:logistic', https://xgboost.readthedocs.io/en/stable/parameter.html.
        'lgbm_objective': str
            objective function if lightgbm model is given in list_base_model.I.e., default='binary', https://lightgbm.readthedocs.io/en/latest/Parameters.html.
        'pred_class': str
            specify problem type, i.e., 'regression' or 'classification'.
        'score_func': str, callable, or None 
            A single string or a callable to evaluate the predictions on the test set, i.e., r2_score, log_loss, etc., See https://scikit-learn.org/stable/modules/model_evaluation.html
        'metric_func': callable, or None. Default for regression is r2_score and for classification is f1_score
            sklearn.metrics function
        'list_base_models' : list of base models to be used to fit on the data. 
            ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']
        'n_trials': int,  default=100
            Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.  
        'boosted_round': int, default=100
            n_estimators parameter for XGBoost and LightGBM.
        'max_depth' : int, default=30
            max tree depth parameter for XGBoost, LightGBM and RandomForest. 
        'rf_n_estimators': int, default=1500
            n_estimators parameter of RandomForest.
        'bagging_estimators': int, default=100
        'n_neighbors': int, n_neighbors of KNN, default=30
        'cv_splits': int, default=3
            Determines the cross-validation splitting strategy.
        'ensemble_bagging_estimators': int, default=50
            n_estimators parameter of Bagging. This is the second baggin method which is used an an ensemble on top of base estimators. 
        'ensemble_n_trials': int, default=50
            Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.
        'attr_algos' : list of xai methods
            ['IG', 'SHAP', 'GradientSHAP', 'knockoffs']
        'fdr': float, default=0.1
            target false discovery rate, 
        'fstats': list of methods to calculate fstats
            ['lasso', 'ridge', 'randomforest']
        'knockoff_runs': int, default=20000
            no. of reruns for each knockoff setting
        :return: dict, dict
            'base_model_scores': list,
                train test score of bases models
            'score': list,
                train test score of ensemble fit
            'xai_model': pandas DataFrame,
                list of cols sorted with respect to their relevance as measured by model-based xai methods.
            'xai_non_model': pandas DataFrame,
                list of cols sorted with respect to their relevance as measured by non model-based knockoff framework method.
    ''' 
    

    base_models = None
    xgb_objective = None
    lgbm_objective = None
    pred_class = None
    score_func = None
    metric_func= None

    n_trials = None
    boosted_round = None
    max_depth = None
    rf_n_estimators = None
    bagging_estimators = None
    n_neighbors = None
    cv_splits = None
    ensemble_bagging_estimators=None
    ensemble_n_trials=None
    
    knockoff_runs = None
    fdr=None
    fstats = None

    
    attr_algos = None
    

    
    if 'base_models' in options:        
        if options['base_models']:
            base_models = options['base_models']
            base_models = [x.lower() for x in base_models]
            result =  all(elem in list_base_models for elem in base_models)
            if not result:
                raise ValueError("Pls select base_models from: " + str(list_base_models))
        else:
            raise ValueError("Pls select base_models from: " + str(list_base_models))

    if base_models:
        if 'pred_class' in options:
            if options['pred_class']:
                if options['pred_class'] not in ['regression', 'classification']:
                    raise ValueError("Pls specify correct prediction class, i.e., ['regression', 'classification']")
                else:
                    pred_class = options['pred_class']
            else:
                raise ValueError("Pls specify correct prediction class, i.e., ['regression', 'classification']")            
        else:
            raise ValueError("Pls specify correct prediction class, i.e., ['regression', 'classification']")            


        if ('briskxgboost' in base_models) or ('slugxgboost' in base_models):
            if 'xgb_objective' in options:
                if options['xgb_objective']:
                    xgb_objective = options['xgb_objective']
                else:
                    raise ValueError("Pls select objective function for xgboost, i.e., xgb_objective='reg:squarederror', https://xgboost.readthedocs.io/en/stable/parameter.html")
            else:
                raise ValueError("Pls select objective function for xgboost, i.e., xgb_objective='reg:squarederror', https://xgboost.readthedocs.io/en/stable/parameter.html")

        if 'sluglgbm' in base_models:
            if 'lgbm_objective' in options:
                if options['lgbm_objective']:
                    lgbm_objective = options['lgbm_objective']
                else:
                    raise ValueError("Pls select objective function for xgboost, i.e., lgbm_objective='regression_l1', https://lightgbm.readthedocs.io/en/latest/Parameters.html")
            else:
                raise ValueError("Pls select objective function for xgboost, i.e., lgbm_objective='regression_l1', https://lightgbm.readthedocs.io/en/latest/Parameters.html")


        if 'score_func' in options:
            if options['score_func']:
                score_func = options['score_func']
            else:
                raise ValueError("Pls specify a scorer, i.e., r2_score/'r2_', log_loss/'neg_log_loss', etc., See https://scikit-learn.org/stable/modules/model_evaluation.html")
        else:
            raise ValueError("Pls specify a scorer, i.e., r2_score/'r2_', log_loss/'neg_log_loss', etc., See https://scikit-learn.org/stable/modules/model_evaluation.html")


        if 'metric_func' in options:
            if options['metric_func']:
                metric_func = options['metric_func']
    

        if 'n_trials' in options:
            n_trials = options['n_trials']
            if n_trials < 0:
                raise ValueError("n_trials > 0 required, you specified: " + str(n_trials))
        else:
            n_trials = 100

        if 'boosted_round' in options:
            boosted_round = options['boosted_round']
            if boosted_round < 0:
                raise ValueError("boosted_round > 0 required, you specified: " + str(boosted_round))
        else:
            boosted_round = 100


        if 'max_depth' in options:
            max_depth = options['max_depth']
            if max_depth < 3:
                raise ValueError("max_depth > 3 required, you specified: " + str(max_depth))
        else:
            max_depth = 10


        if 'rf_n_estimators' in options:
            rf_n_estimators = options['rf_n_estimators']
            if rf_n_estimators < 51:
                raise ValueError("rf_n_estimators > 50 required, you specified: " + str(rf_n_estimators))            
        else:
            rf_n_estimators = 1500

        if 'bagging_estimators' in options:
            bagging_estimators = options['bagging_estimators']
            if bagging_estimators < 11:
                raise ValueError("bagging_estimators > 0 required, you specified: " + str(bagging_estimators))
        else:
            bagging_estimators = 100


        if 'n_neighbors' in options:
            n_neighbors = options['n_neighbors']
            if n_neighbors < 6:
                raise ValueError("n_neighbors > 5 required, you specified: " + str(n_neighbors))
        else:
            n_neighbors = 30
            

        if 'cv_splits' in options:
            cv_splits = options['cv_splits']
            if (cv_splits < 0) or (cv_splits > 5):
                raise ValueError("0 < cv_splits < 5 required, you specified: " + str(cv_splits))
        else:
            cv_splits = 3


        if 'ensemble_bagging_estimators' in options:
            ensemble_bagging_estimators = options['ensemble_bagging_estimators']
            if ensemble_bagging_estimators < 11:
                raise ValueError("ensemble_bagging_estimators > 0 required, you specified: " + str(ensemble_bagging_estimators))            
        else:
            ensemble_bagging_estimators = 50
            

        if 'ensemble_n_trials' in options:
            ensemble_n_trials = options['ensemble_n_trials']
            if ensemble_n_trials < 0:
                raise ValueError("ensemble_n_trials > 0 required, you specified: " + str(ensemble_n_trials))            
        else:
            ensemble_n_trials = 50



    if 'attr_algos' in options:
        attr_algos = options['attr_algos']
        if not attr_algos:
            raise ValueError("Pls select atleast one of the following methods in attr_algos:" + str(list_xai_algos))
        else:
            attr_algos = [x.lower() for x in attr_algos]
            result =  all(elem in list_xai_algos for elem in attr_algos)
            if not result:
                raise ValueError("Pls select attr_algos from: " + str(list_xai_algos))

            if 'knockoffs' in attr_algos:
                if 'fdr' in options:
                    fdr = options['fdr']
                    if (fdr < 0) or (fdr>1.0):
                        raise ValueError(" 0 < fdr < 1 is required")
                else:
                    fdr=0.1

                if 'fstats' in options:
                    fstats = options['fstats']
                    if fstats:
                        fstats = [x.lower() for x in fstats]
                        result =  all(elem in list_fstats for elem in fstats)                
                        if not result:
                            raise ValueError("Pls select fstats from: " + str(list_fstats))
                    else:
                        fstats = list_fstats
                else:
                    fstats = list_fstats                            
        
                if 'knockoff_runs' in options:
                    if options['knockoff_runs']:
                        knockoff_runs = options['knockoff_runs']
                        if knockoff_runs < 0:
                            raise ValueError("knockoff_runs > 0 required, you specified, we recommend knockoff_runs=20000 " + str(knockoff_runs))
                    else:
                        knockoff_runs = 20000    
                else:
                    knockoff_runs = 20000
        
    else:
        raise ValueError("Pls select atleast one of the following methods in attr_algos:" + str(list_xai_algos))
    

    try:                
                
        df_X = df[input_columns]
        df_y = df[target]
        df_knockoffs = None
        
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=config.rand_state)

        ss = StandardScaler()
        X_train_scalar = pd.DataFrame(ss.fit_transform(X_train), columns = X_train.columns)
        X_test_scalar = pd.DataFrame(ss.fit_transform(X_test), columns = X_test.columns)
                    

        if 'knockoffs' in attr_algos:
            customlogger.info('Running knockoffs with the following setting:')
            customlogger.info('     knockoff_runs:' + str(knockoff_runs))
            customlogger.info('     fdr:' + str(fdr))
            customlogger.info('     fstats:' + str(fstats))
            df_knockoffs = simulate_knockoffs(fdr, fstats, knockoff_runs, df_X, df_y)
        
        ### only non-model based xai method was selected so the rest of the code is not to be executed.        
        if base_models is None:
            ret = {
                'base_model_scores': None,
                'score': None,
                'xai_model': None,
                'xai_non_model': df_knockoffs,
            }
            return ret

        customlogger.info('Running Ensemble Modeling with following setting:')
        customlogger.info('     base models: ' + str(base_models))
        customlogger.info('     pred_class: ' + str(pred_class))
        customlogger.info('     xgb_objective: ' + str(xgb_objective))
        customlogger.info('     lgbm_objective:' + str(lgbm_objective))
        customlogger.info('     score_func: ' + str(score_func))
        customlogger.info('     metric_func: ' + str(metric_func))
        customlogger.info('     n_trials: ' + str(n_trials))
        customlogger.info('     boosted_round: ' + str(boosted_round))
        customlogger.info('     max_depth: ' + str(max_depth))
        customlogger.info('     rf_n_estimators: ' + str(rf_n_estimators))
        customlogger.info('     bagging_estimators: ' + str(bagging_estimators))
        customlogger.info('     n_neighbors: ' + str(n_neighbors))
        customlogger.info('     cv_splits: ' + str(cv_splits))
        customlogger.info('     ensemble_bagging_estimators: ' + str(ensemble_bagging_estimators))
        customlogger.info('     ensemble_n_trials: ' + str(ensemble_n_trials))
        

        ensemble_set = Ensemble(   
                                        xgb_objective=xgb_objective,    
                                        lgbm_objective=lgbm_objective,  
                                        pred_class=pred_class,
                                        score_func=score_func,
                                        metric_func=metric_func,
                                        list_base_models= base_models, 
                                        n_trials=n_trials,          
                                        boosted_round=boosted_round,     
                                        max_depth=max_depth, 
                                        rf_n_estimators=rf_n_estimators,
                                        bagging_estimators=bagging_estimators,      
                                        n_neighbors=n_neighbors,                                     #must be > 5
                                        cv_splits=cv_splits,
                                        ensemble_bagging_estimators=ensemble_bagging_estimators,     #must be > 10
                                        ensemble_n_trials=50,
                                        timeout=None
                        )            
            
        
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
        }
        
    except Exception:
        err_trace = traceback.format_exc()
        customlogger.error(err_trace)
        
        ret = {
            'error':err_trace
        }
    
    finally:
        return ret
        
        