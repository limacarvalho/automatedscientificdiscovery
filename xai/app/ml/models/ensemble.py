
from utils import helper, config, logger
import os



import xgboost as xgb
from ml.models.base.slug_ann import SlugANN
from ml.models.base.slug_xgboost import SlugXGBoost
from ml.models.base.brisk_xgboost import BriskXGBoost


from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import numpy as np
import pandas as pd

import signal
import dask
import glob

import asyncio


### change the logger type to save info logs
customlogger = logger.logging.getLogger('console_info')



class BaseUnify:
    def __init__(self, base_models, callback_timer_expired=None, timer=7200) -> None:
        
        self.final_models_path = config.main_dir + '/ml/models/saved/final_models/'
        self.model_save_path = config.main_dir + config.project_name  + "/base/ann/slug"

        if timer < 0:
            timer = 3600

        self.timer = timer ### default is two hours
        self.ensemble_models = None
        self.__callback_timer_expired__ = callback_timer_expired
        
        self.scores = None
        self.df_pred_train = None
        self.df_pred_test = None
        
        self.best_base_model = None
        
        self.base_models = base_models
        self.ensemble_n_trials = 100
        self.scores = []        



    def fetch_models(self, retrain):
        
        self.scores = []
        lazy_results = []
        results = None

        
        #Sets an alarm in t seconds
        #If uncaught will terminate your process.        
        if self.__callback_timer_expired__ is not None:
            signal.signal(signal.SIGALRM, self.__callback_timer_expired__)
            signal.alarm(self.timer)         
        
        
        if not retrain:
            for base_model in self.base_models:
                res = __run_discoveries__(base_model, retrain)
                lazy_results.append(res)
            
            self.base_models = lazy_results
            

        else:
            customlogger.info('ensemble: base models discovery started')

            for base_model in self.base_models:
                res = dask.delayed(__run_discoveries__)(base_model, retrain)
                lazy_results.append(res)

            results = dask.compute(*lazy_results, scheduler='distributed')

            self.base_models = results

        
        for base_model in self.base_models:
            self.scores.append(base_model.score)
    
        result = helper.check_if_all_same(self.scores, None)            
        
        if result:
            customlogger.info('ensemble: no trained base models found')
            return None
        
        self.__get_best_base_model__()

                
    def fit_ensemble(self, retrain=False):
        
        brisk_xgb = BriskXGBoost()

        brisk_xgb.n_trials = self.ensemble_n_trials
        brisk_xgb.temp_path = config.main_dir + config.project_name + "/tmp/xgboost/brisk"
        brisk_xgb.model_save_path = config.main_dir + config.project_name + "/base/xgboost/brisk"

         
        brisk_xgb.X_train, brisk_xgb.X_test, brisk_xgb.y_train, brisk_xgb.y_test = self.__consolidate_predictions__()
        
        #brisk_xgb.X_train = slug_xgboost.df_pred_train.loc[:, slug_xgboost.df_pred_train.columns.values != 'y']
        #brisk_xgb.X_test = slug_xgboost.df_pred_test.loc[:, slug_xgboost.df_pred_test.columns.values != 'y']
        #brisk_xgb.y_train = slug_xgboost.df_pred_train['y']
        #brisk_xgb.y_test = slug_xgboost.df_pred_test['y']

        if not retrain:
            brisk_xgb.get_model_score()
            # brisk_xgb.__load_best_fit__()
            if brisk_xgb.best_fit is None:
                customlogger.info("No BaseUnify model found, please run 'fit_ensemble(retrain=True)'")
                return None
        
        else:    
            customlogger.info("fitting ensemble model using brisk xgboost")              
            brisk_xgb.fetch_model(retrain=True)
            
                
        sum_scores_ensemble_model = sum(brisk_xgb.score)
        sum_scores_base_models = max([sum(i) for i in self.scores])

        customlogger.info("Best score from base models: %s compared to ensemble score:%s",sum_scores_base_models, sum_scores_ensemble_model)                      

        # ensemble model has worst score then any of the final_models, choose the final model as ensemble
        if sum_scores_ensemble_model > sum_scores_base_models:
            self.ensemble_model = brisk_xgb
            customlogger.info("Ensemble selected")  
        else:
            customlogger.info(f'No improvements with Ensemble')
            config.clear_temp_folder(brisk_xgb.model_save_path)
            self.ensemble_model = None                
                

    def __consolidate_predictions__(self):
        slug_xgboost = self.__get_base_model__('SlugXGBoost')
        
        list_preds_train = []
        list_preds_test = []
        for base_model in self.base_models:
            pred_train = base_model.predict(slug_xgboost.X_train)
            pred_test = base_model.predict(slug_xgboost.X_test)
            list_preds_train.append(pred_train)
            list_preds_test.append(pred_test)

        
        df_train = pd.DataFrame(list_preds_train)
        df_train = df_train.T

        df_test = pd.DataFrame(list_preds_test)
        df_test = df_test.T
        
        return df_train, df_test, slug_xgboost.y_train, slug_xgboost.y_test
        
                
                
    def __get_base_model__(self, base_model_name):
        for base_model in self.base_models:
            if base_model_name in str(base_model):
                return base_model
                

    def __get_best_base_model__(self):        
        best_model_idx = np.argmax([sum(i) for i in self.scores])
        self.best_base_model = self.base_models[best_model_idx]

        
                
    def predict(self, df_X):        
        list_preds = []
        if self.ensemble_model is None:
            return self.best_base_model.predict(df_X)
        else:
            for base_model in self.base_models:
                pred = base_model.predic(df_X)
                list_preds.append(pred)
            
            df_pred = pd.DataFrame(list_preds)
            df_pred = df_pred.T
            return self.ensemble_model.predict(df_pred)

                
                
### not a class method, throws error when made otherwise
def __run_discoveries__(base_model, retrain):

    base_model.fetch_model(retrain)
    # base_model.get_model_score()
    return base_model
