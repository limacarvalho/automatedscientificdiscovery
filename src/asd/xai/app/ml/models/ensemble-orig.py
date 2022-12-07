
from utils import helper, config
import os
import logging


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



class Ensemble:
    def __init__(self, list_models, callback_timer_expired=None, timer=7200) -> None:
        
        self.final_models_path = config.main_dir + '/ml/models/saved/final_models/'
        self.model_save_path = config.main_dir + config.project_name  + "/base/ann/slug"

        if timer < 0:
            timer = 3600

        self.timer = timer ### default is two hours
        self.base_models = list_models
        self.ensemble_models = None
        self.__callback_timer_expired__ = callback_timer_expired
        self.df_pred_train = None
        self.df_pred_test = None
        self.scores = None
        self.df_pred_train = None
        self.df_pred_test = None
        
        self.ensemble_model = None
        self.best_fit = None
        
        config.create_project_dirs()



    def fetch_models(self):
        
        # print(f'Cleared previous models at {self.final_models_path}')
        # config.clear_temp_folder(self.final_models_path)

        #Sets an alarm in t seconds
        #If uncaught will terminate your process.        
        if self.__callback_timer_expired__ is not None:
            signal.signal(signal.SIGALRM, self.__callback_timer_expired__)
            signal.alarm(self.timer) 

        lazy_results = []
        results = None

        try:
            for base_model in self.base_models:
                print(f'fetching {type(base_model)} models')
                res = dask.delayed(__run_discoveries__)(base_model)
                lazy_results.append(res)

            results = dask.compute(*lazy_results, scheduler='distributed')


        except Exception as e:
            customlogger.info('Timer Expired Exception')            

        except KeyboardInterrupt as e:
            print('thread interrupt outside of a task')

        finally:
            # self.load_model_scores()
            print('Ensemble training completed')
            



    ### load each trained model along with its metric score and predcitions 
    def load_model_scores(self, score_func=None):
        scores = []
        
        best_models = []
        for base_model in self.base_models:
            score = base_model.get_model_score(score_func)
            scores.append(score)
            best_models.append(base_model.best_fit)

        self.scores = scores
    
        if self.scores is None:
            print(f'No model selected, please reduce the acceptance threshold')
            return None


        ### Specify the best model from the list of finally selected models
        ### we always take max here since our loss functions are either r2 or f1. Additional 'if' condition can be added to see as to take max or min. 
        best_model_idx = np.argmax([sum(i) for i in self.scores])
        self.best_fit = best_models[best_model_idx]
            
            
        df_tmp_train = pd.DataFrame(list_preds_train)
        self.df_pred_train = df_tmp_train.T
        self.df_pred_train['y'] = y_train.values

        df_tmp_test = pd.DataFrame(list_preds_test)
        self.df_pred_test = df_tmp_test.T
        self.df_pred_test['y'] = y_test.values
        
    
    def fit_ensemble(self, retrain=False):
        
        brisk_xgb = BriskXGBoost()

        brisk_xgb.temp_path = config.main_dir + config.project_name + "/tmp/xgboost/brisk"
        brisk_xgb.model_save_path = config.main_dir + config.project_name + "/base/xgboost/brisk"

        brisk_xgb.X_train = self.df_pred_train.loc[:, self.df_pred_train.columns.values != 'y']
        brisk_xgb.X_test = self.df_pred_test.loc[:, self.df_pred_test.columns.values != 'y']
        brisk_xgb.y_train = self.df_pred_train['y']
        brisk_xgb.y_test = self.df_pred_test['y']


        if retrain is False:
            model_files = glob.glob(brisk_xgb.model_save_path+'/*')
            # model_count = brisk_xgb.()
            if model_files == 0:                
                print(f'No pretrained models found. Please execute the "fetch_models".')
                self.ensemble_model = None
                return None
            else:
                print(f'loading pretrained ensemble model.')
                brisk_xgb.get_model_score()
        else:    
            print(f'fitting ensemble model using brisk xgboost')    
            brisk_xgb.n_trails = 100    

            brisk_xgb.fetch_model()
            brisk_xgb.get_model_score()
    
        sum_scores_ensemble_model = sum(brisk_xgb.score)
        sum_scores_base_models = max([sum(i) for i in self.scores])


        # ensemble model has worst score then any of the final_models, choose the final model as ensemble
        if sum_scores_ensemble_model > sum_scores_base_models:
            self.ensemble_model = brisk_xgb.model
            print(f'Ensemble improved r2-score from {sum_scores_base_models} to {sum_scores_ensemble_model}')
        else:
            print(f'No improvements with Ensemble')
            config.clear_temp_folder(brisk_xgb.model_save_path)
            self.ensemble_model = None

    
    ### 
    def predict(self, df_X):            
        ddf_X = xgb.DMatrix(df_X)
        df_X_tensor = helper.df_to_tensor(df_X)
        list_preds = []

        if self.ensemble_model is None:
            return self.best_fit.predict(ddf_X)
        else:
            for model in self.base_models:
                if 'SlugANN' in str(model):
                    pred = model.model(df_X_tensor)
                    pred = helper.torch_tensor_to_numpy(pred)
                    pred = pred.reshape(pred.shape[0], )        
                    list_preds.append(pred)
                if 'SlugXGBoost' in str(model):
                    pred = model.model.predict(ddf_X)
                    list_preds.append(pred)


            df_pred = pd.DataFrame(list_preds)
            df_pred = df_pred.T
            ddf_pred = xgb.DMatrix(df_pred)
            return self.ensemble_model.predict(ddf_pred)




### not a class method, throws error when made otherwise
def __run_discoveries__(base_model, retrain=True):
    
    base_model.fetch_model(retrain)
    
    return base_model
    # print(model)
