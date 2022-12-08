from .base_model import BaseModel
from ml.models import common
from utils import config
from utils.asd_logging import logger as  customlogger


import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score



import os
import glob
import optuna
import xgboost as xgb

import pickle



optuna.logging.set_verbosity(optuna.logging.WARNING)

### change the logger type to save info logs
# customlogger = logger.logging.getLogger('console_info')


class SlugXGBoost(BaseModel):
    def __init__(self,
                    name,
                    timeout=None,
                ) -> None:

        # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ] check "Learning Task Parameters" section at 
        # https://xgboost.readthedocs.io/en/stable/parameter.html

        super().__init__(name)
        
        self.pred_class = "regression"                    
    
        self.objective = "count:poisson" # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]

        self.max_depth  = 30
        self.max_delta_step = 10 # recommended by the algo. documentation
        self.boosted_round = 250
        self.n_trials = 300
        self.cv_splits = 3 # number of folds    
        self.rand_state = 0
        
        self.score_func = None
        self.timeout = timeout
        

            
    def __objective__(self, trial, X_train, X_test, y_train, y_test):

        # X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25)
        param = {
            "objective": self.objective,
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-3, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
            "eta":  trial.suggest_float("eta", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 10.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_depth": trial.suggest_int("max_depth", 1, self.max_depth),
            # "max_delta_step": trial.suggest_int("max_delta_step", 0, Params.max_delta_step),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-3, 1.0, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 1e-3, 1.0, log=True),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 1e-3, 1.0, log=True),        
            "max_bin": trial.suggest_categorical("max_bin", [64, 128, 512, 1024, 2048, 3072, 4096, 8192])
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(param, dtrain, num_boost_round = self.boosted_round, verbose_eval = 1)

        pred_train = model.predict(dtrain)
        pred_test = model.predict(dtest)

        
        if self.pred_class == 'regression':
            err_train = mean_squared_error(y_train, pred_train)
            err_test = mean_squared_error(y_test, pred_test)
        else:
            err_train = f1_score(y_train, pred_train)
            err_test = f1_score(y_test, pred_test)


        weighted_score = common.get_weighted_score(err_train, err_test, self.pred_class)
        
        
        # save model in temp folder
        file_name =  self.temp_path + "/" + self.model_file_name + '_' + str(trial.number) +'.pickle'                
        self.__save_model__(model, file_name)

        return weighted_score



    def __discover_model__(self, X_train, X_test, y_train, y_test):
        
        customlogger.info( self.model_file_name + ': Starting training for trials:%d, boosted rounds: %d, max depth: %d', self.n_trials, self.boosted_round, self.max_depth)

        
        study = optuna.create_study( study_name=config.create_study_name(),
                                            direction="minimize", 
                                            sampler=optuna.samplers.TPESampler(),
                                            pruner=optuna.pruners.MedianPruner()                                                              
                                            )


        obj_func = lambda trial: self.__objective__(trial, X_train, X_test, y_train, y_test)
        
        study.optimize(obj_func, n_trials=self.n_trials, n_jobs=-1, timeout=self.timeout)

        customlogger.info( self.model_file_name + ': Number of trials: %d', len(study.trials))                   
        
        customlogger.info('Best trial:%d', study.best_trial.number)

        customlogger.info('  Params: ')
        for key, value in study.best_trial.params.items():
            customlogger.info('    %s %s', key, value)

                
        # file_name = self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        file_name = self.temp_path + "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        best_model = self.__load_model__(file_name)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)


        pred_train = best_model.predict(dtrain)
        pred_test = best_model.predict(dtest)

        
        ### score_func is none, then score_func is r2
        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, y_train, y_test)

        self.score = [metirc_score_train, metirc_score_test]

        customlogger.info('  test r2 score: %s', metirc_score_test)

        file_name = self.model_save_path + "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        self.__save_model__(best_model, file_name)

        customlogger.info( self.model_file_name + ': Model saved at %s', file_name)

        config.clear_temp_folder(self.temp_path)

        return best_model


    
    
    
    ### perform hyper-parameter search on xgboost model
    def fetch_model(self, X_train, X_test, y_train, y_test, score_func=None, threshold=None):
    
        ### It is important to call this function here so that the directory structure can be built in the remote cluster rather on the local machine    
        super().__create_dir_structure__(self.model_file_name)
        
        self.best_fit = self.__discover_model__(X_train, X_test, y_train, y_test)
        self.model = self.best_fit
                
        self.load_score(X_train, X_test, y_train, y_test, score_func, threshold)            
        return self.best_fit

    
    ### score_func: any sklearn score function, choose in accordance with self.pred_class
    ### persist: save predictions on test and train datasets, accessible via self.pred_train/test, otherwise null
    def load_score(self, X_train, X_test, y_train, y_test, score_func=None, threshold=None):        

        if self.model is None:
            customlogger.info(self.model_file_name + "No trained models found, pls rerun 'fetch_models'")
            return None

            
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        pred_train = self.model.predict(dtrain)
        pred_test = self.model.predict(dtest)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, y_train, y_test)

        if threshold:
            if metirc_score_test > threshold:
                self.best_fit = self.model
                self.score = [metirc_score_train, metirc_score_test]
            else:
                self.best_fit = None
                self.score = []
        else:
            self.best_fit = self.model
            self.score = [metirc_score_train, metirc_score_test]

        customlogger.info(self.model_file_name + ": scores loaded: " + str(self.score) )
        return self.score
    
    
    
    ### predict 
    def predict(self, df_X):

        if self.best_fit is None:
            customlogger.info("no model attached as per your selection threshold. Lower the threshold in the 'load_score' function.")  
            return None        


        ddf_X = xgb.DMatrix(df_X.copy())
        return self.best_fit.predict(ddf_X)
