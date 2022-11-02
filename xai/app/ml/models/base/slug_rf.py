from .base_model import BaseModel
from ml.models import common
from utils import config, logger


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import optuna
import xgboost as xgb

import dask_optuna


import joblib



optuna.logging.set_verbosity(optuna.logging.WARNING)


### change the logger type to save info logs
customlogger = logger.logging.getLogger('console_info')


class SlugRF(BaseModel):
    def __init__(self,
                    name,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    timeout=None,
                ) -> None:
        
        # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ] check "Learning Task Parameters" section at https://xgboost.readthedocs.io/en/stable/parameter.html
        self.pred_class = "regression"
                    
        self.objective = "count:poisson" # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]

        self.max_depth  = 30
        self.max_delta_step = 10 # recommended by the algo. documentation
        self.n_trials = 300
        self.cv_splits = 3 # number of folds    
        self.rand_state = 0
        

        self.score_func = None

        self.timeout = timeout

        temp_path, model_save_path = config.create_dir_structure(name)
        
        super().__init__( X_train, X_test, y_train, y_test, temp_path, model_save_path, name)



            
    def __objective__(self, trial):

        # criterion = “gini” [“gini”, “entropy”, “log_loss”]
    
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, self.max_depth),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }


        if self.pred_class == 'regression':
            cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            score_func = mean_squared_error
            model_rf = RandomForestRegressor(random_state=config.rand_state, **params)

        else:
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            score_func = f1_score
            model_rf =  RandomForestClassifier(random_state=config.rand_state, **params)


        model_rf.fit(self.X_train, self.y_train.values.ravel())
        metirc_score_train = model_rf.score(self.X_train, self.y_train)
        metirc_score_test = model_rf.score(self.X_test, self.y_test)


        weighted_score = common.get_weighted_score(metirc_score_train, metirc_score_test, self.pred_class)

        file_anme =  self.temp_path + "/" + self.model_file_name + '_' + str(trial.number) +'.pickle'

        # save model in temp folder
        self.__save_model__(model_rf, file_anme)
        
        return weighted_score



    def __discover_model__(self):

        customlogger.info( self.model_file_name + ': Starting training for trials:%d, max depth: %d', self.n_trials, self.max_depth)

        storage = dask_optuna.DaskStorage()

        study = optuna.create_study( study_name=config.create_study_name(),
                                            storage=storage, direction="minimize", 
                                            # sampler=optuna.samplers.CmaEsSampler(),
                                            sampler=optuna.samplers.TPESampler(),
                                            pruner=optuna.pruners.MedianPruner()                                                              
                                            )

#        study.enqueue_trial({"max_depth": 10,
#                            "n_estimators": 100,
#                            "min_samples_leaf": 1,
#                            "min_samples_split": 2,}


        with joblib.parallel_backend("dask"):
            study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1, timeout=self.timeout)

        customlogger.info( self.model_file_name + ': Number of trials: %d', len(study.trials))                   
        
        customlogger.info('Best trial:%d', study.best_trial.number)

        customlogger.info('  Params: ')
        for key, value in study.best_trial.params.items():
            customlogger.info('    %s %s', key, value)


        # load model from temp folder
        # file_name =  "/slug_xgboost_lightgbm_{}.pickle".format(study.best_trial.number)
        file_name =  "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        best_model = self.__load_model__(self.temp_path + file_name)

        pred_train = best_model.predict(self.X_train)
        pred_test = best_model.predict(self.X_test)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test)

        self.score = [metirc_score_train, metirc_score_test]
        customlogger.info('  test r2 score: %s', metirc_score_test)

        # save it to permanent folder
        customlogger.info( self.model_file_name + ': Model saved at %s', self.model_save_path + file_name)
        self.__save_model__(best_model, self.model_save_path + file_name)

        config.clear_temp_folder(self.temp_path)

        return best_model


    
    
    ### perform hyper-parameter search on xgboost model
    def fetch_model(self, retrain = True):
    
        self.best_fit = self.__discover_model__()
        self.model = self.best_fit
                
        return self.best_fit

    
    ### score_func: any sklearn score function, choose in accordance with self.pred_class
    ### persist: save predictions on test and train datasets, accessible via self.pred_train/test, otherwise null
    def load_score(self, score_func=None, persist_pred=True, threshold=None):
        

        if self.model is None:
            customlogger.info("No trained models found, pls rerun 'fetch_models'")
            return None

        if self.X_train is None:
            customlogger.info("No train/test dataset found, pls explicity set the parameters.")
            return None
        
        pred_train = self.model.predict(self.X_train)
        pred_test = self.model.predict(self.X_test)
                                    
            
        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, score_func=score_func)

        if persist_pred:
            self.pred_train = pred_train
            self.pred_test = pred_test

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test)

        if threshold:
            if metirc_score_test > threshold:
                self.best_fit = self.model
                self.score = [metirc_score_train, metirc_score_test]
            else:
                self.best_fit = None
                self.score = None
        else:
            self.best_fit = self.model
            self.score = [metirc_score_train, metirc_score_test]

        
        return self.score

    
    
    ### predict 
    def predict(self, df_X):

        if self.best_fit is None:
            customlogger.info("no model attached as per your selection threshold. Lower the threshold in the 'load_score' function.")  
            return None        
        
        if self.X_train is None:
            customlogger.info("No train/test dataset found, pls explicity set the parameters.")
            return None

        
        return self.best_fit.predict(df_X)
