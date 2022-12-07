from .base_model import BaseModel
from ml.models import common
from utils import config
from utils.asd_logging import logger as  customlogger

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)


### change the logger type to save info logs
#customlogger = logger.logging.getLogger('console_info')


class BriskBagging(BaseModel):
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
                    
        self.n_estimators = 30
        self.n_trials = 300
        self.cv_splits = 3 # number of folds            
        self.random_state = 0
        
        self.model_file_name = name
        self.score_func = None

        self.timeout = timeout
        
        
        
        super().__init__( X_train, X_test, y_train, y_test)



            
    def __objective__(self, trial):

        # criterion = “gini” [“gini”, “entropy”, “log_loss”]
    
        params = {
            # 'base_estimator': DecisionTreeRegressor
            'n_estimators': trial.suggest_int('n_estimators', 10, self.n_estimators),
        }


        if self.pred_class == 'regression':
            # cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            self.score_func = mean_squared_error
            model = BaggingRegressor(**params)

        else:
            # cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            self.score_func = f1_score
            model =  BaggingClassifier(**params)


        model.fit(self.X_train, self.y_train.values.ravel())
        
        pred_train = model.predict(self.X_train)
        pred_test = model.predict(self.X_test)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, self.score_func)

        file_name =  self.temp_path + "/" + self.model_file_name + '_' + str(trial.number) +'.pickle'

        # save model in temp folder
        self.__save_model__(model, file_name)
        
        return weighted_score



    def __discover_model__(self):

        customlogger.info( self.model_file_name + ': Starting training for trials:%d, n_estimators  %d', self.n_trials, self.n_estimators)

        study = optuna.create_study( study_name=config.create_study_name(),
                                            direction="minimize", 
                                            sampler=optuna.samplers.TPESampler(),
                                            pruner=optuna.pruners.MedianPruner()                                                              
                                            )

#        study.enqueue_trial({"max_depth": 10,
#                            "n_estimators": 100,
#                            "min_samples_leaf": 1,
#                            "min_samples_split": 2,}



        study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1, timeout=self.timeout)

        customlogger.info( self.model_file_name + ': Number of trials: %d', len(study.trials))                   
        
        customlogger.info('Best trial:%d', study.best_trial.number)

        customlogger.info('  Params: ')
        for key, value in study.best_trial.params.items():
            customlogger.info('    %s %s', key, value)


        # load model from temp folder
        # file_name =  "/slug_xgboost_lightgbm_{}.pickle".format(study.best_trial.number)
        file_name = self.temp_path + "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        best_model = self.__load_model__(file_name)

        pred_train = best_model.predict(self.X_train)
        pred_test = best_model.predict(self.X_test)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test)

        self.score = [metirc_score_train, metirc_score_test]
        customlogger.info('  test r2 score: %s', metirc_score_test)

        # save it to permanent folder
        file_name = self.model_save_path + "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        self.__save_model__(best_model, file_name)
        customlogger.info( self.model_file_name + ': Model saved at %s', file_name)
        
        config.clear_temp_folder(self.temp_path)

        return best_model


    
    
    ### perform hyper-parameter search on random forest model
    def fetch_model(self, score_func=None, threshold=None):
    
        super().__create_dir_structure__(self.model_file_name)
    
        self.best_fit = self.__discover_model__()
        self.model = self.best_fit
    
        self.load_score(score_func, threshold)
    
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
                self.score = []
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
