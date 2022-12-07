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

from hyperopt import hp
# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval



### change the logger type to save info logs
#customlogger = logger.logging.getLogger('console_info')


class BriskBagging(BaseModel):
    def __init__(self,
                    name,
                    timeout=None,
                ) -> None:
        
        super().__init__(name)
        # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ] check "Learning Task Parameters" section at 
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        
        self.pred_class = "regression"
                    
        self.n_estimators = 30
        self.n_trials = 300
        self.cv_splits = 3 # number of folds            
        self.random_state = 0
        
        self.model_file_name = name
        self.score_func = None

        self.timeout = timeout
        
        
        
        



            
    def __objective__(self, params, X_train, X_test, y_train, y_test):

        # criterion = “gini” [“gini”, “entropy”, “log_loss”]
            
        n_estimators = int(params['n_estimators'])
        
        # print(n_estimators)
        
        if self.pred_class == 'regression':
            # cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            self.score_func = mean_squared_error
            model = BaggingRegressor(n_estimators=n_estimators)

        else:
            # cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            self.score_func = f1_score
            model =  BaggingClassifier(n_estimators=n_estimators)

    
        model.fit(X_train, y_train.values.ravel())
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, y_train, y_test, self.score_func)

        #file_name =  self.temp_path + "/" + self.model_file_name + '_' + str(trial.number) +'.pickle'

        # save model in temp folder
#        self.__save_model__(model, file_name)
        
        # return {'loss': weighted_score, 'status': STATUS_OK}
        
        return weighted_score



    def __discover_model__(self, X_train, X_test, y_train, y_test):

        customlogger.info( self.model_file_name + ': Starting training for trials:%d, n_estimators  %d', self.n_trials, self.n_estimators)


#        study.enqueue_trial({"max_depth": 10,
#                            "n_estimators": 100,
#                            "min_samples_leaf": 1,
#                            "min_samples_split": 2,}


        space = {
            # 'base_estimator': DecisionTreeRegressor
            # 'n_estimators': hp.uniform('n_estimators', 10, self.n_estimators),
            'n_estimators': hp.uniform('n_estimators', 10, self.n_estimators),
        }
        

        obj_func = lambda space: self.__objective__(space, X_train, X_test, y_train, y_test)
    
        best_params = fmin(obj_func, space, algo=tpe.suggest, max_evals=100)
                
#         study.optimize(obj_func, n_trials=self.n_trials, n_jobs=-1, timeout=self.timeout)

        #customlogger.info( self.model_file_name + ': Number of trials: %d', len(study.trials))                   
        
        #customlogger.info('Best trial:%d', study.best_trial.number)

        return best_params


    def refit(self, best_params, X_train, X_test, y_train, y_test):
        
        n_estimators = int(best_params['n_estimators'])
        
        if self.pred_class == 'regression':
            # cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            self.score_func = r2_score
            model = BaggingRegressor(n_estimators=n_estimators)

        else:
            # cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=config.rand_state)
            self.score_func = f1_score
            model =  BaggingClassifier(n_estimators=n_estimators)
            
        model.fit(X_train, y_train.values.ravel())
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, y_train, y_test, self.score_func)
        
        customlogger.info('final score :%s', str([metirc_score_train, metirc_score_test]))
        
        return model
            
    
    ### perform hyper-parameter search on random forest model
    def fetch_model(self, X_train, X_test, y_train, y_test, score_func=None, threshold=None):
    
        super().__create_dir_structure__(self.model_file_name)
    
        best_params = self.__discover_model__(X_train, X_test, y_train, y_test)
        self.best_fit = self.refit(best_params, X_train, X_test, y_train, y_test)
                    
        # self.load_score(X_train, X_test, y_train, y_test, score_func, threshold)
    
        return self.best_fit

    
    def predict(self):
        pass
    