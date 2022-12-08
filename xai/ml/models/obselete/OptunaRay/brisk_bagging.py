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


from ray import tune, air
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch

#optuna.logging.set_verbosity(optuna.logging.WARNING)


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

        tune.report(weighted_score=weighted_score)
        




    def __discover_model__(self, X_train, X_test, y_train, y_test):

        customlogger.info( self.model_file_name + ': Starting training for trials:%d, n_estimators  %d', self.n_trials, self.n_estimators)


        params = {
            # 'base_estimator': DecisionTreeRegressor
            'n_estimators': tune.randint(10, self.n_estimators),
        }

        baye_space = {
            # 'base_estimator': DecisionTreeRegressor
            'n_estimators': (10, self.n_estimators),
        }
    
    
#        space = {
#            'width': (0, 20),
#            'height': (-100, 100),
#        }
#        bayesopt = BayesOptSearch(space, metric="mean_loss", mode="min")
#        tuner = tune.Tuner(
#            objective,
#            tune_config=tune.TuneConfig(
#                search_alg=bayesopt,
#            ),
#        )
    
    
        obj_func = lambda params: self.__objective__(params, X_train, X_test, y_train, y_test)
    
        # algo = OptunaSearch(space=params, metric="mean_loss", mode="min")
        bayesopt = BayesOptSearch(space=baye_space, metric="weighted_score", mode="min",  random_state=config.rand_state)
        # algo = ConcurrencyLimiter(algo, max_concurrent=4)
        
        tuner = tune.Tuner(
            obj_func,
            run_config=air.RunConfig(
              name=config.create_study_name(),
              #stop={"training_iteration": 1 if args.smoke_test else 10},
            ),
            tune_config=tune.TuneConfig(
                search_alg=bayesopt,
                num_samples=self.n_trials,
                #checkpoint_dir=None,
            ),
#            param_space=params,
        )
        
        results = tuner.fit()
        best_result = results.get_best_result()
        
        print('############################')
        print("Best hyperparameters: ", best_result)
        print("Best hyperparameters: ", best_result.config)
        
        # best_result = result_grid.get_best_result()
        
        return best_result.config



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
    
        self.best_fit = self.refit(self.__discover_model__(X_train, X_test, y_train, y_test), X_train, X_test, y_train, y_test)        
    
        #self.load_score(X_train, X_test, y_train, y_test, score_func, threshold)
    
        return self.best_fit

    
    ### predict 
    def predict(self, df_X):
        pass
