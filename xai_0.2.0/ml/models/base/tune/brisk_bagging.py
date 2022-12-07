
from ml.models import common
from utils import config
from utils.asd_logging import logger as  customlogger

import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


from tune_sklearn import TuneSearchCV
import ray
from ray import tune



### change the logger type to save info logs
#customlogger = logger.logging.getLogger('console_info')


class BriskBagging():
    '''
    A brisk bagging model using bayesian hyper-param optimization to fit both classification and regression task.
    :param name: str, name of the model
    :param pred_class: pass either 'regression' or classification
    :param score_func: loss function, i.e., r2_scoring = sklearn.metrics.make_scorer(score_func=r2_score, greater_is_better=False)
    :param n_estimators: int, default=10. The number of base estimators in the ensemble.
    :param n_trials: int, default=100. Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.
    :param cv_splits: int, default=3, cv (`cross-validation generator` or `iterable`): Determines the cross-validation splitting strategy.
    :param time_budget_s:  (int|float|datetime.timedelta): Global time budget in seconds after which all trials are stopped. Can also be a ``datetime.timedelta`` object. The stopping condition is checked after receiving a result, i.e. after each training iteration.
    '''
    def __init__(self,
                    name,
                    pred_class,
                    score_func=None,
                    metric_func=None,
                    n_estimators=30,
                    n_trials=100,
                    cv_splits=3,
                    timeout=None,
                ) -> None:
        
        self.pred_class = pred_class                    
        self.n_estimators = n_estimators
        self.n_trials = n_trials
        self.cv_splits = cv_splits # number of folds
        self.random_state = config.rand_state
        
        self.model_file_name = name

        self.score_func = score_func
        self.metric_func = metric_func

        self.scores = []
        self.gs = None

        self.timeout = timeout
        
        

    def __get_model__(self):

        if self.pred_class == 'regression':
            model = BaggingRegressor()
            if self.metric_func is None:
                self.metric_func = r2_score
        else:
            model = BaggingClassifier()
            if self.metric_func is None:
                self.metric_func = f1_score
        
        return model

        
        
    def fit(self, X_train, X_test, y_train, y_test):

        param_dists = {
            'n_estimators': tune.randint(10, self.n_estimators),
        }

        self.gs = TuneSearchCV(self.__get_model__(), 
                                    param_dists, 
                                    n_trials=self.n_trials, 
                                    scoring=self.score_func,
                                    cv=self.cv_splits,
                                    search_optimization ='hyperopt',
                                    time_budget_s=self.timeout
                                    )

        self.gs.fit(X_train, y_train)

        pred_test = self.gs.predict(X_test)
        pred_train = self.gs.predict(X_train)
        
        err_train = self.metric_func(pred_train, y_train)
        err_test = self.metric_func(pred_test, y_test)

        self.scores = [err_train, err_test]
        


    def score(self, X, y, metric_func=None):
        if metric_func is None:
            metric_func = self.metric_func

        pred = self.gs.predict(X)        

        return metric_func(pred, y)



    def predict(self, df_X):
        return self.gs.predict(df_X)
    