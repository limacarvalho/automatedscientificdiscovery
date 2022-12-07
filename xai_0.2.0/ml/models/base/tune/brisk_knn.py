
from ml.models import common
from utils import config
from utils.asd_logging import logger as  customlogger

import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from tune_sklearn import TuneSearchCV
import ray
from ray import tune



### change the logger type to save info logs
#customlogger = logger.logging.getLogger('console_info')


class BriskKNN():
    def __init__(self,
                    name,
                    pred_class,
                    score_func=None,
                    metric_func=None,
                    n_neighbors=30,
                    n_trials=100,
                    cv_splits=3,
                    timeout=None,
                ) -> None:
        
        self.pred_class = pred_class                    
        self.n_neighbors = n_neighbors
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
            model = KNeighborsRegressor()
            if self.metric_func is None:
                self.metric_func = r2_score
        else:
            model = KNeighborsClassifier()
            if self.metric_func is None:
                self.metric_func = f1_score
        
        return model

        
        
    def fit(self, X_train, X_test, y_train, y_test):

        param_dists = {
            'n_neighbors': tune.randint( 5, self.n_neighbors),
            'weights': tune.choice(["uniform", "distance"]),
            'algorithm': tune.choice(["auto", "ball_tree", "kd_tree", "brute"])
        }

        self.gs = TuneSearchCV(self.__get_model__(), 
                                    param_dists, 
                                    n_trials=self.n_trials, 
                                    scoring=self.score_func,
                                    cv=self.cv_splits,
                                    search_optimization ='hyperopt',
                                    time_budget_s=self.timeout,
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
    