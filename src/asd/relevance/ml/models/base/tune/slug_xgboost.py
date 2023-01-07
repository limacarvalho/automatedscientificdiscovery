
from relevance.ml.models import common
from relevance.utils import config
from relevance.utils.asd_logging import logger as  customlogger

import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import xgboost as xgb

from tune_sklearn import TuneSearchCV
import ray
from ray import tune



### change the logger type to save info logs
#customlogger = logger.logging.getLogger('console_info')


class SlugXGBoost():
    def __init__(self,
                    name,
                    objective,
                    pred_class,
                    score_func=None,
                    metric_func=None,
                    n_estimators=100,
                    max_depth=30,
                    n_trials=100,
                    cv_splits=3,
                    timeout=None,
                ) -> None:

        self.objective = objective
        self.pred_class = pred_class
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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
            model = xgb.XGBRegressor(self.objective, booster='gbtree', n_estimators=self.n_estimators)
            if self.metric_func is None:
                self.metric_func = r2_score

        else:
            model = xgb.XGBClassifier(self.objective, booster='gbtree', n_estimators=self.n_estimators)
            if self.metric_func is None:
                self.metric_func = f1_score

        return model



    def fit(self, X_train, X_test, y_train, y_test):

        customlogger.info( self.model_file_name + ': fit')

        param_dists = {
            #"metric": "rmse",
            "lambda": tune.loguniform(1e-3, 0.1),
            "alpha": tune.loguniform(1e-3, 0.1),
            "max_depth": tune.randint(10, self.max_depth),
            "eta": tune.loguniform(1e-3, 0.1),
            "gamma": tune.loguniform(1e-3, 0.1),
            "grow_policy": tune.choice(["depthwise", "lossguide"]),
            "colsample_bytree": tune.loguniform(1e-8, 1.0),
            "colsample_bylevel": tune.loguniform(1e-8, 1.0),
            "colsample_bynode": tune.loguniform(1e-8, 1.0),
            "max_bin": tune.choice([64, 128, 512, 1024, 2048, 3072]),
        }


        self.gs = TuneSearchCV(self.__get_model__(),
                                    param_dists,
                                    n_trials=self.n_trials,
                                    scoring=self.score_func,
                                    cv=self.cv_splits,
                                    loggers= ['csv'],
                                    search_optimization ='hyperopt',
                                    time_budget_s=self.timeout
                                    )

        self.gs.fit(X_train, y_train)

        pred_test = self.gs.predict(X_test)
        pred_train = self.gs.predict(X_train)

        err_train = self.metric_func(pred_train, y_train)
        err_test = self.metric_func(pred_test, y_test)

        self.scores = [err_train, err_test]

        customlogger.info( self.model_file_name + ': score: ' + str(self.scores))


    def score(self, X, y, metric_func=None):
        if metric_func is None:
            metric_func = self.metric_func

        pred = self.gs.predict(X)

        return metric_func(pred, y)



    def predict(self, df_X):
        return self.gs.predict(df_X)
