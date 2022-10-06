
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
import numpy as np
from utils import config

from dask.base import is_dask_collection
import dask.dataframe as dd


class CustomCrossVal(BaseEstimator):
    def __init__(self, estimator, cv, logger):
        self.estimator = estimator
        self.cv = cv
        self.logger = logger
    

    def fit(self, X, y, **fit_kws):

        # Insert more guards here!
        if isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=True)
        elif isinstance(X, pd.DataFrame):
            X = X.values
            
        if is_dask_collection(X):
            from dask_ml.model_selection import train_test_split
        else:
            from sklearn.model_selection import train_test_split


        X_base, X_holdout, y_base, y_holdout = train_test_split(
            X, y, random_state=config.rand_state)
        
        self.split_scores_ = []
        self.holdout_scores_ = []
        self.estimators_ = []            
        
        print(type(X_base))


        for train_idx, test_idx in self.cv.split(X_base, y_base):
            

            # print(test_idx)

            X_test, y_test = X_base[test_idx], y_base[test_idx]
            X_train, y_train = X_base.loc[X_base.index[train_idx]], y_base.loc[y_base.index[train_idx]]
        

            # X_test, y_test = X_base[test_idx], y_base[test_idx]
            # X_train, y_train = X_base[train_idx], y_base[train_idx]

            estimator_ = clone(self.estimator)
            estimator_.fit(X_train, y_train, **fit_kws)

            self.logger.info("... log things ...")
            self.estimators_.append(estimator_)
            self.split_scores_.append(estimator_.score(X_test, y_test))            
            self.holdout_scores_.append(
                estimator_.score(X_holdout, y_holdout))
    
        self.best_estimator_ = \
                self.estimators_[np.argmax(self.holdout_scores_)]
        return self

    