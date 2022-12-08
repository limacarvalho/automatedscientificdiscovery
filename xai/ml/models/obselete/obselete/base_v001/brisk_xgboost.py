from dask_ml.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from utils import config
import numpy as np

import dask.dataframe as ddf


param = {"verbosity": 0, "tree_method": "hist", "objective": "reg:squarederror", ''
                         "max_depth": 10, "learning_rate": 0.01, "max_bin": 128, "num_boost_round" : 200}



class BriskXGBoost():
    def __init__(self, client, cv, logger, pred_type) -> None:
        self.client = client 
        self.cv = cv
        self.logger = logger
        self.mean_train_error = None
        self.mean_test_error = None
        self.holdout_error = None
        self.pred_type = pred_type
        self.model = None
        

    def fit(self, df_X, df_y):

        result_tain_list = []
        result_test_list = []
        

        # X = ddf.to_dask_array(X.values)
        # y = ddf.to_dask_array(y.values)

        no_of_workers = len(self.client.scheduler_info()['workers'])

        # print(no_of_workers)

        X = ddf.from_pandas(df_X, npartitions=no_of_workers)
        y = ddf.from_pandas(df_y, npartitions=no_of_workers)

        X = X.to_dask_array(lengths=True)
        y = y.to_dask_array(lengths=True)

        X_base, X_holdout, y_base, y_holdout = train_test_split(X, y, 
                                                                                                                                    random_state=config.rand_state)

        for train_idx, test_idx in self.cv.split(X_base, y_base):
            X_train, y_train = X_base[train_idx], y_base[train_idx]
            X_test, y_test = X_base[test_idx], y_base[test_idx]
            dtrain = xgb.dask.DaskDMatrix(self.client, X_train, y_train)
            dtest = xgb.dask.DaskDMatrix(self.client, X_test, y_test)

            trained_model = xgb.dask.train(
                self.client,
                param, 
                dtrain,
                num_boost_round=param["num_boost_round"],
                # early_stopping_rounds=50, 
                evals=[(dtrain, "train")],
            )
            
            # booster = output["booster"]
            # print(booster.best_iteration)
            # best_model = booster[: booster.best_iteration]
            train_preds = xgb.dask.predict(self.client, trained_model, dtrain)
            test_preds = xgb.dask.predict(self.client, trained_model, dtest)

            if self.pred_type == 'classification': 
                result_tain_list = -1
                result_test_list = -1
            else:
                result_tain_list.append(mean_squared_error(train_preds, y_train))
                result_test_list.append(mean_squared_error(test_preds, y_test))
        
        
        self.mean_train_err = np.mean(result_tain_list)
        self.mean_test_err = np.mean(result_test_list)
        self.train_test_err_diff = np.abs(self.mean_train_err - self.mean_test_err)


        holdout_pred = xgb.dask.predict(self.client, trained_model, X_holdout)

        if self.pred_type == 'classification': 
            self.holdout_err = -1
        else:
            self.holdout_err = mean_squared_error(holdout_pred, y_holdout)    


        self.model = trained_model

        return {
            "mean_train_err": self.mean_train_err,
            "mean_test_err": self.mean_test_err,
            "train_test_err_diff": self.train_test_err_diff,
            "holdout_err": self.holdout_err
        }


    def get_result_stats(self):

        return {
            "mean_train_err": self.mean_train_err,
            "mean_test_err": self.mean_test_err,
            "train_test_err_diff": self.train_test_err_diff,
            "holdout_err": self.holdout_err
        }

