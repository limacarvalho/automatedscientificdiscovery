
from utils import dasker, helper, config
from pprint import pprint

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from itertools import combinations

import os
import glob

import optuna
import xgboost as xgb

from dask.distributed import Client
import dask_optuna

import joblib
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)



class SlugXGBoost:
    def __init__(self) -> None:
        self.pred_class = "regression"
                    
        # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ] check "Learning Task Parameters" section at https://xgboost.readthedocs.io/en/stable/parameter.html

        self.loss = "count:poisson" # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]

        self.max_depth  = 10
        self.max_delta_step = 10 # recommended by the algo. documentation
        self.boosted_round = 250
        self.n_trails = 300
        self.cv_splits = 3 # number of folds    
        self.rand_state = 0
        self.acceptance_threshold = 0.8
        self.best_fit = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


        self.temp_path = config.main_dir + "/ml/models/saved/temp/xgboost"
        self.model_save_path = config.main_dir + "/ml/models/saved/base/xgboost/slug"



    # Load pickled models
    def load_model(self, file_name):
        with open(file_name, "rb") as fin:
            model = pickle.load(fin)

        return model


    def save_model(self, model, file_name):
        with open(file_name, "wb") as fout:
            pickle.dump(model, fout)



    def __objective__(self, trial):

        # X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25)
        param = {
            "objective": self.loss,
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "eta":  trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_depth": trial.suggest_int("max_depth", 1, self.max_depth),
            # "max_delta_step": trial.suggest_int("max_delta_step", 0, Params.max_delta_step),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-8, 1.0, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 1e-8, 1.0, log=True),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 1e-8, 1.0, log=True),        
            "max_bin": trial.suggest_categorical("max_bin", [64, 128, 512, 1024, 2048, 3072]),
        }


        kf = KFold(n_splits=self.cv_splits)
        
        err_test_list = []

        for train_index, test_index in kf.split(self.X_train, self.y_train):
            X_train, X_test = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
            y_train, y_test = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            bst = xgb.train(param, dtrain, num_boost_round = self.boosted_round, verbose_eval = 1)

            preds = bst.predict(dtest)

            if self.pred_class == 'regression':
                err_test = mean_squared_error(y_test, preds)
            else:
                err_test = f1_score(y_test, preds)
            #print(f'{err_test}')

            err_test_list.append(err_test)

        mean_err = np.mean(err_test_list)
        
        file_anme =  self.temp_path + "/slug_xgboost_{}.pickle".format(trial.number)
        
        # save model in temp folder
        self.save_model(bst, file_anme)

        return mean_err



    def __discover_model__(self):

        print(f'Starting train for trials:{self.n_trails} with boosted rounds:{self.boosted_round}')


        print(f'Cleared previous models in the model save path')
        config.clear_temp_folder(self.model_save_path)

        storage = dask_optuna.DaskStorage()
        study = optuna.create_study(storage=storage, direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

        with joblib.parallel_backend("dask"):
            study.optimize(self.__objective__, n_trials=self.n_trails, n_jobs=-1)

        print("Number of trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("Number of trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


        # load model from temp folder
        file_name =  "/slug_xgboost_{}.pickle".format(study.best_trial.number)
        best_model = self.load_model(self.temp_path + file_name)

        # save it to permanent folder
        print(f"Model saved at:{self.model_save_path + file_name}")
        self.save_model(best_model, self.model_save_path + file_name)

        config.clear_temp_folder(self.temp_path)

        return best_model



    def fetch_model(self):

        client = dasker.get_dask_client()
        print(f"Dask dashboard is available at {client.dashboard_link}")

        self.best_fit = self.__discover_model__()

        return self.best_fit

        
    ### return score on test dataset
    def get_model_score(self, score_func=None):

        metirc_scores = []
        model_files = glob.glob(self.model_save_path+'/*')
        model = self.load_model(model_files[0])    

        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        pred = model.predict(dtest)
        
        if score_func is None:                
            if self.pred_class == 'regression':
                metirc_score = r2_score(pred, self.y_test)

            else:
                metirc_score = f1_score(pred, self.y_test)
        else:
            for func in score_func:
                metirc_score = func(pred, self.y_test)
                metirc_scores.append(metirc_score)

        return metirc_scores
