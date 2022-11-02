
from utils import config, logger

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import os
import glob

import optuna
import xgboost as xgb


import dask_optuna

import joblib
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

### change the logger type to save info logs
customlogger = logger.logging.getLogger('console_info')


class SlugXGBoost:
    def __init__(self) -> None:
        # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ] check "Learning Task Parameters" section at https://xgboost.readthedocs.io/en/stable/parameter.html
        self.pred_class = "regression"                    
    
        self.objective = "count:poisson" # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]

        self.max_depth  = 10
        self.max_delta_step = 10 # recommended by the algo. documentation
        self.boosted_round = 250
        self.n_trials = 300
        self.cv_splits = 3 # number of folds    
        self.rand_state = 0
        
        self.best_fit = None
        self.score = None        
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.pred_train = None
        self.pred_test = None        
                
        self.temp_path = config.main_dir + config.project_name + "/tmp/xgboost/slug"
        self.model_save_path = config.main_dir + config.project_name  + "/base/xgboost/slug"

        



    # Load pickled models
    def __load_model__(self, file_name):
        with open(file_name, "rb") as fin:
            model = pickle.load(fin)

        return model


    
    def __save_model__(self, model, file_name):
        with open(file_name, "wb") as fout:
            pickle.dump(model, fout)


            
    def __load_best_fit__(self):        
        if self.best_fit is None:            
            model_files = glob.glob(self.model_save_path+'/*')
            if len(model_files) == 0:
                return None
            else:
                self.best_fit = self.__load_model__(model_files[0])                

            
    def __objective__(self, trial):

        # X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25)
        param = {
            "objective": self.objective,
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
        self.__save_model__(bst, file_anme)

        return mean_err



    def __discover_model__(self):

        customlogger.info('slug xgboost: Starting train for trials:%d with boosted rounds: %d', self.n_trials, self.boosted_round)

        customlogger.info('slug xgboost: Cleared previous models in the model save path')        
        config.clear_temp_folder(self.model_save_path)

        
        storage = dask_optuna.DaskStorage()
        study = optuna.create_study(storage=storage, direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

        with joblib.parallel_backend("dask"):
            study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1)

        customlogger.info('xgboost: Number of trials: %d', len(study.trials))
                   
        customlogger.info('Best trial:')
        trial = study.best_trial


        customlogger.info('  xgboost Params: ')
        for key, value in trial.params.items():
            customlogger.info('    %s %s', key, value)



        # load model from temp folder
        file_name =  "/slug_xgboost_{}.pickle".format(study.best_trial.number)
        best_model = self.__load_model__(self.temp_path + file_name)

        # save it to permanent folder
        customlogger.info('xgboost: Model saved at %s', self.model_save_path + file_name)
        self.__save_model__(best_model, self.model_save_path + file_name)

        config.clear_temp_folder(self.temp_path)
        return best_model


    
    ### perform hyper-parameter search on xgboost model
    def fetch_model(self, retrain = True):
    
        if retrain:                
            self.best_fit = self.__discover_model__()
        else:
            self.__load_best_fit__()
            if self.best_fit is None:
                customlogger.info("xgboost: no saved models found, please rerun the 'fetch_model' first.")
                return None
            
        self.get_model_score()
                
        return self.best_fit


    
    ### score_func: any sklearn score function, choose in accordance with self.pred_class
    ### persist: save predictions on test and train datasets, accessible via self.pred_train/test, otherwise null
    def get_model_score(self, score_func=None, persist_pred=True):
        
        self.__load_best_fit__()
        
        if self.best_fit is None:
            customlogger.info("xgboost: no saved models found, please rerun the 'fetch_model' first.")
            return None

            
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        
        pred_train = self.best_fit.predict(dtrain)
        pred_test = self.best_fit.predict(dtest)
                
        if persist_pred:
            self.pred_train = pred_train
            self.pred_test = pred_test
        
        
        if score_func is None:                
            if self.pred_class == 'regression':
                metirc_score_train = r2_score(pred_train, self.y_train)
                metirc_score_test = r2_score(pred_test, self.y_test)
            else:
                metirc_score_train = f1_score(pred_train, self.y_train)
                metirc_score_test = f1_score(pred_test, self.y_test)                
        else:
            metirc_score_train = score_func(pred_train, self.y_train)
            metirc_score_test = score_func(pred_test, self.y_test)            

        self.score = [metirc_score_train, metirc_score_test]

        return self.score

    
    def get_predictions(self, model):
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        pred_train = model.predict(dtrain)
        pred_test = model.predict(dtest)

        return pred_train, pred_test, self.y_train, self.y_test    
    
    
    ### predict 
    def predict(self, df_X):

        self.__load_best_fit__()        
        if self.best_fit is None:
            customlogger.info("xgboost: no saved models found, please rerun the 'fetch_model' first.")            
            return None
        
        
        ddf_X = xgb.DMatrix(df_X.copy())
        return self.best_fit.predict(ddf_X)
