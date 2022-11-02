

from ml.models import common
from utils import helper, config, logger, dasker

from ml.models.base.brisk_xgboost import BriskXGBoost
from ml.models.base.slug_ann import SlugANN
from ml.models.base.slug_lightgbm import SlugLGBM
from ml.models.base.slug_xgboost import SlugXGBoost
from ml.models.base.slug_rf import SlugRF


from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import numpy as np
import pandas as pd

import signal
import dask




### change the logger type to save info logs
customlogger = logger.logging.getLogger('console_info')

class Ensemble:
    def __init__(self, 
                 df_X,
                 df_y,
                 project_name,
                 num_reruns=5,
                 epochs=150,
                 boosted_round=100,
                 ensemble_boosted_round=100,
                 ensemble_num_reruns=5,
                 timeout=7200) -> None:


        self.score = None
        self.base_model_scores = []
        self.slug_rf = None
        self.ensemble_boosted_round = ensemble_boosted_round
        self.ensemble_num_reruns = ensemble_num_reruns


        self.df_X_train_ensemble = pd.DataFrame()
        self.df_X_test_ensemble = pd.DataFrame() 

        
        config.project_name = '/' + project_name #house_prices_advanced_regression_techniques_ensemble'


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_X, 
                                                                                                                                df_y, test_size=0.33,
                                                                                                                                random_state=config.rand_state)


        self.X_train_scalar = StandardScaler().fit_transform(self.X_train.copy())
        self.X_test_scalar = StandardScaler().fit_transform(self.X_test.copy())



        self.brisk_xgb = BriskXGBoost('brisk_xgb', self.X_train, self.X_test, self.y_train, self.y_test, timeout=timeout)
        self.brisk_xgb.boosted_round = boosted_round


        self.slug_xgb = SlugXGBoost('slug_xgb', self.X_train, self.X_test, self.y_train, self.y_test, timeout=timeout)
        self.slug_xgb.boosted_round = boosted_round


        self.slug_ann = SlugANN('slug_ann', self.X_train_scalar, self.X_test_scalar, self.y_train, self.y_test, timeout=timeout)
        self.slug_ann.epochs = epochs


        self.base_models = [self.slug_ann, self.brisk_xgb, self.slug_xgb]




    def fetch_models(self):
        
        lazy_results = []
        results = None


        for base_model in self.base_models:
            res = dask.delayed(__run_discoveries__)(base_model)
            lazy_results.append(res)

        results = dask.compute(*lazy_results, scheduler='distributed')

        # self.base_models = results

        customlogger.info("Ensemble training completed")
        self.load_scores()



    def load_scores(self, score_func=None, persist_pred=True, threshold=None):
        self.base_model_scores = []
        for base_model in self.base_models:        
            base_model.load_scores(score_func, persist_pred, threshold)
            if not helper.check_if_all_same(base_model.scores, None):
                temp_scores = base_model.scores
                temp_scores.append(base_model)
                self.base_model_scores.append(temp_scores)

        customlogger.info("please re-run 'fit' model")
        self.score = None


    def fit(self):

        for base_model in self.base_models:
            temp_X_train = base_model.df_pred_train #.loc[:, base_model.df_pred_train.columns!='y']
            temp_X_test = base_model.df_pred_test #.loc[:, base_model.df_pred_test.columns!='y']

            self.df_X_train_ensemble = pd.concat([self.df_X_train_ensemble, temp_X_train], axis=1)
            self.df_X_test_ensemble = pd.concat([self.df_X_test_ensemble, temp_X_test], axis=1 )



        ### rename cols
        col_range = range(0, self.df_X_train_ensemble.shape[1])
        col_ids = []

        for i in col_range:
            col_ids.append(str(i))
        self.df_X_train_ensemble.columns = col_ids
        self.df_X_test_ensemble.columns = col_ids


        ### fit Random forest on the all the base models to get one single outcome
        self.slug_rf = SlugRF('slug_rf', self.df_X_train_ensemble, self.df_X_test_ensemble, self.y_train, self.y_test)
        self.slug_rf.num_reruns = self.ensemble_num_reruns

        self.slug_rf.fetch_models()

        self.slug_rf.persist_best()

        self.score = self.slug_rf.scores[0]



    def predict(self, df_X, df_y) :

        df_pred = pd.DataFrame()
        df_X_scalar = StandardScaler().fit_transform(df_X.copy())


        for base_model in self.base_models:
            if 'SlugANN' in str(base_model):
                pred = base_model.predict(df_X_scalar)
            else:
                pred = base_model.predict(df_X)

            #print(str(base_model))
            #for col in pred:
            #    print(r2_score(pred[col], df_y.values))                
                
            #print(pred.shape)
            #print(df_y.shape)
            
            df_pred = pd.concat([df_pred, pred], axis=1)



        ### rename cols
        col_range = range(0, df_pred.shape[1])
        col_ids = []

        for i in col_range:
            col_ids.append(str(i))
        df_pred.columns = col_ids
        df_pred.columns = col_ids

        

        return self.slug_rf.predict(df_pred)


    

### not a class method, throws error when made otherwise
def __run_discoveries__(base_model):
    base_model.fetch_models()
    return base_model

