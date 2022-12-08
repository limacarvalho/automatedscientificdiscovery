

from ml.models import common
from utils import helper, config
from utils.asd_logging import logger as  customlogger

from ml.models.base.brisk_xgboost import BriskXGBoost
from ml.models.base.slug_ann import SlugANN
from ml.models.base.slug_xgboost import SlugXGBoost
from ml.models.base.slug_rf import SlugRF
from ml.models.base.slug_knn import SlugKNN
from ml.models.base.brisk_bagging import BriskBagging


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import numpy as np
import pandas as pd

import signal
import ray



### change the logger type to save info logs
# customlogger = logger.logging.getLogger('console_info')

class Ensemble:
    def __init__(self, 
                 df_X,
                 df_y,
                 list_base_models=[],
                 n_trials=100,
                 epochs=15,
                 boosted_round=10,
                 
                 ensemble_boosted_round=10,
                 ensemble_n_trials=10,
                 timeout=7200) -> None:


        self.score = None
        self.base_model_scores = []
        self.slug_rf = None
        self.ensemble_boosted_round = ensemble_boosted_round
        self.ensemble_n_trials = ensemble_n_trials
        
        # config.project_name = '/' + project_name #house_prices_advanced_regression_techniques_ensemble'

        self.df_X = df_X
        self.df_y = df_y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_X, 
                                                                                df_y, test_size=0.33,
                                                                                random_state=config.rand_state)

        # ss = StandardScaler()
        #self.X_train_scalar = pd.DataFrame(ss.fit_transform(self.X_train), columns = self.X_train.columns)
        # self.X_test_scalar = pd.DataFrame(ss.fit_transform(self.X_test), columns = self.X_test.columns)
        
        self.base_models = []
        
        if len(list_base_models)==0:
            # list_base_models = ['BriskXGBoost', 'SlugXGBoost', 'SlugANN', 'SlugRF', 'SlugKNN', 'BriskBagging']
            list_base_models = ['briskxgboost', 'slugxgboost', 'slugann', 'slugrf', 'slugknn', 'briskbagging']
        
        for model in list_base_models:
            if model=='briskxgboost':
                brisk_xgb = BriskXGBoost('brisk_xgb', self.X_train, self.X_test, self.y_train, self.y_test, timeout=timeout)
                brisk_xgb.boosted_round = boosted_round
                brisk_xgb.n_trials = n_trials
                self.base_models.append(brisk_xgb)
            if model=='slugxgboost':
                slug_xgb = SlugXGBoost('slug_xgb', self.X_train, self.X_test, self.y_train, self.y_test, timeout=timeout)
                slug_xgb.boosted_round = boosted_round
                slug_xgb.n_trials = n_trials
                self.base_models.append(slug_xgb)
            if model=='slugann':
                slug_ann = SlugANN('slug_ann', self.X_train, self.X_test, self.y_train, self.y_test, timeout=timeout)
                slug_ann.epochs = epochs
                slug_ann.n_trials = n_trials
                self.base_models.append(slug_ann)
            if model=='slugrf':
                slug_rf = SlugRF('slug_rf', self.X_train, self.X_test, self.y_train, self.y_test)
                slug_rf.max_n_estimators = 1500
                slug_rf.n_trials = n_trials
                self.base_models.append(slug_rf)
            if model=='slugknn':
                slug_knn = SlugKNN('slug_knn', self.X_train, self.X_test, self.y_train, self.y_test)
                slug_knn.n_neighbors = 50
                slug_knn.n_trials = n_trials # 2000
                self.base_models.append(slug_knn)
            if model=='briskbagging':                
                brisk_bagging = BriskBagging('brisk_bagging', self.X_train, self.X_test, self.y_train, self.y_test)
                brisk_bagging.n_estimators = 50
                brisk_bagging.n_trials = n_trials # 2000                
                self.base_models.append(brisk_bagging)
        
        
        #self.base_models = [self.brisk_xgb, self.slug_xgb, self.slug_ann]

                
        
    def fetch_models(self, score_func=None, threshold=None):
        
        lazy_results = []
        results = None
        self.base_model_scores = []
                
        customlogger.info("Ensemble: starting discovery process for models " + str(self.base_models))
        
        for base_model in self.base_models:
            lazy_results.append(__run_discoveries__.remote(base_model, score_func, threshold))
            

        self.base_models = ray.get(lazy_results)
                        
        customlogger.info("base model training completed")
        
        for base_model in self.base_models:
            if not helper.check_if_all_same(base_model.score, None):
                temp_scores = base_model.score
                temp_scores.append(base_model)
                self.base_model_scores.append(temp_scores)


        customlogger.info("Ensemble: base model scores loaded, access via 'base_model_scores'")
        
        


    def load_scores(self, score_func=None, threshold=None):
        self.base_model_scores = []
        for base_model in self.base_models:
            base_model.load_score(score_func, threshold)
            if len(base_model.score)>0:
                temp_scores = base_model.score
                temp_scores.append(base_model)
                self.base_model_scores.append(temp_scores)


        customlogger.info("Ensemble: base model scores loaded, access via 'base_model_scores'")                
        #self.score = None

        
    
    def fit(self):

        df_X_train_ensemble = pd.DataFrame()
        df_X_test_ensemble = pd.DataFrame()
        
        for base_model in self.base_models:
            temp_X_train = pd.DataFrame(base_model.predict(self.X_train))
            temp_X_test = pd.DataFrame(base_model.predict(self.X_test))
            
            df_X_train_ensemble = pd.concat([df_X_train_ensemble, temp_X_train], axis=1)
            df_X_test_ensemble = pd.concat([df_X_test_ensemble, temp_X_test], axis=1 )


            
        customlogger.info("Ensemble: fiting ensemble on " + str(df_X_train_ensemble.shape[1]) + " models")  

        ### rename cols
        col_range = range(0, df_X_train_ensemble.shape[1])
        col_ids = []

        for i in col_range:
            col_ids.append(str(i))
        df_X_train_ensemble.columns = col_ids
        df_X_test_ensemble.columns = col_ids

        
        ### fit Random forest on the all the base models to get one single outcome
        self.ensemble = BriskBagging('ensemble_brisk_bagging', df_X_train_ensemble, df_X_test_ensemble, self.y_train, self.y_test)
        self.ensemble.n_estimators = 50
        self.ensemble.n_trials = self.ensemble_n_trials

        self.ensemble = ray.get(__run_discoveries__.remote(self.ensemble, None, None))
                
        self.score = self.ensemble.score
        #self.ensemble = self.brisk_xgb


    
    
    def predict(self, df_X) :

        df_pred = pd.DataFrame()
        
        ss = StandardScaler()
        df_X_scalar = pd.DataFrame(ss.fit_transform(df_X), columns = df_X.columns)
        

        for base_model in self.base_models:
            if 'SlugANN' in str(base_model):
                pred = pd.DataFrame(base_model.predict(df_X_scalar))
            else:
                pred = pd.DataFrame(base_model.predict(df_X))
            
            df_pred = pd.concat([df_pred, pred], axis=1)

        

        ### rename cols
        col_range = range(0, df_pred.shape[1])
        col_ids = []

        for i in col_range:
            col_ids.append(str(i))
        df_pred.columns = col_ids
        df_pred.columns = col_ids
        
        return self.ensemble.predict(df_pred)
    
    
### not a class method, throws error when made otherwise
@ray.remote
def __run_discoveries__(base_model, score_func, threshold):
    base_model.fetch_model(score_func, threshold)
#    base_model.load_score()
    return base_model

