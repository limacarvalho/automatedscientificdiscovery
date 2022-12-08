

from ml.models import common
from utils import helper, config
from utils.asd_logging import logger as  customlogger

from ml.models.base.v2.brisk_xgboost import BriskXGBoost
from ml.models.base.v2.slug_ann import SlugANN
from ml.models.base.v2.slug_xgboost import SlugXGBoost
from ml.models.base.v2.slug_rf import SlugRF
from ml.models.base.v2.slug_knn import SlugKNN
from ml.models.base.v2.brisk_bagging import BriskBagging



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
                 list_base_models=[],
                 
                 n_trials=100,          ### common param
                 epochs=15,             ### ANN param
                 boosted_round=10,      ### boosting tree param
                 max_depth=30,          ### boosting tree param
                 max_n_estimators=1500, ### rf param
                 n_estimators=50,       ### bagging param
                 n_neighbors=50,        ### knn param
                 
                 ensemble_n_estimators=10,
                 ensemble_n_trials=10,
                 timeout=7200) -> None:

        self.score = None
        self.base_model_scores = []
        self.slug_rf = None
        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_n_trials = ensemble_n_trials
        
        self.base_models = []
        
        
        # print(list_base_models)
        
        list_base_models = [x.lower() for x in list_base_models]
        
        
        if len(list_base_models)==0:            
            list_base_models = ['briskxgboost', 'slugxgboost', 'slugann', 'slugrf', 'slugknn', 'briskbagging']
            # ['briskxgboost', 'slugxgboost', 'slugann', 'slugrf', 'slugknn', 'briskbagging']
            
        
        for model in list_base_models:
            if model=='briskxgboost':
                brisk_xgb = BriskXGBoost('brisk_xgb', timeout=timeout)
                brisk_xgb.boosted_round = boosted_round
                brisk_xgb.n_trials = n_trials
                self.base_models.append(brisk_xgb)
            if model=='slugxgboost':
                slug_xgb = SlugXGBoost('slug_xgb', timeout=timeout)
                slug_xgb.boosted_round = boosted_round
                slug_xgb.max_depth = max_depth
                slug_xgb.n_trials = n_trials
                self.base_models.append(slug_xgb)
            if model=='slugann':
                slug_ann = SlugANN('slug_ann', timeout=timeout)
                slug_ann.epochs = epochs
                slug_ann.n_trials = n_trials
                self.base_models.append(slug_ann)
            if model=='slugrf':
                slug_rf = SlugRF('slug_rf', timeout=timeout)
                slug_rf.max_n_estimators = max_n_estimators
                slug_rf.n_trials = n_trials
                self.base_models.append(slug_rf)
            if model=='slugknn':
                slug_knn = SlugKNN('slug_knn', timeout=timeout)
                slug_knn.n_neighbors = n_neighbors
                slug_knn.n_trials = n_trials # 2000
                self.base_models.append(slug_knn)
            if model=='briskbagging':
                brisk_bagging = BriskBagging('brisk_bagging', timeout=timeout)
                brisk_bagging.n_estimators = n_estimators
                brisk_bagging.n_trials = n_trials # 2000                
                self.base_models.append(brisk_bagging)
        
        
    
        #self.base_models = [self.brisk_xgb, self.slug_xgb, self.slug_ann]

                
        
    def fetch_models(self, X_train, X_test, y_train, y_test, threshold=None):
        
        lazy_results = []
        # results = None
        self.base_model_scores = []
                
        customlogger.info("Ensemble: starting discovery process for models " + str(self.base_models))
                                    
        X_train_id = ray.put(X_train)
        y_train_id = ray.put(y_train)
        X_test_id = ray.put(X_test)
        y_test_id = ray.put(y_test)
        
        
        for base_model in self.base_models:
            lazy_results.append(__run_discoveries__.remote(base_model, X_train_id, X_test_id, y_train_id, y_test_id, None, threshold))
            

        self.base_models = ray.get(lazy_results)
                        
        customlogger.info("base model training completed")
        
        for base_model in self.base_models:
            if not helper.check_if_all_same(base_model.score, None):
                temp_scores = base_model.score
                temp_scores.append(base_model)
                self.base_model_scores.append(temp_scores)


        customlogger.info("Ensemble: base model scores loaded, access via 'base_model_scores'")
        
        self.fit(X_train, X_test, y_train, y_test)
        
        
        customlogger.info("Ensemble: Cleanning storage")
        del X_train_id
        del X_test_id
        del y_train_id
        del y_test_id
        


    def load_scores(self,  X_train, X_test, y_train, y_test, score_func=None, threshold=None):
        self.base_model_scores = []
        for base_model in self.base_models:
            base_model.load_score(X_train, X_test, y_train, y_test, score_func, threshold)
            if len(base_model.score)>0:
                temp_scores = base_model.score
                temp_scores.append(base_model)
                self.base_model_scores.append(temp_scores)


        customlogger.info("Ensemble: base model scores loaded, access via 'base_model_scores'")                
        #self.score = None

        
    
    def fit(self,  X_train, X_test, y_train, y_test):

        df_X_train_ensemble = pd.DataFrame()
        df_X_test_ensemble = pd.DataFrame()
        
        for base_model in self.base_models:
            temp_X_train = pd.DataFrame(base_model.predict(X_train))
            temp_X_test = pd.DataFrame(base_model.predict(X_test))            
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
        self.ensemble = BriskBagging('ensemble_brisk_bagging')
        self.ensemble.n_estimators = self.ensemble_n_estimators
        self.ensemble.n_trials = self.ensemble_n_trials

        self.ensemble = ray.get(__run_discoveries__.remote(self.ensemble, df_X_train_ensemble, df_X_test_ensemble, y_train, y_test, None, None))
                
        self.score = self.ensemble.score
        #self.ensemble = self.brisk_xgb


    
    
    def predict(self, df_X):

        df_pred = pd.DataFrame()
        
        #ss = StandardScaler()
        # df_X_scalar = pd.DataFrame(ss.fit_transform(df_X), columns = df_X.columns)
        
        for base_model in self.base_models:
            if 'SlugANN' in str(base_model):
                pred = pd.DataFrame(base_model.predict(df_X))
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
def __run_discoveries__(base_model, X_train, X_test, y_train, y_test, score_func, threshold):
    base_model.fetch_model(X_train, X_test, y_train, y_test, score_func, threshold)
#    base_model.load_score()
    return base_model

