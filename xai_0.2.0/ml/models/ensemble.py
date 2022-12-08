

from ml.models import common
from utils import helper, config
from utils.asd_logging import logger as  customlogger

from ml.models.base.tune import BriskBagging, BriskKNN, BriskXGBoost, SlugXGBoost, SlugLGBM, SlugRF


import numpy as np
import pandas as pd


import ray



### change the logger type to save info logs
# customlogger = logger.logging.getLogger('console_info')

class Ensemble:
    def __init__(self, 
                 xgb_objective,
                 lgbm_objective,
                 pred_class,
                 score_func=None,
                 metric_func=None,
                 list_base_models=[],
                 n_trials=100,          ### common param
                 epochs=15,             ### ANN param
                 boosted_round=10,      ### boosting tree param
                 max_depth=30,          ### boosting tree param
                 max_n_estimators=1500, ### rf param
                 n_estimators=30,       ### bagging param
                 n_neighbors=30,        ### knn param
                 cv_splits=3,
                 ensemble_n_estimators=10,
                 ensemble_n_trials=10,
                 timeout=None) -> None:


        self.scores = None
        self.base_model_scores = []


        self.xgb_objective = xgb_objective
        self.lgbm_objective = lgbm_objective        


        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_n_trials = ensemble_n_trials
        self.ensemble = None

        
        self.base_models = []

        self._base_models = []
                        
        
        list_base_models = [x.lower() for x in list_base_models]
        
        
        if len(list_base_models)==0:            
            list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']
            # ['briskxgboost', 'slugxgboost', 'slugann', 'slugrf', 'slugknn', 'briskbagging']
            
        
        for model in list_base_models:
            if model=='briskbagging':
                brisk_bagging = BriskBagging(name='brisk_bagging',
                                                                        pred_class=pred_class, 
                                                                        n_estimators=n_estimators,
                                                                        n_trials=n_trials,
                                                                        score_func=score_func,
                                                                        metric_func=None,
                                                                        cv_splits=cv_splits,
                                                                        timeout=timeout)

                self._base_models.append(brisk_bagging)

            if model=='briskknn':
                brisk_knn = BriskKNN(name='brisk_knn',
                                                            pred_class=pred_class, 
                                                            score_func=score_func,
                                                            metric_func=None,
                                                            n_neighbors=n_neighbors,
                                                            n_trials=n_trials,
                                                            timeout=timeout,
                                                            )
                self._base_models.append(brisk_knn)

            if model=='briskxgboost':
                brisk_xgb = BriskXGBoost(name='brisk_xgboost',
                                                                    objective=self.xgb_objective, 
                                                                    pred_class=pred_class, 
                                                                    n_estimators=boosted_round,
                                                                    n_trials=n_trials,
                                                                    score_func=score_func,
                                                                    metric_func=None,
                                                                    cv_splits=cv_splits,
                                                                    timeout=timeout,
                                                                    )
                self._base_models.append(brisk_xgb)

            if model=='slugxgboost':
                slug_xgb = SlugXGBoost(name='slug_xgboost',
                                                                objective=self.xgb_objective, 
                                                                pred_class=pred_class,
                                                                score_func=score_func,
                                                                metric_func=None,                    
                                                                n_estimators=boosted_round,
                                                                max_depth=max_depth,
                                                                n_trials=n_trials,
                                                                cv_splits=cv_splits,
                                                                timeout=None,
                                                                    )
                self._base_models.append(slug_xgb)

        
            if model=='sluglgbm':
                slug_lgbm = SlugLGBM(name='slug_lgbm',
                                                                objective=self.lgbm_objective,
                                                                pred_class=pred_class,
                                                                score_func=score_func,
                                                                metric_func=None,
                                                                n_estimators=boosted_round,
                                                                max_depth=max_depth,
                                                                n_trials=n_trials,
                                                                cv_splits=cv_splits,
                                                                timeout=None,
                                                                    )
                self._base_models.append(slug_lgbm)
        
            if model=='slugrf':
                slug_rf = SlugRF(name='slug_rf',
                                                    pred_class=pred_class,
                                                    score_func=score_func,
                                                    metric_func=None,                    
                                                    max_depth=max_depth,
                                                    max_n_estimators=max_n_estimators,
                                                    n_trials=n_trials,
                                                    cv_splits=cv_splits,
                                                    timeout=None,
                                                )
                self._base_models.append(slug_rf)



        self.ensemble = BriskBagging(name='ensemble_brisk_bagging',
                                                                    pred_class=pred_class, 
                                                                    n_estimators=ensemble_n_estimators,
                                                                    n_trials=ensemble_n_trials,
                                                                    score_func=score_func,
                                                                    metric_func=None,
                                                                    cv_splits=cv_splits,
                                                                    timeout=timeout)

    

                
        
    def fetch_models_pll(self, X_train, X_test, y_train, y_test, threshold=None):
        
        lazy_results = []
        # results = None
                
        customlogger.info("Ensemble: starting discovery process for models " + str(self._base_models))
                                    
        X_train_id = ray.put(X_train)
        y_train_id = ray.put(y_train)
        X_test_id = ray.put(X_test)
        y_test_id = ray.put(y_test)

                
        for base_model in self._base_models:
            lazy_results.append(__run_discoveries__.remote(base_model, X_train_id, X_test_id, y_train_id, y_test_id))
            

        self._base_models = ray.get(lazy_results)

        self.select(threshold)

        self.fit(X_train, X_test, y_train, y_test)
        
        customlogger.info("base model training completed")


        del X_train_id
        del X_test_id
        del y_train_id
        del y_test_id


    def fetch_models(self, X_train, X_test, y_train, y_test, threshold=None):
                        
        customlogger.info("Ensemble: starting discovery process for models " + str(self._base_models))
                                    
                
        for base_model in self._base_models:
            base_model.fit(X_train, X_test, y_train, y_test)
            
        self.select(threshold)

        self.fit(X_train, X_test, y_train, y_test)
        
        customlogger.info("base model training completed")





    def select(self, threshold=None):

        self.base_models = []
        self.base_model_scores = []


        if threshold is None:
            for base_model in self._base_models:
                self.base_models.append(base_model)
                self.base_model_scores.append(base_model.scores)
            return None


        if (threshold < 0.0) or (threshold > 1.0):
            for base_model in self._base_models:
                self.base_models.append(base_model)
                self.base_model_scores.append(base_model.scores)
            raise ValueError("0 < threshold < 1 ")


        for base_model in self._base_models:
            if base_model.scores[1] > threshold:
                self.base_models.append(base_model)
                self.base_model_scores.append(base_model.scores)

        customlogger.info("filtered out " +  str( len(self._base_models) - len(self.base_models) ) + " models.")


    
    def fit(self,  X_train, X_test, y_train, y_test):

        if self.base_models is None:
            customlogger.info("Fit base models first.")  
            return None

        if len(self.base_models)==1:
            self.ensemble = self.base_models[0]


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


        self.ensemble.fit(df_X_train_ensemble, df_X_test_ensemble, y_train, y_test)
        self.scores = self.ensemble.scores



    
    def predict(self, df_X):

        if self.base_models is None:
            customlogger.info("Fit base models first.")  
            return None

        if len(self.base_models)==1:
            return pd.DataFrame(self.base_models[0].predict(df_X))


        df_pred = pd.DataFrame()
        
        #ss = StandardScaler()
        # df_X_scalar = pd.DataFrame(ss.fit_transform(df_X), columns = df_X.columns)
        
        for base_model in self.base_models:
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
@ray.remote(num_returns=1)
def __run_discoveries__(base_model, X_train, X_test, y_train, y_test):
    base_model.fit(X_train, X_test, y_train, y_test)
    return base_model

