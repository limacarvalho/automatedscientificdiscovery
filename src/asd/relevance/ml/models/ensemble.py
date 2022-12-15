

from asd.relevance.ml.models import common
from asd.relevance.utils import helper, config
from asd.relevance.utils.asd_logging import logger as  customlogger

from asd.relevance.ml.models.base.tune import BriskBagging, BriskKNN, BriskXGBoost, SlugXGBoost, SlugLGBM, SlugRF

from sklearn.metrics import make_scorer
from sklearn import metrics

import numpy as np
import pandas as pd


import ray


### change the logger type to save info logs
# customlogger = logger.logging.getLogger('console_info')

list_base_models = ['briskxgboost', 'slugxgboost', 'slugrf', 'briskknn', 'briskbagging', 'sluglgbm']
                              


class Ensemble:
    def __init__(self,
                 xgb_objective,
                 lgbm_objective,
                 pred_class,
                 score_func=None,
                 metric_func=None,
                 list_base_models=[],
                 n_trials=100,
                 boosted_round=100,
                 max_depth=30,
                 rf_n_estimators=1500,
                 bagging_estimators=100,
                 n_neighbors=30,
                 cv_splits=3,
                 ensemble_bagging_estimators=50,
                 ensemble_n_trials=50,
                 timeout=None) -> None:
        '''    
        The goal of ensemble class is to fit several base models and then fit an ensemble model to achieve automated learning.\n
        Arg:\n
        \txgb_objective (str): objective function if xgboost model is given in list_base_model. I.e., default='binary:logistic', see doc. of XGBoost for more details.
        \tlgbm_objective (str): objective function if lightgbm model is given in list_base_model.I.e., default='binary', see doc. of LightGBM for more details.
        \tpred_class (str): specify problem type, i.e., 'regression' or 'classification'.
        \tscore_func (str, callable, or None): A single string or a callable to evaluate the predictions on the test set. See https://scikit-learn.org/stable/modules/model_evaluation.html
        \tmetric_func (str): sklearn.metrics
        \tlist_base_models (list): list of base models to be used to fit on the data. I.e., ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']
        \tn_trials (int): . Default to 100
        \tboosted_round (int): n_estimators parameter for XGBoost and LightGBM. Default to 100
        \tmax_depth (int): max tree depth parameter for XGBoost, LightGBM and RandomForest. Default to 30
        \trf_n_estimators (int): n_estimators parameter of RandomForest. Default to 1500
        \tbagging_estimators (int): n_estimators parameter of Bagging. Default to 100
        \tn_estimators (int): The number of trees in the forest parameter of RandomForest. Default to 1500
        \tn_neighbors (int): n_neighbors of KNN. Default to 30
        \tcv_splits (int): Determines the cross-validation splitting strategy. I.e., cv_split=3
        \tensemble_bagging_estimators (int): n_estimators parameter of Bagging. This is the second baggin method which is used an an ensemble on top of base estimators. Default to 50
        \tensemble_n_trials (int): Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution. Default to 50.
        '''

        self.scores = None
        self.base_model_scores = []


        self.xgb_objective = xgb_objective
        self.lgbm_objective = lgbm_objective        


        self.ensemble_bagging_estimators = ensemble_bagging_estimators
        self.ensemble_n_trials = ensemble_n_trials
        self.ensemble = None

        
        self.base_models = []

        self._base_models = []
                        
        
        list_base_models = [x.lower() for x in list_base_models]
        
        
        if (score_func is None) and (pred_class=='regression'): 
            score_func = 'neg_mean_absolute_error'

        if (score_func is None) and (pred_class=='classification'): 
            score_func =  make_scorer(metrics.log_loss, greater_is_better=False)


        if len(list_base_models)==0:            
            list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']
            # ['briskxgboost', 'slugxgboost', 'slugann', 'slugrf', 'slugknn', 'briskbagging']
            
        
        for model in list_base_models:
            if model=='briskbagging':
                brisk_bagging = BriskBagging(name='brisk_bagging',
                                                                        pred_class=pred_class, 
                                                                        n_estimators=bagging_estimators,
                                                                        n_trials=n_trials,
                                                                        score_func=score_func,
                                                                        metric_func=metric_func,
                                                                        cv_splits=cv_splits,
                                                                        timeout=timeout)

                self._base_models.append(brisk_bagging)

            if model=='briskknn':
                brisk_knn = BriskKNN(name='brisk_knn',
                                                            pred_class=pred_class, 
                                                            score_func=score_func,
                                                            metric_func=metric_func,
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
                                                                    metric_func=metric_func,
                                                                    cv_splits=cv_splits,
                                                                    timeout=timeout,
                                                                    )
                self._base_models.append(brisk_xgb)

            if model=='slugxgboost':
                slug_xgb = SlugXGBoost(name='slug_xgboost',
                                                                objective=self.xgb_objective, 
                                                                pred_class=pred_class,
                                                                score_func=score_func,
                                                                metric_func=metric_func,                    
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
                                                                metric_func=metric_func,
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
                                                    metric_func=metric_func,                    
                                                    max_depth=max_depth,
                                                    max_n_estimators=rf_n_estimators,
                                                    n_trials=n_trials,
                                                    cv_splits=cv_splits,
                                                    timeout=None,
                                                )
                self._base_models.append(slug_rf)



        self.ensemble = BriskBagging(name='ensemble_brisk_bagging',
                                                                    pred_class=pred_class, 
                                                                    n_estimators=ensemble_bagging_estimators,
                                                                    n_trials=ensemble_n_trials,
                                                                    score_func=score_func,
                                                                    metric_func=None,
                                                                    cv_splits=cv_splits,
                                                                    timeout=timeout)

    

                
        
    def _fetch_models_pll(self, X_train, X_test, y_train, y_test, threshold=None):

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
        '''
        Fit the base and ensemble model on the training dataset and evaluate on test dataset.\n
        Arg:\n
        \tX_train: train variable dataset
        \t X_test: label trainset
        \ty_train: test variable dataset
        \t y_test: label testset
        '''

        customlogger.info("Ensemble: starting discovery process for models " + str(self._base_models))
                                            
        for base_model in self._base_models:
            base_model.fit(X_train, X_test, y_train, y_test)


        self.select(threshold)

        self.fit(X_train, X_test, y_train, y_test)
        
        customlogger.info("base model training completed")





    def select(self, threshold=None):
        '''
        Select only the models with goodness of fit > threshold.\n
        Arg:\n
        \tthreshold: 0 < threshold < 1
        '''

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
        '''
        Fit the ensemble model on the trained base models.\n
        Arg:\n
        \tX_train: train variable dataset
        \tX_test:  label trainset
        \ty_train: test variable dataset
        \ty_test:  label testset
        '''

        if self.base_models is None:
            customlogger.info("Fit base models first.")  
            return None

        if len(self.base_models)==1:
            self.ensemble = self.base_models[0]
            self.scores = self.base_models[0].scores
            return None


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
        '''
        Return predictions.\n
        Arg:\n
        \tdf_X: input dataframe
        '''

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

