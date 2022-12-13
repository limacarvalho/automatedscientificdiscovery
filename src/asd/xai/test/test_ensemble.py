



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import helper, rayer
from ml.models.ensemble import Ensemble

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from ml.models import common

from test import dataset_handler

import pandas as pd

import time

# list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']



def use_tokamat_ds():
    df = dataset_handler.get_tokamat_dataset()
    df = df.reset_index()

    df = common.label_encode(df)
    df = df.fillna(-1)

    potential_targets = ['WTOT', 'WTH', 'PLTH']
    df_y = df[potential_targets[0]]

    df_X = df[df.columns[~df.columns.isin(potential_targets)]]
    df_X = df_X.drop(['TOK_ID', 'LCUPDATE', 'DATE', 'NEL', 'ENBI'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33)

    ens_mdl = Ensemble(   
                                xgb_objective='count:poisson',  # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]
                                lgbm_objective='poisson',    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
                                pred_class='regression',
                                score_func='neg_mean_absolute_error',
                                metric_func= mean_squared_error,
                                list_base_models= ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm', 'slugrf'],
                                n_trials=100,
                                boosted_round=100,
                                max_depth=30,
                                rf_n_estimators=1500,
                                bagging_estimators=100,
                                n_neighbors=30,
                                cv_splits=3,
                                ensemble_bagging_estimators=50,
                                ensemble_n_trials=50,
                                timeout=None
                 )

    return ens_mdl, X_train, X_test, y_train, y_test


def use_covid_ds():
    df_X, df_y = helper.get_covid_dataset()
    df_X = df_X.drop(['location'], axis = 1)

    df_X = df_X.reset_index()

    df_X = common.label_encode(df_X)
    df_X = df_X.fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33)
    
    ens_mdl = Ensemble(   
                                xgb_objective='count:poisson',  # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]
                                lgbm_objective='poisson',    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
                                pred_class='regression',
                                score_func='neg_mean_absolute_error',
                                metric_func='r2',
                                list_base_models=['briskbagging', 'briskknn'],
                                n_trials=100,
                                boosted_round=100,
                                max_depth=30,
                                rf_n_estimators=1500,
                                bagging_estimators=100,
                                n_neighbors=30,
                                cv_splits=3,
                                ensemble_bagging_estimators=50,
                                ensemble_n_trials=50,
                                timeout=None
                 )

    
    ens_mdl.fetch_models(X_train, X_test, y_train, y_test)

    print('########### Base Model Scores ###################')
    print(ens_mdl.base_model_scores)

    print('############# Ensemble Score #################')
    print(ens_mdl.scores)


    return ens_mdl, X_train, X_test, y_train, y_test






def main():

    rayer.get_global_cluster(num_cpus=50)

    # r2_scoring = make_scorer(score_func=r2_score, greater_is_better=False)

    st = time.time()

    # ens_mdl, X_train, X_test, y_train, y_test = use_transaction_predictions_ds()     
    ens_mdl, X_train, X_test, y_train, y_test = use_tokamat_ds()
    #ens_mdl, X_train, X_test, y_train, y_test = use_covid_ds()

    ens_mdl.fetch_models(X_train, X_test, y_train, y_test)

    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('########### Execution time ###################')
    print(str(elapsed_time) + ' seconds')

    print('########### Base Model Scores ###################')
    print(ens_mdl.base_model_scores)

    print('############# Ensemble Score #################')
    print(ens_mdl.scores)

    print('############## Predictions ################')
    print(ens_mdl.predict(X_test))

    rayer.stop_global_cluster()

if __name__ == '__main__':
    main()