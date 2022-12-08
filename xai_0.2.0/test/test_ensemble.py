


import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is


from utils import helper, config, rayer, kaggle_dataset_helper

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ml.models.ensemble import Ensemble

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from ml.models import common



import pandas as pd

import time


def use_tokamat_ds():
    df = helper.get_tokamat_dataset()
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
                                score_func=None,
                                metric_func=None,
                                list_base_models=[],
                                n_trials=100,          ### common param
                                epochs=15,             ### ANN param
                                boosted_round=10,      ### boosting tree param
                                max_depth=30,          ### boosting tree param
                                max_n_estimators=1500, ### rf param
                                n_estimators=30,       ### bagging param, must be > 10 
                                n_neighbors=30,        ### knn param, must be > 5

                                ensemble_n_estimators=30,  ###  must be > 10
                                ensemble_n_trials=10,
                                timeout=None
                 )

    return ens_mdl, X_train, X_test, y_train, y_test


def use_covid_ds():
    df_X, df_y = helper.get_covid_dataset()
    df_X = df_X.drop(['location'], axis = 1)

    return train_test_split(df_X, df_y, test_size=0.33)



def use_house_pricing_ds():
    ds_train, ds_test = kaggle_dataset_helper.get_house_prices_dataset()
    ds_train = common.label_encode(ds_train)
    ds_test = common.label_encode(ds_test)

    ds_train = ds_train.fillna(-1)
    ds_test = ds_test.fillna(-1)


    df_X = ds_train.loc[:, ds_train.columns != 'SalePrice']
    df_y = ds_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33)

    
    ens_mdl = Ensemble(   
                                xgb_objective='count:poisson',  # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]
                                lgbm_objective='poisson',    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
                                pred_class='regression',
                                score_func=None,
                                metric_func=None,
                                list_base_models=[],
                                n_trials=10,           ### common param
                                epochs=15,             ### ANN param
                                boosted_round=10,      ### boosting tree param
                                max_depth=30,          ### boosting tree param
                                max_n_estimators=1500, ### rf param
                                n_estimators=30,       ### bagging param, must be > 10 
                                n_neighbors=30,        ### knn param, must be > 5

                                ensemble_n_estimators=30,  ###  must be > 10
                                ensemble_n_trials=10,
                                timeout=None
                 )

    return ens_mdl, X_train, X_test, y_train, y_test



def use_transaction_predictions_ds():
    ds_train, ds_test = kaggle_dataset_helper.get_transaction_predictions_dataset()
    ds_train = common.label_encode(ds_train)
    ds_test = common.label_encode(ds_test)

    ds_train = ds_train.fillna(-1)
    ds_test = ds_test.fillna(-1)

    df_X = ds_train.loc[:, ds_train.columns != 'target']
    df_y = ds_train['target']
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=config.rand_state)

    r2_scoring = make_scorer(score_func=r2_score, greater_is_better=False)

    ens_mdl = Ensemble(   
                                xgb_objective='binary:logistic',  # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]
                                lgbm_objective='binary',    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
                                pred_class='classification',
                                score_func=None,
                                metric_func=None,
                                list_base_models=[],
                                n_trials=10,          ### common param
                                epochs=15,             ### ANN param
                                boosted_round=10,      ### boosting tree param
                                max_depth=30,          ### boosting tree param
                                max_n_estimators=1500, ### rf param
                                n_estimators=30,       ### bagging param, must be > 10 
                                n_neighbors=30,        ### knn param, must be > 5

                                ensemble_n_estimators=30,  ###  must be > 10
                                ensemble_n_trials=10,
                                timeout=None
                 )

    return ens_mdl, X_train, X_test, y_train, y_test



def main():

    rayer.get_global_cluster(num_cpus=45)

    # r2_scoring = make_scorer(score_func=r2_score, greater_is_better=False)

    st = time.time()

    # ens_mdl, X_train, X_test, y_train, y_test = use_transaction_predictions_ds()     
    ens_mdl, X_train, X_test, y_train, y_test = use_tokamat_ds()

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



if __name__ == '__main__':
    main()