


import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from asd.xai.ml.models.ensemble import Ensemble

from asd.xai.ml.xai.model import Explainable


from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from asd.xai.ml.models import common


import time
import pandas as pd

import time

# list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']

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
                                list_base_models=['briskbagging', 'sluglgbm'],
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


def use_covid_ds():
    df_X, df_y = helper.get_covid_dataset()
    df_X = df_X.drop(['location'], axis = 1)

    return train_test_split(df_X, df_y, test_size=0.33)


def main():
    rayer.get_global_cluster()

    # r2_scoring = make_scorer(score_func=r2_score, greater_is_better=False)

    st = time.time()

    ensemble_set, X_train, X_test, y_train, y_test = use_tokamat_ds()

    ensemble_set.fetch_models_pll(X_train, X_test, y_train, y_test)

    et = time.time()

    # get the execution time
    elapsed_time_model_dis = et - st



    st = time.time()
    attr_algos = ['IG', 'SHAP', 'GradientSHAP'] #, 'knockoff']

    ex = Explainable(ensemble_set, X_train)             
    ex.get_attr(attr_algos)

    et = time.time()

    # get the execution time
    elapsed_time_xai = et - st


    ret = {
        'base_model_scores': ensemble_set.base_model_scores,
        'score': ensemble_set.scores,
        'xai_model': ex.df_scores,
        #'xai_non_model': df_knockoffs,
#            'raw': ex.raw
    }

    print('############# Final Resultset #################')
    print(ret)


    print('########### Model Discovery Execution time ###################')
    print(str(elapsed_time_model_dis) + ' seconds')


    print('########### Base Model Scores ###################')
    print(ensemble_set.base_model_scores)


    print('############# Ensemble Score #################')
    print(ensemble_set.scores)


    print('########### Xai Execution time ###################')
    print(str(elapsed_time_xai) + ' seconds')



    # print('############## Predictions ################')
    # print(ens_mdl.predict(X_test))



if __name__ == '__main__':
    main()