
import pandas as pd


from pprint import pprint
from sklearn import metrics
from sklearn.metrics import make_scorer

from asd.relevance import relevance
from asd.relevance.ml.models import common
# from ml.models import common
import dataset_handler
# from test import dataset_handler
from asd.relevance.utils import rayer

# from utils import rayer


def use_tokamat_ds():
    df = dataset_handler.get_tokamat_dataset()
    df = df.reset_index()

    df = common.label_encode(df)
    df = df.fillna(-1)

    potential_targets = ['WTOT', 'WTH', 'PLTH']
    df_y = df[potential_targets[0]]

    df_X = df[df.columns[~df.columns.isin(potential_targets)]]
    df_X = df_X.drop(['TOK_ID', 'LCUPDATE', 'DATE', 'NEL', 'ENBI'], axis = 1)
    
    return df_X, df_y


def use_covid_ds():
    df_X, df_y = dataset_handler.get_covid_dataset()
    df_X = df_X.drop(['location'], axis = 1)

    df_X = df_X.reset_index()

    df_X = common.label_encode(df_X)
    df_X = df_X.fillna(-1)


    return df_X, df_y


def test_case_1():
    df_X, df_y = use_covid_ds()
    df = pd.concat([df_X, df_y], axis=1)

    input_columns = df_X.columns.values
    target = df_y.columns.values


    mean_squared_error = make_scorer(score_func=metrics.mean_squared_error, greater_is_better=False)
    

    ### the above two are small datasets so better keep the tree depths low
    options = {
        'base_models' : ['briskxgboost', 'slugxgboost', 'slugrf', 'briskknn', 'sluglgbm', 'briskbagging'],
        'pred_class': 'regression',
        'xgb_objective': 'count:poisson',
        'lgbm_objective': 'poisson',
        'score_func': mean_squared_error,
        'metric_func': metrics.r2_score,
        'n_trials' : 300,
        'boosted_round': 200,
        'max_depth': 20,
        'rf_n_estimators': 10000,
        'bagging_estimators' : 50,
        'n_neighbors': 50,
        'cv_splits': 3,
        'ensemble_n_estimators': 10,
        'ensemble_n_trials': 10,
        
        'attr_algos' : ['IG', 'SHAP', 'GradientSHAP', 'knockoffs'],
        'fdr': 0.1,
        'fstats': ['lasso', 'ridge', 'randomforest'],
        'knockoff_runs' : 20000
    }        

    # rayer.get_local_cluster(num_cpus=4)

    ret = relevance(df, input_columns, target, options)

    pprint(ret)


def test_case_2():
    df_X, df_y = use_tokamat_ds()
    df = pd.concat([df_X, df_y], axis=1)

    input_columns = df_X.columns.values
    target = 'WTOT'

    r2_scoring = make_scorer(score_func=metrics.r2_score, greater_is_better=True)
    ### slugrf (rf_n_estimators=5000, max_depth=20), KNN(n_neighbors=50) sluglgbm(boosted_round=100, max_depth=20), briskxgboost(boosted_round=1000)
    options = {
        'base_models' : ['briskxgboost', 'slugxgboost', 'slugrf', 'briskknn', 'sluglgbm', 'briskbagging'],
        'pred_class': 'regression',
        'xgb_objective': 'count:poisson',
        'lgbm_objective': 'poisson',
        'score_func': r2_scoring,
        'metric_func': metrics.r2_score,
        'n_trials' : 100,
        'boosted_round': 100,
        'max_depth': 20,
        'rf_n_estimators': 1000,
        'bagging_estimators' : 50,
        'n_neighbors': 50,
        'cv_splits': 3,
        'ensemble_n_estimators': 50,
        'ensemble_n_trials': 50,
        
        'attr_algos' : ['IG', 'SHAP', 'GradientSHAP', 'knockoffs'],
        'fdr': 0.1,
        'fstats': ['lasso', 'ridge', 'randomforest'],
        'knockoff_runs' : 2000
    }

    ret = relevance.relevance(df, input_columns, target, options)

    pprint(ret)

        
if __name__ == '__main__':    
    # rayer.get_local_cluster()
    # rayer.get_global_cluster()
    test_case_2()
    
    
    