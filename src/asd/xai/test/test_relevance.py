
import pandas as pd
from asd.xai.ml.models import common
from asd.xai.utils.asd_logging import logger as  customlogger
from asd.xai.utils import dataset_handler, rayer
from asd.xai.relevance import relevance

from pprint import pprint




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

    return df_X, df_y


def test_case_1():
    df_X, df_y = use_covid_ds()
    df = pd.concat([df_X, df_y], axis=1)

    input_columns = df_X.columns.values
    target = df_y.columns.values


    options = {
        'threshold': None,
        'base_models' : None,

        'n_trials' : 100,
        'boosted_round': 15,
        'max_depth': 30,
        'max_n_estimators': 150,
        'n_estimators' : 500,
        'n_neighbors': 50,
        'ensemble_n_estimators': 100,
        'ensemble_n_trials': 100,
        
        'attr_algos' : ['IG', 'SHAP', 'GradientSHAP', 'knockoffs'], 
        'fdr': 0.2,
        'fstats': ['lasso', 'ridge', 'randomforest'],
        'knockoff_runs' : -1
    }        

    # rayer.get_local_cluster(num_cpus=4)

    ret = relevance(df, input_columns, target, options)

    pprint(ret)



def test_case_2():
    df_X, df_y = use_tokamat_ds()
    df = pd.concat([df_X, df_y], axis=1)

    input_columns = df_X.columns.values
    target = df_y.columns.values


    options = {
        'threshold': None,
        'base_models' : None,

        'n_trials' : 100,
        'boosted_round': 15,
        'max_depth': 30,
        'max_n_estimators': 150,
        'n_estimators' : 500,
        'n_neighbors': 50,
        'ensemble_n_estimators': 100,
        'ensemble_n_trials': 100,
        
        'attr_algos' : ['IG', 'SHAP', 'GradientSHAP', 'knockoffs'], 
        'fdr': None,
        'fstats': ['lasso', 'ridge', 'randomforest'],
        'knockoff_runs' : 10
    }        

    ret = relevance(df, input_columns, target, options)

    pprint(ret)

        
if __name__ == '__main__':

#    rayer.get_local_cluster(num_cpus=4)

    test_case_1()
    
    
    