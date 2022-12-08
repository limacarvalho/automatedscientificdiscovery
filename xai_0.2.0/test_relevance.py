import ray
import pandas as pd
from utils import helper, config, kaggle_dataset_helper

import traceback
import sys, getopt

from ml.models import common
from utils.asd_logging import logger as  customlogger
from pprint import pprint
from relevance import relevance

        
        
if __name__ == '__main__':
    
    inputfile=0
    opt, arg = getopt.getopt(sys.argv, "i")
    
    if arg[1] == '-i':
        inputfile = arg[2]
            
    if inputfile == '0':
        df_X, df_y = helper.get_covid_dataset()
        df_X = df_X.drop(['location'], axis = 1)
        df = pd.concat([df_X, df_y], axis=1)

        input_columns = df_X.columns.values
        target = df_y.columns.values
    
    #elif inputfile=='1':        
    else:
        ds_train, ds_test = kaggle_dataset_helper.get_house_prices_dataset()
        ds_train = common.label_encode(ds_train)
        ds_test = common.label_encode(ds_test)
        ds_train = ds_train.fillna(-1)
        ds_test = ds_test.fillna(-1)

        df_X = ds_train.loc[:, ds_train.columns != 'SalePrice']
        df_y = ds_train['SalePrice']

        df = pd.concat([df_X, df_y], axis=1)
        
        input_columns = df_X.columns.values
        target = 'SalePrice'
        
    
    
    print('dataset id:' + str(inputfile))
    print('dataset shape:' + str(df.shape))
    

    # list_fstats = ['lasso', 'ridge', 'randomforest']
    
    options = {
        'threshold': None,
        'base_models' : ['BriskXGBoost', 'slugxgboost', 'slugann', 'slugknn', 'briskbagging'],
        #'naiv_models' : ['Lasso', 'Ridge', 'Elastic', 'Nova', 'slugknn', 'briskbagging'],

        'n_trials' : 100,
        'epochs': 150,
        'boosted_round': 15,

        'max_depth': 30,
        'max_n_estimators': 150,

        'n_estimators' : 500,
        'n_neighbors': 50,

        'ensemble_n_estimators': 100,
        'ensemble_n_trials': 100,

        # 'attr_algos' : ['knockoff', 'SHAP'],
        
        'attr_algos' : ['IG', 'SHAP', 'GradientSHAP', 'knockoffs'], # if knockoff is given, fdr, fstats and knockoff_runs can also be provided, defaults are fdr=0.1, knockoff_runs=1000, fstats=['lasso', 'ridge', 'randomforest']
        'fdr': 0.1,
        'fstats': ['lasso', 'ridge', 'randomforest'],
        'knockoff_runs' : 10000
    }        
    
    pprint(options)
    
    ret = relevance(df, input_columns, target, options)

    pprint(ret)
    
    if ret is None:
        sys.exit(1)
        
    pprint(ret)
    ret['df_scores'].to_csv('df_scores.csv', sep=';')
    
    
    