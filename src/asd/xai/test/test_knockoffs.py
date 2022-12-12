
import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is

import os
import numpy as np
import pandas as pd
import traceback

from sklearn.model_selection import train_test_split

from ml.models import common

from ml.xai.non_model import KnockoffSetting, simulate_knockoffs




from utils import helper, config, rayer, kaggle_dataset_helper
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
    
    return df_X, df_y


def use_covid_ds():
    df_X, df_y = helper.get_covid_dataset()
    df_X = df_X.drop(['location'], axis = 1)

    return df_X, df_y



def use_transaction_predictions_ds():
    ds_train, ds_test = kaggle_dataset_helper.get_transaction_predictions_dataset()
    ds_train = common.label_encode(ds_train)
    ds_test = common.label_encode(ds_test)

    ds_train = ds_train.fillna(-1)
    ds_test = ds_test.fillna(-1)

    df_X = ds_train.loc[:, ds_train.columns != 'target']
    df_y = ds_train['target']

    return df_X, df_y
    



def use_house_pricing_ds():
    ds_train, ds_test = kaggle_dataset_helper.get_house_prices_dataset()
    ds_train = common.label_encode(ds_train)
    ds_test = common.label_encode(ds_test)

    ds_train = ds_train.fillna(-1)
    ds_test = ds_test.fillna(-1)

    df_X = ds_train.loc[:, ds_train.columns != 'SalePrice']
    df_y = ds_train['SalePrice']

    return df_X, df_y

    


if __name__ == '__main__':
    
    print('########### Connecting ray cluster ###################')

    rayer.get_global_cluster(num_cpus=45)

    try:

        st = time.time()

        df_X, df_y = use_covid_ds()

        np.random.seed(KnockoffSetting.SEED)

        X = df_X
        y = df_y


        itr = 20000
        fdr = 0.1
        fstats = ['lasso', 'ridge', 'randomforest']


        for i in range(0, 1000):
            df_knockoffs = simulate_knockoffs(fdr, fstats, itr=itr, df_X=X, df_y=y)
            file_name='df_knockoffs_' + str(i) + '_' +  str(fdr) +'.csv'
            df_knockoffs.to_csv(file_name, sep=';')

        et = time.time()


        # get the execution time
        elapsed_time = et - st
        print('########### Execution time ###################')
        print(str(elapsed_time/60) + ' seconds')


    except:
    # printing stack trace
        print(traceback.print_exc())