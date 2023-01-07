



import numpy as np
import pandas as pd
import traceback

from sklearn.model_selection import train_test_split

from relevance.ml.models import common
import dataset_handler
from relevance.ml.xai.non_model import KnockoffSetting, simulate_knockoffs
from relevance.utils import helper, rayer
import time




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



if __name__ == '__main__':

    print('########### Connecting ray cluster ###################')

    rayer.get_local_cluster()

    try:

        st = time.time()

        df_X, df_y = use_covid_ds()

        np.random.seed(KnockoffSetting.SEED)

        X = df_X
        y = df_y


        itr = 20
        fdr = 0.1
        fstats = ['lasso', 'ridge', 'randomforest']


        for i in range(0, 10):
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