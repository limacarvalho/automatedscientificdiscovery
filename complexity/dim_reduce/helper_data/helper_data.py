import numpy as np
import pandas as pd
import random
import traceback
import time
import datetime
import math
import os
import json
from datetime import datetime
from typing import Union
from sklearn.preprocessing import StandardScaler
from dimension_tools.dimension_suite.dim_reduce.helper_data.global_vars import *


class Preprocess_data:

    '''
    utility class for preprocessing data: scaling size reduction etc.
    '''

    def __init__(self):
        pass


    def reduce_helper(
              self,
              data: np.array,
              nrows: int,
              nrows_reduced: int
        ) -> np.array:
        '''
        helper for reduction of data size.
        :param np.array data: raw data, need to be numeric
        :param int nrows: number of rows of np.array (shape[0])
        :param int nrows_reduced:  number of target rows
        :return: np.array with reduced number of rows
        '''
        if nrows < 500:
            return data
        else:
            idx_rows = sorted(random.sample(range(0, nrows), nrows_reduced))
        return data[idx_rows]


    def reduce_file_size(self,
                         data: np.array,
                         percent_of_rows: Union[str,int]
                         ) -> np.array:
        '''
        reducing file size for the first steps of the dim reduction sequence with hundrets
        of dimensionality reductions.
        in case the percentage is 'auto': data size is 5 rows per column, in case of very
        small datasets (<250 rows) we keep all columns. Rows are choosen randomly.
        :param np.array data: high dimensional data
        :param int percent_of_rows: default 'auto' for automatic calculation.
            in case a value 0...100 is provided it means the percentage of rows to use.
        :return: np.array dataset with every nth row of original dataset
        '''
        min_nrows = 250
        nrows = data.shape[0]
        ncols = data.shape[1]
        data_reduced = data
        start = time.time()

        # we calculate the size of the small dataset
        if percent_of_rows == 'auto':

            # if number of rows of original dataset is smaller than minimum number of rows
            if nrows < min_nrows:
                data_reduced = data

            # if number of rows of original dataset is bigger than minimum number of rows
            else:
                try:
                    nrows_reduced = min(int(ncols * 5), nrows)

                    # dataset with less than 50 columns but more than 250 rows.
                    if nrows_reduced < min_nrows <= nrows:
                        # 1500 is set arbitrary, it should work for datasets less than aprox 15.000 rows
                        nrows_reduced = max(0.2 * nrows, 1500)
                    data_reduced = self.reduce_helper(data, nrows-1, nrows_reduced)
                except:
                    print(globalstring_error + ' DATA SIZE REDUCE')
                    print(traceback.format_exc())

        # the customer has predefined the percentage of rows
        else:
            if isinstance(percent_of_rows, int) and (0 < percent_of_rows <= 100):
                if percent_of_rows == 100:
                    data_reduced = data
                else:
                    nrows_reduced = min(int(nrows * int(percent_of_rows/100)), nrows)
                    data_reduced = self.reduce_helper(data, nrows-1, nrows_reduced)
            else:
                print(globalstring_error + 'DATA SIZE REDUCE nth row must be integer, and between'
                                           ' 0 and 100; dataset is not reduced')
                print(traceback.format_exc())
                data_reduced = data

        # TODO: remove prints, make logger message
        print(globalstring_info, 'REDUCE NUMBER OF ROWS FROM: ',
              data.shape[0], ' TO: ', data_reduced.shape[0], 'NCOLS: ', data_reduced.shape[1],
              'TIMIT: ', round(time.time() - start, 2))

        return data_reduced


    def scaling(self, X_train: Union[np.array, pd.DataFrame],
                      X_test: Union[np.array, pd.DataFrame] = None) -> np.array:
        '''
        standart scaling of data (X_train, default) and X_test if provided.
        speed : 10.000rows * 100cols = 0.0sec >>>

        :param X_train: array wit data to be scaled
        :param X_test: array wit data to be scaled
        :return: array with scaled data
        '''
        start = time.time()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        print('SCALING, TIMIT: ', round(time.time() - start, 2))
        if X_test != None:
            X_test  = sc.transform(X_test)
            return X_train, X_test
        else:
            return X_train


    def check_scale_status_data(self, data: np.array) -> list:
        '''
        function to observe if columns are scaled, normalized or not preprocessed,
        and how many are.
        :return: list with status (scaled, normalized or not preprocessed)
                 and percentage of those columns
        '''
        data = pd.DataFrame(data)
        try:
            data_numeric = data.select_dtypes(include=[np.number])
            names = ['NOT_PREPROCESSED', 'STD_SCALED', 'NORMALIZED']
            no_preprocess, data_scaled, data_normalized = 0,0,0
            for col in data_numeric.columns:
                mea = np.round(np.mean(data[col]),1)
                std = np.round(np.std(data[col]),1)
                min = np.round(np.min(data[col]),1)
                max = np.round(np.max(data[col]),1)
                if max-min == 1:
                    data_normalized = data_normalized + 1
                elif mea == 0 and std == 1:
                    data_scaled = data_scaled + 1
                else: no_preprocess = no_preprocess +1
            numbers = [no_preprocess, data_scaled, data_normalized]
            Max = np.max(numbers)
            max_idx = numbers.index(Max)
            status = names[max_idx]
            cols = round(Max/data_numeric.shape[1]*100, 0)
            print('DATA preprocess:', data.shape, status, 'columns: ', cols)
            return [status, cols]
        except:
            print('STATUS OF DATAFRAME ESTIMATION ERROR')
            print(traceback.format_exc())
            return [0, 0]


    def preprocess_scaling(self, data: Union[np.array, pd.DataFrame]) -> (np.array, list):
        '''
        scaling of data and check how many columns are scaled.
        :param data: raw data that need to be scaled
        :return: array of scaled data and scaling status of columns
        '''
        data_scaled = self.scaling(X_train=data)
        status = self.check_scale_status_data(data_scaled)
        return data_scaled, status


    def positive_scale(self, data: np.array) -> np.array:
        '''
        converts negative values in dataframe into positive values.
        Used in non-negative-matrix-factorization.
        :param data: DataFame potentially containing negative values
        :return: array with only positive values
        '''
        min_value = data.min()
        if min_value.min() < 0:
            data = data + abs(min_value)
        return data



class Helper:

    def __init__(self):
        pass


    def save_csv(self, data: Union[np.array, pd.DataFrame], path: str):
        '''
        save csv file
        :param data: data table to be saved
        :param path: complete path 'dirbla/blabla.csv'
        '''
        data.to_csv(path, index=False)


    def erase_file(self, path: str):
        '''
        remove file if exists
        :param path: string of path
        '''
        if os.path.exists(path) == True:
            os.remove(path)


    def check_if_dir_exists(self, path: str):
        '''
        check if directory exist and print error message if dont.
        :param path: directory path
        '''
        if os.path.isdir(path) == True:
            pass
        else:
            print('DIRECTORY DOES NOT EXIT path:', path)



def check_dim(ndim: int, ncol: int) -> bool:
    '''
    ckecks if dimension has the correct format and is within the correct range: 1...ncols
    :param ndim: target dimension for reduction
    :param ncol: number of columns in dataframe
    :return: False if not correct, True if correct
    '''
    if type(ndim) != int or math.isinf(ndim) or math.isnan(ndim):
        return False
    # now numric checks, cant be more than ncol or less than 1
    elif ndim > ncol or ndim < 1:
        return False
    else:
        return True


def empty_dict(fun_id: str, dim_low: Union[int, None], kmax: int) -> dict:
    '''
    Returns empty dictionary in case something failes in one of the functions.
    -------

    '''
    dict_empty = {'Q': np.array([[1, 2], [1, 2]]),
                  'kmax': kmax,
                  'trust': 0,
                  'cont': 0,
                  'lcmc': 0,
                  'mae': 0,
                  'mae_norm': 0,
                  'rmse': 0,
                  'r2': 0,
                  'hyperparameters': json.dumps({'empty': 'empty'}), # dict
                  'time': 0,
                  'fun_id': fun_id,
                  'dim_low': dim_low,
                  'init_steps': 0,
                  'iter_steps': 0
                  }
    return dict_empty



def flatten_list_dicts(list_dicts: list) -> list:
    '''
    converts inputs into list of dicts
    :param list_dicts: list of dicts which sometimes is not the case
    :return: list of dicts
    '''
    list_dicts_flat = []

    if isinstance(list_dicts, list):
        for items in list_dicts:
            # results is list of lists of dicts
            if isinstance(items, list):
                for it in items:
                    list_dicts_flat.append(it)
            # results is lists of dicts
            elif isinstance(items, dict):
                list_dicts_flat.append(items)
            else:
                list_ = list_dicts
                print(globalstring_error, 'PLOT ITEMS OF LIST, UNKNOWN DATA-TYPE, must be dict or list')

    # results lists of dicts
    elif isinstance(list_dicts, dict):
        list_dicts_flat.append(list_dicts)
    #
    else:
        list_dicts_flat = list_dicts
        print(globalstring_error, 'PLOT ITEMS OF LIST, UNKNOWN DATA-TYPE, must be dict or list')
    return list_dicts_flat



########### DECORATORS
# timing of wrapped function
def timit_(fun_id):
    def function(function_to_be_decorated):
        def real_wrapper(*args, **kwargs):
           start = time.time()
           result = function_to_be_decorated(*args, **kwargs)
           print('timit: ', fun_id, round(time.time()- start,2))
           return result
        return real_wrapper
    return function



''' INFORMATION
json.loads take a string as input and returns a dictionary as output.

json.dumps take a dictionary as input and returns a string as output.
'''

# def daytime_day():
#     '''
#     returns the actual time in format: DAY-MONTH-YEAR: 12-07-1979
#     :return: string of actual day
#     '''
#     DAY = datetime.now().day
#     MONTH = datetime.now().month
#     YEAR = datetime.now().year
#     return str(str(DAY) + '-' + str(MONTH) + '-' + str(YEAR))