import numpy as np
import pandas as pd
import traceback
import time
import datetime
import math
import os
import json
from datetime import datetime
from typing import Union
from sklearn.preprocessing import StandardScaler
from helper_data.global_vars import *
from asd_logging import logger


class class_preprocess_data:

    '''
    utility class for preprocessing data
    '''

    def __init__(self):
        pass


    def reduce_file_size(self, data, nth_row):
        '''
        reducing file size for the first steps of the dim reduction method.
        data size is determined as ncols*nrows.

        :param data: high dimensional data
        :param nth_row: int with number of rows to keep, if 3 for example it takes every 3rd row
                        'auto' for automatic calculation or int for every nth row.
        :return: dataset with every nth row of custom dataset
        '''
        lowest_limit = 10000
        if nth_row == 'auto':
            try:
                data_size = data.shape[0] * data.shape[1]
                if data_size <= lowest_limit:
                    nth_row = 1
                else:
                    nth_row = 1 + int(round(data_size/lowest_limit,0))
            except:
                logger.error(f"{globalstring_error} DATA SIZE REDUCE", exc_info=True)

        else:
            if not isinstance(nth_row, int):
                logger.info(f"{globalstring_error}DATA SIZE REDUCE nth row must be integer, take every row instead")
                nth_row = 1 #

        # reduce number of rows by taking every nth row
        data_reduced = data[::nth_row]

        logger.info(f"{globalstring_info}REDUCE NUMBER OF ROWS FROM: {data.shape[0]} TO: {data_reduced.shape[0]} NCOLS: {data_reduced.shape[1]}")
        return data_reduced



    def scaling(self, X_train: Union[np.array, pd.DataFrame],
                      X_test: Union[np.array, pd.DataFrame] = None) -> np.array:
        '''
        standart scaling of data (X_train, default) and X_test if provided.
        :param X_train: array wit data to be scaled
        :param X_test: array wit data to be scaled
        :return: array with scaled data
        '''
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
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
            logger.info(f"DATA preprocess: {data.shape} {status} columns: {cols}")
            return [status, cols]
        except:
            logger.error("STATUS OF DATAFRAME ESTIMATION ERROR", exc_info=True)
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


    # def normalizing(self, X_train, X_test=None):
    #     '''
    #     standart scaling of data (X_train, default) and X_test if provided.
    #     :param X_train:
    #     :param X_test:
    #     :return:
    #     '''
    #     sc = MinMaxScaler()
    #     X_train = sc.fit_transform(X_train)
    #     if X_test != None:
    #         X_test  = sc.transform(X_test)
    #         return X_train, X_test
    #     else:
    #         return X_train


    # def main_preprocess(self, data, method):
    #     '''
    #     preprocessing of data
    #     :param data: data to be preprocessed, pd.DataFrame or Array
    #     :param method: which method to be used.
    #     :return:
    #     '''
    #     if method == 'scaling':
    #         return self.scaling(X_train=data)
    #     elif method == 'normalize':
    #         return self.normalizing(X_train=data)
    #     elif method == None:
    #         return data
    #     else: print('data preprocesss method not specified')





class helper:

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
            logger.info(f"DIRECTORY DOES NOT EXIT path: {path}")



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



########### DECORATORS
# timing of wrapped function
def timit_(fun_id):
    def function(function_to_be_decorated):
        def real_wrapper(*args, **kwargs):
           start = time.time()
           result = function_to_be_decorated(*args, **kwargs)
           logger.info(f"timit: {fun_id} {round(time.time()- start,2)}")
           return result
        return real_wrapper
    return function


def daytime_day():
    '''
    returns the actual time in format: DAY-MONTH-YEAR: 12-07-1979
    :return: string of actual day
    '''
    DAY = datetime.now().day
    MONTH = datetime.now().month
    YEAR = datetime.now().year
    return str(str(DAY) + '-' + str(MONTH) + '-' + str(YEAR))


''' INFORMATION
json.loads take a string as input and returns a dictionary as output.

json.dumps take a dictionary as input and returns a string as output.
'''

