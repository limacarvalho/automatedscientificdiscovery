import numpy as np
import pandas as pd
import random
import time
import math
import os
import json
from typing import Union
from sklearn.preprocessing import StandardScaler
from utils_logger import logger


class Preprocess_data:

    '''
    utility class for preprocessing data: scaling, size reduction etc.
    '''

    def __init__(self):
        pass


    def reduce_helper(self, data: np.array, nrows: int, nrows_reduced: int ) -> np.array:
        '''
        reduction of number of rows of dataset with random selection of row indices.
        :param data: np.array raw data, need to be numeric
        :param nrows: int, number of rows to select for reduced dataset
        :param nrows_reduced: int, number of target rows
        :return: np.array with reduced number of rows
        '''
        if nrows < 500:
            return data
        else:
            idx_rows = sorted(random.sample(range(0, nrows), nrows_reduced))
        return data[idx_rows]


    def reduce_file_size(self, data: np.array, percent_of_rows: Union[str,int]) -> np.array:
        '''
        reducing file size for the first steps of the dim reduction sequence with hundrets
        of dimensionality reductions.
        in case the percentage is 'auto': data size is 5 rows per column, in case of very
        small datasets (<250 rows) we keep all columns. Rows are choosen randomly.
        :param data: np.array, high dimensional data
        :param percent_of_rows: int, default 'auto' for automatic calculation.
            in case a value 0...100 is provided it means the percentage of rows to use.
        :return: np.array, reduced dataset randomly selected rows
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
                    nrows_reduced = max( min(int(ncols * 5), nrows), int(0.08*nrows) )
                    data_reduced = self.reduce_helper(data, nrows-1, nrows_reduced)
                except:
                    logger.error(msg='reduce data size', exc_info=True)

        # the customer has predefined the percentage of rows
        else:
            if isinstance(percent_of_rows, int) and (0 < percent_of_rows <= 100):
                if percent_of_rows == 100:
                    data_reduced = data
                else:
                    nrows_reduced = min(int(nrows * int(percent_of_rows/100)), nrows)
                    data_reduced = self.reduce_helper(data, nrows-1, nrows_reduced)
            else:
                data_reduced = data
                logger.error(msg='nth row must be integer, and 0...100, dataset is not reduced', exc_info=True)

        logger.info(msg=('REDUCE NUMBER OF ROWS FROM: ' + str(data.shape[0])
                         + ' TO: ' + str(data_reduced.shape[0])
                         + ' NCOLS: ' + str(data_reduced.shape[1])
                         + ' TIMIT: ' + str(round(time.time() - start, 2)) ))

        return data_reduced


    def scaling(self, X_train: Union[np.array, pd.DataFrame],
                      X_test: Union[np.array, pd.DataFrame] = None) -> np.array:
        '''
        standart scaling of data (X_train, default) and X_test if provided.
        speed : 10.000rows * 100cols = 0.0sec >>>

        :param X_train: Union[np.array, pd.DataFrame], data to be scaled
        :param X_test: Union[np.array, pd.DataFrame], if test data are provided
            they will be scaled with the same model lie the train data.
        :return: np.array, scaled data, in case of test data are provided,
            train and test data will be returned
        '''
        start = time.time()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        logger.info(msg=(' SCALING, TIMIT: ' + str(round(time.time() - start, 2))))
        if X_test != None:
            X_test  = sc.transform(X_test)
            return X_train, X_test
        else:
            return X_train


    def check_scale_status_data(self, data: np.array) -> list:
        '''
        function to observe if columns are scaled, normalized or not preprocessed,
        and how many are.
        :return: list, status (scaled, normalized or not preprocessed) and percentage
                of those columns
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
            logger.info(msg=(' DATA PREPROCESS: ' + str(data.shape) + ' ' + str(status) + ' columns: ' + str(cols)))
            return [status, cols]
        except:
            logger.error(msg='calculation status of column', exc_info=True)
            return [0, 0]


    def check_array(self, array_scaled: np.array):
        '''

        :param array_scaled:
        :param columns:
        :return:
        '''
        try:
            cols_not_finite = []
            for i in range(array_scaled.shape[1]):
                if np.isinf(array_scaled[:, i]).any() or np.isnan(array_scaled[:, i]).any():
                    cols_not_finite.append(i)
            return cols_not_finite
        except:
            logger.error(msg='checking array for infinitye values', exc_info=True)
            return []



    def preprocess_scaling(self, data: Union[np.array, pd.DataFrame]) -> (np.array, list):
        '''
        scaling of data and check how many columns are scaled.
        :param data: Union[np.array, pd.DataFrame], raw data that need to be scaled
        :return: (np.array, list), array of scaled data and scaling status of columns
        '''
        ##
        if isinstance(data, pd.DataFrame):
            colnames = data.columns
        else:
            colnames = list(data.dtype.names)

        data_scaled = self.scaling(X_train=data)
        nan_columns = self.check_array(data_scaled)

        if len(nan_columns) > 0:
            data_scaled = np.delete(data_scaled, nan_columns, axis=1)
            colnames_nan = [colnames[i] for i in nan_columns]
            logger.error(msg=('remove columns with infinite or NaN values after scaling: '+str(colnames_nan)))

        status = self.check_scale_status_data(data_scaled)
        return data_scaled, status


    def positive_scale(self, data: np.array) -> np.array:
        '''
        converts negative values in dataframe into positive values.
        Used in non-negative-matrix-factorization.
        :param data: np.array, containing negative values
        :return: np.array, with positive scaled values
        '''
        min_value = data.min()
        if min_value.min() < 0:
            data = data + abs(min_value)
        return data



class Helper:
    'utilities for data handling'

    def __init__(self):
        pass


    def save_csv(self, data: Union[np.array, pd.DataFrame], path: str):
        '''
        save csv file without indizes column
        :param data: Union[np.array, pd.DataFrame], data table to be saved
        :param path: str, complete path 'dirbla/blabla.csv'
        '''
        data.to_csv(path, index=False)


    def erase_file(self, path: str):
        '''
        remove file if exists
        :param path: str, string of path
        '''
        if os.path.exists(path):
            os.remove(path)


    def check_if_dir_exists(self, path: str):
        '''
        check if directory exist and print error message if dont.
        :param path: str, directory path
        '''
        if os.path.isdir(path):
            pass
        else:
            logger.info(msg=('directory does not exist, path: ' + path))



def check_dim(ndim: int, ncol: int) -> bool:
    '''
    ckecks if dimension has the correct format and is within the correct range: 1...ncols
    :param ndim: int, target dimension for reduction
    :param ncol: int, number of columns in dataframe
    :return: bool, False if not correct, True if correct
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
    empty dictionary which will be returned in case something goes wrong during
    dim reduction or quality assessment. In this way, the function is tracked
    and program runs smoothly.
    :param fun_id: str, function identifier
    :param dim_low: int,
    :param kmax: int, number of neighbors
    :return: dict, dictionary with dummy variables
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
                logger.error(msg='flatten items of list, unknown data type, must be dict or list')

    # results lists of dicts
    elif isinstance(list_dicts, dict):
        list_dicts_flat.append(list_dicts)
    #
    else:
        list_dicts_flat = list_dicts
        logger.error(msg='flatten items of list, unknown data type, must be dict or list')
    return list_dicts_flat



########### DECORATORS
# timing of wrapped function
def timit_(fun_id):
    def function(function_to_be_decorated):
        def real_wrapper(*args, **kwargs):
           start = time.time()
           result = function_to_be_decorated(*args, **kwargs)
           logger.info(msg=(' timit: ' + fun_id + str(round(time.time()- start,2)) + 'sec'))
           return result
        return real_wrapper
    return function