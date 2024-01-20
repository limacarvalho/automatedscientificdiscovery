import json
import logging
import math
import random
import time
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()


class Preprocess_data:
    """
    utility class for preprocessing data: scaling, size reduction etc.
    """

    def __init__(self):
        pass

    def reduce_helper(self, data: np.array, nrows: int, nrows_reduced: int) -> np.array:
        """
        reduction of number of rows of dataset with random selection of row indices.
        :param data: np.array raw data,
            needs to be numeric
        :param nrows: int,
            number of rows to select for reduced dataset
        :param nrows_reduced: int,
            number of target rows
        :return: np.array,
            with reduced number of rows
        """
        if nrows < 500:
            return data
        else:
            idx_rows = sorted(random.sample(range(0, nrows), nrows_reduced))
        return data[idx_rows]

    def reduce_file_size(self, data: np.array) -> np.array:
        """
        reducing file size for speeding up the search of the target dimension and hyperparameter
        optimization of dimensionality reduction functions.
        in case the percentage is 'auto': the number of rows is calculated as follows:
        A) small dataset with <250 rows: no reduction
        B) dataset with >250 rows: max( min(int(ncols * 5), nrows), int(0.08*nrows) )
        Rows are choosen randomly.
        :param data: np.array,
            high dimensional data
        :return: np.array,
            reduced dataset
        """
        min_nrows = 250
        nrows = data.shape[0]
        ncols = data.shape[1]
        data_reduced = data
        start = time.time()

        # if number of rows of original dataset is smaller than minimum number of rows
        if nrows < min_nrows:
            data_reduced = data

        # if number of rows of original dataset is bigger than minimum number of rows
        else:
            try:
                nrows_reduced = max(min(int(ncols * 5), nrows), int(0.08 * nrows))
                data_reduced = self.reduce_helper(data, nrows - 1, nrows_reduced)
            except:
                logging.error("reduce data size")

        logging.info(
            f"REDUCE NUMBER OF ROWS FROM: {str(data.shape[0])} TO: {str(data_reduced.shape[0])} NCOLS: {str(data_reduced.shape[1])} TIMIT: {str(round(time.time() - start, 2))}"
        )
        return data_reduced

    def scaling(self, X_train: Union[np.array, pd.DataFrame], X_test: Union[np.array, pd.DataFrame] = None) -> np.array:
        """
        standart scaling of data (X_train, default) and X_test if provided.
        speed : 10.000rows * 100cols = 0.0sec >>>

        :param X_train: Union[np.array, pd.DataFrame],
            data to be scaled
        :param X_test: Union[np.array, pd.DataFrame],
            if test data are provided (not used here)
            they will be scaled with the same model lie the train data.
        :return: np.array,
            scaled high dimensional data
        """
        start = time.time()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        logging.info(f" SCALING, TIMIT: {str(round(time.time() - start, 2))}")
        if X_test is not None:
            X_test = sc.transform(X_test)
            return X_train, X_test
        else:
            return X_train

    def check_scale_status_data(self, data: np.array) -> list:
        """
        observe if the scaling of the data: check which percentage of columns is scaled, normalized
        or not preprocessed.
        Standart scaling: In case the percentage is lower than 100 percent: its an indication
        that data are not well distributed (categorical values) or columns with only one value etc.
        :return: list,
            status (scaled, normalized or not preprocessed) and percentage
        """
        data = pd.DataFrame(data)
        try:
            data_numeric = data.select_dtypes(include=[np.number])
            names = ["NOT_PREPROCESSED", "STD_SCALED", "NORMALIZED"]
            no_preprocess, data_scaled, data_normalized = 0, 0, 0
            for col in data_numeric.columns:
                mea = np.round(np.mean(data[col]), 1)
                std = np.round(np.std(data[col]), 1)
                min = np.round(np.min(data[col]), 1)
                max = np.round(np.max(data[col]), 1)
                if max - min == 1:
                    data_normalized = data_normalized + 1
                elif mea == 0 and std == 1:
                    data_scaled = data_scaled + 1
                else:
                    no_preprocess = no_preprocess + 1
            numbers = [no_preprocess, data_scaled, data_normalized]
            Max = np.max(numbers)
            max_idx = numbers.index(Max)
            status = names[max_idx]
            cols = round(Max / data_numeric.shape[1] * 100, 0)
            logging.info(f" DATA PREPROCESS: {str(data.shape)} {str(status)} columns: {str(cols)}")
            return [status, cols]
        except:
            logging.error("calculation status of column")
            return [0, 0]

    def check_array(self, array_scaled: np.array):
        """
        observes array for any infinito or nan values
        :param array_scaled: np.array,
            scaled dataset
        :return: list,
            list with columns containing infonite of nan values
        """
        try:
            cols_not_finite = []
            for i in range(array_scaled.shape[1]):
                if np.isinf(array_scaled[:, i]).any() or np.isnan(array_scaled[:, i]).any():
                    cols_not_finite.append(i)
            return cols_not_finite
        except:
            logging.error("checking array for infinitye values")
            return []

    def preprocess_scaling(self, data: Union[np.array, pd.DataFrame]) -> (np.array, list):
        """
        Preprocessing of data and scaling with some posterior checks.
        1)
        :param data: Union[np.array, pd.DataFrame],
            high dimensional data
        :return: (np.array, list),
            np.array:
                array of scaled data
            list:
                scaling status of columns
        """
        # get column names and remove non numeric columns
        # pd.DataFrame, remove object columns
        if isinstance(data, pd.DataFrame):
            colnames = data.columns
            data_ = data.select_dtypes(exclude=["object"])
            colnames_objects = list(set(data.columns).difference(data_.columns))
            logging.warning(f"remove object type columns PRIOR scaling: {str(colnames_objects)}")
            data = data_
        # np.array
        elif data.dtype.names:
            colnames = list(data.dtype.names)
        # else: columns get numbers as colnames 0...n columns-1
        else:
            colnames = range(0, data.shape[1])

        # standart scaling
        data_scaled = self.scaling(X_train=data)
        # remove columns with inf, nan values
        nan_columns = self.check_array(data_scaled)
        if len(nan_columns) > 0:
            data_scaled = np.delete(data_scaled, nan_columns, axis=1)
            colnames_nan = [colnames[i] for i in nan_columns]
            logging.warning(f"remove columns with infinite or NaN values AFTER scaling: {str(colnames_nan)}")

        # check scaling status of columns
        status = self.check_scale_status_data(data_scaled)
        return data_scaled, status

    def positive_scale(self, data: np.array) -> np.array:
        """
        converts negative values in dataframe into positive values.
        Used in non-negative-matrix-factorization.
        :param data: np.array,
            containing negative values
        :return: np.array,
            with positive scaled values
        """
        min_value = data.min()
        if min_value.min() < 0:
            data = data + abs(min_value)
        return data


def save_csv(data: Union[np.array, pd.DataFrame], path: str):
    """
    save csv file without indizes added by pandas.
    :param data: Union[np.array, pd.DataFrame],
        dataframe to be saved
    :param path: str,
        complete path for saving data 'dir_bla/filename_bla.csv'
    """
    data.to_csv(path, index=False)


def check_low_dimension(dim_low: int, ncol: int) -> bool:
    """
    ckecks if dimension has the correct format and is within the correct range: 1...ncols
    :param dim_low: int,
        target dimension for reduction
    :param ncol: int,
        number of columns in dataframe
    :return: bool,
        False if not correct, True if correct
    """
    if type(dim_low) != int or math.isinf(dim_low) or math.isnan(dim_low):
        return False
    # now numric checks, cant be more than ncol or less than 1
    elif dim_low > ncol or dim_low < 1:
        return False
    else:
        return True


def empty_dict(fun_id: str, dim_low: Union[int, None]) -> dict:
    """
    dictionary with dummy values that will be returned in case something goes wrong during
    dimensionality reduction or quality assessment.
    :param fun_id: str,
        function identifier
    :param dim_low: int,
        low dimension
    :return: dict,
        dictionary with dummy variables
    """
    dict_empty = {
        "Q": np.array([[1, 2], [1, 2]]),  # np.array: coranking matrix
        "rel_err": 0.0,  # float: relative error 0...1 (1=perfect)
        "r2": 0.0,  # float: r-squared value
        "hyperparameters": json.dumps({"empty": "empty"}),  # str: string of hyperparameters 'hyperparameter=value'
        "time": 0.0,  # float: time for dimensionality reduction in seconds
        "fun_id": fun_id,  # str: function identifier, example: 'py_pca'
        "dim_low": dim_low,  # int: dimension (ncolumns)of low dimensional dataset
        "init_steps": 0,  # int: initialiation steps for hyerparameter optimization
        "iter_steps": 0,  # int: iteration steps for hyerparameter optimization
    }
    return dict_empty


def flatten_list_dicts(list_dicts: list) -> list:
    """
    converts inputs into list of dicts (which sometimes is not the case).
    :param list_dicts:
        should be a list of dicts
    :return: list,
        list of dicts
    """
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
                logging.error("flatten items of list, unknown data type, must be dict or list")

    # results lists of dicts
    elif isinstance(list_dicts, dict):
        list_dicts_flat.append(list_dicts)
    #
    else:
        list_dicts_flat = list_dicts
        logging.error("flatten items of list, unknown data type, must be dict or list")
    return list_dicts_flat


########### DECORATOR
# timing of wrapped function
def timit_(fun_id):
    def function(function_to_be_decorated):
        def real_wrapper(*args, **kwargs):
            start = time.time()
            result = function_to_be_decorated(*args, **kwargs)
            logging.info(f" timit: {fun_id} {str(round(time.time() - start, 2))} sec")
            return result

        return real_wrapper

    return function
