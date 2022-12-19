from helper_data.utils_data import Preprocess_data
import numpy as np
from utils_logger import  logger

class Py_helper:

    '''helper to run python dimensionality reduction functions.'''

    def __init__(self, fun_id: str, data_high: np.array, dim_low: int, function):
        '''
        :param fun_id: str,
            function identifier
        :param data_high: np.array,
            high dimensional data
        :param dim_low: int,
            low dimension
        '''
        self.fun_id = fun_id
        self.data_high = data_high
        self.dim_low = dim_low
        self.function = function


    def exe_python_functions(self, params_py_dict: dict) -> (np.array, dict):
        '''
        runs the fit_transform function.
        in case there are hyperparameters provided, the default hyperparameters of the function
        are overwriten. In case the function is nmf (non-negative-matrix factorization) we scale the
        data to positive values by adding the lowest negative number to all numbers
        of the dataset.
        :param params_py_dict: dict,
            parameters, values of hyperparameters of dim reduce function
        :return: np.array, dict,
            np.array:
                reduced data at target dimension,
            dict:
                dictionary with hyperparameters
        '''
        # set the hyperparameters for dimred function in case there are
        if len(params_py_dict) != 0:
            self.function = self.function.set_params(**params_py_dict)
        self.function = self.function.set_params(n_components=self.dim_low)

        # nmf works with only positive values, rescale the data (only positive numbers)
        logger.info(msg='start dimreduction: ' +self.fun_id)
        if self.fun_id == "py_nmf":
            data_pos = Preprocess_data().positive_scale(self.data_high)
            data_low = self.function.fit_transform(data_pos)

        # all other functions
        else:
            data_low = self.function.fit_transform(self.data_high)

        return data_low, params_py_dict
