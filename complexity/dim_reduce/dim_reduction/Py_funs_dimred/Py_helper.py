from helper_data.helper_data import class_preprocess_data
from dimred_call_functions import call_dimred_functions
import numpy as np
import pyper as pr
r = pr.R(use_pandas = True)
r('library(Rdimtools)')
r('Sys.setenv(LANGUAGE="en")')


class class_run_py_functions:


    def __init__(self,
                 fun_id: str,
                 data_high: np.array,
                 dim_low: int,
                 function):
        '''
        class to run python functions.

        :param fun_id: function identifier
        :param data_high: high dimensional data
        :param dim_low: low dimension
        '''

        self.fun_id = fun_id
        self.data_high = data_high
        self.dim_low = dim_low
        self.function = function


    def exe_python_functions(self, params_py_dict: dict) -> (np.array, dict):
        '''
        execution of python dim reduce functions with the fit_transform function.
        in case there are hyperparameters provided, we overwrite the default
        hyperparameters of the function.
        Incase the function is nmf (non-negative-matrix factorization) we scale the
        data to positive values by adding the lowest negative number to all numbers
        of the dataset.

        Parameters
        ----------
        params_py : dictionary with hyperparameters of dim reduce function

        Returns: array of data of lower dimension, dictionary with hyperparameters
        -------

        '''
        # set the hyperparameters in case there are
        if len(params_py_dict) != 0:
            self.function = self.function.set_params(**params_py_dict)
        self.function = self.function.set_params(n_components=self.dim_low)

        # nmf needs data with only positive values, scale the data to only positive numbers
        if self.fun_id == 'py_nmf':
            data_pos = class_preprocess_data().positive_scale(self.data_high)
            data_low = self.function.fit_transform(data_pos)

        # all other functions
        else:
            data_low = self.function.fit_transform(self.data_high)

        return data_low, params_py_dict




    ''' 
    here is some extra code for kernl pca, which is not used anymore due
    to too many errors
    kernel pca runs very unstable with a lot of error messages, removing 'poly' and
    related hyperparameters kernel reduces a few
    
    if self.fun_id == 'pca_kernel':  # replace nan and inf values
        print('parms', params_py)
        data_nonan = np.nan_to_num(self.data_high, nan=0, posinf=10, neginf=-10)
        data_low = self.fun.fit_transform(data_nonan)
    set parameters
    '''