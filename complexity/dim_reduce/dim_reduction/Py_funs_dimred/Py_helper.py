from dimension_tools.dimension_suite.dim_reduce.helper_data.helper_data import (
    Preprocess_data,
)
from dimension_tools.dimension_suite.dim_reduce.dim_reduction.dimred_call_functions import (
    call_dimred_functions,
)
import numpy as np
import pyper as pr

r = pr.R(use_pandas=True)
r("library(Rdimtools)")
r('Sys.setenv(LANGUAGE="en")')


class Py_helper:

    def __init__(self,
                 fun_id: str,
                 data_high: np.array,
                 dim_low: int, function
                 ):

        """
        helper to run python dim reduction functions.
        :param str  fun_id: function identifier
        :param np.array  data_high: high dimensional data
        :param int  dim_low: low dimension
        """

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

        :param dict params_py_dict: dictionary with hyperparameters of dim reduce function
        :return: np.array  reduced data at target dimension, dictionary with hyperparameters
        '''
        # set the hyperparameters in case there are
        if len(params_py_dict) != 0:
            self.function = self.function.set_params(**params_py_dict)
        self.function = self.function.set_params(n_components=self.dim_low)

        # nmf needs data with only positive values, scale the data to only positive numbers
        if self.fun_id == "py_nmf":
            data_pos = Preprocess_data().positive_scale(self.data_high)
            data_low = self.function.fit_transform(data_pos)

        # all other functions
        else:
            data_low = self.function.fit_transform(self.data_high)

        return data_low, params_py_dict
