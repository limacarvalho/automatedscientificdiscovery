import numpy as np
from .R_funs_dimred.R_helper import R_helper
from .Py_funs_dimred.Py_helper import Py_helper
from hyperparameters.hyperparameter_initialization import hyperparameter_init
from dim_reduction.dimred_call_functions import call_dimred_functions
from helper_metrix.metrics_dimred import Metrics, fun_kmax
from helper_data.global_vars import *
from utils_logger import logger
from typing import Union
import time
import json
# important, do not erase, otherwise logs will be full with warnings
import warnings
warnings.filterwarnings('ignore')



class Dimreduction:
    '''
    Class to perform dimensionality reduction with one dim_red function and one dimension.
    '''

    def __init__(self, fun_id: str, data_high: np.array, dim_low: int):
        '''
        :param fun_id: str, function_identifier
        :param data_high: np.array, high dimensional data
        :param dim_low: int, low dimension
        '''

        self.fun_id = fun_id
        self.data_high = data_high
        self.dim_low = int(dim_low)

        # call function to receive the A) function object B) function default hyperparameters
        # C) Hyperparameter ranges for optimization
        dict_fun = call_dimred_functions(self.fun_id, self.data_high)
        self.fun, self.params_fun, hyperpars = list(dict_fun.values())[0]


    def hyperparameters_r_py(self, params: Union[str, dict], step: str) -> (dict, str):
        '''
        convert R and Python hyperparameters into the correct formt before calling the function.
        3 options: hyperparameters are used for
        A) the hyperparameter optimization step_2, in this case some of the hyperparameters will
            be provided in the wrong format (float) and need to be converted into the correct
            format first (category) by using the hyperparameter_initialization.py function.
        B) Other steps than hyperparameter optimization:
            in case of R hyperparameters its possible that they are provided as dictionaries and need
            to be converted first into a string.
        C) Other steps than hyperparameter optimization: In case of python hyperparameters its possible
            that they come as string and need to be converted into a dictionary format.
        :param params: Union[str, dict], hyperparameters: string for R functions, dict for Python functions
        :param step: str, for what is the function used?
        :return: (dict, str) hyperparameters for Python and R functions: string for R (params_r_str) and
            dict for Python (params_py_dict)
        '''
        # initialization
        params_r_str = ''
        params_py_dict = {}

        try:
        # Option 1: the parameters are used for hyperparameter optimization.
            # the params are returned by bayesian optimization program as floats and
            # need to be converted into categorical or integer values.
            if step == globalstring_step2:
                params_r_str, params_py_dict = hyperparameter_init(
                    params = params,
                    fun_id = self.fun_id,
                    dim = int(self.dim_low),
                    data_high = self.data_high
                )

        # Option 2: R-functions NOT used for hyperparameter optimization
            elif step != globalstring_step2 and isinstance(self.params_fun, str): # and self.params_fun[:4] == 'Rfun':
                # its a dict, convert to string (we need R hyperparameters in sring format)
                if isinstance(params, dict):
                    params_r_str = R_helper().dict_to_r_string(params)
                # its a string, do nothing
                else:
                    params_r_str = params

        # Option 3: Python-functions NOT used for hyperparameter optimization
            else:
                # its a dictionary, do nothing
                if isinstance(params_py_dict, dict):
                    params_py_dict = params
                # its a string, convert to dict (python functions need Python hyperparameters in dict format)
                else:
                    params_py_dict = json.loads(params)
        except:
            logger.error(msg=('INIT HYPERPARAMETER ' + step + self.fun_id + ' dim: ' + str(self.dim_low)),
                         exc_info=True)

        # we always return two things although one is always empty
        return params_py_dict, params_r_str


    def exe_dimreduce(self, params: dict, step: str) -> (np.array, dict):
        '''
        Single dimensionality reduction with given fun_id, low_dim and params.
        This main function reduces the dimensionality of the data and evaluates the
        quality. It returns the loss of choice (mae_normalized) and keeps all
        the quality information (losses, time, Q-matrix).

        :param params: dict, hyperparameters for dim reduce function
        :param step: str,
            'dim_reduce' one reduction with given fun_id, low_dim and params
            'hyperparameters' for hyperparameter optimization.
             categorical parameters are provided as floats by the bayes opt program and need to be converted
             to their original values.
             more info: hyperparameters -> hyperparameter_init
        :return: dict, dictionary with results and quality of dim reduction
        '''
        # initialization
        hyperparameters = ''
        data_low = np.array
        start = time.time()

        # preprocess the functions hyperparameters
        params_py_dict, params_r_str = self.hyperparameters_r_py(params, step)

        try:
            # R functions
            if isinstance(self.params_fun, str): # [:4] == 'Rfun':
                data_low, hyperparameters = R_helper().r_function_exe(
                    self.fun_id,
                    self.params_fun,
                    params_r_str,
                    self.data_high,
                    self.dim_low
                )
            # Python functions
            else:
                Py_funs = Py_helper(self.fun_id, self.data_high, self.dim_low, self.fun)
                data_low, hyperparameters = Py_funs.exe_python_functions(params_py_dict)
        except:
            logger.error(msg=('DIMREDUCE ' + step + self.fun_id + ' dim:' + str(self.dim_low)), exc_info=True)

        # time for dim reduction
        stop = round(time.time() - start, 3)

        # Measure quality of dim reduction, helper_metrix class retrurns empty dict in case of problems
        kmax = fun_kmax(self.data_high)
        metrix = Metrics(self.fun_id, self.data_high, data_low, kmax)
        dict_results = metrix.metrix_all()
        if step == globalstring_step3:
            logger.info(msg=('finito: ', self.fun_id))

        '''
        append results to list and return with the exception we make sure that something is returned 
        in case one or a few calculations fail.
        '''

        # step of the dim reduction sequence
        try:
            dict_results['step'] = step
        except:
            logger.error(msg=' adding step to dict, add whitespace instead')
            dict_results['step'] = ' '

        # string with function identifier
        try:
            dict_results['fun_id'] = self.fun_id
        except:
            logger.error(msg=' adding fun_id to dict, add empty instead')
            dict_results['fun_id'] = 'empty'

        # string with time spend for dim reduction
        try:
            dict_results['time'] = stop
        except:
            logger.error(msg=' adding execution time to dict, add 0 instead')
            dict_results['time'] = 0

        # string with shape of the data used for dim reduction
        try:
            dict_results['rows_cols'] = str(self.data_high.shape[0])+'_'+str(self.data_high.shape[1])
        except:
            logger.error(msg=' adding rows_cols to dict, add 0 instead')
            dict_results['rows_cols'] = 0

        # int with target dimension for dim reduction
        try:
            dict_results['dim_low'] = self.dim_low
        except:
            logger.error(msg=' adding dim_low to dict')
            dict_results['dim_low'] = 0

        # hyperparameters as string
        try:
            dict_results['hyperparameters'] = json.dumps(hyperparameters) # dict to string
        except:
            logger.error(msg=' adding hyperparamers-string to dict, add empty string')
            dict_results['hyperparameters'] = json.dumps({'empty': 'empty'})

        # step3: add intrinsic dimensionality to the results dict
        if step == globalstring_step3:
            try:
                dict_results['data_lowdim'] = data_low
            except:
                logger.error(msg='adding data_lowdim to dict, add NaN instead')
                dict_results['data_lowdim'] = 'NaN'

        return dict_results
