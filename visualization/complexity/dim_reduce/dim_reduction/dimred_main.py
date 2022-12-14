import numpy as np
from .R_funs_dimred.R_helper import R_helper
from .Py_funs_dimred.Py_helper import Py_helper
from hyperparameters.hyperparameter_initialization import hyperparameter_init
from .R_funs_dimred.R_funs_dimred import Dimred_functions_R
from dim_reduction.Py_funs_dimred.Py_funs_dimred import Dimred_functions_python
from helper_metrix.metrics_dimred import Metrics
from helper_data.global_vars import *
from utils_logger import logger
from typing import Union
import time
import json
# ! do not erase, otherwise logs will be full with warnings
import warnings
warnings.filterwarnings('ignore')


def exclude_functions(functions: Union[str, list]) -> bool:
    '''
    check if function list or string provided by the customer starts with '!'.
    These functions will be excluded from the list. If this is not the case,
    the functions will be included
    :param functions: Union[str, list],
        functions provided by the customer.
    :return: bool,
        True in case functions starts with '!', else: False
    '''
    excludes = False
    for fun_id in functions:
        if fun_id.startswith('!'):
            excludes = True
    return excludes



def call_dimred_functions(custom_functions: Union[str, list], data_high: np.array) -> dict:
    '''
    returns dimensionality reduction functions and identifiers.
    :param data_high: np.array,
        high dimensional data
    :param custom_functions: Union[str, list],
        list of strings or string with function identifiers:
        example: 'py_pca', default: 'all_functions'
    :return: dict,
        function identifiers and function calls
    '''
    nrows = data_high.shape[0]
    ncols = data_high.shape[1]
    # call python function class
    python = Dimred_functions_python(nrows=nrows, ncols=ncols, all_hp=False)
    # call R function class
    rdim = Dimred_functions_R(nrows=nrows, ncols=ncols, all_hp=False)

    # dictionary with function identifiers (keys) and function calls (values)
    dim_reduce_functions = {
        'py_pca': python.py_pca(),
        'py_pca_sparse': python.py_pca_sparse(),
        'py_pca_incremental': python.py_pca_incremental(),
        'py_truncated_svd': python.py_truncated_svd(),
        'py_crca': python.py_crca(),
        'py_sammon': python.py_sammon(),
        'r_adr': rdim.funR_adr(),
        'r_mds':rdim.funR_mds(),
        'r_ppca': rdim.funR_ppca(),
        'r_rpcag': rdim.funR_rpcag(),
        'r_ispe': rdim.funR_ispe(),
    }

    # message strings
    string_except = 'choosing this dimred function(s) failed.\n' \
                    'check the correct spelling of function(s).\n' \
                    'functions are listed in above'

    # in case a function is provided as a string, make it a list
    if isinstance(custom_functions, str):
        custom_functions = [custom_functions]

    # default: calls all functions listed above
    if custom_functions[0] == 'all_functions':
        try:
            return dim_reduce_functions
        except:
            logger.error(msg=' error in choosing functions: all functions', exc_info=True)

    else:
        # exclude functions in case the first item of list custom_functions starts with '!'.
        exclude = exclude_functions(custom_functions)
        if exclude:
            try:
                dict_funs = dim_reduce_functions
                for fun_id in custom_functions:
                    # remove the leading '!' and delete function dictionary
                    fun_to_remove = fun_id.replace('!', '')
                    del dict_funs[fun_to_remove]
                logger.info(msg=('excluding functions: '+str(custom_functions)))
                return dict_funs
            except:
                logger.error(msg=string_except, exc_info=True)

        # in case there is no leading '!', include the functions in custom_functions
        else:
            # make list in case its a string
            if isinstance(custom_functions, str):
                custom_functions = [custom_functions]
            # loop through the list with function identifiers
            try:
                dict_funs = {}
                for fun_id in custom_functions:
                    # make sure the function is found in the list above
                    if fun_id in dim_reduce_functions.keys():
                        for id, function_call in dim_reduce_functions.items():
                            if id == fun_id:
                                dict_funs[id] = function_call
                    # function is not found
                    else:
                        logger.error(msg=(fun_id + ' ' + string_except), exc_info=True)
                # make sure something is returned, if not stopp
                if dict_funs:
                    return dict_funs
                else:
                    logger.critical(msg=string_except, exc_info=True)
            except:
                logger.error(msg=string_except, exc_info=True)






class Dimreduction:
    '''
    Utility for dimensionality reduction.
    '''

    def __init__(self, fun_id: str, data_high: np.array, dim_low: int):
        '''
        :param fun_id: str,
            function_identifier
        :param data_high: np.array,
            high dimensional data
        :param dim_low: int,
            low dimension
        '''
        self.fun_id = fun_id
        self.data_high = data_high
        self.dim_low = int(dim_low)


        # dimreduction wrapper which returns:
        #   fun: function call
        #   params_fun: updated default hyperparameters
        #   hyperpars: hyperparameter ranges
        dict_fun = call_dimred_functions(self.fun_id, self.data_high)
        self.fun, self.params_fun, hyperpars = list(dict_fun.values())[0]


    def convert_hyperparameters_r_py(self, params: Union[str, dict], step: str) -> (dict, str):
        '''
        converts R and Python ahyperparameters into the correct format.
        Python: dictionary
        R: string
        The returns depend on the step and the platform (R or Python).
        :param params: Union[str, dict],
            hyperparameters: string for R functions, dict for Python functions
        :param step: str,
            step of dim reduction process
        :return: (dict, str)
            dict: python hyperparameters
            str: R hyperparameters

        '''
        # initialization
        params_r_str = ''
        params_py_dict = {}

        try:
        # Option 1: the parameters are used for step2: hyperparameter optimization.
            # the params are returned by bayesian optimization program as floats and some
            # need to be converted into the correct format (categorical or integer values).
            if step == globalstring_step2:
                params_r_str, params_py_dict = hyperparameter_init(
                    params = params,
                    fun_id = self.fun_id,
                    dim = int(self.dim_low),
                    data_high = self.data_high
                )

        # Option 2: R FUNCTIONS NOT used for hyperparameter optimization
            elif step != globalstring_step2 and isinstance(self.params_fun, str):
                # if its a dictionary: convert it to a string
                if isinstance(params, dict):
                    params_r_str = R_helper().dict_to_r_string(params)
                # its a string: do nothing
                else:
                    params_r_str = params

        # Option 3: PYTHON FUNCTIONS NOT used for hyperparameter optimization
            else:
                # its a dictionary, do nothing
                if isinstance(params_py_dict, dict):
                    params_py_dict = params
                # its a string, convert to dict (python functions need Python hyperparameters in dict format)
                else:
                    params_py_dict = json.loads(params)
        except:
            logger.error(msg=('INIT HYPERPARAMETER '+step+self.fun_id+' dim: '+str(self.dim_low)),exc_info=True)
        # we always return two things (one is always empty)
        return params_py_dict, params_r_str


    def exe_dimreduce(self, params: dict, step: str) -> (np.array, dict):
        '''
        Main function for dimensionality reduction.
        1) converts hyperparameters into the correct format.
           the format depends on ste step and the functions platform (R, Python)
        2) dimensionality reduction which returns the reduced dataset.
        3) measures the quality of the reduced dataset and returns a dictionary
           with different losses, co-ranking matrix and documentation.
        4) finally, add information about the dimensionality reduction to the dictionary.

        :param params: dict,
            hyperparameters for dim reduce function
        :param step: str,
            'dim_reduce' one reduction with given fun_id, low_dim and params
            'hyperparameters' for hyperparameter optimization.
             categorical parameters are provided as floats by the bayes opt program and need to be converted
             to their original values.
             more info: hyperparameters -> hyperparameter_init
        :return: dict,
            dictionary with results and quality of dim reduction:
                'Q': Q,                 # np.array: coranking matrix
                'rel_err': rel_err,     # float: relative error 0...1 (1=perfect dimensionality reduction)
                'r2': r2                # float: r-squared value
                'step': step            # str: step of the dim reduction process
                'fun_id': fun_id        # str: function identifier, example: 'py_pca'
                'time': seconds         # float: time for dimensionality reduction in seconds
                'rows_cols': 'rows_cols' # str: 'nrows_ncolumns' of high dimensional dataset
                'dim_low': dim_low       # int: dimension (ncolumns)of low dimensional dataset
                'hyperparameters': 'hyperparameters'   # str: string of hyperparameters 'hyperparameter=value'
                'data_lowdim': data_lowdim   # np.array: low dimensional data
        '''
        # initialization
        hyperparameters = ''
        data_low = np.array
        start = time.time()

        # preprocess the functions hyperparameters
        params_py_dict, params_r_str = self.convert_hyperparameters_r_py(params, step)

        try:
            # R functions
            if isinstance(self.params_fun, str):
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
            logger.error(msg=('DIMREDUCE '+step+' '+self.fun_id+' dim:'+str(self.dim_low)), exc_info=True)

        # time for dim reduction
        stop = round(time.time() - start, 3)

        # Measure quality of dim reduction, metrics returns an empty dict in case of exceptions
        metrix = Metrics(self.fun_id, self.data_high, data_low)
        dict_results = metrix.metrix_all()

        # append results or return a empty value in case of exceptions
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

        # string with time of dim reduction [seconds]
        try:
            dict_results['time'] = stop
        except:
            logger.error(msg=' adding execution time to dict, add 0 instead')
            dict_results['time'] = 0

        # string with shape of the high dimensional data
        try:
            dict_results['rows_cols'] = str(self.data_high.shape[0])+'_'+str(self.data_high.shape[1])
        except:
            logger.error(msg=' adding rows_cols to dict, add 0 instead')
            dict_results['rows_cols'] = 0

        # integer with dimension of low dimensional data
        try:
            dict_results['dim_low'] = self.dim_low
        except:
            logger.error(msg=' adding dim_low to dict')
            dict_results['dim_low'] = 0

        # hyperparameters in string format 'hyperparameter=value'
        try:
            dict_results['hyperparameters'] = json.dumps(hyperparameters) # dict to string
        except:
            logger.error(msg=' adding hyperparamers-string to dict, add empty string')
            dict_results['hyperparameters'] = json.dumps({'empty': 'empty'})

        # step3: low dimensional data (array)
        if step == globalstring_step3:
            try:
                dict_results['data_lowdim'] = data_low
            except:
                logger.error(msg='adding data_lowdim to dict, add NaN instead')
                dict_results['data_lowdim'] = 'NaN'

        if step == globalstring_step3:
            logger.info(msg=('finito: '+self.fun_id
                            +' dim:'+str(self.dim_low)
                            +' loss:'+str(dict_results[globalvar_loss_function])
                            +' time:'+str(stop)))
        return dict_results