from .R_funs_dimred.R_funs_dimred import Dimred_functions_R
from dim_reduction.Py_funs_dimred.funs.Py_funs_dimred import Dimred_functions_python
import numpy as np
from typing import Union
from utils_logger import logger


def exclude_functions(functions: Union[str, list]) -> bool:
    '''
    checks if functions need to be excluded or not.
    :param functions: Union[str, list], functions provided by the customer.
    :return: bool, True in case functions starts with !
    '''
    excludes = False
    for fun_id in functions:
        if fun_id.startswith('!'):
            excludes = True
    return excludes



def call_dimred_functions(custom_functions: Union[str, list], data_high: np.array) -> dict:
    '''
    function caller with function_identifier: function call (returns function,
    default parameters, hyperparameters (hyperparameter: range*))
    *categorical hyperparametes are presented as integer, we are using bayes-opt
    which returns floats, which need to be translated into categories.
    some of the functions need data information (shape, distributions etc).
    Please note: we have tested most of the functions available in September 2022 from sources:
    R: https://kisungyou.com/Rdimtools
    Python: https://scikit-learn.org
    The functions implented here are those working best for our purposes. However, there are many
    more functions available which are not implemented here.

    :param data_high: np.array, high dimensional data
    :param custom_functions: Union[str, list] list of strings or string with function identifiers:
           example: 'py_pca', default: 'all_functions'
    :return: dict, function identifiers and function calls
    '''
    nrows = data_high.shape[0]
    ncols = data_high.shape[1]

    # call python function class
    python = Dimred_functions_python(nrows=nrows, ncols=ncols)

    # call R function class
    rdim = Dimred_functions_R(nrows=nrows, ncols=ncols)

    dim_reduce_functions = {
        'py_pca': python.py_pca(),
        'py_pca_sparse': python.py_pca_sparse(),
        'py_pca_sparse_mb': python.py_pca_sparse_mini_batch(),
        'py_fa': python.py_factor_analysis(),
        'py_fastica':  python.py_fast_ica(),
        'py_nmf': python.py_non_negative_matrix_factorization(),
        'py_truncated_svd': python.py_truncated_svd(),
        # 'py_mds': python.py_mds(), # TODO currently implemented
        'py_crca': python.py_crca(),
        #'py_rpca': python.py_rpca(), # TODO currently implemented
        'r_adr': rdim.funR_adr(),
        'r_lmds': rdim.funR_lmds(),
        'r_mds':rdim.funR_mds(),
        'r_npca': rdim.funR_npca(),
        'r_olpp':rdim.funR_olpp(),
        'r_ppca':rdim.funR_ppca(),
        'r_rndproj':rdim.funR_rndproj(),
        'r_rpcag':rdim.funR_rpcag(),
        'r_udp':rdim.funR_udp(),
        'r_cisomap':rdim.funR_cisomap(),
        'r_fastmap':rdim.funR_fastmap(),
        'r_ispe':rdim.funR_ispe(),
        'r_lamp':rdim.funR_lamp(),
        'r_sammon': rdim.funR_sammon(),
        'r_spe':rdim.funR_spe()
    }

    # message strings
    string_except = 'choosing this dimred function(s) failed.\n' \
                    'check the correct spelling of function(s).\n' \
                    'functions are listed in dimred_call_functions.py'

    # in case a functions string is provided instead of a list, make it list
    if isinstance(custom_functions, str):
        custom_functions = [custom_functions]

    # calls all functions of the above list, uncheck them if not needed
    if custom_functions[0] == 'all_functions':
        try:
            return dim_reduce_functions
        except:
            logger.error(msg=' error in choosing functions: all functions', exc_info=True)

    else:
        # exclude custom functions: exlude the functions from the funcs dictionary if at least one
        # function needs to start with: '!'
        exclude = exclude_functions(custom_functions)
        if exclude:
            try:
                dict_funs = dim_reduce_functions
                for fun_id in custom_functions:
                    # remove the leading '!' and delete fnction from 'all_functions' dictionary
                    fun_to_remove = fun_id.replace('!', '')
                    del dict_funs[fun_to_remove]
                logger.info(msg=('excluding functions: '+str(custom_functions)))
                return dict_funs
            except:
                logger.error(msg=string_except, exc_info=True)


        # functions choosen by customer
        else:
            # make list in case its a string
            if isinstance(custom_functions, str):
                custom_functions = [custom_functions]
            # loop through the list with function names
            try:
                dict_funs = {}
                for fun_id in custom_functions:
                    # make sure the function is found in the list above
                    if fun_id in dim_reduce_functions.keys():
                        for key, value in dim_reduce_functions.items():
                            if key == fun_id:
                                dict_funs[key] = value
                    # function is not found above
                    else:
                        logger.error(msg=(fun_id + ' ' + string_except), exc_info=True)
                # make sure something is returned, if not stopp
                if dict_funs:
                    return dict_funs
                else:
                    logger.critical(msg=string_except, exc_info=True)
            except:
                logger.error(msg=string_except, exc_info=True)