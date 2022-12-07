from dim_reduction.dimred_call_functions import call_dimred_functions
from dim_reduction.dimred_main import Dimreduction
from hyperparameters.hyperparameter_optimization import Hyperparameter_optimization
from helper_data.global_vars import *
from helper_data.utils_data import timit_
from utils_logger import logger
import numpy as np
import json
import ray



@ray.remote
def hp_optimization(
        fun_id: str,
        fun,
        high_data: np.array,
        low_dim: int,
        cutoff_loss: float
    ) -> list:
    '''
    hyperparameter optimization function for parallel processing on multiple cores.
    Each core runs the hyperparameter optimization of one dimred function.
    :param fun_id: str, function identifier example: 'py_pca'
    :param fun: (object, dict,  fun), returns the following:
            1) the function call 2) default hyperparameters 3) hyperparameter ranges for optimization
    :param high_data: np.array, high dimensional data
    :param low_dim: int, target dimension for low dimensional data
    :param cutoff_loss: float, cutoff
    :return: list, list with results of hyperparmeter optimization [dict_best_dim_reduction, lods_all_results]:
        dict_best_dim_reduction: dictionary with best dim reduction result
        lods_all_results: list of dicts with all results.
        dict result:
    '''
    hp = Hyperparameter_optimization(cutoff_loss)
    function, params, hyperparams = fun
    # hyperparameter optimization with bayesian method
    dict_best_result, lods_results = hp.hp_optimization_bayes(fun_id, hyperparams, high_data, low_dim)
    list_results = [dict_best_result, lods_results]
    return list_results

@timit_('step2_hyperparameter_optimization ')
def multi_optimization(functions: list,
                       data_high: np.array,
                       dim_low: int,
                       cutoff_loss: float) -> dict:
    '''
    getter function for parallel processed hyperparameter optimization function.
    Each CPU performs the hyperparameter optimization of one dim reduction function ('py_pca').
    The results are received together with the ray.get function in a list.
    The function (hp.hp_optimization_bayes) returns two results for each function:
    1) dictionary of best dim reduction including the dimension, function and hyperparameters
    2) list of dictionaries of results of all dim reductions
    Here we seperate them and append them with the results from the other functions.

    :param functions: list, list of dim reduce function identifiers
    :param data_high: np.array, high dimensional data
    :param dim_low: int, target dimension for low dimensional daa
    :param cutoff_loss: float, quality cutoff for q-matrix parameter default: mae_norm
    :return: dict with: -list with dicts of best results of each function
                        -list of lists of all results for each function
    '''
    # set ray multiprocessing, get resources: ray.available_resources()['GPU'] 'CPU'
    # here we use all cpus when more than 10 functions else we use one cpu per function
    n_cpus = globalvar_n_cpus
    ray.shutdown()
    ray.init(num_cpus=n_cpus, num_gpus=globalvar_n_gpus)
    logger.info(msg=('MULTIPROCESSING ON: ' + str(n_cpus) +  ' CPUs'))

    # returns dictionary with fun_id: fun; fun returns: function, params, hyperparams
    dict_fun = call_dimred_functions(functions, data_high)

    list_of_list_of_results = ray.get([
                                hp_optimization.remote(
                                    fun_id, fun,
                                    data_high,
                                    dim_low,
                                    cutoff_loss
                                )  for fun_id, fun in dict_fun.items() ])

    # create list of dictionaries (lods) with best and all results, and make dictionary
    lods_best_results, lods_all_results = [],[]
    for results_fun in list_of_list_of_results:
        # dicts with results of best dim reduction during hp optimization, one for each function
        lods_best_results.append(results_fun[0])
        # dicts with all results of hyperparameter optimization, all for each function
        lods_all_results.append(results_fun[1])

    # dictionary with results
    results_step_2 = {
        'best_results': lods_best_results,
        'all_results': lods_all_results
        }

    return results_step_2




#######################################################################################
# parallell dim reduction without hyperparameter optimization

@ray.remote
def worker_dim_reduce(dict_fun: dict,
                      data: np.array,
                      dim_low: int) -> dict:
    '''
    worker function for dim reduction of full size dataset with one function.
    If many functions are provided, the will be processed in parallell.
    we take the hyperparameters from the results dicionary of the previous step.

    Parameters
    ----------
    dict_fun : dictionary with all utilities for dim reduction with one function
    data : high dimensional data
    dim_low : dimension of low dimensional data
    cutoff_loss : loss cutoff 0...1

    Returns
        dictionary with results and information about the dim reduction
    -------

    '''
    fun_id = dict_fun['fun_id']
    # string to dict, hyperparameters from hyperparameter optimization step
    hyperparameters = json.loads(dict_fun['hyperparameters'])
    # dim reduction
    class_dimred = Dimreduction(fun_id=fun_id, data_high=data, dim_low=dim_low)
    dict_results = class_dimred.exe_dimreduce(params=hyperparameters, step=globalstring_step3)
    return dict_results



def multiprocess_dimred(dicts_funs: list, data: np.array, dim_low: int) -> list:
    '''
    getter function for ray processed hyperparameter hp_optimization function.
    This handles the optimization of several dim reduce functions in parallel.
    It takes the dictionary of each dim reduction which also contains the function
    identifier and hyperparameters.

    :param dicts_funs: list of dictionaries with all utilities for dim reduction with one function
    :param data: high dimensional data
    :param dim_low: dimension of low dimensional data
    :param cutoff_loss: loss cutoff 0...1
    :return: list with dicts o all results
    '''
    # initialize ray, wit hnumber of cpus, gpus
    ray.shutdown()
    ray.init(num_cpus=globalvar_n_cpus, num_gpus=globalvar_n_gpus)
    # lods: list of dicts
    lods_results = ray.get([worker_dim_reduce.remote(dict_fun, data, dim_low)
                            for dict_fun in dicts_funs
                            ])
    return lods_results
