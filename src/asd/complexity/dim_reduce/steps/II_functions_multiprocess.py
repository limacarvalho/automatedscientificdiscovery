from dim_reduction.dimred_main import Dimreduction, call_dimred_functions
from hyperparameters.hyperparameter_optimization import Hyperparameter_optimization
from helper_data.global_vars import *
from helper_data.utils_data import timit_
from utils_logger import logger
import numpy as np
import json
import ray



def initialize_ray(step, timeout):
    '''
    logger message
    :param step: str,
        string indicating the step of the process
    :param timeout: int,
        programmed timeout [seconds] in case the calculations arent finished.
    :return: str,
        string with message
    '''
    ray.shutdown()
    ray.init(num_cpus=globalvar_n_cpus, num_gpus=globalvar_n_gpus)
    msg = step + ' MULTIPROCESSING ON: ' + str(globalvar_n_cpus) + ' CPUs ' \
    + str(globalvar_n_gpus) + ' GPUs,  timeout [seconds]: ' + str(timeout)
    logger.info(msg=msg)


@ray.remote
def ray_worker_optimization(
        fun_id: str,
        fun,
        high_data: np.array,
        dim_low: int,
        cutoff_loss: float
    ) -> list:
    '''
    Hyperparameter optimization (ray remote function running in parallel on on multiple cores).
    Each core works on hyperparameter optimization of one dimensionality reduction function:
    example: core1: py_pca, core2: py_crca ...
    :param fun_id: str,
        function identifier example: 'py_pca'
    :param fun: (object, dict, fun),
            object:
                the function call object (example: pca())
            dict:
                default hyperparameters
            hyperparams:
                hyperparameter definitions (ranges etc.) for optimization
    :param high_data: np.array,
        high dimensional data
    :param dim_low: int,
        target dimension for low dimensional data
    :param cutoff_loss: float,
        loss function cutoff (dim reduction quality)
    :return: list, list with two items:
        item 1:
            dict_best_dim_reduction: dictionary with best dim reduction result
        item 2:
            lods_all_results: list of dictionaries with results of dimensionality reductions.
    '''
    hp = Hyperparameter_optimization(cutoff_loss)
    function, params, hyperparams = fun
    # hyperparameter optimization with bayesian method
    dict_best_result, lods_results = hp.hp_optimization_bayes(fun_id, hyperparams, high_data, dim_low)
    list_results = [dict_best_result, lods_results]
    return list_results



@timit_('step2_hyperparameter_optimization ')
def ray_get_optimization(functions: list, data_high: np.array, dim_low: int, cutoff_loss: float) -> dict:
    '''
    function to get a list of remote objects from ray_worker_optimization function.
    We include a timeout because the computation of some functions might take too long.
    Two lists wit ray objects will be returned by ray.get():
        1) list_of_list_of_results:
            each list contains dictionaries with the results of the hyperparameter optimization
            of a specific dimensionality reduction function (example: py_crca).
            In case the timeout is reached, it returns the the data from hyperparameter optimizations
            that are completed.
        2) not_ready:
            list of objects that are not ready after timeout is reached.
            Note: timeout does not work when ray.get(not_ready) is called.

    :param functions: list,
        list of strings of dim reduce function identifiers
     :param data_high: np.array,
        high dimensional dataset
    :param dim_low: int,
        dimension of low dimensional data
    :param cutoff_loss: float,
        loss function cutoff (dim reduction quality)
    :return: dict,
        dictionary with 2 lists:
            - 'best_results': list with dicts of best results of each function.
            - 'all_results': list of lists of all results.
    '''
    # set ray multiprocessing, get resources: ray.available_resources()['GPU'] 'CPU'
    # TODO: substitute with statement from the asd-instance (Joao)
    initialize_ray(step='STEP2', timeout=globalvar_ray_timeout_step2)

    # returns dictionary with fun_id: fun; fun returns: function, params, hyperparams
    dict_fun = call_dimred_functions(functions, data_high)

    list_of_list_of_results, not_ready = ray.wait(
        [ ray_worker_optimization.remote(fun_id, fun, data_high, dim_low, cutoff_loss)
            for fun_id, fun in dict_fun.items() ],
        timeout=globalvar_ray_timeout_step2,  num_returns=len(dict_fun)
    )

    # create list of dictionaries (lods) with best and all results, and make dictionary
    lods_best_results, lods_all_results = [],[]
    try:
        for results_fun in ray.get(list_of_list_of_results):
            # dicts with results of best dim reduction during hp optimization, one for each function
            lods_best_results.append(results_fun[0])
            # dicts with all results of hyperparameter optimization, all for each function
            lods_all_results.append(results_fun[1])
    except:
        # empty lists will be returned
        logger.error(msg='timeout without results at step2 (hyperparameter optimization)', exc_info=True)

    # dictionary with results
    results_step_2 = {
        'best_results': lods_best_results,
        'all_results': lods_all_results
    }
    # logger not working with ray.get(not_ready), if we call that here timeout will not work!
    ray.shutdown()
    return results_step_2




#######################################################################################
# parallel dimensionality reduction without hyperparameter optimization

@ray.remote
def worker_dim_reduce(dict_fun: dict, data_high: np.array, dim_low: int) -> dict:
    '''
    worker function for dimensionality reduction (ray remote function running in parallel on on multiple cores).
    Usage: dim reduction of full size dataset.
    Each core works on reduction of dataset with one dimensionality reduction function:
    example: core1: py_pca, core2: py_crca ...

    :param dict_fun: dict,
        dictionary with results from previous dimensionality reduction.
        we take the fun id and hyperparameters (optimized in previous step) from there.
    :param data_high: np.array,
        high dimensional data
    :param dim_low: int,
        target dimension for low dimensional data
    :return: dict:
        dictionary with results of dim reduction
    '''
    fun_id = dict_fun['fun_id']
    # converts the string of dictionary of hyperparmaeters to a python dictionary
    hyperparameters = json.loads(dict_fun['hyperparameters'])
    # dimensionality reduction
    class_dimred = Dimreduction(fun_id=fun_id, data_high=data_high, dim_low=dim_low)
    dict_results = class_dimred.exe_dimreduce(params=hyperparameters, step=globalstring_step3)
    return dict_results



def ray_get_dimreduction(dicts_funs: list, data: np.array, dim_low: int) -> list:
    '''
    function to get a list of remote objects from ray_worker_reduce function.
    We include a timeout because the computation of some functions might take too long.
    Two lists wit ray objects will be returned by ray.get():
        1) lods_results (list of dictionaries):
            each list contains dictionaries with the results of the one dimensionality reduction.
            In case the timeout is reached, it returns the the data from dim reductions that are finished.
        2) not_ready:
            list of objects that are not ready after timeout is reached.
            Note: timeout does not work when ray.get(not_ready) is called.

    :param dicts_funs: list,
        list of dictionaries with all utilities for dim reduction with one function
    :param data: np.array,
        high dimensional dataset
    :param dim_low: int,
        dimension of low dimensional data
    :return: list,
        list with dictionaries (lods) of results
    '''
    # initialize ray, wit hnumber of cpus, gpus
    # TODO: substitute with statement from the asd-instance (Joao)
    initialize_ray(step='STEP3 ', timeout=globalvar_ray_timeout_step3)

    # lods: list of dictionaries
    lods_results, not_ready = ray.wait(
        [worker_dim_reduce.remote(dict_fun, data, dim_low) for dict_fun in dicts_funs],
        timeout=globalvar_ray_timeout_step3, num_returns=len(dicts_funs)
    )
    try:
        lods_results = ray.get(lods_results)

    except:
        # in this case the best result of the 'target_dimesnion step)' will be used.
        lods_results = False
        logger.error(msg='timeout without results at multiprocessing', exc_info=True)
    ray.shutdown()
    return lods_results
