from dimension_tools.dimension_suite.dim_reduce.dim_reduction.dimred_call_functions import call_dimred_functions
from dimension_tools.dimension_suite.dim_reduce.dim_reduction.dimred_main import Dimreduction
from dimension_tools.dimension_suite.dim_reduce.hyperparameters.hyperparameter_optimization import Hyperparameter_optimization
from dimension_tools.dimension_suite.dim_reduce.helper_data.global_vars import *
import numpy as np
import json
import ray

# import multiprocessing
# from threading import Thread
# import dask
# from dask.distributed import Client, progress
# client = Client(threads_per_worker=4, n_workers=10)

######################################################
# parallell hyperparameter optimization

@ray.remote
def hp_optimization(fun_id: str,
                    fun,
                    high_data: np.array,
                    low_dim: int,
                    cutoff_loss: float) -> list:
    '''
    hyperparameter optimization function for parallel processing on multiple cores.
    Each core runs the hyperparameter optimization of one dimred function.

    :param str fun_id: function identifier example: 'py_pca'
    :param (object, dict,  fun: returns 1) the function call 2) default hyperparameters 3)
    :param np.array high_data: high dimensional data
    :param low_dim: target dimension for low dimensional data
    :param cutoff_loss: cutoff
    :return: list with [dct_best_dim_reduction, lods_all_results]
    '''
    hp = Hyperparameter_optimization(cutoff_loss)
    function, params, hyperparams = fun
    # hyperparameter optimization with bayesian method
    dict_best_result, lods_results = hp.hp_optimization_bayes(fun_id,
                                                              hyperparams,
                                                              high_data,
                                                              low_dim
                                                              )
    list_results = [dict_best_result, lods_results]
    return list_results


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

    :param functions: list of dim reduce function identifiers
    :param data_high: np.array high dimensional data
    :param dim_low: int target dimension for low dimensional daa
    :param cutoff_loss: float quality cutoff for q-matrix parameter default: mae_norm
    :return: dict with: -list with dicts of best results of each function
                        -list of lists of all results for each function
    '''
    # set ray multiprocessing, get resources: ray.available_resources()['GPU'] 'CPU'
    # here we use all cpus when more than 10 functions else we use one cpu per function
    n_cpus = globalvar_n_cpus
    ray.shutdown()
    ray.init(num_cpus=n_cpus, num_gpus=globalvar_n_gpus)
    print(globalstring_info + 'MULTIPROCESS ON:', n_cpus,  ' CPUs')

    # returns dictionary with fun_id: fun; fun returns: function, params, hyperparams
    dict_fun = call_dimred_functions(functions, data_high)

    list_of_list_of_results = ray.get([
                                hp_optimization.remote(fun_id, fun,
                                                       data_high,
                                                       dim_low,
                                                       cutoff_loss)
                                for fun_id, fun in dict_fun.items()
                                ])

    # create list of dictionaries (lods) with best and all results, and make dictionary
    lods_best_results, lods_all_results = [],[]
    for results_fun in list_of_list_of_results:
        # dicts with results of best dim reduction during hp optimization, one for each function
        lods_best_results.append(results_fun[0])
        # dicts with all results of hyperparameter optimization, all for each function
        lods_all_results.append(results_fun[1])

    # dictionary with results
    results_step_2 = {'best_results': lods_best_results,
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



def multiprocess_dimred(dicts_funs: list,
                        data: np.array,
                        dim_low: int) -> list:
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




















# def for_loop(dict_fun, data, dim):
#     list_of_lists_ = []
#     for fun_id, fun in dict_fun.items():
#         list_results = dask.delayed(optimization(fun_id, fun, data, dim))
#         list_of_lists_.append(list_results)
#     return list_of_lists_
#
# def multi_optimization_dask(functions, data, dim):
#     print(global_vars.globalstring_info + 'FUNCTIONS: ', functions)
#     print(global_vars.globalstring_info + 'CPU-RESOURCES', ray.available_resources())
#     dict_fun = call_functions_dimred(functions, data, dim)
#
#     list_of_lists = []
#     # sequential, not parallel
#     # for fun_id, fun in dict_fun.items():
#     #     list_results = dask.delayed(optimization(fun_id, fun, data, dim))
#     #     list_of_lists_.append(list_results)
#     # c = for_loop(dict_fun, data, dim)
#     # list_of_lists = c.compute()
#
#     # tons of error messages, slow, looks sequential
#     # for fun_id, fun in dict_fun.items():
#     #     futures = client.submit(for_loop, (fun_id, fun))
#     #     futures.append(futures)
#     # list_of_lists = client.gather(futures)
#
#     # get dask results
#     list_best_results, list_of_dict_results = [],[]
#     for results_fun in list_of_lists:
#         results_fun = results_fun.compute()
#         list_best_results.append(results_fun[0])
#         list_of_dict_results.append(results_fun[1])
#
#     return list_best_results, list_of_dict_results


# class step_2_functions:
#
#     def __init__(self, functions, data, dim, loss_fun):
#         self.loss_fun = loss_fun
#         self.functions = functions
#         self.data = data
#         self.dim = dim
#         self.max_cpu = multiprocessing.cpu_count()-1
#
#
#
#     @ray.remote
#     def optimization(self, fun_id, fun, i):
#         '''
#
#         Parameters
#         ----------
#         fun_id :
#         fun :
#         i :
#
#         Returns
#         -------
#
#         '''
#         function, params, hyperparams = fun
#         hp = hyperparameters(loss=self.loss_fun)
#         try:
#             results = hp.hp_optimization(fun_id, function, hyperparams, self.data, self.dim)
#             self.results_list[i] = results
#         except:
#             print('ERROR', fun_id, i)
#             print(traceback.format_exc())
#             # results is list: [trust, cont, lcmc, mae, mae_norm, rmse, r2, str_pars, Q]
#
#
#     def chunks(self, dict_fun):
#         '''
#         l = [1, 2, 3, 4, 5, 6, 7, 8, 9] ... [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
#         Parameters
#         ----------
#         dict_fun :
#
#         Returns
#         -------
#
#         '''
#         n = self.max_cpu
#         chunks = [list(dict_fun.items())[i:i + n] for i in range(0, len(dict_fun), n)]
#         return chunks
#
#
#
#
#     # remove the chunk stuff!
#     def multi_optimization_(self):
#         '''
#
#         Returns
#         -------
#
#         '''
#         print('FUNCTIONS: ', self.functions)
#         dict_fun = call_functions_dimred(self.functions, self.data, self.dim)
#         self.results_list_all = []
#
#         chunks = self.chunks(dict_fun)
#
#         for dict_fun_chunk in chunks:
#             threads: List[int] = [0] * min(len(dict_fun), self.max_cpu)
#             self.results_list = [0] * min(len(dict_fun), self.max_cpu)
#
#             for i in range(len(dict_fun_chunk)):
#                 fun_id = dict_fun_chunk[i][0]
#                 fun = dict_fun_chunk[i][1]
#                 # Thread(target=self.optimization, args=(fun_id, fun, i)) 99sec
#                 threads[i] = multiprocessing.Process(target=self.optimization, args=(fun_id, fun, i)) # 54sec
#                 threads[i].start()
#
#             # do some other stuff
#             for i in range(len(threads)):
#                 threads[i].join()
#
#             for l in range(len(self.results_list)):
#                 self.results_list_all.append(self.results_list[l])
#
#             print('RL', len(self.results_list))
#             print('RLC', len(self.results_list_all))
#
#             # control function
#             # for n in range(len(self.results_list)):
#             #     # fun_id, trust, cont, lcmc, mae, mae_norm, rmse, r2, string_print, Q, time_fun
#             #     print('H', self.results_list[n][0], self.results_list[n][8])
#
#
#     def multi_optimization(self):
#         '''
#
#         Returns
#         -------
#
#         '''
#         print('FUNCTIONS: ', self.functions)
#         dict_fun = call_functions_dimred(self.functions, self.data, self.dim)
#         self.results_list_all = []
#         print(dict_fun)
#
#         i = -1
#         for fun_id, fun in dict_fun.items():
#             i = i + 1
#             self.optimization.remote(parameter=(fun_id, fun, i))
#             # threads[i] = multiprocessing.Process(target=self.optimization, args=(fun_id, fun, i)) # 54sec
#             # threads[i].start()
#
#         # do some other stuff
#         # for i in range(len(threads)):
#         #     threads[i].join()
#
#         for l in range(len(self.results_list)):
#             self.results_list_all.append(self.results_list[l])
#
#         print('RL', len(self.results_list))
#         print('RLC', len(self.results_list_all))
