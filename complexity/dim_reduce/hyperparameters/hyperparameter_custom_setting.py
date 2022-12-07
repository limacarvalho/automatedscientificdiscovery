from dimension_tools.dimension_suite.dim_reduce.dim_reduction.dimred_call_functions import call_dimred_functions
from dimension_tools.dimension_suite.dim_reduce.dim_reduction.dimred_main import Dimreduction
from ..helper_data.global_vars import *



# TODO: custom hyperparameter tuning function with early stop condition
def custom_factor_analysis(data_high, dim_low, dict_params):

    fun_id = 'py_fa'

    # hyperparameters
    hyperparameters = {
    'max_iter': 1000,  # [5,1000], tolerance for expectation–maximization algorithm (log-likelihood)
    'tol': 0.01,  # [0.001, 0.1], single value decomposition, more accurate, slow: ‘lapack’, speed: ‘randomized’ (default)
    'svd_method': 'randomized',  # [0,1], svd iteration: only for: 'randomized'_svd; iterations to make: m-by-k matrix Y
    'iterated_power': 3  # [0, 10],
    }

    # check if is dict

    # 2 - update hyperparameters
    for key, value in dict_params:

        try:
            hyperparameters[key] = value
        except:
            message = '''
                        thats rubbish! please make sure that the parameter exists ad is spelled correctly!'
                        'max_iter': 1000,  # [5,1000], tolerance for expectation–maximization algorithm (log-likelihood)
                        'tol': 0.01,  # [0.001, 0.1], single value decomposition, more accurate, slow: ‘lapack’, speed: ‘randomized’ (default)
                        'svd_method': 'randomized',  # [0,1], svd iteration: only for: 'randomized'_svd; iterations to make: m-by-k matrix Y
                        'iterated_power': 3  # [0, 10],
                    '''
            print(message)

    dimred = Dimreduction(fun_id=fun_id, data_high=data_high, dim_low=dim_low)
    results = dimred.exe_dimreduce(params=hyperparameters, step=globalstring_step1)
    return results





    # def dim_reduce_worker(self, dim_low):
    #     '''
    #     call dim reduce function and reduce the dimension with 'pca'.
    #     this function returns the results in dictionary format containing Q matrix, mae_norm,
    #     time etc. The results are saved in a list (list of dicts).
    #     :param dim_low: target dimension
    #     :return: loss of dimensionality reduction quality test
    #     '''
    #     # returns: dict with
    #     dimred = Dimreduction(fun_id='py_pca', data_high=self.data, dim_low=dim_low)
    #     results = dimred.exe_dimreduce(params={}, step=globalstring_step1)
    #     self.list_of_dicts.append(results)
    #     return results[self.loss_fun]

######## custom function, will substitute the optimizer function in the future
# def fun_custom_maximize(self, hyperparameters):
#     '''
#     check also: bayesian theorem update - less iterations
#
#     :param hyperparameters:
#     :return:
#     '''
#     optimizer = BayesianOptimization(
#         f=None,
#         pbounds=hyperparameters,
#         verbose=2,
#         random_state=1,
#     )
#
#     utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
#
#     next_point_to_probe = optimizer.suggest(utility)
#
#     target = self.black_box_dimreduce(**next_point_to_probe)
#
#     optimizer.register(
#         params=next_point_to_probe,
#         target=target,
#     )
#
#     for i in range(self.init_steps + self.iter_steps):
#         next_point = optimizer.suggest(utility)
#         target = self.black_box_dimreduce(**next_point)
#         if i == self.init_steps and target >= self.cutoff_loss:
#             return optimizer
#         else:
#             optimizer.register(params=next_point, target=target)
#     return optimizer


