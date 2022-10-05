from ..dim_reduction.dimred_main import Dimreduction
from ..helper_data.global_vars import *
from ..helper_data.helper_data import empty_dict
from ..helper_data.helper_data import check_dim
from ..helper_metrix.loss_functions import fun_kmax
from bayes_opt import BayesianOptimization
import traceback
import numpy as np


class Hyperparameter_optimization:
    '''
    opt

    '''

    def __init__(self, cutoff_loss):
        self.loss_fun = globalvar_loss_function
        self.cutoff_loss = cutoff_loss


    def black_box_reduce(self, **params: dict):
        '''
        Black box function run within the BayesianOptimization function.
        It receives hyperparameters from the bayes optimization package and returns a
        quality loss which is maximized during the optimization process.
        The results of all dim reductions are stored in self.list_of_dict_results .
        :param params: dict parameters provided by the bayesian optimization method
        :return: float mae_norm 0.0 ... 1.0
        '''
        # dim reduction and quality measurement
        dimred = Dimreduction(self.fun_id, self.data_high, self.dim_low)
        dict_results = dimred.exe_dimreduce(params, step=globalstring_step2)

        # number of init and iteration steps
        try:
            dict_results['init_steps'] = self.init_steps
            dict_results['iter_steps'] = self.iter_steps
        except:
            print(globalstring_error + 'adding init steps or iter steps to dict')
            dict_results['init_steps'] = 0
            dict_results['iter_steps'] = 0

        dict_results['dim_low'] = self.dim_low
        # this dictionary is updated after every dim reduction
        self.list_of_dict_results.append(dict_results)
        # return the loss value for hyperparameter optimization
        return dict_results[self.loss_fun]


    def get_init_iterations(self, hyperparameters: dict) -> (int, int):
        '''
        BayesianOptimization function has two main steps: A) initialization, where the hyperparameters
        are choosen randomly and B) iteration where the bayesian mthod is used to optimize the hyper
        parameters from step to step. here we calculate the number of initial steps and iterations.
        Number depends on the number of hyperparameters.
        :param hyperparameters: dictionary with hyperparameter, value range items
        :return: init steps, iter steps
        '''
        if ('empty' in hyperparameters) or ('empty_py' in hyperparameters):
            init, iterations = 1, 0
        else:
            n_hps = len(hyperparameters)
            init = 3 + (n_hps * 2)
            iterations = n_hps * 5
        print(globalstring_info + self.fun_id, ' init:', init, ' iterations:', iterations)
        return init, iterations


    def fun_maximize_optimizer(self, hyperparameters: dict) -> object:
        '''
        runs the BayesianOptimization function by maximizing the black box output (loss)
        in case of no changes after te init steps, the function breaks.
        :param hyperparameters: dict hyperparameters, hyperparameter: value_range
        :return: object optimizer.max
        '''
        optimizer = BayesianOptimization(
            f = self.black_box_reduce,
            pbounds = hyperparameters,
            random_state = 1,
            verbose = 0)
        try:
            optimizer.maximize(init_points=self.init_steps, n_iter=self.iter_steps)
        except:
            print(globalstring_warning + 'NO CHANGE WITH HYPERPARAMETER TUNING', self.fun_id)
            pass
        return optimizer


    def get_best_optimizer_result(self, optimizer):
        '''
        function to return the dictionary of the best dimesnionality reduction eg. the
        optimizer.max function.
        The function loops through the optimizer results (optimizer.res) which are
        chronologically sorted. It searches for the n-th entry in list of results with the
        best result (hyperparameters of result == hyperparameters optimizer.res == hyperparameters
        optimizer.max)
        The hyperparameters are floats with a lot of float values, so its very unlikely to have the
        same combination of hyperparamters.
        :param optimizer: dict optimizer
        :return: dict of best results
        '''
        best_result = {}
        try:
            best_params = optimizer.max['params']
            for i, row in enumerate(optimizer.res):
                if row['params'] == best_params:
                    best_result = self.list_of_dict_results[i]
        except:
            best_result = {}
            print(globalstring_error + 'FINDING BEST RESULTS', self.fun_id)
            print(traceback.format_exc())
        return best_result


    def hp_optimization_bayes(self,
                              fun_id: str,
                              hyperparameters: dict,
                              data: np.array,
                              ndim: int
                              ) -> (dict,list):
        '''
        - - - DESCRIPTION - - -
        Bayesian global hyperparameter optimization with gaussian processes.
        2 steps: random exploration (intialization) and iteration.

        init_points: How many steps of random exploration you want to perform.
                     Random exploration can help by diversifying the exploration space.
                     If no change, a error is thrown after this phase and the iterations
                     are not initialized.

        n_iter: How many steps of bayesian optimization you want to perform.
                The more steps the more likely to find a good maximum you are.

        Total number of steps: is init_poins + n_iter

        - - - COMMON ERRORS - - -
        Errors: 'StopIteration: Queue is empty, no more objects to retrieve.' AND connected:
                'ValueError: array must not contain infs or NaNs'
                    -> is raised in case of no change of loss in exploration phase.

        - - - INFORMATION - - -
        https://github.com/fmfn/BayesianOptimization

        '''
        self.fun_id = fun_id
        self.data_high = data
        self.dim_low = ndim
        self.list_of_dict_results = []
        self.init_steps, self.iter_steps = self.get_init_iterations(hyperparameters)

        # checks if dimension is correct (True, False)
        dimcheck = check_dim(self.dim_low, self.data_high.shape[1])

        # dim sensitive functions
        dimsensitive_py = [''] # 'py_truncated_svd'

        # something is wrong with the dimension or function
        if dimcheck == False or str(self.fun_id) in dimsensitive_py:
            kmax = fun_kmax(self.data_high)
            self.list_of_dict_results = [empty_dict(self.fun_id, self.dim_low, kmax)]
            dict_best_results = self.list_of_dict_results[0]

        # dim reduce functions without hyperparameters
        elif ('empty' in hyperparameters) or ('empty_py' in hyperparameters):
            _ = self.black_box_reduce(**hyperparameters)
            # there is only one dict here
            dict_best_results = self.list_of_dict_results[0]

        # optimize hyperparameters of functions with hyperparameters
        else:
            # optimize
            optimizer = self.fun_maximize_optimizer(hyperparameters)
            if not optimizer:
                dict_best_results = {}
                self.list_of_dict_results = []
            else:
                dict_best_results = self.get_best_optimizer_result(optimizer)

            # future function, will come with early stop condition
            # optimizer = self.fun_custom_maximize(hyperparameters) # funzt

        return dict_best_results, self.list_of_dict_results



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
#     target = self.black_box_reduce(**next_point_to_probe)
#
#     optimizer.register(
#         params=next_point_to_probe,
#         target=target,
#     )
#
#     for i in range(self.init_steps + self.iter_steps):
#         next_point = optimizer.suggest(utility)
#         target = self.black_box_reduce(**next_point)
#         if i == self.init_steps and target >= self.cutoff_loss:
#             return optimizer
#         else:
#             optimizer.register(params=next_point, target=target)
#     return optimizer




'''
functions with mae_norm < cutoff after init and mae_norm > cutoff after iter 
*** top3 results    ** 3...10 top results    * no good results
isomap ***
pca_sparse **
pca_sparse_mb **
spmds *
dictlearn_mb *
mdproj *
lmds *

'''




