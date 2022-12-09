from dim_reduction.dimred_main import Dimreduction
from helper_data.global_vars import *
from helper_data.utils_data import empty_dict, check_low_dimension
from utils_logger import logger
from bayes_opt import BayesianOptimization
import numpy as np


class Hyperparameter_optimization:
    '''
    hyperparameter optimization of dimensionality reduction functions with the bayes-opt package.
    thanks to Fernando Nogueira for the bayes_opt package.
    repository:  https://github.com/fmfn/BayesianOptimization
    '''

    def __init__(self, cutoff_loss):
        '''
        :param cutoff_loss: float, default=0.99
            cutoff for loss function 0...1 (perfect dimensionality reduction)
        '''
        self.loss_fun = globalvar_loss_function
        self.cutoff_loss = cutoff_loss


    def black_box_dimreduce(self, **params: dict) -> float:
        '''
        Black box function to run within the BayesianOptimization method.
        1) dimensionality reduction with hyperparameters calculated by bayes-opt.
        2) qulaity measurement and returns the loss whcih is maximized by bayes-opt.
        The results of all dim reductions are documented in list_of_dict_results.
        :param params: dict,
            parameters provided by the bayesian optimization method
        :return: float,
            loss value
        '''
        # dim reduction and measuring the loss
        dimred = Dimreduction(self.fun_id, self.data_high, self.dim_low)
        dict_results = dimred.exe_dimreduce(params, step=globalstring_step2)

        # number of init and iteration steps
        try:
            dict_results['init_steps'] = self.init_steps
            dict_results['iter_steps'] = self.iter_steps
        except:
            logger.error(msg='adding init steps or iter steps to dict', exc_info=False)
            dict_results['init_steps'] = 0
            dict_results['iter_steps'] = 0

        # update dictionary
        dict_results['dim_low'] = self.dim_low

        # append results to list
        self.list_of_dict_results.append(dict_results)

        # return the loss value
        return dict_results[self.loss_fun]


    def get_init_iterations(self, hyperparameters: dict) -> (int, int):
        '''
        BayesianOptimization function has two main steps: A) initialization, where the hyperparameters
        are choosen randomly and B) iteration where the bayesian mthod is used to optimize the hyper-
        parameters from step to step. here we calculate the number of initial steps and iterations.
        initial steps: adds 5 for the first hyperparameter and add 2 for any other hyperparameter.
        iteration steps: adds 5 per hyperparameter.
        :param hyperparameters:
            dictionary with hyperparameters and values or value ranges
        :return: int, int,
            int:
                number of initialization steps
            int:
                number of iteration steps
        '''
        n_hps, init, iterations = 0, [], []
        try:
            for key, value in hyperparameters.items():
                if isinstance(value, list):
                    if not init:
                        init.append(3 + 2)
                    else:
                        init.append(2)
                    iterations.append(5)
                else:
                    init.append(1)
                    iterations.append(0)
            # sum lists
            init = sum(init)
            iterations = sum(iterations)
        except:
            init, iterations = 1, 0
            logger.error(msg=(self.fun_id + ' n_hyperpars:' + str(n_hps)), exc_info=True)
        logger.info(msg=(self.fun_id+' n_hyperpars:'+str(n_hps)+' init:'+str(init)+' iterations:'+str(iterations)))
        return init, iterations


    def fun_maximize_optimizer(self, hyperparameters: dict) -> object:
        '''
        runs the BayesianOptimization function by maximizing the black box output (loss)
        in case of no changes after te init steps, the function breaks.
        :param hyperparameters: dict,
            dictionary with hyperparameters: 'hyperparameter': list(value_range)
        :return: objec,
            optimizer.max
        '''
        optimizer = BayesianOptimization(
            f = self.black_box_dimreduce,
            pbounds = hyperparameters,
            random_state = 1,
            verbose = 0)
        try:
            optimizer.maximize(init_points=self.init_steps, n_iter=self.iter_steps)
        except:
            logger.info(msg=('no loss change with hyperparameter tuning ', self.fun_id))
            pass
        return optimizer


    def get_best_optimizer_result(self, optimizer):
        '''
        function to return the dictionary of the best dimesnionality reduction.
        The function loops through the optimizer results (optimizer.res) which are
        chronologically sorted. It searches for the n-th entry in list of results with the
        best result (hyperparameters of result == hyperparameters optimizer.res == hyperparameters
        optimizer.max)
        :param optimizer:
            optimizer object
        :return: dict,
            dictionary with best result (highest loss)
        '''
        best_result = {}
        try:
            best_params = optimizer.max['params']
            for i, row in enumerate(optimizer.res):
                if row['params'] == best_params:
                    best_result = self.list_of_dict_results[i]
        except:
            best_result = {}
            logger.error(msg=('find best results ' + self.fun_id), exc_info=True)
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
                -> is raised in case of no change of loss afer initialization phase. Ignore!

        - - - SOURCE / INFORMATION - - -
        https://github.com/fmfn/BayesianOptimization

        :param fun_id: str,
            function identifier
        :param hyperparameters: dict,
            dictionaries with hyperparameters and values
        :param data: np.array,
            high dimensional data
        :param ndim: int,
            low dimension
        :return: (dict,list),
            dict:
                dictionary of best dim reduction results
            list:
                list of dicts with results of all dimensionality reductions.
        '''
        self.fun_id = fun_id
        self.data_high = data
        self.dim_low = ndim
        self.list_of_dict_results = []
        self.init_steps, self.iter_steps = self.get_init_iterations(hyperparameters)

        # checks if dimension has the correct format (returns: True or False)
        dimcheck = check_low_dimension(self.dim_low, self.data_high.shape[1])

        # dimension has incorrect format
        if not dimcheck:
            self.list_of_dict_results = [empty_dict(self.fun_id, self.dim_low)]
            dict_best_results = self.list_of_dict_results[0]

        # dimensionality reduction of functions without hyperparameters.
        elif (self.init_steps + self.iter_steps) == 1:
            _ = self.black_box_dimreduce(**hyperparameters)
            # there is only one dict here
            dict_best_results = self.list_of_dict_results[0]

        # dimensionality reduction of functions with hyperparameters.
        else:
            optimizer = self.fun_maximize_optimizer(hyperparameters)
            if not optimizer:
                dict_best_results = {}
                self.list_of_dict_results = []
            else:
                dict_best_results = self.get_best_optimizer_result(optimizer)

        return dict_best_results, self.list_of_dict_results