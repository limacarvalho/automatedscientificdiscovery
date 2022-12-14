import numpy as np
import pandas as pd
from .I_target_dimension import factor_stepsize
from .II_functions_multiprocess import ray_get_dimreduction
from helper_data.global_vars import *
from helper_data.utils_data import timit_
from utils_logger import logger


class dimreduce_full_data:
    '''
    In the previous 2 steps we identified the target dimension (stepI) and best dimesnionality
    reduction functions (stepII) with a reduced dataset in order to save time.
    Here in stepIII the best functions are used to reduce the full dataset.
    '''
    def __init__(self, cutoff_loss: float ):
        '''
        :param cutoff_loss: float,
            loss function cutoff (dim reduction quality)
        '''
        self.loss_fun = globalvar_loss_function
        self.cutoff_loss = cutoff_loss


    # def track_progress(self, dict_result_fun: dict):
    #     '''
    #     logs results dim reduction (function_id, dimension, loss parameter and loss value)
    #     :param dict_result_fun: dict,
    #         dim reduction results.
    #     '''
    #     fun = dict_result_fun['fun_id'] # function id
    #     dim = dict_result_fun['dim_low'] # dimension used in dim reduction
    #     loss = dict_result_fun[globalvar_loss_function]
    #     logger.info(msg=(str(fun).strip()+' dim: '+str(dim)+' '+globalvar_loss_function+': '+str(loss)))


    def worker_reduce(self, lods_best_functions: list, high_data: np.array, dim_low: int ) -> list:
        '''
        Dimensionality reductions with all functions in list of dictionaries of best functions
        (lods_best_functions).
        Updates the results_step3 list with the results and filters the results with
        sufficient quality of dim reduction (loss >= loss cutoff).
        :param lods_best_functions: list,
            list of dictionaries with best dim reduction results eg. best functions and hyperparameters
        :param high_data: np.array,
            high dimensional data
        :param dim_low: int,
            target dimension low dimensional data
        :return: list,
            list of dictionaries of best dim reductions
        '''
        lods_best_funs_new = []
        # reduce the full dataset with the traget dimension and the best functions
        lods_results = ray_get_dimreduction(lods_best_functions, high_data, dim_low)
        # update the results
        self.results_step3.append(lods_results)
        # select those functions that are above the loss cutoff
        for i in lods_results:
            # filter out functions with insufficent loss
            if i[globalvar_loss_function] >= self.cutoff_loss:
                lods_best_funs_new.append(i)
        # save only the best results
        self.track_lods_best_funs_new.append(lods_best_funs_new)
        return lods_best_funs_new



    @timit_('step3_full_data ')
    def intrinsic_dimension_full_data(self, lods_best_functions: list, data_high: np.array, dim_low: int) \
            -> (list, pd.DataFrame):
        '''
        computes the intrinsic dimension of the dataset.
        in previous steps we found the target dimension and the best functions with a reduced
        size dataset. Here we start with testing these functions on the full dataset at the target dimension.
        If the quality of the dim reduction of at least one function is sufficient we reduce the dimension
        and test again until we find the loss cutoff is reached.
        If the quality for all functions is not sufficient we increase the dimension until we find the one
        were at least one function reduces to a sufficient quality.
        :param lods_best_functions: list,
            list of dictionaries with results containing information such as hyperparameters, function identifier etc.
        :param data_high: np.array,
            high dimensional data
        :param dim_low: int,
            dimension of low dimensional data
        :return: list,
            list of lists of results (dictionaries), each list for results of functions tested at a specific dimension.
        '''
        # how many dimensios stepsize, as higher the dimension as higher the stepsize
        factor = factor_stepsize(dim_low)
        self.results_step3 = []
        self.track_lods_best_funs_new = []
        best_results_final = []

        # we start with testing the best functions (stepII) at the target dim (stepI)
        lods_best_funs_new = self.worker_reduce(lods_best_functions, data_high, dim_low)

        # there are two options:
        # A) there are results with the target dim -> go down one 'stepsize' of dimensions and try again
        if lods_best_funs_new:
            # check for functions that are higher than cutoff
            results = [s for s in lods_best_funs_new if s[globalvar_loss_function] > self.cutoff_loss]

            # go down one 'stepsize' of dimensions and try again
            if results:
                lods_best_funs_new = results
                factor = -factor

            # there are functions with loss == cutoff (it cant be much lower), stop here
            else:
                results_step_3 = {
                    'results': self.results_step3,
                    'best_results': lods_best_funs_new,
                    'intrinsic_dimension': dim_low
                }
                return results_step_3


        # B) no satisfying results, lets increase the number of dimensions.
        else:
            lods_best_funs_new = lods_best_functions
            factor = factor

        # 30 is just a random number
        for i in range(30):

            # new dim_low
            dim_low = dim_low + factor
            logger.info(msg=('step3 dim_low', dim_low)) # TODO: remove

            # break if dim_low is < 1
            if dim_low <= 1:
                dim_low = 1
                best_results_final = lods_best_funs_new
                logger.info(msg=('step3 A', dim_low, factor)) # TODO: remove
                break

            # break if dim_low is > n columns
            elif dim_low >= data_high.shape[1]:
                dim_low = data_high.shape[1]
                best_results_final = lods_best_funs_new
                logger.info(msg=('step3 B', dim_low, factor)) # TODO: remove
                break
            else:
                pass

            # reduce the dimensionality with all functions in dicts_best_funs with new dim low
            lods_best_funs_new = self.worker_reduce(lods_best_funs_new, data_high, dim_low)

            # there are results with a smaller dimension, append results and lower the dimension by one stepsize.
            if lods_best_funs_new and factor < 0:
                tmp = [s for s in lods_best_funs_new if s[globalvar_loss_function] > self.cutoff_loss]
                lods_best_funs_new = tmp
                logger.info(msg=('step3 C', dim_low, factor)) # TODO: remove
                continue

            # there are no results with smaller dimension, but there are results with the
            # previously tested higher dimension. stop here.
            elif not lods_best_funs_new and factor < 0:
                dim_low = dim_low - factor
                best_results_final = self.track_lods_best_funs_new[-2]
                logger.info(msg=('step3 D', dim_low, factor)) # TODO: remove
                break

            # there are results with the higher dimension (and not before), break here
            elif lods_best_funs_new and factor > 0 :
                dim_low = dim_low
                best_results_final = self.track_lods_best_funs_new[-1]
                logger.info(msg=('step3 E', dim_low, factor))  # TODO: remove
                break

            # there are no satisfying results with dim_low (and havnt been before).
            # go one stepsize of dimensions up with and try again with the functions we started with.
            elif not lods_best_funs_new and factor > 0:
                lods_best_funs_new = lods_best_functions
                logger.info(msg=('step3 F', dim_low, factor)) # TODO: remove
                continue
            else:
                logger.info(msg=('step3 G  something completelly unexpected happened!', dim_low, factor)) # TODO: remove

        # each list contains the results for one dimension.
        # as we increase or decrese the dimensions we add a new list to the list.
        results_step_3 = {
            'results': self.results_step3, #  list of lists of dictionaries
            'best_results': best_results_final,
            'intrinsic_dimension': dim_low
        }
        return results_step_3