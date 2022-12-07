import numpy as np
import pandas as pd
from .II_functions_multiprocess import multiprocess_dimred
from helper_data.global_vars import *
from helper_data.utils_data import timit_
from utils_logger import logger


class Full_data:
    '''
    Reduction of dimensionality of full dataset with target dimension and
    the best functions (with best hyperparameters).
    '''

    def __init__(self, cutoff_loss: float ):

        '''
        :param cutoff_loss: float, cutoff for loss
        '''
        self.loss_fun = globalvar_loss_function
        self.cutoff_loss = cutoff_loss


    def track_progress(self, dict_result_fun: dict):
        '''
        print some data after testing each new dim in order to keep track
        of the dim reduction procedures and identify problems quickly.
        :param dict_result_fun: dict, dim reduction results.
        '''
        fun = dict_result_fun['fun_id']
        dim = dict_result_fun['dim_low']
        loss = dict_result_fun[globalvar_loss_function]
        logger.info(msg=(str(fun).strip()+' dim: '+str(dim)+ ' '+globalvar_loss_function+': '+str(loss)))


    def worker_reduce(self,
          lods_best_functions: list,
          high_data: np.array,
          dim_low: int
        ) -> list:
        '''
        Worker function which reduces the dimension for the best functions and
        measures the quality of the dm reduction.
        In the process it saves the results of all dim reductions in the
        self.results_step3 list and returns only dictionaries of the best functions
        quality of dim reduction >= loss cutoff.

        :param factor:
        :param lods_best_functions: list, list of dictionaries with best dim reduction results
                                    eg. best functions and hyperparameters
        :param high_data: np.array, high dimensional data
        :param dim_low: int, target dimension low dimensional data
        :return: list, list of dictionaries of best dim reductions
        '''
        # init
        lods_best_funs_new = []

        # reduce the full dataset with the traget dimension and the best functions
        lods_results = multiprocess_dimred(lods_best_functions, high_data, dim_low)
        # update the results
        self.results_step3.append(lods_results)

        # select those functions that are above the loss cutoff
        for i in lods_results:
            # print results of each function
            self.track_progress(i)

            # only functions with cutoff > cutoff_loss
            if i[globalvar_loss_function] >= self.cutoff_loss:
                lods_best_funs_new.append(i)

        # save only the best results
        self.track_lods_best_funs_new.append(lods_best_funs_new)

        return lods_best_funs_new



    @timit_('step3_full_data ')
    def dim_reduction_best_functions(self,
             lods_best_functions: list,
             data_high: np.array,
             dim_low: int
        ) -> (list, pd.DataFrame):
        '''
        finds the intrinsic dimension of the dataset.
        in previous steps we found the target dimension and the best functions with a reduced
        size dataset. Here we test them on the full dataset. If the dim reduction quality is
        higher than the loss cutoff for at one or mor functions. We repeat the dim reduction with a lower
        dimension until the qality is too low.
        In the opposite case (quality is to low) we go one dimension up until we find at least
        one function which reduces the dimension with quality >= loss  cutoff.
        We handle exceptions were dimensions are 1 or the number of columns of the high_data.
        Loggings are used to track the function, there have been a few exceptions with some datasets
        so we want to know better whats going on.

        :param lods_best_functions: list, list of dictionaries with results containing information such as hyperparameters,
                                function identifier etc.
        :param data_high: np.array, high dimensional data
        :param dim_low: int, dimension of low dimensional data
        :return: list, list of lists of results (dictionaries) for each dimension testet here
        '''
        # TODO retrieve reduced dataset with best quality

        self.results_step3 = []
        self.track_lods_best_funs_new = []
        best_results_final = []

        # we start with the dim reduction and quality assessment of the best functions
        # and the target dim with the full dataset (in step 2 we used only a small portion of the dataset).
        # There should be always an input in lods_best_functions.
        # Returns list of dicts (lods) of best functions in case results are
        # good.

        lods_best_funs_new = self.worker_reduce(lods_best_functions, data_high, dim_low)

        # two options:
        # A) there are results with the target dim -> lets go down one dimension and try again
        if lods_best_funs_new:
            # check for functions taht are higher than cutoff
            tmp = [s for s in lods_best_funs_new if s[globalvar_loss_function] > self.cutoff_loss]

            # there are function higher then cutoff, continue with those and a lower dim
            # Note: this filters out functions with loss == cutoff (a lower dim will likelly give a result lower than cutoff)
            if tmp:
                lods_best_funs_new = tmp
                factor = -1

            # there are only functions with loss == cutoff, (a lower dim will likelly give a result lower than cutoff)
            # return those
            else:
                results_step_3 = {
                    'results': self.results_step3,
                    'best_results': lods_best_funs_new,
                    'intrinsic_dimension': dim_low
                }
                return results_step_3


        # B) no satisfying results, lets go up to higher dims.
        # This might be the case when pca works fine with a lower dimension on the reduced size
        # dataset but not here on the full size dataset.
        else:
            lods_best_funs_new = lods_best_functions
            factor = 1


        # higher or lower the dim to achieve minimum dimension
        # 100 is just a random number
        for i in range(20):

            # new dim_low
            dim_low = dim_low + factor
            logger.info(msg=('step3 dim_low', dim_low)) # TODO: remove

            # break if dim_low is smaller 1 or higher ncols
            if dim_low < 1:
                dim_low = 1
                best_results_final = lods_best_funs_new
                logger.info(msg=('step3 A', dim_low, factor)) # TODO: remove
                break
            elif dim_low >= data_high.shape[1]:
                dim_low = data_high.shape[1]
                best_results_final = lods_best_funs_new
                logger.info(msg=('step3 B', dim_low, factor)) # TODO: remove
                break
            else:
                pass

            # reduce the dimensionality with all functions in dicts_best_funs with new dim low
            lods_best_funs_new = self.worker_reduce(lods_best_funs_new, data_high, dim_low)

            # there are results with a smaller than the initial dimension, append results and
            # lower the dimension by one. Remove
            if lods_best_funs_new and factor == -1:
                tmp = [s for s in lods_best_funs_new if s[globalvar_loss_function] > self.cutoff_loss]
                lods_best_funs_new = tmp
                print('3', lods_best_funs_new)
                logger.info(msg=('step3 C', dim_low, factor)) # TODO: remove
                continue

            # there are no results with smaller dimension, but there are results with the
            # previously tested higher dimension. stop here.
            elif not lods_best_funs_new and factor == -1:
                dim_low = dim_low - factor
                best_results_final = self.track_lods_best_funs_new[-2]
                logger.info(msg=('step3 D', dim_low, factor)) # TODO: remove
                break

            # there are results with the higher dimension (and not before), break here
            elif lods_best_funs_new and factor == 1:
                dim_low = dim_low
                best_results_final = self.track_lods_best_funs_new[-1]
                logger.info(msg=('step3 E', dim_low, factor))  # TODO: remove
                break

            # there are no satisfying results with dim_low (and havnt been before).
            # go one dimension up with the best funcions we started with.
            elif not lods_best_funs_new and factor == 1:
                lods_best_funs_new = lods_best_functions
                logger.info(msg=('step3 F', dim_low, factor))
                continue
            else:
                logger.info(msg=('step3 G  something completelly unexpected happened!', dim_low, factor)) # TODO: remove


        # results: list of lists of dictionaries
        # each list contains the results of dim reductions with all functions for one dimension.
        # as we increase or decrese the dimensions we add a new list to the list :)
        results_step_3 = {
            'results': self.results_step3,
            'best_results': best_results_final,
            'intrinsic_dimension': dim_low
        }
        return results_step_3