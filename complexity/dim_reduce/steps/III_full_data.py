import numpy as np
import pandas as pd
from dimension_tools.dimension_suite.dim_reduce.steps.II_functions_multiprocess import multiprocess_dimred
from dimension_tools.dimension_suite.dim_reduce.helper_data.global_vars import *


class Full_data:
    '''
    Reduction of dimensionality of full dataset with target dimension and
    the best functions (with best hyperparameters).
    '''

    def __init__(self,
                 cutoff_loss: float,
                 ):

        '''
        :param cutoff_loss: cutoff for loss
        :param loss_fun: loss function usually mae_norm
        '''
        self.loss_fun = globalvar_loss_function
        self.cutoff_loss = cutoff_loss


    def track_progress(self, dict_result_fun: dict):
        '''

        :param dict_result_fun:
        :return:
        '''
        fun = dict_result_fun['fun_id']
        dim = dict_result_fun['dim_low']
        loss = dict_result_fun[globalvar_loss_function]
        print(fun, ' dim:', dim, ' ', globalvar_loss_function, loss)



    def worker_reduce(self,
                      lods_best_functions: list,
                      high_data: np.array,
                      dim_low: int) -> list:
        '''
        Worker function which reduces the dimension for the best functions and
        measures the quality of the dm reduction.
        In the process it saves the results of all dim reductions in the
        self.results_step3 list and returns only dictionaries of the best functions
        quality of dim reduction >= loss cutoff.

        :param lods_best_functions: list of dictionaries with best dim reduction results
                                    eg. best functions and hyperparameters
        :param high_data: high dimensional data
        :param dim_low: target dimension low dimensional data
        :return: lis of dictionaries of best dim reductions
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

            if i[globalvar_loss_function] >= self.cutoff_loss:
                lods_best_funs_new.append(i)

        # save only the best results
        self.track_lods_best_funs_new.append(lods_best_funs_new)

        return lods_best_funs_new


    def dim_reduction_best_functions(self,
                                     lods_best_functions: list,
                                     data_high: np.array,
                                     dim_low: int) -> (list, pd.DataFrame):
        '''
        finds the intrinsic dimension of the dataset.
        in previous steps we found the target dimension and the best functions with a reduced
        size dataset. Here we test them on the full dataset. If the dim reduction quality is
        higher than the loss cutoff for at one or mor functions. We repeat the dim reduction with a lower
        dimension until the qality is too low.
        In the opposite case (quality is to low) we go one dimension up until we find at least
        one function which reduces the dimension with quality >= loss  cutoff.
        We handle exceptions were dimensions are 1 or the number of columns of the high_data.

        :param lods_best_functions: list dictionaries with results containing information such as hyperparameters,
                                function identifier etc.
        :param data_high: np.array high dimensional data
        :param dim_low: int dimension of low dimensional data
        :return: list of lists of results (dictionaries) for each dimension testet here
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
            factor = -1

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
            print('step3 dim_low', dim_low)

            # break if dim_low is smaller 1 or higher ncols
            if dim_low < 1:
                dim_low = 1
                best_results_final = lods_best_funs_new
                print('step3 A', dim_low, factor) # TODO remove
                break
            elif dim_low >= data_high.shape[1]:
                dim_low = data_high.shape[1]
                best_results_final = lods_best_funs_new
                print('step3 B', dim_low, factor) # TODO remove
                break
            else:
                pass

            # reduce the dimensionality with all functions in dicts_best_funs with new dim low
            lods_best_funs_new = self.worker_reduce(lods_best_funs_new, data_high, dim_low)

            # there are results with a smaller than the initial dimension, append results and
            # lower the dimension by one.
            if lods_best_funs_new and factor == -1:
                print('step3 C', dim_low, factor) # TODO remove
                continue


            # there are no results with smaller dimension, but there are results with the
            # previously tested higher dimension. stop here.
            elif not lods_best_funs_new and factor == -1:
                dim_low = dim_low - factor
                best_results_final = self.track_lods_best_funs_new[-2]
                print('step3 D', dim_low, factor) # TODO remove
                break

            # there are results with the higher dimension (and never before) so break here
            elif lods_best_funs_new and factor == 1:
                dim_low = dim_low
                best_results_final = self.track_lods_best_funs_new[-1]
                print('step3 E', dim_low, factor) # TODO remove
                break

            # there are no satisfying results with dim_low (and havnt been before).
            # go one dimension up with the best funcions we started with.
            elif not lods_best_funs_new and factor == 1:
                lods_best_funs_new = lods_best_functions
                print('step3 F', dim_low, factor) # TODO remove
                continue

            else:
                # TODO do something more here
                print('step3 G something completelly unexpected happened!', dim_low, factor) # TODO remove


        # results: list of lists of dictionaries
        # each list contains the results of dim reductions with all functions for one dimension.
        # as we increase or decrese the dimensions we add a new list to the list :)
        results_step_3 = {'results': self.results_step3,
                          'best_results': best_results_final,
                          'intrinsic_dimension': dim_low}
        return results_step_3