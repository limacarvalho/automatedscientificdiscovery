import numpy as np
import pandas as pd
from II_multiprocessing import multiprocess_dimred
from helper_data.global_vars import *


class class_dimreduce_full_data:

    def __init__(self,
                 cutoff_loss=0.99,
                 loss_fun=globalvar_loss_function
                 ):

        '''
        :param cutoff_loss: cutoff for loss
        :param loss_fun: loss function usually mae_norm
        '''
        self.loss_fun = loss_fun
        self.cutoff_loss = cutoff_loss


    def worker_reduce(self, lods_best_functions: list, high_data: np.array, dim_low: int) -> list:
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
            if i[globalvar_loss_function] >= self.cutoff_loss:
                lods_best_funs_new.append(i)

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

        :param dicts_best_funs: dictionaries with results containing information such as hyperparameters,
                                function identifier etc.
        :param data: high dimensional data
        :param dim_low: dimension of low dimensional data
        :return: list of lists of results (dictionaries) for each dimension testet here
        '''
        self.results_step3 = []
        # we start with the dim reduction and quality assessment of the best functions
        # and the target dim. Returns list of dicts (lods) of best functions.
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
        for i in range(100):

            # new dim_low and mak sure its bewtween 1 and number of columns
            dim_low = dim_low + factor

            # check if dim_low is smaller than 1 or same as n columns od data_high
            if dim_low < 1:
                dim_low = 1
                break
            elif dim_low >= data_high.shape[1]:
                dim_low = data_high.shape[1]
                break
            else:
                pass

            # reduce the dimensionality with all functions in dicts_best_funs with new dim low
            lods_best_funs_new = self.worker_reduce(lods_best_funs_new, data_high, dim_low)

            # there are results with a smaller than the initial dimension, append results and
            # lower the dimension by one.
            if lods_best_funs_new and factor == -1:
                continue

            # there are no results with smaller dimension, but there are results with the
            # previous dimension. stop here.
            elif not lods_best_funs_new and factor == -1:
                dim_low = dim_low - factor
                break

            # there are results with the higher dimension (and never before) so break here
            elif lods_best_funs_new and factor == 1:
                dim_low = dim_low
                break

            # there are no results with the higher dimension (and havnt been before)
            # go one dimension up
            else:
                continue


        # results: list of lists of dictionaries
        # each list contains the results of dim reductions with all functions for one dimension.
        # as we increase or decrese the dimensions we add a new list to the list :)
        results_step_3 = {'results': self.results_step3,
                          'intrinsic_dimension': dim_low}
        return results_step_3