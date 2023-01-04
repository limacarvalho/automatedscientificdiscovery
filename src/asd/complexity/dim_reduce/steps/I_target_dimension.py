from dim_reduction.dimred_main import Dimreduction
from helper_data.global_vars import *
from helper_data.utils_data import empty_dict, timit_, check_low_dimension
from utils_logger import logger
from typing import Union
import numpy as np



def factor_stepsize(ncols):
    '''
    calculates the stepsize eg. number of dimensions changed in each step to increase the
    speed with high dimensional data. As higher the number of dimensions (columns) as higher the factor.
    :param: ncols: int,
        number of columns of the dataset
    :return: int,
        factor (number of dimensions changed in each step)
    '''
    if ncols > 150:
        factor = int(round(np.sqrt(np.sqrt(ncols)), 0) - 1)
    else:
        factor = 1
    return factor



class Target_dimension:
    '''
    Find the target dimension for the next step (hyperparameter optimization).
    Target dimension is the samllest possible dimension with sufficient quality.
    PCA (py_pca) function is used because its accurate and very fast.
    A new dimension is calculated (dim_low), then the dataset is reduced to dim_low and the loss
    is measured. If the loss is higher than the cutoff (cutoff_loss) the dim_low is reduced.
    If its lower, the diemsnion is increased.
    These steps are repeated until the best dimension is found.

    2 main steps:
    In order to increase the speed and quality we first calculate the target dimension with
    the reduced dataset and adjust it using the full dataset.
    '''

    def __init__(self,
                 data_high: np.array,
                 data_high_small: np.array,
                 data_id: str = 'data',
                 cutoff_loss: float = 0.99,
                 loss_fun: str = globalvar_loss_function
                 ):
        '''
        :param data_high: np.array,
            high dimensional data
        :param data_id: str, default = random generated string
            identifier for dataset
        :param cutoff_loss: float, default = 0.99
            cutoff for quality (loss)
        :param loss_fun: str, default = relative error 0...1 (best)
            loss function
        '''
        # init
        self.data_id = data_id
        self.data_high = data_high
        self.data_high_small = data_high_small
        self.ncols = data_high.shape[1]
        self.loss_fun = loss_fun
        self.cutoff_loss = cutoff_loss


    def dim_reduce_worker(self, dim_low: int, data: np.array) -> float:
        '''
        call dim reduce function and reduce the dimension with 'py_pca'.
        this function returns the results in dictionary format containing Q matrix, rel_err,
        time etc. The results are saved in a list (list of dicts).
        :param: data: np.array,
            dataset (scaled)
        :param: dim_low: int,
            target dimension
        :return: float,
            loss of dimensionality reduction
        '''
        # returns: dict with
        dimred = Dimreduction(fun_id='py_pca', data_high=data, dim_low=dim_low)
        results = dimred.exe_dimreduce(params={}, step=globalstring_step1)
        self.list_of_dicts.append(results)
        return results[self.loss_fun]


    def bisect_dimension(self, dim: int) -> int:
        '''
        calculates new dimension as 50 percent of actual dimension.
        :param dim: int,
            actual dimension
        :return: int,
            new_dimension
        '''
        # the round step is necesarry, python dont know basic mathmatics.
        dim_new = abs(int(round(dim * 0.5, 0)))
        return dim_new



    def target_dimension_small_dataset(self) -> (list, int):
        '''
        Find target dimension using the reduced dataset (speed).
        Search for target dimension of the dataframe following these steps:
        1) pca with dimension x on dataframe
        2) quality measurement (loss)
        3) evaluate results and increase or decrease the dimension
        4) repeat 1-3) until best dimension is found eg. smallest possible dimension
           with sufficient quality.
        In order to reduce the number of dim reduction steps we adjust the stepsize (factor).
        :return:  list, int
             list_of_dicts:
                list of dicts of results of dim reductions
             int:
                target dimension
        '''
        self.list_of_dicts = []
        ndims = [self.ncols]
        dim_low = ndims[-1]
        factor = factor_stepsize(self.ncols)

        # when dimension greater than 1: new dimension by bisection
        if dim_low > 1:
            dim_low = int(round(dim_low * 0.5, 0))
            ndims.append(dim_low)

        # if dim is 1, break here, target dim = 1
        else:
            target_dim = 1
            return self.list_of_dicts, target_dim

        # run a while loop until stop condition (stop=1) is reached
        stop = None
        while not stop:

            # calculate range of dimensions for bisection
            dim_range = abs(ndims[-2] - ndims[-1])

            # reduce dimension to dim_low and measure the quality
            loss = self.dim_reduce_worker(dim_low, data=self.data_high_small)

            ## OPTION 0: dimension is number of features, test n_featurs-1 with full dataset in next step
            if dim_low >= self.ncols and not loss:
                dim_low = self.ncols - factor # 1
                break

            ## OPTION 1: dimension is one and loss is higher than cutoff, stop here, target_dim = 1
            elif dim_low == 1 and loss > self.cutoff_loss:
                break

            ## OPTION 2: dim_range is 1 and the quality is higher than cutoff, stop here, target_dim = dim_low
            elif dim_range == 1 and loss > self.cutoff_loss:
                break

            ## OPTION 3: loss is exactly the cutoff, it cant be better. stop here, target_dim = dim_low
            elif loss == self.cutoff_loss:
                break

            ## OPTION 4: loss is lower than cutoff
            elif loss < self.cutoff_loss:

                # loss is lower than cutoff
                # check if we have already data from dim_low + factor
                # if yes and its >= than cutoff: stop here, target_dim = dim_low + factor
                for i in self.list_of_dicts:
                    if i['dim_low'] == dim_low + factor and i[self.loss_fun] >= self.cutoff_loss:
                        dim_low = dim_low + factor
                        self.list_of_dicts.append(i)
                        break

                # loss is lower than cutoff, if dim low is ncols -1, ncols is the dimension
                if dim_low == ndims[0] - factor:
                    dim_low = dim_low + factor
                    continue

                # loss is lower than cutoff
                # calculate new dim and start again
                elif not stop:
                    dim_range = self.bisect_dimension(dim=dim_range)
                    if dim_range == 0:
                        dim_low = ndims[-1] + factor
                    else:
                        dim_low = ndims[-1] + dim_range
                    ndims.append(dim_low)
                    continue

            ## OPTION 5: loss > cutoff_loss
            # make sure there is no smaller dimension than this one
            else :
                # 1: loss > cutoff_loss, it cant get lower than 1, stop here, 1 is the target_dimension
                # 2: 2 is too high, so dim must be 1 or 2, 2 is higher than cutoff, set dim_low to 1 and check!
                if dim_low <= 2:
                    dim_low = 1
                    break

                # loss > cutoff_loss, calculate new dim and check again
                else:
                    dim = self.bisect_dimension(dim=dim_range)
                    dim_low = ndims[-1] - dim
                    ndims.append(dim_low)
                    continue

        return self.list_of_dicts, dim_low



    def target_dimension_full_dataset(self, dim_low):
        '''
        Finetune target dimension obtained in step 1 with the small dataset, but here we use the full dataset.
        Search for target dimension of the dataframe following these steps:
        1) pca with dimension x on dataframe
        2) quality measurement (loss)
        3) evaluate results and increase or decrease the dimension
        4) repeat 1-3) until best dimension is found eg. smallest possible dimension
           with sufficient quality.
        In order to reduce the number of dim reduction steps we adjust the stepsize (factor).
        :param dim_low: int,
            target dimension
        :return: list, int
            list:
                list of dictionaries (lods) with results
             int:
                target dimension
        '''
        ndims = []
        self.list_of_dicts = []
        factor = factor_stepsize(dim_low)


        # when dimension greater than 1: new dimension by bisection
        if dim_low > 1:
            ndims.append(dim_low)

        # if dim is 1, break here, target dim = 1
        else:
            target_dim = 1
            return self.list_of_dicts, target_dim

        # run a while loop until stop condition (stop=1) is reached
        stop = None
        while not stop:

            # reduce dimension to dim_low and measure the quality
            loss = self.dim_reduce_worker(dim_low, self.data_high)

            ## 1: dimension is the number of features
            if dim_low >= self.ncols and not loss:
                dim_low = self.ncols - factor # 1
                break

            ## 1: dimension is one and loss is higher than cutoff, stop here, target_dim = 1
            if dim_low <= 1 and loss > self.cutoff_loss:
                break

            ## 2: the quality is higher than cutoff but dim_low-1 is already testet, (and therefore lower).
            elif loss > self.cutoff_loss and dim_low-factor in ndims: # 1
                break

            ## 3: the quality is higher than cutoff and dim_low-1 is not testet, therefore target_dim must be higher.
            elif loss > self.cutoff_loss and dim_low-factor not in ndims:
                dim_low = dim_low - factor
                continue


            ## OPTION 3: loss is exactly the cutoff, it cant be better. stop here, target_dim = dim_low
            elif loss == self.cutoff_loss:
                break

            ## OPTION 4: loss is lower than cutoff
            elif loss < self.cutoff_loss:

                # loss is lower than cutoff, check if we have already data from dim_low + factor
                # if yes and its >= than cutoff: stop here, target_dim = dim_low + factor
                for i in self.list_of_dicts:
                    if i['dim_low'] == dim_low + factor and i[self.loss_fun] >= self.cutoff_loss:
                        dim_low = dim_low + factor
                        self.list_of_dicts.append(i)
                        break

                # loss is lower than cutoff, if dim low is ncols -1, ncols is the dimension
                if dim_low == ndims[0] - factor:
                    dim_low = dim_low + factor
                    continue

                # loss is lower than cutoff, calculate new dim and start again
                elif not stop:
                    dim_low = ndims[-1] + factor
                    ndims.append(dim_low)
                    continue

        return self.list_of_dicts, dim_low



    @timit_('step1_target_dimension ')
    def main_target_dimension(self) -> (dict, float):
        '''
        main function to calculate the target dimension.
        In case there is no satisfying result an empty dict is returned.
        :return: list, int
            list:
                list of dicts (lods) with results
             int:
                target dimension
        '''
        try:
            # calculate target dimension with the reduced dataset
            lods_results_, target_dim_ = self.target_dimension_small_dataset()
            # fine tuning of target dimension with the full size dataset
            lods_results, target_dim = self.target_dimension_full_dataset(target_dim_)
            results_step_1 = {'results': lods_results_ + lods_results, 'target_dim': target_dim}
            loss = self.cutoff_loss
        except:
            results_step_1 = {'results': [empty_dict(fun_id='py_pca', dim_low=None)], 'target_dim': None}
            loss = self.cutoff_loss
            logger.error(msg=('target dimension: '+self.data_id+'something went wrong calculating the target dimension.'),
                         exc_info=True)

        return results_step_1, loss
