from dim_reduction.dimred_main import Dimreduction
from helper_data.global_vars import *
from helper_data.utils_data import empty_dict, timit_
from utils_logger import logger
from helper_metrix.metrics_dimred import fun_kmax
from typing import Union
import numpy as np


class Target_dimension:
    '''
    Find the target dimension for dimensionality reduction.
    The target dimension is the smallest dimension to reduce the dataset with sufficient quality.
    PCA (py_pca) function is used because its accurate and very fast.
    A new dimension is calculated (dim_low), then the dataset is reduced to dim_low and the loss
    is measured.
    If the loss is higher than the cutoff (cutoff_loss) the dim_low is reduced if its lower its
    increased until the best dim is found.
    The first dim_low is 50 percent of the number of columns.
    '''

    def __init__(self,
                 data_high: np.array,
                 data_high_small: np.array,
                 data_id: str = 'data',
                 cutoff_loss: float = 0.99,
                 loss_fun: str = globalvar_loss_function
                 ):

        '''
        :param data_high: np.array,high dimensional data
        :param data_id: str, data_identifier (default=data)
        :param cutoff_loss: float, cutoff for loss (default=0.99)
        :param loss_fun: str, loss function (default=mae_norm)
        '''

        # init
        self.data_id = data_id
        self.data_high = data_high
        self.data_high_small = data_high_small
        self.ncols = data_high.shape[1]
        self.loss_fun = loss_fun
        self.cutoff_loss = cutoff_loss


    def dim_reduce_worker(self, dim_low, data):
        '''
        call dim reduce function and reduce the dimension with 'py_pca'.
        this function returns the results in dictionary format containing Q matrix, mae_norm,
        time etc. The results are saved in a list (list of dicts).
        :param data:
        :param dim_low: int, target dimension
        :return: float, loss of dimensionality reduction (default=mae_norm)
        '''
        # returns: dict with
        dimred = Dimreduction(fun_id='py_pca', data_high=data, dim_low=dim_low)
        results = dimred.exe_dimreduce(params={}, step=globalstring_step1)
        self.list_of_dicts.append(results)
        return results[self.loss_fun]


    def bisect_dimension(self, dim: int) -> int:
        '''
        calculates new dimension (50 percent of dim).
        :param dim: int, dimension
        :return: int, new_dimension
        '''
        # the round step is necesarry, python dont know basic mathmatics.
        dim_new = abs(int(round(dim * 0.5, 0)))
        return dim_new


    def target_dimension_small_dataset(self) -> (list, int):
        '''
        Masterpiece!
        I tried several other approaches (bayesian optimization) but this one works best.
        Search for target dimension of the dataframe following these steps:

        1) pca with dimension x on dataframe
        2) quality measurement
        3) evaluate results and increase or decrease the dimension
        4) repeat 1-3) until best dimension is found eg the minimum dimension with dim
        reduction results above the loss cutoff.

        In order to reduce the number of dim reductions we always reduce the dimension range
        by two until we can only go up or down with one dimension.
        We always check if the dimension is 1 or ncols and if it mut be a diemension we
        have already checkt. Therefore the script looks very chaotic.

        :return: list_of_dicts: list, list of dicts of results of all dim reductions
                 dim_low: int, target dimension
        '''
        # TODO: remove prints
        self.list_of_dicts = []
        ndims = [self.data_high.shape[1]]
        dim_low = ndims[-1]

        # when dimension greater than 1: new dimension by bisection
        if dim_low > 1:
            dim_low = int(round(dim_low * 0.5, 0))
            ndims.append(dim_low)

        # if dim is 1, break here, target dim = 1
        else:
            target_dim = 1
            # print('A', dim_low, target_dim)
            return self.list_of_dicts, target_dim

        # run a while loop until stop condition (stop=1) is reached

        stop = None
        while not stop:
            # print('start', dim_low, ndims)

            # calculate range of dimensions for bisection
            dim_range = abs(ndims[-2] - ndims[-1])

            # reduce dimension to dim_low and measure the quality
            loss = self.dim_reduce_worker(dim_low, data=self.data_high_small)
            # print('loop1', dim_low, 'loss:', loss, 'ndims:', ndims, 'range:', dim_range)

            ## OPTION 1: dimension is one and loss is higher than cutoff, stop here, target_dim = 1
            if dim_low == 1 and loss > self.cutoff_loss:
                # print('stopped A:', dim_low, 'loss:', loss, 'ndims:', ndims)
                break

            ## OPTION 2: dim_range is 1 and the quality is higher than cutoff, stop here, target_dim = dim_low
            elif dim_range == 1 and loss > self.cutoff_loss:
                # print('stopped B :', dim_low, 'loss:', loss, 'ndims:', ndims)
                break

            ## OPTION 3: loss is exactly the cutoff, it cant be better. stop here, target_dim = dim_low
            elif loss == self.cutoff_loss:
                # print('stopped C:', dim_low, 'loss:', loss, 'ndims:', ndims)
                break

            ## OPTION 4: loss is lower than cutoff
            elif loss < self.cutoff_loss:

                # loss is lower than cutoff
                # check if we have already data from dim_low + 1
                # if yes and its >= than cutoff: stop here, target_dim = dim_low + 1
                # print('loop2 dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                for i in self.list_of_dicts:
                    if i['dim_low'] == dim_low + 1 and i[self.loss_fun] >= self.cutoff_loss:
                        dim_low = dim_low + 1
                        self.list_of_dicts.append(i)
                        # print('stopped D:', dim_low, 'loss:', i[self.loss_fun], 'ndims:', ndims)
                        break

                # loss is lower than cutoff, if dim low is ncols -1, ncols is the dimension
                if dim_low == ndims[0] - 1:
                    dim_low = dim_low + 1
                    continue

                # loss is lower than cutoff
                # calculate new dim and start again
                elif not stop:
                    dim_range = self.bisect_dimension(dim=dim_range)
                    if dim_range == 0:
                        dim_low = ndims[-1] + 1
                    else:
                        dim_low = ndims[-1] + dim_range
                    ndims.append(dim_low)
                    # print('loop2 A dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                    continue

            ## OPTION 5: loss > cutoff_loss
            # make sure there is no smaller dimension than this one
            else :
                # loss > cutoff_loss, it cant get lower than 1, stop here, 1 is the target_dimension
                if dim_low == 1:
                    break

                # loss > cutoff_loss
                # 2 is too high, so dim must be 1 or 2, 2 is higher than cutoff, set dim_low to 1 and check!
                elif dim_low == 2:
                    dim_low = 1
                    break

                # loss > cutoff_loss, calculate new dim and check again
                else:
                    dim = self.bisect_dimension(dim=dim_range)
                    dim_low = ndims[-1] - dim
                    ndims.append(dim_low)
                    # print('loop3 A dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                    continue
        return self.list_of_dicts, dim_low



    def long_search_target_dim(self, dim_low):
        # TODO: remove prints
        ndims = []
        self.list_of_dicts = []


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
            # print('--loop1', dim_low, 'loss:', loss, 'ndims:', ndims)

            ## 1: dimension is one and loss is higher than cutoff, stop here, target_dim = 1
            if dim_low == 1 and loss > self.cutoff_loss:
                # print('--stopped 0:', dim_low, 'loss:', loss, 'ndims:', ndims)
                break

            ## 2: the quality is higher than cutoff but dim_low-1 is already testet, (and therefore lower).
            elif loss > self.cutoff_loss and dim_low-1 in ndims:
                # print('--stopped A:', dim_low, 'loss:', loss, 'ndims:', ndims)
                break

            ## 3: the quality is higher than cutoff and dim_low-1 is not testet, therefore target_dim must be higher.
            elif loss > self.cutoff_loss and dim_low-1 not in ndims:
                dim_low = dim_low - 1
                # print('--stopped B :', dim_low, 'loss:', loss, 'ndims:', ndims)
                continue


            ## OPTION 3: loss is exactly the cutoff, it cant be better. stop here, target_dim = dim_low
            elif loss == self.cutoff_loss:
                # print('--stopped C:', dim_low, 'loss:', loss, 'ndims:', ndims)
                break

            ## OPTION 4: loss is lower than cutoff
            elif loss < self.cutoff_loss:

                # loss is lower than cutoff, check if we have already data from dim_low + 1
                # if yes and its >= than cutoff: stop here, target_dim = dim_low + 1
                # print('--loop2 dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                for i in self.list_of_dicts:
                    if i['dim_low'] == dim_low + 1 and i[self.loss_fun] >= self.cutoff_loss:
                        dim_low = dim_low + 1
                        stop = 1
                        self.list_of_dicts.append(i)
                        print('stopped D:', dim_low, 'loss:', i[self.loss_fun], 'ndims:', ndims)
                        break

                # loss is lower than cutoff, if dim low is ncols -1, ncols is the dimension
                if dim_low == ndims[0] - 1:
                    dim_low = dim_low + 1
                    continue

                # loss is lower than cutoff, calculate new dim and start again
                elif not stop:
                    dim_low = ndims[-1] + 1
                    ndims.append(dim_low)
                    # print('--loop2 A dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                    continue

        return self.list_of_dicts, dim_low



    @timit_('step1_target_dimension ')
    def getter_target_dimension(self, target_dim: Union[int, str]) -> dict:
        '''
        gets the output from above function and handles exceptions.
        In case there is no satisfying result an empty dict is returned.
        :param: target_dim: Union[int, str], value provided by the customer
            target dimension for optimization of dimred function hyperparameters
            (default='auto') calculates the target dim on a dataset with less rows
        :return: lods_results: list, list of dicts (lods) of results of all dim reductions
                 dim_low: int, target dimension
        '''
        def exception_target_dim():
            '''
            empty dictionary and logger message for exceptions.
            :return: dict with empty results and None as target dim
            '''
            kmax = fun_kmax(self.data_high)
            results_step_1 = {
                'results': empty_dict(fun_id='py_pca', dim_low=None, kmax=kmax),
                'target_dim': None
            }
            # message for logger
            msg = 'target dimension: ' + self.data_id + \
                  'please provide valid target dimension [int] or set target_dim to auto.\n' + \
                  'dim_reduce_main -> intrinsic_dimension -> params'
            return results_step_1, msg

        # automatic calculation of target dim
        if target_dim == 'auto':
            try:
                lods_results_, target_dim_ = self.target_dimension_small_dataset()
                lods_results, target_dim = self.long_search_target_dim(target_dim_)
                results_step_1 = {
                    'results': lods_results_+lods_results,
                    'target_dim': target_dim
                }
            except:
                results_step_1, msg = exception_target_dim()
                logger.error(
                    msg=('target dimension. ' + str(self.data_id) + ' returns empty [dict] and no target_dim'),
                    exc_info=True)

        # target dim is provided by the customer
        elif isinstance(target_dim, int) and self.ncols >= target_dim >= 1:
            try:
                kmax = fun_kmax(self.data_high)
                results_step_1 = {
                    'results': empty_dict(fun_id='py_pca', dim_low=None, kmax=kmax),
                    'target_dim': target_dim
                }
            except:
                results_step_1, msg = exception_target_dim()
                logger.error(msg=msg, exc_info=True)

        # wrong value provided by the customer
        else:
            results_step_1, msg = exception_target_dim()
            logger.error(msg=msg, exc_info=True)

        # return dictionary with results and target_dim
        return results_step_1

