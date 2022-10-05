from dimension_tools.dimension_suite.dim_reduce.dim_reduction.dimred_main import Dimreduction
from dimension_tools.dimension_suite.dim_reduce.helper_data.global_vars import *
from dimension_tools.dimension_suite.dim_reduce.helper_data.helper_data import empty_dict
from dimension_tools.dimension_suite.dim_reduce.helper_metrix.loss_functions import fun_kmax
import traceback
from typing import Union


class Target_dimension:
    '''
    Here we find the target dimension for dimensionality reduction.
    The target dimension is the smallest possible dimension above the loss cutoff.
    We start with dim = 50 percent of n-columns, if quality of dimreduction is higher than cutoff
    we go down again by 50 percent. When its lower than cutoff we go up. and so on
    '''

    def __init__(self,
                 data_high,
                 data_id = 'data',
                 cutoff_loss = 0.99,
                 loss_fun = globalvar_loss_function
                 ):

        '''
        :param data_high: high dimensional data
        :param data_id: data_identifier
        :param cutoff_loss: cutoff for loss
        :param loss_fun: loss function usually mae_norm
        '''

        # init
        self.data_id = data_id
        self.data_high = data_high
        self.ncols = data_high.shape[1]
        self.loss_fun = loss_fun
        self.cutoff_loss = cutoff_loss


    def dim_reduce_worker(self, dim_low):
        '''
        call dim reduce function and reduce the dimension with 'pca'.
        this function returns the results in dictionary format containing Q matrix, mae_norm,
        time etc. The results are saved in a list (list of dicts).
        :param dim_low: target dimension
        :return: loss of dimensionality reduction quality test
        '''
        # returns: dict with
        dimred = Dimreduction(fun_id='py_pca', data_high=self.data_high, dim_low=dim_low)
        results = dimred.exe_dimreduce(params={}, step=globalstring_step1)
        self.list_of_dicts.append(results)
        return results[self.loss_fun]


    def bisect_dimension(self, dim: int) -> int:
        '''
        calculates new dimension as 50 percent of input.
        the round step is necesarry, python dont know basic mathmatics.
        :param dim: dimension
        :return: new_dimension
        '''
        dim_new = abs(int(round(dim * 0.5, 0)))
        return dim_new


    def quick_search_target_dimension(self) -> (list, int):
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

        :return: list_of_dicts: list of dicts of results of all dim reductions
                 dim_low: target dimension
        '''
        self.list_of_dicts = []
        ndims = [self.data_high.shape[1]]
        dim_low = ndims[-1]

        # when dimenion greater than 1: new dimension by bisection
        if dim_low > 1:
            dim_low = int(round(dim_low * 0.5, 0))
            ndims.append(dim_low)

        # if dim is 1, break here, target dim = 1
        else:
            target_dim = 1
            # print('A', dim_low, target_dim)
            return self.list_of_dicts, target_dim

        # run a while loop until stop condition is reached
        stop = None
        while not stop:
            # print('start', dim_low, ndims)

            # calculate range of dimensions for bisection
            dim_range = abs(ndims[-2] - ndims[-1])

            # reduce dimension to dim_low and measure the quality
            loss = self.dim_reduce_worker(dim_low)
            # print('loop1', dim_low, 'loss:', loss, 'ndims:', ndims, 'range:', dim_range)

            ## OPTION 1: dimension is one and loss is higher than cutoff
            #            stop here, target_dim = 1
            if dim_low == 1 and loss > self.cutoff_loss:
                stop = 1
                # print('stopped A:', dim_low, 'loss:', loss, 'ndims:', ndims)
                continue

            ## OPTION 2: dim_range is 1 and the quality is higher than cutoff
            #            stop here, target_dim = dim_low
            elif dim_range == 1 and loss > self.cutoff_loss:
                stop = 1
                # print('stopped B :', dim_low, 'loss:', loss, 'ndims:', ndims)
                continue

            ## OPTION 3: loss is exactly the cutoff, it cant be better.
            #            stop here, target_dim = dim_low
            elif loss == self.cutoff_loss:
                # print('stopped C:', dim_low, 'loss:', loss, 'ndims:', ndims)
                stop = 1
                continue

            ## OPTION 4: loss is lower than cutoff
            elif loss < self.cutoff_loss:

                # loss is lower than cutoff
                # check if we have already data from dim_low + 1
                # if yes and its >= than cutoff: stop here, target_dim = dim_low + 1
                # print('loop2 dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                for i in self.list_of_dicts:
                    if i['dim_low'] == dim_low + 1 and i[self.loss_fun] >= self.cutoff_loss:
                        dim_low = dim_low + 1
                        stop = 1
                        self.list_of_dicts.append(i)
                        # print('stopped D:', dim_low, 'loss:', i[self.loss_fun], 'ndims:', ndims)
                        break

                # loss is lower than cutoff
                # if dim low is ncols -1, ncols is the dimension
                if dim_low == ndims[0] - 1:
                    dim_low = dim_low + 1
                    continue # correct? shouldnt it be break? but stop here has no effect?!

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
                # loss > cutoff_loss
                # it cant get lower than 1, stop here, 1 is the target_dimension
                if dim_low == 1:
                    stop = 1
                    continue

                # loss > cutoff_loss
                # 2 is too high, so dim must be 1 or 2, 2 is higher than cutoff,
                # set dim_low to 1 and check!
                elif dim_low == 2:
                    dim_low = 1
                    continue

                # loss > cutoff_loss
                # calculate new dim and check again
                else:
                    dim = self.bisect_dimension(dim=dim_range)
                    dim_low = ndims[-1] - dim
                    ndims.append(dim_low)
                    # print('loop3 A dim:', dim_low, 'loss:', loss, 'ndims:', ndims)
                    continue

        return self.list_of_dicts, dim_low



    def getter_target_dimension(self, target_dim: Union[int, str]) -> dict:
        '''
        gets the output from above function and handles exceptions.
        In case there is no satisfying result an empty dict is returned.
        :param: Union[int, str] target_dim: value provided by the customer
            target dimension for optimization of dimred function hyperparameters
            default: 'auto' calculates the target dim on a dataset with less rows
        :return: lods_results: list of dicts (lods) of results of all dim reductions
                 dim_low: target dimension
        '''
        def exception_target_dim():
            '''
            build dictionary for exception.
            :return: dict with empty results and None as target dim
            '''
            kmax = fun_kmax(self.data_high)
            results_step_1 = {
                'results': empty_dict(fun_id='py_pca', dim_low=None, kmax=kmax),
                'target_dim': None
            }
            return results_step_1

        # automatic calculation of target dim
        if target_dim == 'auto':
            try:
                lods_results, target_dim = self.quick_search_target_dimension()
                results_step_1 = {
                    'results': lods_results,
                    'target_dim': target_dim
                }
            except:
                results_step_1 = exception_target_dim()
                print(globalstring_error + 'TARGET DIMENSION', self.data_id,
                      'returns empty [dict] and no target_dim')
                print(traceback.format_exc())

        # target dim is provided by the customer
        elif isinstance(target_dim, int) and self.ncols >= target_dim >= 1:
            try:
                kmax = fun_kmax(self.data_high)
                results_step_1 = {
                    'results': empty_dict(fun_id='py_pca', dim_low=None, kmax=kmax),
                    'target_dim': target_dim
                }
            except:
                results_step_1 = exception_target_dim()
                print(globalstring_error + 'TARGET DIMENSION', self.data_id,
                      'please provide valid target dimension or default: "auto"')
                print(traceback.format_exc())

        # wrong value provided by the customer
        else:
            print(globalstring_error + 'TARGET DIMENSION', self.data_id,
                  'please provide valid target dimension or default: "auto"')
            results_step_1 = exception_target_dim()
        # return dictionary with results and target_dim
        return results_step_1

