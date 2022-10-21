import numpy as np
from R_helper import class_run_r_functions
from Py_helper import class_run_py_functions
from hyperparameter_initialization import hyperparameter_init
from dimred_call_functions import call_dimred_functions
from helper_metrix import loss_functions as metrix_
from loss_functions import fun_kmax
from helper_data.global_vars import *
from typing import Union
import traceback
import time
import pyper as pr
import json
from asd_logging import logger
r = pr.R(use_pandas = True)
r('library(Rdimtools)')
r('Sys.setenv(LANGUAGE="en")')


class class_dimreduce_main:
    '''
    main function to perform dimensionality reduction for a single function and a single
    dimension.


    '''

    def __init__(self, fun_id: str, data_high: np.array, dim_low=int):
        '''

        :param fun_id: function_identifier
        :param data_high: high dimensional data
        :param dim_low: low dimension
        '''

        self.fun_id = fun_id
        self.data_high = data_high
        self.dim_low = int(dim_low)

        # call function to receive the A) function object B) function default hyperparameters
        # C) Hyperparameter ranges for optimization
        dict_fun = call_dimred_functions(self.fun_id, self.data_high)
        self.fun, self.params_fun, hyperpars = list(dict_fun.values())[0]


    def hyperparameters_r_py(self, params: Union[str, dict], step: str) -> (dict, str):
        '''
        convert R and Python hyperparameters into the correct formt before calling the function.
        3 options: we need the hyperparameters for A) the hyperparameter optimization step2, some
        of the hyperparameters will be given in the wrong format (float) and need to be converted
        into the correct format first (category) by using the hyperparameter_initialization.py function.
        B) In case of R hyperparameters its possible that they come as dictionary and need to be
        converted first into a string.
        C) In case of python hyperparameters its possible that they come as string and need to be
        converted into dictionary.
        :param params: hyperparameters: string for R functions, dict for Python functions
        :param use: for what is the function used?
        :return: params_r_str, params_py_dict
        '''
        # initialization
        params_r_str = ''
        params_py_dict = {}

        try:
            ### hyperparameter optimization
            # the params are the parameters provided by the bayesian optimization method
            if step == globalstring_step2:
                # convert float to categorical values
                params_r_str, params_py_dict = hyperparameter_init(params, self.fun_id,
                                                                   int(self.dim_low), self.data_high)

            ### dim reduction (not hyperparameter optimization)
            # r functions
            elif step != globalstring_step2 and self.params_fun == 'Rfun':
                if isinstance(params, dict):
                    # convert dict of hyperparameters to string
                    params_r_str = class_run_r_functions().dict_to_r_string(params)
                else:
                    params_r_str = params  # params is str

            # python functions
            else:
                if isinstance(params_py_dict, dict):
                    params_py_dict = params
                else:
                    # convert string to dict
                    params_py_dict = json.loads(params)
        except:
            logger.error(f"{globalstring_error}INIT HYPERPARAMETER{step}{self.fun_id} dim:{self.dim_low}", exc_info=True)

        return params_py_dict, params_r_str


    def exe_dimreduce(self, params: dict,  step: str) -> (np.array, dict):
        '''
        Single dim reduction with given fun_id, low_dim and params.
        This main function reduces the dimensionality of the data and evaluates the
        quality. It returns the loss of choice (mae_normalized) and keeps all
        the quality information (losses, time, Q-matrix).

        Parameters
        ----------
        params : dict parameters by customer of bayesian method
        use :   'dim_reduce' one reduction with given fun_id, low_dim and params
                'hyperparameters' for hyperparameter optimization.
                categorical parameters are provided as floats and need to be converted.
                see: hyperparameter_init

        Returns dictionary with results of dim reduction quality measurement
        -------

        '''
        # initialization
        hyperparameters = ''
        data_low = np.array
        start = time.time()

        # preprocess the functions hyperparamters
        params_py_dict, params_r_str = self.hyperparameters_r_py(params, step)

        try:
            # R functions
            if self.params_fun == 'Rfun':
                class_r_funs = class_run_r_functions()
                data_low, hyperparameters = class_r_funs.r_function_exe(self.fun_id,
                                                                        params_r_str,
                                                                        self.data_high,
                                                                        self.dim_low)
            # Python functions
            else:
                class_py_funs = class_run_py_functions(self.fun_id, self.data_high, self.dim_low, self.fun)
                data_low, hyperparameters = class_py_funs.exe_python_functions(params_py_dict)
        except:
            logger.error(f"{globalstring_error}DIMREDUCE{step}{self.fun_id} dim:{self.dim_low}", exc_info=True)

        # time for dim reduction
        stop = round(time.time() - start, 3)

        # Measure quality of dim reduction, helper_metrix class retrurns empty dict in case of problems
        kmax = fun_kmax(self.data_high)
        metrix = metrix_.class_metrix_dim_reduce(self.fun_id, self.data_high, data_low, kmax)
        dict_results = metrix.metrix_all()

        # append results to list and return
        # we make sure that something is returned
        try:
            dict_results['step'] = step
        except:
            logger.error(f"{globalstring_error}adding step to dict", exc_info=True)
            dict_results['fun_id'] = ' '

        try:
            dict_results['fun_id'] = self.fun_id
        except:
            logger.error(f"{globalstring_error}adding fun_id to dict", exc_info=True)
            dict_results['fun_id'] = 'empty'

        try:
            dict_results['time'] = stop
        except:
            logger.error(f"{globalstring_error}adding timing function to dict", exc_info=True)
            dict_results['time'] = 0

        try:
            dict_results['dim_low'] = self.dim_low
        except:
            logger.error(f"{globalstring_error}adding dim_low to dict", exc_info=True)
            dict_results['dim_low'] = 0

        try:
            dict_results['hyperparameters'] = json.dumps(hyperparameters) # dict to string
        except:
            logger.error(f"{globalstring_error}adding hyperparamers-string to dict", exc_info=True)
            dict_results['hyperparameters'] = json.dumps({'empty': 'empty'})

        return dict_results






# ## First part
# try:
#     # function is used for hyperparameter optimization
#     # the params are the parameters provided by the bayesian optimization method
#     if use == 'hyperparameters':
#         # convert float to categorical values
#         params_r_str, params_py_dict = hyperparameter_init(params, self.fun_id,
#                                                            int(self.dim_low), self.data_high)
#     #
#     # r
#     elif use == 'dim_reduce' and self.params_fun == 'Rfun':
#         if isinstance(params, dict):
#             params_r_str = class_run_r_functions().dict_to_r_string(params)
#         else:
#             params_r_str = params # params is str
#
#     # python
#     else:
#         if isinstance(params_py_dict, dict):
#             params_py_dict = params
#         else:
#             params_py_dict = json.loads(params)  # string to dict
# except:
#     print(globalstring_error + 'INIT HYPERPARAMETER', use, self.fun_id,
#           ' dim:', self.dim_low)
#     print(traceback.format_exc())




''' ERRORS that occur frequently during hyperparameter optimization

FUNCTIONS: Python: dictlearn_mb, lle, sammon
           R: asi, iltsa, ladakm, cisomap, crda, lapeig, spmds, keca, truncated svd, spe, 
           
           
ray.exceptions.RayTaskError(RuntimeError): ray::loop_dim_reduce() (pid=15680, ip=127.0.0.1)
RuntimeError: The remote function failed to import on the worker. This may be because needed library dependencies are not installed in the worker environment:

Kernel PCA
AttributeError: 'KernelPCA' object has no attribute 'lambdas_'

numpy.linalg.LinAlgError: Matrix is singular.
 raise LinAlgError('Matrix is singular.')
(optimization pid=14524) numpy.linalg.LinAlgError: Matrix is singular.

Truncated SVD
ValueError: k must be between 1 and min(A.shape), k=11

Truncated SVD
ValueError: n_components must be < n_features; got 11 >= 11

PCA
ValueError: n_components=11 must be strictly less than min(n_samples, n_features)=11 with svd_solver='arpack'


!! ERROR DIMREDUCE hyperparameters lle
Gi[:, 1:] = v[:, :n_components]
(optimization pid=92987) ValueError: could not broadcast input array from shape (5,5) into shape (5,11)

R-ERROR:  olpp "x"
(optimization pid=92527) "1" "Error in dt_pca(X, pcadim, FALSE): * do.pca : 'ndim' should be in [1,ncol(X)).


R-ERROR:  ppca, spe, ispe "x" -> ndim = ncol or string
(optimization pid=92527) "1" "Error in do.ppca(as.matrix(rdata), ndim = 11): *do.pca : 'ndim' is a positive integer in [1,#(covariates)).

 R-ERROR:  ispe "x"
(optimization pid=92527) "1" "Error in do.ispe(as.matrix(rdata), ndim = 11, C = 495, S = 497, cutoff = 9.23153365713723, : * do.ispe : 'ndim' is a positive integer in [1,#(covariates)).
(optimization pid=92527) "

 R-ERROR:  spe "x"
(optimization pid=92522) "1" "Error in do.spe(as.matrix(rdata), ndim = 11, C = 213, S = 344, drate = 0.204452249731517, : * do.spe : 'ndim' is a positive integer in [1,#(covariates)).
(optimization pid=92522) "


R-ERROR:  asi "x" n=3
(optimization pid=24303) "1" "Error in Uold - Unew: non-conformable arrays


 ## INFO: elpp2  init: 1  iterations: 0   n=3
(optimization pid=24303) R-ERROR:  asi "x"
(optimization pid=24303) "1" "Error in Uold - Unew: non-conformable arrays


R-ERROR:  ppca "x"   n=1
"1" "Error in (eigS$vectors[, 1:q]) %*% (diag(eigS$values[1:q] - mlsig2)): non-conformable arguments

R-ERROR:  ppca "x"
(optimization pid=92983) "1" "Error in do.ppca(as.matrix(rdata), ndim = 11): *do.pca : 'ndim' is a positive integer in [1,#(covariates)).

R-ERROR:  asi "x" 
(optimization pid=18037) "1" "Error in P[, i]: subscript out of bounds


R-ERROR:  crca "x", AND crda 'x'
"1" "Error in do.crca(as.matrix(rdata), ndim = 1, alpha = 4.2, lambda = 7.23121248507737, : * do.crca : 'ndim' is a positive integer in (1,#(covariates)).
"


R-ERROR:  spmds "x"
(optimization pid=15353) "1" "Error in solve.default(LHS, RHS): system is computationally singular: reciprocal condition number = 1.40073e-16



R-ERROR:  crda "x"
(optimization pid=15344) "1" "Error in aux.shortestpath(wD): * aux.shortestpath : input 'dist' should be either (n*n) matrix or 'dist' class object.
(optimization pid=15344) "



R-ERROR:  keca "x"
(optimization pid=15351) "1" "Error in eigen(K): infinite or missing values in 'x'



R-ERROR:  cisomap "x"
(optimization pid=15352) "1" "Error in aux.shortestpath(wDconformal): * aux.shortestpath : input 'dist' should be either (n*n) matrix or 'dist' class object.



R-ERROR:  iltsa "x"
(optimization pid=15353) "1" "Error in eigs_real_sym(A, nrow(A), k, which, sigma, opts, mattype = \"sym_matrix\", : 'k' must satisfy 0 < k < nrow(A)
(optimization pid=15353) "


lapeig "x"
(optimization pid=25566) "1" "Error in do.lapeig(as.matrix(rdata), ndim = 17, kernelscale = 0, symmetric = \"asymmetric\", : * do.lapeig : 'kernelscale' is a positive real value.


R-ERROR:  lapeig "x"
(optimization pid=25566) "1" "Error in do.lapeig(as.matrix(rdata), ndim = 17, kernelscale = 0, symmetric = \"asymmetric\", : * do.lapeig : 'kernelscale' is a positive real value.


R-ERROR:  ldakm "x"
(optimization pid=15345) "1" "Error in kmeans(projected, k): more cluster centers than distinct data points.



!! ERROR DIMREDUCE hyperparameters dictlearn_mb  dim: 9
line 512, in orthogonal_mp_gram
(optimization pid=18036)     raise ValueError("The number of atoms cannot be more than the number "
(optimization pid=18036) ValueError: The number of atoms cannot be more than the number of features



 R-ERROR:  sammon "x"
(optimization pid=92984) "1" "Error in MASS::sammon(dX, y = initY, k = ndim, trace = FALSE): invalid initial configuration



!! ERROR DIMREDUCE hyperparameters lle  dim: 9
RuntimeError: Factor is exactly singular


!! ERROR DIMREDUCE hyperparameters lle  dim: 9
ValueError: Error in determining null-space with ARPACK. Error message: 'Factor is exactly singular'. Note that method='arpack' can fail when the weight matrix is singular or otherwise ill-behaved.  method='dense' is recommended. See online documentation for more information.
(optimization pid=18038)


!! ERROR DIMREDUCE hyperparameters lle  dim: 23
Gi[:, 1:] = v[:, :n_components]
(optimization pid=20586) ValueError: could not broadcast input array from shape (5,5) into shape (5,23)


!! ERROR DIMREDUCE hyperparameters dictlearn_mb  dim: 9
line 512, in orthogonal_mp_gram
(optimization pid=18036)     raise ValueError("The number of atoms cannot be more than the number "
(optimization pid=18036) ValueError: The number of atoms cannot be more than the number of features

->x if tol is None and n_nonzero_coefs > X.shape[1]:
solution: update scilearn to 1.1.2 and add tol parameter with exception!


FileNotFoundError: [Errno 2] No such file or directory: '/Users/kay/Desktop/W7X/project/results/r_errors/rerror.txt'
'''