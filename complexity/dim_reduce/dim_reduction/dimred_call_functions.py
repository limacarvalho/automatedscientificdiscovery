from .R_funs_dimred.R_funs_dimred import Dimred_functions_R
from .Py_funs_dimred.Py_funs_dimred import Dimred_functions_python
import pandas as pd
import numpy as np
from ..helper_data import global_vars
from typing import Union

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import warnings
warnings.filterwarnings('ignore')


def exclude_functions(functions: Union[str, list]) -> bool:
    '''
    checks if functions need to be excluded or not.
    :param functions: list functions provided by the customer.
    :return: bool is True in case functions starts with !
        is False in case it doesnt start with
    '''
    excludes = False
    for fun_id in functions:
        if fun_id.startswith('!'):
            excludes = True
    return excludes



def call_dimred_functions(
        functions: Union[str, list],
        data_high: np.array
    ) -> dict:
    '''
    function caller with function_identifier: function call (returns function,
    default parameters, hyperparameters (hyperparameter: range*))
    *categorical hyperparametes are presented as integer, we are using bayes-opt
    which returns floats, which need to be translated into categories.
    some of the functions need data information (shape, distributions etc)
    :param np.array data_high: high dimensional data
    :param Union[str, list] functions: list of strings with function identifiers:
        'py_pca' or just string
    :return: dict dictionary with function identifiers and function calls
    '''

    nrows = data_high.shape[0]
    ncols = data_high.shape[1]

    # call python function class
    python = Dimred_functions_python(nrows=nrows, ncols=ncols)

    # call R function class
    rdim = Dimred_functions_R(nrows=nrows, ncols=ncols)

    funcs = {
        'py_pca': python.pca(),  # keep always on! 1# 'py_pca_incremental': python.pca_incremental(), # !
        'py_pca_sparse': python.pca_sparse(),
        'py_pca_sparse_mb': python.pca_sparse_mini_batch(),
        'py_dictlearn_mb': python.dictionary_learning_mini_batch(),
        'py_fa': python.factor_analysis(),
        'py_fastica':  python.fast_ica(),
        'py_nmf': python.non_negative_matrix_factorization(),
        'py_spectral_embedding': python.spectral_embedding(),
        'py_truncated_svd': python.truncated_svd(), #1
        'r_adr': rdim.funR_adr(),
        'r_asi': rdim.funR_asi(),
        'r_elpp2': rdim.funR_elpp2(), #1
        'r_extlpp': rdim.funR_extlpp(),
        'r_ldakm': rdim.funR_ldakm(),
        'r_lmds': rdim.funR_lmds(),
        'r_lpp': rdim.funR_lpp(),
        'r_mds':rdim.funR_mds(),
        'r_npca': rdim.funR_npca(),
        'r_olpp':rdim.funR_olpp(),
        'r_pflpp':rdim.funR_pflpp(),
        'r_ppca':rdim.funR_ppca(),
        'r_rndproj':rdim.funR_rndproj(),
        'r_rpcag':rdim.funR_rpcag(),
        'r_sdlpp':rdim.funR_sdlpp(),
        'r_udp':rdim.funR_udp(),
        'r_cisomap':rdim.funR_cisomap(),
        'r_crca':rdim.funR_crca(),
        'r_crda':rdim.funR_crda(),
        'r_fastmap':rdim.funR_fastmap(),
        'r_idmap':rdim.funR_idmap(),
        'r_ispe':rdim.funR_ispe(),
        'r_keca':rdim.funR_keca(),
        'r_lamp':rdim.funR_lamp(),
        'r_lapeig':rdim.funR_lapeig(),
        'r_nnp': rdim.funR_nnp(),
        'r_sammon': rdim.funR_sammon(), # initialize= random properly called
        'r_spe':rdim.funR_spe(),
        'r_spmds':rdim.funR_spmds()

        # ## 'py_tsne': python.tsne(), # takes too long, no good results
        # ## 'pca_kernel': python.pca_kernel(), # many errors, no good results
        # ## 'py_isomap':  python.isomap(), # good results, slow
        # ## 'py_lle': python.locally_linear_embedding(), # many errors, no good results

    }

    # in case a functions string is provided instead of a list, make it list
    if isinstance(functions, str):
        functions = [functions]

    # message strings
    string_except = 'coosing dimred functions failed, please check the correct spelling.' \
                    'Please also check the instructions'

    # calls all functions of the above list, uncheck them if not needed
    if functions[0] == 'all_functions':
        try:
            # print(string_try+'all functions: ', funcs.keys())
            return funcs
        except:
            print(global_vars.globalstring_error, 'error in choosing functions: all functions')

    else:
        # exclude custom functions: exlude the functions from the funcs dictionary if at least one
        # function needs to start with: '!'
        exclude = exclude_functions(functions)
        if exclude:
            try:
                dict_funs = funcs
                for fun_id in functions:
                    # remove the leading '!' and delete fnction from 'all_functions' dictionary
                    fun_to_remove = fun_id.replace('!', '')
                    del dict_funs[fun_to_remove]
                # print(string_try+'exclude custom functions', dict_funs.keys())
                return dict_funs
            except:
                print(global_vars.globalstring_error, string_except)

        # include custom functions: 'py_pca' for example
        # customers need to choose
        else:
            try:
                # make list in case its a string
                if isinstance(functions, str):
                    functions = [functions]
                # loop through the list with function names
                dict_funs = {}
                for fun_id in functions:
                    for key, value in funcs.items():
                        if key == fun_id:
                            dict_funs[key] = value
                # print(string_try+'custom functions', dict_funs.keys())
                return dict_funs
            except:
                print(global_vars.globalstring_error, string_except)







## NOT IMPLEMENTED
## 'smacov': python.smacof(n=ndim), # no fit function, dificult to implement
## 'umap': python.umap(n=ndim), # not installable on M1
## 'neural_network': nn.custom_neural_network(data, dim_out=ndim, epochs=500).neural_network_dim_reduce(),


# OTHERS, NOT USED
## 'latent_da': python.latent_ditrichlet_allocation(n=ndim), # Textverarbeitung Topics
## 'lisomap':rdim.funR_lisomap(), # only errors with covid
## 'umap' python, error implementation
## 'iltsa': rdim.funR_iltsa(),  # too many errors with cause propably ndim

# TEXTVERARBEITUNG
# 'latent_da', 'lda'
# LABEL REQUIERED
# if function == 'lda':  # check function for hyperparameters
#     fun = disc.linear_discriminant_analysis(n=ndim)
#
# if function == 'qda':  # no hyperparameters
#     fun = disc.quadratic_discriminant_analysis(n=ndim)

# SLOW PYTHON
# 'dictlearn': python.dictionary_learning(n=ndim)


# SLOW R
# 'cnpe', # slow
# 'spca', # slow
# 'dictlearn', # slow
# 'isoproj', # slow
# 'kudp', # slow
# 'lltsa', # slow
# 'spp', # not working covid, spectra
# 'nonpp', # not working covid, spectra
# 'bmds', # better than pca but 3000 times slower
# 'cisomap', # slow
# 'dm', # slow
# 'dve', # slow
# 'iltsa', # slow
# 'keca', # slow
# 'kpca', # slow
# 'mve', # slow
# 'mvu', # slow
# 'phate', # slow
# 'ree', # slow
# 'sne', # slow
# 'spp': rdim.funR_spp(), # very slow! 50sec (pca: 0.2), inacurate: covid: 0.772 (pca: 0.991)
# 'bmds':rdim.funR_bmds(), # covid: very slow 1400sec