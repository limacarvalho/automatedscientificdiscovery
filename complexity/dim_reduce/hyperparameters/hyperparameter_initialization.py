from helper_data import global_vars
import numpy as np


def hyperparameter_init(params: dict, fun_id: str, dim: int, data_high: np.array) -> (str, dict):
    '''
    The bayes_opt package processes and returns continuous values, not categorical values.
    Our around is to convert the categorical input and numeric output into integers or categories
    that are used in some hyperparaeters.
    Example:  'eigen_solver': ['arpack','lobpcg','amg'] as [0, 2] and convert the bayes_opt
    output (1.66 for example) into: ['arpack','lobpcg','amg'][2] = 'amg'.
    Thats not optimal but the bayes_opt should learn from that.
    :param params: dict, dict with hyperparameters and values as floats
    :param fun_id: str, function identifier
    :param dim: int, low dimension
    :param data_high: np.array, high dimensional data
    :return: (str, dict), updated hyperparameters as string (R) and dicionary (Python)
    '''
    arg_string = ''

    for key, value in params.items():

        ########
        if key == 'eigen_solver':
            if fun_id == 'py_spectral_embedding':
                params[key] = ['arpack','lobpcg','amg'][int(round(value,0))]
            else:
                # isomap, lle: 'arpack', 'dense', default='auto' 0...1
                # 'randomized': kernel pca 0...2
                params[key] = ['arpack','dense','randomized'][int(round(value,0))]

        ######## PYTHON  dictionary_learning_mini_batch
        if key == 'fit_algorithm':
            params[key] = ['cd','lars'][int(round(value,0))]

        ######## PYTHON     dictionary_learning_mini_batch
        if key == 'transform_n_nonzero_coefs':
            params[key] = int(value)

        ######## PYTHON     dictionary_learning_mini_batch
        if key == 'transform_algorithm':
            params[key] = ['lars','lasso_cd','threshold'][int(round(value,0))]
            # ['omp','lars','lasso_lars','lasso_cd','threshold']
        ######## PYTHON     fastica
        if key == 'fun':
            params[key] = ['logcosh','exp','cube'][int(round(value,0))]

        ######## PYTHON     sparse pca
        if key == 'max_iter':
            params[key] = int(value)

        ######## PYTHON     mb sparse pca
        if key == 'n_iter':
            params[key] = int(value)

        ######## PYTHON     mb sparse pca
        if key == 'mc_iter':
            params[key] = int(value)

        ######## PYTHON     isomap
        if key == 'path_method':
            params[key] = ['FW','D'][int(round(value,0))]

        ######## PYTHON    isomap
        if key == 'neighbors_algorithm':
            params[key] = ['brute','kd_tree','ball_tree'][int(round(value,0))]

        ######## PYTHON    tsne
        # default: 'minkowski' there are dozens of options, I just choose the two most common ones
        # also determined y 'p' in isomap
        if key == 'metric':
            params[key] = ['minkowski','manhattan','euclidean'][int(round(value,0))]

        ######## PYTHON    isomap
        if key == 'p':
            params[key] = int(value)

        ######## PYTHON    factor analysis,
        if key == 'svd_method':
            params[key] = ['lapack','randomized'][int(round(value,0))]

        ########
        if key == 'algorithm':
            # truncated svd
            if fun_id == 'py_truncated_svd':
                params[key] = ['arpack', 'randomized'][int(round(value,0))]
            # fastICA
            if fun_id == 'py_fastica':
                params[key] = ['parallel','deflation'][int(round(value,0))]

        ######## PYTHON     isomap, lle, spectral_embedding, isomap, umap
        # important that it is here ate the beginning of the list, key== method can change it
        if key == 'n_neighbors':
            params[key] = int(value)
            ### LLE
            if fun_id == 'py_lle':
                # general condition: n_neighbors < n_samples
                if params[key] >= data_high.shape[0]:
                    params[key] = int(data_high.shape[0]-1)

                # n_neighbors of following condition is too high for many datasets
                if params['method'] == 'hessian':
                    # condition hessian: n_neighbors > n_components * (n_components + 3) / 2
                    cond = dim * (dim + 3) / 2
                    if params[key] <= dim + cond:
                        params[key] = int(dim + cond + 1)

                if params['method'] == 'modified':
                    # condition modified: n_neighbors >= n_components
                    params[key] = int(max(dim, params['n_neighbors']))


        ######## PYTHON    LLE
        if key == 'method' and fun_id == 'py_lle':
            # 'hessian' is not practical, required n-neighbors exceeds n-rows
            params[key] = ['standard','modified','ltsa'][int(round(value,0))]


        ######## PYTHON    SPARSE PCA
        if key == 'method' and (fun_id == 'py_pca_sparse' or fun_id == 'py_pca_sparse_mb'):
            params[key] = ['lars','cd'][int(round(value,0))]

        ######## PYTHON    NNF
        if key == 'init':
            if fun_id == 'py_nmf':
                params[key] = ['random', 'nndsvd', 'nndsvdar'][int(round(value,0))]
            # TSNE
            else:
                params[key] = ['random','pca'][int(round(value,0))]

        ######## PYTHON    NNF
        if key == 'solver':
            params[key] = ['cd', 'mu'][int(round(value,0))]
            if params['solver'] == 'cd':
                params['beta_loss'] = 'frobenius'

        ######## PYTHON    NNF
        if key == 'beta_loss':
            if params['solver'] == 'cd':
                params[key] = 'frobenius'
            else:
                params[key] = ['frobenius', 'kullback-leibler', 'itakura-saito'][int(round(value,0))]

        ######## PYTHON    NNF
        if key == 'l1_ratio':
            params[key] = round(value,2)

        ######## PYTHON    factor analysis, PCA
        if key == 'iterated_power':
            params[key] = int(round(value,0))

        ######## PYTHON    PCA
        if key == 'svd_solver':
            # only for arpack
            params[key] = ['full','arpack','randomized'][int(round(value,0))]
            if fun_id == 'py_pca' and params[key] == 'arpack':
                if dim >= min(data_high.shape):
                    pass

        ######## PYTHON    KERNEL-PCA
        if key == 'kernel' and fun_id == 'py_pca_kernel':
            params[key] = ['linear','rbf','sigmoid','cosine'][int(round(value,0))] # 'poly'

        ######## PYTHON    KERNEL-PCA
        if key == 'fit_inverse_transform':
            params[key] = [False, True][int(round(value,0))]

        ######## PYTHON    SPECTRAL EMBEDDING
        if key == 'affinity':
            params[key] = ['nearest_neighbors','rbf'][int(round(value,0))]

        ######## PYTHON    TSNE
        if key == 'perplexity':
            params[key] = int(round(value,0))

        ######## PYTHON    TSNE
        if key == 'n_iter_without_progress':
            params[key] = [50,100,150,200,250,300,350,400,450,500][int(round(value,0))]


        ######## PYTHON    MANY PYTHON FUNCTIONS
        if key == 'tol' and not fun_id == 'py_mve':
            params[key] = value

        # incremental_pca
        if key == 'empty_py':
            # set a dump parameter
            params['whiten'] = False
            # remove empty key
            try:
                params.pop('empty_py')
            except:
                print(global_vars.globalstring_error + 'no such key: empty_py')
                pass



# R-functions ########   ########   ########   ########   ########   ########

        ######## R  adr, asi, ldakm, nonpp, mve, ree, rpca, sne
        if key == 'maxiter':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  rpcag, llp, sdlpp
        if key == 't':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  double functions: crca
        if key == 'alpha':
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  double functions: llp, crca, rpca
        if key == 'lambda':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  double functions: crca
        if key == 'tolerance':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  rndproj
        if key == 's':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  rpcag
        if key == 'k':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  extlpp
        if key == 'numk':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  lmds, lisomap
        if key == 'npoints':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  idmap
        if key == 'engine':
            val = str(['"NNP"','"FastMap"'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  adr, asi, ldakm, ree, rpca
        if key == 'abstol':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  spp, nonpp
        if key == 'reltol':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  lpp, iltsa, spmds "intersect","union","asymmetric"
        # olpp: "intersect","union"
        # tsne: TRUE, FALSE
        if key == 'symmetric':
            if fun_id == 'r_tsne':
                val = str(['FALSE','TRUE'][int(round(value,0))])
            else:
                val = str(['"intersect"','"union"','"asymmetric"'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  lpp, nonpp, olpp, udp, iltsa, rndproj, mvu, spmds
        # important: no spaces in c(...)
        if key == 'type':
            if fun_id == 'r_rndproj':
                vals = ['"gaussian"','"achlioptas"','"sparse"']

            elif fun_id == 'r_olpp' or fun_id ==  'r_spmds':
                vals = ['c("knn",5)', 'c("proportion",0.3)']

            elif fun_id == 'r_udp':
                vals = ['c("knn",5)', 'c("enn",1)', 'c("proportion",0.1)']

            else:
                vals = [
                       'c("knn",5)', 'c("knn",50)',
                       'c("enn",1)', 'c("enn",3)', # nonpp
                       'c("proportion",0.1)', 'c("proportion",0.3)']
            val = str(vals[int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  bmds
        if key == 'par.a':
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  bmds
        if key == 'par.alpha':
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  bmds
        if key == 'par.step':
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  bmds
        if key == 'par.iter':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  lisomap
        if key == 'ltype':
            val = str(['"random"','"MaxMin"'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  cisomap, lisomap
        if key == 'weight':
            val = str(['FALSE','TRUE'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  dm
        if key == 'multiscale':
            val = str(['FALSE','TRUE'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  dm
        if key == 'timescale':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  ispe
        if key == 'proximity':
            # --- instead of = because its a split identifier in argstring_R_to_dict
            # 'function(x){dist(x,method="manhattan")}' not working well for most datasets
            val = str(['function(x){dist(x,method="euclidean")}',
                       'function(x){dist(x,method="minkowski", p=3)}',
                       'function(x){dist(x,method="manhattan")}' ][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  ispe
        if key == 'C':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  ispe
        if key == 'S':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  ispe
        if key == 'drate':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  ispe
        if key == 'cutoff':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  keca
        # there are 20 types + parameters**,
        # more kernels: https://kisungyou.com/Rdimtools/reference/aux_kernelcov.html
        # how can we make bayesian search effective here? 2 searches?
        # here I choose the 2 default options and the 3 w/o parameters
        if key == 'kernel' and fun_id == 'r_keca':
            val = str(['c("gaussian",5)',
                       'c("laplacian",1)',
                       'c("chisq")',
                       'c("spline")'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  lapeig
        if key == 'weighted':
            val = str(['FALSE','TRUE'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  lapeig
        if key == 'kernelscale':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  mve kwidth
        if key == 'knn':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  mve
        if key == 'kwidth':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  mve
        if key == 'tol' and fun_id == 'mve':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  mvu
        if key == 'projtype':
            val = str(['"spectral"','"kpca"'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  ree
        if key == 'dmethod':
            val = str(['"euclidean"',
                       '"maximum"',
                       '"manhattan"',
                       '"canberra"',
                       '"binary"',
                       '"minkowski"'][int(round(value,0))])
            arg_string = arg_string + str(key) + '=' + val + ', '

        ######## R  ree TODO: method not choosen, remove
        if key == 'initc':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  rpca
        if key == 'mu':
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  sammon
        if key == 'initialize':
            val = str(['"random"', '"pca"'][int(round(value, 0))])
            arg_string = arg_string + str(key) + '=' + val + ', '
            # arg_string = arg_string + 'initialize="random", ' TODO: remove

        ######## R  spmds
        if key == 'neigs':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  spmds
        if key == 'ratio':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  rpca
        if key == 'empty':
            arg_string = 'empty=empty'

        # # tsne, not used, too many hyperparameters and low quality
        # if key == 'perplexity':
        #     arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '
        #
        # # tsne
        # if key == 'eta':
        #     arg_string = arg_string + str(key) + '=' + str(value) + ', '
        #
        # # tsne
        # if key == 'jitter':
        #     arg_string = arg_string + str(key) + '=' + str(value) + ', '
        #
        # # tsne
        # if key == 'jitterdecay':
        #     arg_string = arg_string + str(key) + '=' + str(value) + ', '
        #
        # # tsne
        # if key == 'momentum':
        #     arg_string = arg_string + str(key) + '=' + str(value) + ', '
        #
        # # tsne
        # if key == 'pca':
        #     val = str(['FALSE','TRUE'][int(round(value,0))])
        #     arg_string = arg_string + str(key) + '=' + val + ', '
        #
        # # tsne
        # if key == 'pcascale':
        #     val = str(['FALSE','TRUE'][int(round(value,0))])
        #     arg_string = arg_string + str(key) + '=' + val + ', '
        #
        # # tsne
        # if key == 'BHuse':
        #     val = str(['FALSE','TRUE'][int(round(value,0))])
        #     arg_string = arg_string + str(key) + '=' + val + ', '
        #
        # # tsne
        # if key == 'BHtheta':
        #     val = str(['FALSE','TRUE'][int(round(value,0))])
        #     arg_string = arg_string + str(key) + '=' + val + ', '

    if arg_string.endswith(', '):
        arg_string = arg_string[:-2]

    # R: arg_string  Python: params
    return arg_string, params

# TODO: remove
# def dict_to_r_string(dict_params):
#     '''
#     converts a dictionary to a string
#     :param dict_params:
#     :return:
#     '''
#     arg_string = ''
#
#     for key, value in dict_params.items():
#         if key == 'empty':
#             return arg_string
#         else:
#             arg_string = arg_string + str(key) + '=' + str(value) + ', '
#
#     if arg_string.endswith(', '):
#         arg_string = arg_string[:-2]
#
#     return arg_string