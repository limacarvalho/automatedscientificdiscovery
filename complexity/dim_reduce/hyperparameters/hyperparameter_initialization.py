from ..helper_data import global_vars
import numpy as np

def hyperparameter_init(params: dict, fun_id: str, dim: int, data: np.array) -> (str, dict):
    '''
    The bayes_opt package processes and returns only continuous values, not categorical values.
    My work around is to convert the bayes_opt numeric output into integers or categories
    that are used in some hyperparaeters.
    Example:  'eigen_solver': ['arpack','lobpcg','amg'] as [0, 2] and convert the bayes_opt
    output (1.66 for example) into: ['arpack','lobpcg','amg'][2] = 'amg'.
    Thats not optimal but the bayes_opt should learn from that.
    :param params: dict with hyperparameters and values as floats
    :return: updated hyperparameters as string (R) and dicionary (Python)
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
                if params[key] >= data.shape[0]:
                    params[key] = int(data.shape[0]-1)

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
                params[key] = ['random', 'nndsvd', 'nndsvda', 'nndsvdar'][int(round(value,0))]
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
                if dim >= min(data.shape):
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
        if key == 'alpha' and (fun_id == 'r_crca' or fun_id == 'r_phate'):
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  double functions: llp, crca, rpca
        if key == 'lambda':
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
            else:
                vals = ['c("knn",5)', 'c("knn",50)',
                       'c("enn",1)', 'c("enn",3)',
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
            val = str(['function(x){dist(x,method="euclidean")}',
                       'function(x){dist(x,method="manhattan")}',
                       'function(x){dist(x,method="minkowski")}'][int(round(value,0))])
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
                       #'c("histintx")', only positive values
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

        ######## R  ree
        if key == 'initc':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  rpca
        if key == 'mu':
            arg_string = arg_string + str(key) + '=' + str(round(value,1)) + ', '

        ######## R  sammon
        if key == 'initialize':
            arg_string = arg_string + 'initialize="random", '

        ######## R  spmds
        if key == 'neigs':
            arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '

        ######## R  spmds
        if key == 'ratio':
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

        ######## R  rpca
        if key == 'empty':
            arg_string = 'empty=empty'

        # # tsne
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


def dict_to_r_string(dict_params):
    '''

    Parameters
    ----------
    dict_params :

    Returns
    -------

    '''
    arg_string = ''

    for key, value in dict_params.items():
        if key == 'empty':
            return arg_string
        else:
            arg_string = arg_string + str(key) + '=' + str(value) + ', '

    if arg_string.endswith(', '):
        arg_string = arg_string[:-2]

    return arg_string


















'''
kernel**
    linear
    c("linear",c)

    polynomial
    c("polynomial",c,d)

    gaussian
    c("gaussian",c)

    laplacian
    c("laplacian",c)

    anova
    c("anova",c,d)

    sigmoid
    c("sigmoid",a,b)

    rational quadratic
    c("rq",c)

    multiquadric
    c("mq",c)

    inverse quadric
    c("iq",c)

    inverse multiquadric
    c("imq",c)

    circular
    c("circular",c)

    spherical
    c("spherical",c)

    power/triangular
    c("power",d)

    log
    c("log",d)

    spline
    c("spline")

    Cauchy
    c("cauchy",c)

    Chi-squared
    c("chisq")

    histogram intersection
    c("histintx")

    generalized histogram intersection
    c("ghistintx",c,d)

    generalized Student-t
    c("t",d)

eigen_solver{'auto', 'arpack', 'dense'}, default='auto'    -> finds eigenvalues/vectors
    'auto' : Attempt to choose the most efficient solver for the given problem.
    'arpack' : Use Arnoldi decomposition to find the eigenvalues and eigenvectors. (speed accuracy by tol and max_iter)
    'dense' : Use a direct solver (i.e. LAPACK) for the eigenvalue decomposition.
    - ARPACK 1 is a Fortran package which provides routines for quickly finding a few eigenvalues/eigenvectors 
      of large sparse matrices. 
    https://docs.scipy.org/doc/scipy/tutorial/arpack.html

###############   
neighbors_algorithm{'auto', 'brute', 'kd_tree', 'ball_tree'}, default='auto'
    Algorithm to use for nearest neighbors search, passed to neighbors. NearestNeighbors instance.

###############
n_jobs int or None, default=None
    The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. 
    -1 means using all processors.

###############
metric str, or callable, default=”minkowski”
    The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, 
    it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. 
    If metric is “precomputed”, X is assumed to be a distance matrix and must be square.

###############           
p int, default=2
    Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. When p = 1, 
    this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. 
    For arbitrary p, minkowski_distance (l_p) is used.

###############
path_method{'auto', 'FW', 'D'}, default='auto'
    Method to use in finding shortest path between all vertices in the weighted graph.
    'auto' : attempt to choose the best algorithm automatically.
    'FW' : Floyd-Warshall algorithm.
    'D' : Dijkstra's algorithm.    
     shortest path between all vertices in the weighted graph

    https://pythonwife.com/all-pair-shortest-path-problem-in-python/
    Dijkstra's Algorithm is one example of a single-source shortest or SSSP algorithm, i.e., given a source 
        vertex it finds shortest path from source to all other vertices.
    Floyd Warshall Algorithm is an example of all-pairs shortest path algorithm, meaning it computes the 
        shortest path between all pair of nodes.

    Time Complexity of Dijkstra's Algorithm: O(E log n)
    Time Complexity of Floyd Warshall: O(n^3) - faster

    We can use Dijskstra's shortest path algorithm for finding all pair shortest paths by running it for every vertex. 
    But time complexity of this would be O(mn Log n) which can go (n^3 Log n) in worst case.
    Another important differentiating factor between the algorithms is their working towards distributed systems. 
    Unlike Dijkstra's algorithm, Floyd Warshall can be implemented in a distributed system, making it suitable for data 
    structures such as Graph of Graphs (Used in Maps).
    Lastly Floyd Warshall works for negative edge but no negative cycle, whereas Dijkstra's algorithm don't work for 
    negative edges.   

###############
tolerance (Eigenvalue, Eigenvector)
    ARPACK is a Fortran package which provides routines for quickly finding a few eigenvalues/eigenvectors of large sparse matrices.
    Residual tolerances of the computed eigenvalues. 
    less restrictive Tolerance e-2, e-6 more restrictive: e-15, 
    https://docs.scipy.org/doc/scipy/tutorial/arpack.html



R-FUNCTIONS
###############
maxiter
    maximum number of iterations (default: 100).
    -I tested 100 and 1000 with Covid (190rows), 1000 is much slower, same result.

###############
abstol
     absolute tolerance stopping criterion (default: 1e-8).
     -I tested -8 and -12 with Covid (190rows), -12 a bit faster, same result.

'''