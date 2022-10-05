from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning, dict_learning_online
from sklearn.decomposition import LatentDirichletAllocation, NMF, FastICA, FactorAnalysis, TruncatedSVD
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, KernelPCA, MiniBatchSparsePCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, smacof, SpectralEmbedding, MDS, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np
# from umap import UMAP # rosetta error



class Dimred_functions_python:
    '''
    dimensionality reduction functions
    '''

    def __init__(self, nrows, ncols):
        self.nrows = nrows  # data.shape[0]
        self.ncols = ncols  # data.shape[1]


    def dictionary_learning_mini_batch(self) -> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        Dictionary learning is the method of learning a matrix, called a dictionary, such that we can write
        a signal as a linear combination of as few columns from the matrix as possible.
        Finds a dictionary (a set of atoms) that performs well at sparsely encoding the fitted data.
        Solves the optimization problem:
        (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components
        ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm which
        is the sum of the absolute values of all the entries in the matrix.

        All variations of dictionary learning implement the following
        transform methods, controllable via the transform_method initialization parameter:
        -Orthogonal matching pursuit (Orthogonal Matching Pursuit (OMP))
        -Least-angle regression (Least Angle Regression)
        -Lasso computed by least-angle regression
        -Lasso using coordinate descent (Lasso)
        -Thresholding
        use: image denoising, image and timeseries decomposition

        MiniBatchDictionaryLearning implements a faster, but less accurate version of the dictionary learning
        algorithm that is better suited for large datasets.

        - - - PARAMETERS - - -
        # XXX: these should be optimized, as they can be a bottleneck.
        alpha float, default=1.0
            Sparsity controlling parameter.

        max_iter int, default=1000
            Maximum number of iterations to perform.

        tol float, default=1e-8
            Tolerance for numerical error.

        fit_algorithm{‘lars’, ‘cd’}, default=’lars’
            'lars': uses the least angle regression method to solve the lasso problem (lars_path);
            'cd': uses the coordinate descent method to compute the Lasso solution (Lasso). Lars will be faster
                 if the estimated components are sparse.

        # XXX: these should be optimized, as they can be a bottleneck.
        transform_algorithm{‘lasso_lars’, ‘lasso_cd’, ‘lars’, ‘omp’, ‘threshold’}, default=’omp’
            Algorithm used to transform the data:
            'lars': uses the least angle regression method (lars_path);
            'lasso_lars': uses Lars to compute the Lasso solution.
            'lasso_cd': uses the coordinate descent method to compute the Lasso solution (Lasso). 'lasso_lars' will
                        be faster if the estimated components are sparse.
            'omp': uses orthogonal matching pursuit to estimate the sparse solution.
            'threshold': squashes to zero all coefficients less than alpha from the projection dictionary * X'.

        transform_n_nonzero_coefs int, default=None
            Number of nonzero coefficients to target in each column of the solution. This is only used by algorithm='lars'
            and algorithm='omp'. If None, then transform_n_nonzero_coefs=int(n_features / 10).

        transform_alpha float, default=None
            If algorithm='lasso_lars' or algorithm='lasso_cd', alpha is the penalty applied to the L1 norm.
            If algorithm='threshold', alpha is the absolute value of the threshold below which coefficients will be squashed to zero.
            If None, defaults to alpha.
        ...
        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
        '''
        fun = MiniBatchDictionaryLearning(
            n_components=None,
            alpha=0.1,
            n_iter=1000,
            fit_algorithm='lars',
            n_jobs=None,
            batch_size=3,
            shuffle=True,
            dict_init=None,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=None,
            transform_alpha=None,
            verbose=False,
            split_sign=False,
            random_state=42,
            positive_code=False,
            positive_dict=False,
            transform_max_iter=1000
        )
        # Hyperparameters
        hyperpars = {
             'alpha': [0.1, 10], # XXX controls sparsity
             'n_iter': [5,1000], #
             'fit_algorithm': [0, 1], # {‘lars’, ‘cd’}, default=’lars’
             'transform_algorithm': [0, 2], # XXX 'lars','lasso_cd','threshold' / others: omp, lasso-lars
             'transform_n_nonzero_coefs': [2, int(self.ncols/2) ], # int! only for 'lars' and 'omp', default None == int(n_features / 10)
             'transform_alpha': [0.005, 6], #
             }
        params = fun.get_params()
        return fun, params, hyperpars


    def factor_analysis(self)-> (object, dict, dict):
        '''
         - - - DESCRIPTION - - -
        FactorAnalysis
            is a statistical method used to describe variability among observed, correlated variables
            FactorAnalysis performs a maximum likelihood estimate of the so-called loading matrix, the transformation of the
            latent variables to the observed ones, using SVD based approach.
            This allows better model selection than probabilistic PCA in the presence of heteroscedastic noise
            USE: model analisis, image decomposition, psychometrics, marketing

        - - - PARAMETERS - - -
        tol float, default=1e-2
            Stopping tolerance for log-likelihood increase.

        copy bool, default=True
            Whether to make a copy of X. If False, the input X gets overwritten during fitting.

        max_iter int, default=1000
            Maximum number of iterations.

        noise_variance_init ndarray of shape (n_features,), default=None
            The initial guess of the noise variance for each feature. If None, it defaults to np.ones(n_features).

        # XXX: these should be optimized, as they can be a bottleneck.
        svd_method {‘lapack’, ‘randomized’}, default=’randomized’
            Which SVD method to use. If ‘lapack’ use standard SVD from scipy.linalg, if ‘randomized’ use fast
            randomized_svd function. Defaults to ‘randomized’. For most applications ‘randomized’ will be
            sufficiently precise while providing significant speed gains. Accuracy can also be improved by
            setting higher values for iterated_power. If this is not sufficient, for maximum precision you
            should choose ‘lapack’.

        # XXX: these should be optimized, as they can be a bottleneck.
        iterated_power int, default=3
            Number of iterations for the power method. 3 by default. Only used if svd_method equals ‘randomized’.

        rotation{‘varimax’, ‘quartimax’}, default=None
            If not None, apply the indicated rotation. Currently, varimax and quartimax are implemented.
            See “The varimax criterion for analytic rotation in factor analysis” H. F. Kaiser, 1958.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
        '''
        fun = FactorAnalysis(
             n_components=None,
             tol=0.05, # default: 0.01
             copy=True,
             max_iter=100, # default=1000,
             noise_variance_init=None, # custom noise variances for each feature.
             svd_method='randomized',  # {‘lapack’, ‘randomized’}
             iterated_power=0, # default =3
             random_state=42
        )

        hyperpars = {
            'empty_py': [0]
             # 'max_iter': [5,1000], # maximum iterations
             # 'tol': [0.001, 0.1], # tolerance for expectation–maximization algorithm (log-likelihood)
             # 'svd_method': [0,1], # single value dcomposition, more accurate, slow: ‘lapack’, speed: ‘randomized’ (default)
             # 'iterated_power': [0, 10], # svd iteration: only for: 'randomized'_svd; iterations to make: m-by-k matrix Y
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def fast_ica(self)-> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        FastICA: a fast algorithm for Independent Component Analysis.
        Method is to efficiently find a projection of given data which minimises entropy.
        ICA is an algorithm that finds directions in the feature space corresponding to projections with high
        non-Gaussianity. These directions need not be orthogonal in the original feature space, but they are
        orthogonal in the whitened feature space, in which all directions correspond to the same variance.
        PCA, on the other hand, finds orthogonal directions in the raw feature space that correspond to directions
        accounting for maximum variance.
        Running ICA corresponds to finding a rotation in this space to identify the directions of largest
        non-Gaussianity.
        use:
        estimate sources given noisy measurements. Imagine 3 instruments playing simultaneously and 3 microphones
        recording the mixed signals. ICA is used to recover the sources ie. what is played by each instrument.
        Importantly, PCA fails at recovering our instruments since the related signals reflect non-Gaussian processes.

        - - - PARAMETERS - - -
        algorithm{‘parallel’, ‘deflation’}, default=’parallel’
            Apply parallel or deflational algorithm for FastICA.

        whiten bool, default=True
            If whiten is false, the data is already considered to be whitened, and no whitening is performed.

        # XXX: these should be optimized, as they can be a bottleneck.
        fun{‘logcosh’, ‘exp’, ‘cube’} or callable, default=’logcosh’
            The functional form of the G function used in the approximation to neg-entropy.
            The approximation to the negentropy used in fastICA dramatically decreases the computational time.
            Could be either ‘logcosh’, ‘exp’, or ‘cube’. You can also provide your own function. It should return a tuple containing
            the value of the function, and of its derivative, in the point.
            Example:
            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)

        fun_args dict, default=None
            Arguments to send to the functional form. If empty and if fun=’logcosh’, fun_args will take value {‘alpha’ : 1.0}.

        max_iter int, default=200
            Maximum number of iterations during fit.

        tol float, default=1e-4
            Tolerance on update at each iteration.

        w_init ndarray of shape (n_components, n_components), default=None
            The mixing matrix to be used to initialize the algorithm.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
        '''
        fun = FastICA(
              n_components=None,
              algorithm='parallel',  # ‘parallel’ (default), ‘deflation’
              whiten=True, # !!! not needed but if not set to True: n_components will be set to n_features
              fun='logcosh',  # XXX functional form of the G function used in the approximation to neg-entropy
              fun_args=None,
              max_iter=200,
              tol=0.0001,
              w_init=None, # custom: The mixing matrix to be used to initialize the algorithm.
              random_state=42
        )
        # does not work: dtypes ok, hps ok,
        hyperpars = {
             # 'algorithm': [0, 1], # fastica algorithm: ‘parallel’ (default), ‘deflation’
             'fun': [0, 2], # 'logcosh','exp','cube'
             'max_iter': [5, 1000], # maximum iterations
             'tol': [1e-8, 0.1], # Tolerance on update at each iteration.
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def isomap(self)-> (object, dict, dict):
            '''
            - - - DESCRIPTION - - -
            Isomap Embedding.
            Non-linear dimensionality reduction through Isometric Mapping
            One of the earliest approaches to manifold learning is the Isomap algorithm, short for Isometric Mapping.
            Isomap can be viewed as an extension of Multi-dimensional Scaling (MDS) or Kernel PCA. Isomap seeks a
            lower-dimensional embedding which maintains geodesic distances between all points. Isomap can be performed
            with the object Isomap.


            - - - PARAMETERS - - -
            n_neighbors : int or None, default=5
                Number of neighbors to consider for each point. If `n_neighbors` is an int,
                then `radius` must be `None`.
            radius : float or None, default=None
                Limiting distance of neighbors to return. If `radius` is a float,
                then `n_neighbors` must be set to `None`.
                .. versionadded:: 1.1
            n_components : int, default=2
                Number of coordinates for the manifold.
            eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
                'auto' : Attempt to choose the most efficient solver
                for the given problem.
                'arpack' : Use Arnoldi decomposition to find the eigenvalues
                and eigenvectors.
                'dense' : Use a direct solver (i.e. LAPACK)
                for the eigenvalue decomposition.
            tol : float, default=0
                Convergence tolerance passed to arpack or lobpcg.
                not used if eigen_solver == 'dense'.
            max_iter : int, default=None
                Maximum number of iterations for the arpack solver.
                not used if eigen_solver == 'dense'.
            path_method : {'auto', 'FW', 'D'}, default='auto'
                Method to use in finding shortest path.
                'auto' : attempt to choose the best algorithm automatically.
                'FW' : Floyd-Warshall algorithm.
                'D' : Dijkstra's algorithm.
            neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                                  default='auto'
                Algorithm to use for nearest neighbors search,
                passed to neighbors.NearestNeighbors instance.
            n_jobs : int or None, default=None
                The number of parallel jobs to run.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.
            metric : str, or callable, default="minkowski"
                The metric to use when calculating distance between instances in a
                feature array. If metric is a string or callable, it must be one of
                the options allowed by :func:`sklearn.metrics.pairwise_distances` for
                its metric parameter.
                If metric is "precomputed", X is assumed to be a distance matrix and
                must be square. X may be a :term:`Glossary <sparse graph>`.
                .. versionadded:: 0.22
            p : int, default=2
                Parameter for the Minkowski metric from
                sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
                equivalent to using manhattan_distance (l1), and euclidean_distance
                (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
                .. versionadded:: 0.22
            metric_params : dict, default=None
                Additional keyword arguments for the metric function.
                .. versionadded:: 0.22


             - - - INFORMATION - - -
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
            '''
            fun = Isomap(
                 n_components=None,
                 n_neighbors=5,
                 eigen_solver='auto', # eigenvector, eigenvalue {‘auto’, ‘arpack’, ‘dense’}, default=’auto’
                 tol=0, # Convergence tolerance of arpack, tol <e-12 slower, >e-7 faster, not used if eigen_solver == ‘dense’.
                 max_iter=None, # not used if eigen_solver == ‘dense’
                 path_method='auto', # {‘auto’, ‘FW’, ‘D’}, default=’auto’; FW faster?
                 neighbors_algorithm='auto', # auto is fastest, ('auto', 'brute', 'kd_tree', 'ball_tree')
                 n_jobs=None, # spectra dataset: 64.49s, 63.99s no effect
                 metric='minkowski', # metric is determined by p value!
                 p=2, # for metric='minkowski': manhatten distance (p=1), euclidean distance (p=2)
                 metric_params=None
            )

            # Hyperparameters
            neighbors_max = int(min(self.nrows/2, 100))
            hyperpars = {
                'n_neighbors': [5, neighbors_max],
                 'p': [1, 2],  # for metric: 'minkowski' 1-minkowski 2-euclidean
                 'max_iter': [5, 1000],
                 'path_method': [0, 1], # 'FW', 'D' speed!
                 'neighbors_algorithm': [0, 2], # 'brute', 'kd_tree', 'ball_tree'
                 'tol': [1e-15, 0.1],   # tol <e-12 slower, >e-7 faster
                # 'eigen_solver': [0, 1], # ‘arpack’, ‘dense’ not important and a lot of errors
            }
            params = fun.get_params()
            return fun, params, hyperpars



    def locally_linear_embedding(self)-> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        Locally linear embedding (LLE) seeks a lower-dimensional projection of the data which preserves distances within
        local neighborhoods. It can be thought of as a series of local Principal Component Analyses which are globally
        compared to find the best non-linear embedding.
        1) Weight Matrix Construction O [D N k^3].
         N number of training data points
         D input dimension
         k number of nearest neighbors
        2) The construction of the LLE weight matrix involves the solution of a kxk linear equation for each of the N
           local neighborhoods
        3) Partial Eigenvalue Decomposition. reduces the
        use: the groups are super seperated from each other
        LLE is able to learn the global structure of nonlinear manifolds like face images or text documents.

        #### MODIFIED LLE
        MLLE: Modified Locally Linear Embedding: use multiple weight vectors in each neighborhood.
            It requires n_neighbors > n_components.
            One well-known issue with LLE is the regularization problem. When the number of neighbors is greater than the number
            of input dimensions, the matrix defining each local neighborhood is rank-deficient. To address this, standard LLE applies
            an arbitrary regularization parameter r, which is chosen relative to the trace of the local weight matrix.
            1) Nearest Neighbors Search. Same as standard LLE
            2) Weight Matrix Construction.
            3) Partial Eigenvalue Decomposition. Same as standard LLE

        #### HESSIAN LLE
        HLLE: Modified Locally Linear Embedding: use multiple weight vectors in each neighborhood.
            n_neighbors > n_components * (n_components + 3) / 2
            Hessian Eigenmapping (also known as Hessian-based LLE: HLLE) is another method of solving the regularization problem of LLE.
            It revolves around a hessian-based quadratic form at each neighborhood which is used to recover the locally linear structure.
            Though other implementations note its poor scaling with data size, sklearn implements some algorithmic improvements which
            make its cost comparable to that of other LLE variants for small output dimension. HLLE can be performed with function
            locally_linear_embedding or its object-oriented counterpart LocallyLinearEmbedding, with the keyword method = 'hessian'.
            It requires n_neighbors > n_components * (n_components + 3) / 2.
            1) Nearest Neighbors Search. Same as standard LLE
            2) Weight Matrix Construction.
            3) Partial Eigenvalue Decomposition. Same as standard LLE

        #### Local Tangent Space Alignment
        LTSA: is algorithmically similar enough to LLE
            Rather than focusing on preserving neighborhood distances as in LLE, LTSA seeks to characterize the local geometry at each
            neighborhood via its tangent space, and performs a global optimization to align these local tangent spaces to learn the
            embedding. LTSA can be performed with function locally_linear_embedding or its object-oriented counterpart
            LocallyLinearEmbedding,
            eigensolver changed to dense (better for some datasets)

        - - - PARAMETERS - - -
        n_neighbors int
            number of neighbors to consider for each point.

        n_components int
            number of coordinates for the manifold.

        reg float, default=1e-3
            regularization constant, multiplies the trace of the local covariance matrix of the distances.

        eigen_solver{‘auto’, ‘arpack’, ‘dense’}, default=’auto’
            auto : algorithm will attempt to choose the best method for input data

        arpack use arnoldi iteration in shift-invert mode.
            For this method, M may be a dense matrix, sparse matrix, or general linear operator.
            Warning: ARPACK can be unstable for some problems. It is best to try several random seeds in order to check results.

        dense use standard dense matrix operations for the eigenvalue
            decomposition. For this method, M must be an array or matrix type. This method should be avoided for large problems.

        tol float, default=1e-6
            Tolerance for ‘arpack’ method Not used if eigen_solver==’dense’.

        max_iter int, default=100
            maximum number of iterations for the arpack solver.

        method{‘standard’, ‘hessian’, ‘modified’, ‘ltsa’}, default=’standard’
            standard use the standard locally linear embedding algorithm. see reference [1]

            'hessian' use the Hessian eigenmap method. This method requires
                n_neighbors > n_components * (1 + (n_components + 1) / 2. see reference [2]
                (is not working, this is too high!

            'modified' use the modified locally linear embedding algorithm.
                see reference [3]

            'ltsa' use local tangent space alignment algorithm
                see reference [4]
                ** not working for spectral and covid dataset. see error

        hessian_tol float, default=1e-4
            Tolerance for Hessian eigenmapping method. Only used if method == ‘hessian’

        modified_tol float, default=1e-12
            Tolerance for modified LLE method. Only used if method == ‘modified’

        random_state int, RandomState instance, default=None
            Determines the random number generator when solver == ‘arpack’. Pass an int for reproducible results
            across multiple function calls. See Glossary.

        n_jobs int or None, default=None
            The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib. parallel_backend
            context. -1 means using all processors. See Glossary for more details.

        - - - ERROR - - -
        covid, spectral: method = 'ltsa'
            Gi[:, 1:] = v[:, :n_components]
            ValueError: could not broadcast input array from shape (5,5) into shape (5,17)

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html
        '''
        fun = LocallyLinearEmbedding(
             n_components=None,
             n_neighbors=5,
             reg=0.001,
             eigen_solver='dense',
             tol=1e-06,
             max_iter=100,
             method='auto',
             hessian_tol=0.0001,
             modified_tol=1e-12,
             random_state=42,
             n_jobs=None
        )
        # Hyperparameters
        neighbors_max = int(min(self.nrows/2, 100))
        # n_components * (n_components + 3) / 2
        hyperpars = {
             'n_neighbors': [5, neighbors_max],
             'reg': [0.00001, 0.1], # regularization constant
             'tol': [1e-10, 0.1], # ‘arpack’ only, tolerance
             'max_iter': [5, 1000],
             'method': [0, 2], # ‘standard’,‘modified’,‘ltsa’, default=’standard’ (‘hessian’ requieres too many n-neighbors)
             'modified_tol': [1e-10, 0.1], # if method == ‘modified’
             # 'eigen_solver': [0, 1], # ‘arpack’, ‘dense’ arpack complains a lot and not important
             # 'hessian_tol': [1e-10, 0.1], # if method == ‘hessian’
             }
        params = fun.get_params()
        return fun, params, hyperpars



    def non_negative_matrix_factorization(self) -> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X.
        USE: dimensionality reduction, source separation or topic extraction

        - - - PARAMETERS - - -
        init{‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’}, default=None
            Method used to initialize the procedure. Default: None.
            None: ‘nndsvda’ if n_components <= min(n_samples, n_features), otherwise random. (v.11)
            'random': non-negative random matrices, scaled with: sqrt(X.mean() / n_components)
            'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD) initialization (better for sparseness)
            'nndsvda': NNDSVD with zeros filled with the average of X (better when sparsity is not desired)
            'nndsvdar' NNDSVD with zeros filled with small random values (generally faster, less accurate alternative to
             NNDSVDa for when sparsity is not desired)
            'custom': use custom matrices W and H

        solver{‘cd’, ‘mu’}, default=’cd’
            Numerical solver to use: ‘cd’ is a Coordinate Descent solver. ‘mu’ is a Multiplicative Update solver.

        beta_loss float or {‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’}, default=’frobenius’
            Beta divergence to be minimized, measuring the distance between X and the dot product WH. Note that values different
            from ‘frobenius’ (or 2) and ‘kullback-leibler’ (or 1) lead to significantly slower fits. Note that for beta_loss <= 0
            (or ‘itakura-saito’), the input matrix X cannot contain zeros. Used only in ‘mu’ solver.

        tol float, default=1e-4
            Tolerance of the stopping condition (based on change of H)

        max_iter int, default=200
            Maximum number of iterations before timing out.

        random_state int, RandomState instance or None, default=None
            Used for initialisation (when init == ‘nndsvdar’ or ‘random’), and in Coordinate Descent. Pass an int for
            reproducible results across multiple function calls. See Glossary.

        alpha float, default=0.0
            Constant that multiplies the regularization terms. Set it to zero to have no regularization. When using alpha
            instead of alpha_W and alpha_H, the regularization terms are not scaled by the n_features (resp. n_samples)
            factors for W (resp. H).
            Deprecated since version 1.0: Use alpha_W and alpha_H instead.

        alpha_H float or “same”, default=”same” New in version 1.0.
            Constant that multiplies the regularization terms of H. Set it to zero to have no regularization on H.
            If “same” (default), it takes the same value as alpha_W.

        l1_ratio float, default=0.0 New in version 1.0.
            The regularization mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an elementwise L2
            penalty (aka Frobenius Norm). For l1_ratio = 1 it is an elementwise L1 penalty. For 0 < l1_ratio < 1, the penalty
            is a combination of L1 and L2.

        verbose int, default=0
            Whether to be verbose.

        shuffle bool, default=False
            If true, randomize the order of coordinates in the CD solver.

        regularization{‘both’, ‘components’, ‘transformation’, None}, default=’both’
            Select whether the regularization affects the components (H), the transformation (W), both or none of them.

        - - - INFORMATION - - -
        MORE: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF
        '''
        fun = NMF(
              n_components=None,
              init=None, # None: ‘nndsvd’ if n_components <= min(n_samples, n_features), otherwise random. ‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’
              solver='cd', # Numerical solver 'cd' oordinate descent ,'mu' multuplicative update
              beta_loss='frobenius', # distance between X and W*H ‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’ (kl slow)
              tol=0.0001,
              max_iter=200,
              random_state=42, # default None
              alpha=1,
              l1_ratio=0.0,
              verbose=0,
              shuffle=False
              )
        hyperpars = {
             'init': [0, 3], # ‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’
             'solver': [0, 1], # ‘cd’, ‘mu’
             'beta_loss': [0, 1], # ‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’
             'tol': [1e-6, 0.1], #
             'max_iter': [5, 1000], #
             'alpha': [0.1, 10], # from v1.0 on substituted by alpha_w, alpha_h
             'l1_ratio': [0.0, 1.0], # For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def pca(self) -> (object, dict, dict):
        '''
         - - - DESCRIPTION - - -
        n=0.95...0.99 means percent of explained variance in pca's
        n=3 number of pca's

        - - - PARAMETERS - - -
        n_components: int, float or 'mle', default=None
            Number of components to keep.
            if float: 0.95...0.99 percent of explained variance in pca's, if int: number of pcas

        copy : bool, default=True
            If False, data passed to fit are overwritten and running
            fit(X).transform(X) will not yield the expected results,
            use fit_transform(X) instead.

        whiten : bool, default=False
            When True (False by default) the `components_` vectors are multiplied
            by the square root of n_samples and then divided by the singular values
            to ensure uncorrelated outputs with unit component-wise variances.
            Whitening will remove some information from the transformed signal
            (the relative variance scales of the components) but can sometime
            improve the predictive accuracy of the downstream estimators by
            making their data respect some hard-wired assumptions.

        svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
            If auto :
                The solver is selected by a default policy based on `X.shape` and
                `n_components`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient 'randomized'
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full :
                run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing
            If arpack :
                run SVD truncated to n_components calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                0 < n_components < min(X.shape)
            If randomized :
                run randomized SVD by the method of Halko et al.
        tol : float, default=0.0
            Tolerance for singular values computed by svd_solver == 'arpack'.
            Must be of range [0.0, infinity).
        iterated_power : int or 'auto', default='auto'
            Number of iterations for the power method computed by
            svd_solver == 'randomized'.
            Must be of range [0, infinity).
            .. versionadded:: 0.18.0
        random_state : int, RandomState instance or None, default=None
            Used when the 'arpack' or 'randomized' solvers are used. Pass an int
            for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        '''
        fun = PCA(
              n_components=None,
              copy=True,
              whiten=False,
              svd_solver='auto',
              tol=0.0, #
              iterated_power='auto',
              random_state=42
        )
        hyperpars = {
            'empty_py': [0]
             # 'svd_solver': [0, 2], # ‘full’, ‘arpack’, ‘randomized’, default=’auto’
             # 'tol': [0.0, 100.0], # !arpack only [0.0, infinity]
             # 'iterated_power': [0, 100], # !randomized only
             }
        params = fun.get_params()
        return fun, params, hyperpars


    def pca_incremental(self)-> (object, dict, dict):
        '''
         - - - DESCRIPTION - - -
        Incremental principal components analysis (IPCA).
        USE: Dataset is too large for memory
        Efficiency dpends on the size of the input data.
        This algorithm has constant memory complexity, on the order of batch_size * n_features, enabling use of
        np.memmap files without loading the entire file into memory.
        For sparse matrices, the input is converted to dense in batches (in order to be able to subtract the mean)
        which avoids storing the entire dense matrix at any one time.
        The computational overhead of each SVD is O(batch_size * n_features ** 2), but only 2 * batch_size samples
        remain in memory at a time. There will be n_samples / batch_size SVD computations to get the principal
        components, versus 1 large SVD of complexity O(n_samples * n_features ** 2) for PCA.
        Linear dimensionality reduction using Singular Value Decomposition of the data, keeping only the most
        significant singular vectors to project the data to a lower dimensional space. The input data is centered
        but not scaled for each feature before applying the SVD.

        - - - PARAMETERS - - -
        n_components : int, default=None
            Number of components to keep. If ``n_components`` is ``None``,
            then ``n_components`` is set to ``min(n_samples, n_features)``.
        whiten : bool, default=False
            When True (False by default) the ``components_`` vectors are divided
            by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
            with unit component-wise variances.
            Whitening will remove some information from the transformed signal
            (the relative variance scales of the components) but can sometimes
            improve the predictive accuracy of the downstream estimators by
            making data respect some hard-wired assumptions.
        copy : bool, default=True
            If False, X will be overwritten. ``copy=False`` can be used to
            save memory but is unsafe for general use.
        batch_size : int, default=None
            The number of samples to use for each batch. Only used when calling
            ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
            is inferred from the data and set to ``5 * n_features``, to provide a
            balance between approximation accuracy and memory consumption.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA
        '''
        fun = IncrementalPCA(
             n_components=None,
             whiten=False,
             copy=True,
             batch_size=None # 5 * n_features, provides a balance between accuracy and memory use.
        )
        hyperpars = {
             'empty_py': [0]
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def pca_sparse(self) -> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        USE: better interpretation of the features,
        DETAILS: Finds the set of sparse components that can optimally reconstruct the data.
        The amount of sparseness is controllable by the coefficient of the L1 penalty, given by the parameter alpha.
        L1 penalty avoids overfitting by panelizing the regression coefficients of high value.
        mushroom dataset: alpha=0.1, ridge=0.01

        - - - PARAMETERS - - -
        n_components : int, default=None
            Number of sparse atoms to extract. If None, then ``n_components``
            is set to ``n_features``.
        alpha : float, default=1
            Sparsity controlling parameter. Higher values lead to sparser
            components.
        ridge_alpha : float, default=0.01
            Amount of ridge shrinkage to apply in order to improve
            conditioning when calling the transform method.
        max_iter : int, default=1000
            Maximum number of iterations to perform.
        tol : float, default=1e-8
            Tolerance for the stopping condition.
        method : {'lars', 'cd'}, default='lars'
            Method to be used for optimization.
            lars: uses the least angle regression method to solve the lasso problem
            (linear_model.lars_path)
            cd: uses the coordinate descent method to compute the
            Lasso solution (linear_model.Lasso). Lars will be faster if
            the estimated components are sparse.
        n_jobs : int, default=None
            Number of parallel jobs to run.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        U_init : ndarray of shape (n_samples, n_components), default=None
            Initial values for the loadings for warm restart scenarios. Only used
            if `U_init` and `V_init` are not None.
        V_init : ndarray of shape (n_components, n_features), default=None
            Initial values for the components for warm restart scenarios. Only used
            if `U_init` and `V_init` are not None.
        verbose : int or bool, default=False
            Controls the verbosity; the higher, the more messages. Defaults to 0.
        random_state : int, RandomState instance or None, default=None
            Used during dictionary learning. Pass an int for reproducible results
            across multiple function calls.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
        '''
        fun = SparsePCA(
            n_components=None,
            alpha=1,
            ridge_alpha=0.01, # L2 Regularization, adds “squared magnitude” of coefficient as penalty term to the loss function
            max_iter=1000,
            tol=1e-08,
            method='lars', # lars: uses the least angle regression method to solve the lasso problem.
                           # cd: uses the coordinate descent method to compute the Lasso solution.
            n_jobs=None, # loadings for warm restart scenarios
            U_init=None, # loadings for warm restart scenarios
            V_init=None, # loadings for warm restart scenarios
            verbose=False,
            random_state=42
        )
        hyperpars = {
             'alpha': [0.0001, 0.5], # controls sparsity, higher: more sparse
             'ridge_alpha': [0.0001, 10], # L2 Regularization
             'tol': [0.001, 0.1], # Tolerance for the stopping condition
             'max_iter': [100,800], #
             'method': [0,1], # ‘lars’, ‘cd’
        }
        params = fun.get_params()
        return fun, params, hyperpars


    def pca_sparse_mini_batch(self) -> (object, dict, dict):
        '''
         - - - DESCRIPTION - - -

        Mini-batch sparse PCA (MiniBatchSparsePCA)
        DETAILS: MB-SparsePCA is a variant of SparsePCA that is faster but less accurate.
        The increased speed is reached by iterating over small chunks of the set of features, for a given number of iterations.
        results are very different on higher or lower alphas.
        mushroom dataset: alpha=0.1, ridge=0.01
        - - - PARAMETERS - - -
         n_components : int, default=None
            Number of sparse atoms to extract. If None, then ``n_components``
            is set to ``n_features``.
        alpha : int, default=1
            Sparsity controlling parameter. Higher values lead to sparser
            components.
        ridge_alpha : float, default=0.01
            Amount of ridge shrinkage to apply in order to improve
            conditioning when calling the transform method.
        n_iter : int, default=100
            Number of iterations to perform for each mini batch.
        callback : callable, default=None
            Callable that gets invoked every five iterations.
        batch_size : int, default=3
            The number of features to take in each mini batch.
        verbose : int or bool, default=False
            Controls the verbosity; the higher, the more messages. Defaults to 0.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting it in batches.
        n_jobs : int, default=None
            Number of parallel jobs to run.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        method : {'lars', 'cd'}, default='lars'
            Method to be used for optimization.
            lars: uses the least angle regression method to solve the lasso problem
            (linear_model.lars_path)
            cd: uses the coordinate descent method to compute the
            Lasso solution (linear_model.Lasso). Lars will be faster if
            the estimated components are sparse.
        random_state : int, RandomState instance or None, default=None
            Used for random shuffling when ``shuffle`` is set to ``True``,
            during online dictionary learning. Pass an int for reproducible results
            across multiple function calls.

        - - - INFORMATION - - -
        '''
        fun = MiniBatchSparsePCA(
             n_components=None,
             alpha=1,
             ridge_alpha=0.01,
             n_iter=100,
             callback=None,
             batch_size=3,
             verbose=False,
             shuffle=True,
             n_jobs=None,
             method='lars',
             random_state=42
        )
        hyperpars = {
             'alpha': [0.1, 10], # controls sparsity, higher: more sparse
             'ridge_alpha': [0.5, 10], # L2 Regularization
             'n_iter': [50,250], #
             'method': [0,1], # ‘lars’, ‘cd’
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def pca_kernel(self) -> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        Kernel PCA is able to find a projection of the data that makes data linearly separable
        Example: group1 -inner circle, group2 -outer circle
        Gamma: defines how far the influence of a single training example reaches, with low values
        meaning 'far' and high values meaning 'close'.
        RBF-Kernel = exp( dist(A,B) / 2*rho^2) )
        if rho is small - region of similarity, big - region of dissimilarity

        - - - PARAMETERS - - -
        n_components int, default=None
            Number of components. If None, all non-zero components are kept.

        kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
            Kernel used for PCA.

        gamma float, default=None
            Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels. If gamma is None, then it is set to 1/n_features.

        degree int, default=3
            Degree for poly kernels. Ignored by other kernels.

        coef0 float, default=1
            Independent term in poly and sigmoid kernels. Ignored by other kernels.

        kernel_params dict, default=None
            Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels.

        alpha float, default=1.0
            Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).

        fit_inverse_transform bool, default=False
            Learn the inverse transform for non-precomputed kernels (i.e. learn to find the pre-image of a point).

        eigen_solver{‘auto’, ‘dense’, ‘arpack’, ‘randomized’}, default=’auto’
            Select eigensolver to use. If n_components is much less than the number of training samples, randomized
            (or arpack to a smaller extend) may be more efficient than the dense eigensolver. Randomized SVD is performed
            according to the method of Halko et al [3].

            auto: the solver is selected by a default policy based on n_samples (the number of training samples) and n_components:
            if the number of components to extract is less than 10 (strict) and the number of samples is more than 200 (strict),
            the ‘arpack’ method is enabled. Otherwise the exact full eigenvalue decomposition is computed and optionally
            truncated afterwards (‘dense’ method).

            dense: run exact full eigenvalue decomposition calling the standard LAPACK solver via scipy.linalg.eigh, and
            select the components by postprocessing

            arpack: run SVD truncated to n_components calling ARPACK solver using scipy.sparse.linalg.eigsh. It requires strictly
            0 < n_components < n_samples

            randomized:
            run randomized SVD by the method of Halko et al. [3]. The current implementation selects eigenvalues based on their module;
            therefore using this method can lead to unexpected results if the kernel is not positive semi-definite. See also [4].

        tol float, default=0
            Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.

        max_iter int, default=None
            Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.

        iterated_power int >= 0, or ‘auto’, default=’auto’
            Number of iterations for the power method computed by svd_solver == ‘randomized’. When ‘auto’, it is set to 7 when
            n_components < 0.1 * min(X.shape), other it is set to 4.

        remove_zero_eig bool, default=False
            If True, then all components with zero eigenvalues are removed, so that the number of components in the output may
            be < n_components (and sometimes even zero due to numerical instability). When n_components is None, this parameter is
            ignored and components with zero eigenvalues are removed regardless.

        random_state int, RandomState instance or None, default=None
            Used when eigen_solver == ‘arpack’ or ‘randomized’. Pass an int for reproducible results across multiple function calls.

        copy_X bool, default=True
            If True, input X is copied and stored by the model in the X_fit_ attribute. If no further changes will be done to X, setting
            copy_X=False saves memory by storing a reference.

        n_jobs int, default=None
            The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA
        '''
        fun = KernelPCA(
            n_components=None,
            kernel='linear',
            gamma=10,
            degree=3,
            coef0=1,
            kernel_params=None, # Parameters (keyword arguments and values) for kernel passed as callable object.
            alpha=1.0,
            fit_inverse_transform=False,
            eigen_solver='auto',
            tol=0,
            max_iter=None,
            remove_zero_eig=False,
            random_state=42,
            copy_X=True,
            n_jobs=None
        )
        #
        gamma_min = round(1 / max([5, self.ncols]), 4)

        hyperpars = {
             'kernel': [0, 3], # ‘linear’, ‘rbf’, ‘sigmoid’, ‘cosine’, not used: ‘precomputed’, 'poly'-errors
             'gamma': [gamma_min, 10], # if None: 1/n_features
             # 'degree': [1, 4], # only for ‘poly’-errors
             # 'coef0': [1.0, 10.0], # only for poly and sigmoid -errors
             'alpha': [0.0001, 10], # controls ridge regression, only when fit_inverse_transform=True.
             'fit_inverse_transform': [0,1], # False - True
             # 'eigen_solver': [0, 2], # ‘dense’, ‘arpack’, ‘randomized’ not important and a lot of errors
             'tol': [0, 0.1], #  only arpack, Residual tolerances of the computed eigenvalues. default: 0
             'max_iter': [5,1000], # only arpack, Maximum number of iterations
             # 'remove_zero_eig': [0,1], # True, False if to remove zero eigenvectors (n might be smaller than n_components)
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def smacof(self):
        '''
         - - - DESCRIPTION - - -
        SMACOF - Scaling by MAjorizing a COmplicated Function
        is a multidimensional scaling algorithm which minimizes an objective function (the stress) using a
        majorization technique. Stress majorization, also known as the Guttman Transform, guarantees a monotone
        convergence of stress, and is more powerful than traditional techniques such as gradient descent.
        The SMACOF algorithm for metric MDS can be summarized by the following steps:
        1) Set an initial start configuration, randomly or not.
        2) Compute the stress
        3) Compute the Guttman Transform
        ) Iterate 2 and 3 until convergence.
        The nonmetric algorithm adds a monotonic regression step before computing the stress.

        - - - PARAMETERS - - -
        dissimilarities : ndarray of shape (n_samples, n_samples)
            Pairwise dissimilarities between the points. Must be symmetric.
        metric : bool, default=True
            Compute metric or nonmetric SMACOF algorithm.
        n_components : int, default=2
            Number of dimensions in which to immerse the dissimilarities. If an
            ``init`` array is provided, this option is overridden and the shape of
            ``init`` is used to determine the dimensionality of the embedding
            space.
        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the algorithm. By
            default, the algorithm is initialized with a randomly chosen array.
        n_init : int, default=8
            Number of times the SMACOF algorithm will be run with different
            initializations. The final results will be the best output of the runs,
            determined by the run with the smallest final stress. If ``init`` is
            provided, this option is overridden and a single run is performed.
        n_jobs : int, default=None
            The number of jobs to use for the computation. If multiple
            initializations are used (``n_init``), each run of the algorithm is
            computed in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        max_iter : int, default=300
            Maximum number of iterations of the SMACOF algorithm for a single run.
        verbose : int, default=0
            Level of verbosity.
        eps : float, default=1e-3
            Relative tolerance with respect to stress at which to declare
            convergence.
        random_state : int, RandomState instance or None, default=None
            Determines the random number generator used to initialize the centers.
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.
        return_n_iter : bool, default=False
            Whether or not to return the number of iterations.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html#
        '''
        fun = smacof(
             dissimilarities=np.zeros(shape=(2, 2)),
             metric=True,
             n_components=None,
             init=None,  # ndarray with Starting configuration
             n_init=8,
             n_jobs=None,
             max_iter=300,
             verbose=0,
             eps=0.001,
             random_state=42,
             return_n_iter=False
        )
        hyperpars = {
             'metric': [0.0, 1.0], # False, True (metric or non-metric)
             'n_init': [2,32], # n initializations (best one choosen, default=8
             'max_iter': [5,1000], # only arpack, Maximum number of iterations
             'eps': [0.00001, 0.1],
        }
        params = fun.get_params()
        return fun, params, hyperpars


    def spectral_embedding(self) -> (object, dict, dict):
        '''
        - - - DESCRIPTION - - -
        Spectral embedding for non-linear dimensionality reduction.
        Forms an affinity matrix given by the specified function and applies spectral decomposition to the corresponding
        graph laplacian. The resulting transformation is given by the value of the eigenvectors for each data point.
        Note : Laplacian Eigenmaps is the actual algorithm implemented here.
        1) Weighted Graph Construction. Transform the raw input data into graph representation using affinity matrix.
        2) Graph Laplacian Construction (Degree matrix - Afinity Matrix)
        3) Partial Eigenvalue Decomposition on graph Laplacian

        - - - PARAMETERS - - -
        affinity: ‘nearest_neighbors’, ‘rbf’, ‘precomputed’, ‘precomputed_nearest_neighbors’, default=’nearest_neighbors’
            ‘nearest_neighbors’ : construct the affinity matrix by computing a graph of nearest neighbors.
            ‘rbf’ : construct the affinity matrix by computing a radial basis function (RBF) kernel.
            ‘precomputed’ : interpret X as a precomputed affinity matrix.
            ‘precomputed_nearest_neighbors’ : interpret X as a sparse graph of precomputed nearest neighbors, and constructs
                                              the affinity matrix by selecting the n_neighbors nearest neighbors.

        gamma float, default=None
            Kernel coefficient for rbf kernel (Radial Basis Function (RBF) kernel SVM).
            If None, gamma will be set to 1/n_features.
            https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

        random_state int, RandomState instance or None, default=None
            eigen_solver == 'amg', and for the K-Means initialization. Use an int to make the results deterministic across calls.
            Note When using eigen_solver == 'amg', it is necessary to also fix the global numpy seed with np.random.seed(int)
            to get deterministic results. See https://github.com/pyamg/pyamg/issues/139 for further information.

        eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None * should not have influence on output
            The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed. It can be faster on very large, sparse problems.
            If None, then 'arpack' is used.

        n_neighbors int, default=None
            Number of nearest neighbors for nearest_neighbors graph building. If None, n_neighbors will be set to max(n_samples/10, 1).

        n_jobs int, default=None
            The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
        '''
        ####
        fun = SpectralEmbedding(
            n_components=None,
            affinity='nearest_neighbors', #
            gamma=None,
            random_state=42,
            eigen_solver=None, # should have no influence on result
            n_neighbors=None,
            n_jobs=None
        )
        ### init
        inv_feat = 1/self.ncols
        neighbors_max = int(min(self.nrows/2, 100))
        # hyerparameter ranges, further processed in hyperparameter -> hyperparameter_initialization.py
        hyperpars = {
             'affinity': [0, 1], #‘ nearest_neighbors’, ‘rbf’
             'gamma': [inv_feat/100, 10], # only rbf, default: 1/n_features
             'n_neighbors': [5, neighbors_max], # only nearest neighbors, default: 1/n_features
            # 'eigen_solver': [0, 2], # CAUSES MANY MATRIX ERRORS ‘arpack’, ‘lobpcg’, ‘amg’
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def truncated_svd(self) -> (object, dict, dict):
        '''
         - - - DESCRIPTION - - -
            TruncatedSVD implements a variant of singular value decomposition (SVD) that only computes the k
            largest singular values, where k is a user-specified parameter.
            In linear algebra, the singular value decomposition (SVD) is a factorization of a real or complex matrix.
            It generalizes the eigendecomposition of a square normal matrix with an orthonormal eigenbasis to any M x n
            m\times n matrix. It is related to the polar decomposition.

            This transformer performs linear dimensionality reduction by means of truncated singular value
            decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing
            the singular value decomposition. This means it can work with sparse matrices efficiently.
            In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in
            sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA).
            This estimator supports two algorithms: a fast randomized SVD solver, and a “naive” algorithm that
            uses ARPACK as an eigensolver on X * X.T or X.T * X, whichever is more efficient.

            TruncatedSVD is very similar to PCA, but differs in that the matrix X
            does not need to be centered. When the columnwise (per-feature) means of X
            are subtracted from the feature values, truncated SVD on the resulting matrix is equivalent to PCA.
            In practical terms, this means that the TruncatedSVD transformer accepts scipy.sparse matrices without
            the need to densify them, as densifying may fill up memory even for medium-sized document collections.

        - - - PARAMETERS - - -
        n_components : int, default=2
            Desired dimensionality of output data.
            Must be strictly less than the number of features.
            The default value is useful for visualisation. For LSA, a value of
            100 is recommended.
        algorithm : {'arpack', 'randomized'}, default='randomized'
            SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
            (scipy.sparse.linalg.svds), or "randomized" for the randomized
            algorithm due to Halko (2009).
        n_iter : int, default=5
            Number of iterations for randomized SVD solver. Not used by ARPACK. The
            default is larger than the default in
            :func:`~sklearn.utils.extmath.randomized_svd` to handle sparse
            matrices that may have large slowly decaying spectrum.
        random_state : int, RandomState instance or None, default=None
            Used during randomized svd. Pass an int for reproducible results across
            multiple function calls.
            See :term:`Glossary <random_state>`.
        tol : float, default=0.0
            Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
            SVD solver.

        - - - INFORMATION - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#
        '''
        fun = TruncatedSVD(
            n_components=None,
            algorithm='randomized',
            n_iter=5,
            random_state=42,
            tol=0.0
            )
        hyperpars = {
             'algorithm': [0, 1], # algorithm with other options is already choosen, 'arpack', 'randomized'
             'n_iter': [5,1000], #
             'tol': [0, 0.1], #  only arpack, Residual tolerances of the computed eigenvalues. default: 0
        }
        params = fun.get_params()
        return fun, params, hyperpars



# def tsne(self) -> (object, dict, dict):
#     '''
#     !!! ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
#      - - - DESCRIPTION - - -
#
#
#     - - - PARAMETERS - - -
#     perplexity float, default=30.0
#         The perplexity is related to the number of nearest neighbors that is used in other manifold learning
#         algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between
#         5 and 50. Different values can result in significantly different results.
#     early_exaggeration float, default=12.0
#         Controls how tight natural clusters in the original space are in the embedded space and how much
#         space will be between them. For larger values, the space between natural clusters will be larger
#         in the embedded space. Again, the choice of this parameter is not very critical. If the cost
#         function increases during initial optimization, the early exaggeration factor or the learning rate
#         might be too high.
#     learning_rate float or ‘auto’, default=200.0
#         The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high,
#         the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours.
#         If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
#         If the cost function gets stuck in a bad local minimum increasing the learning rate may help.
#         Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE, etc.) use a definition of
#         learning_rate that is 4 times smaller than ours. So our learning_rate=200 corresponds to
#         learning_rate=800 in those other implementations. The ‘auto’ option sets the learning_rate to
#         max(N / early_exaggeration / 4, 50) where N is the sample size, following [4] and [5].
#         This will become default in 1.2.
#     n_iter int, default=1000
#         Maximum number of iterations for the optimization. Should be at least 250.
#     n_iter_without_progress int, default=300
#         Maximum number of iterations without progress before we abort the optimization, used after 250 initial
#         iterations with early exaggeration. Note that progress is only checked every 50 iterations so this value
#         is rounded to the next multiple of 50.
#     min_grad_norm float, default=1e-7
#         If the gradient norm is below this threshold, the optimization will be stopped.
#     metric str or callable, default=’euclidean’
#         The metric to use when calculating distance between instances in a feature array. If metric is a string,
#         it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter,
#         or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is “precomputed”, X is assumed
#         to be a distance matrix. Alternatively, if metric is a callable function, it is called on each pair of
#         instances (rows) and the resulting value recorded. The callable should take two arrays from X as input
#         and return a value indicating the distance between them. The default is “euclidean” which is interpreted
#         as squared euclidean distance.
#     metric_params dict, default=None
#         Additional keyword arguments for the metric function.
#     init{‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default=’random’
#         Initialization of embedding. Possible options are ‘random’, ‘pca’, and a numpy array of shape
#         (n_samples, n_components). PCA initialization cannot be used with precomputed distances and is usually
#         more globally stable than random initialization. init='pca' will become default in 1.2.
#     verbose int, default=0
#         Verbosity level.
#     random_state int, RandomState instance or None, default=None
#         Determines the random number generator. Pass an int for reproducible results across multiple function calls.
#         Note that different initializations might result in different local minima of the cost function.
#     method str, default=’barnes_hut’
#         By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time.
#         method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should
#         be used when nearest-neighbor errors need to be better than 3%. However, the exact method cannot scale
#         to millions of examples.
#     angle float, default=0.5
#         Only used if method=’barnmetrices_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
#         ‘angle’ is the angular size (referred to as theta in [3]) of a distant node as measured from a point.
#         If this size is below ‘angle’ then it is used as a summary node of all points contained within it.
#         This method is not very sensitive to changes in this parameter in the range of 0.2 - 0.8.
#         Angle less than 0.2 has quickly increasing computation time and angle greater 0.8 has quickly increasing error.
#         n_jobs int, default=None
#         The number of parallel jobs to run for neighbors search. This parameter has no impact when metric="precomputed"
#          or (metric="euclidean" and method="exact"). None means 1 unless in a joblib.parallel_backend context. -1 means
#           using all processors.
#     - - - INFORMATION - - -
#     https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
#
#     '''
#     fun = TSNE(
#        n_components=None,
#        perplexity=30.0,
#        early_exaggeration=12.0, # the choice of this parameter is not very critical
#        learning_rate='auto',
#        n_iter=1000,
#        n_iter_without_progress=300, # checked every 50 iterations
#        min_grad_norm=1e-07,
#        metric='euclidean',
#        init='pca',
#        verbose=0,
#        random_state=42,
#        method='exact', # the 'exact' method cannot scale, barnes_hut is only for 1,2,3 dimensions
#        angle=0.5,
#        n_jobs=None
#     )
#     hyperpars = {
#          'perplexity': [5, 50], # default: 30
#          'learning_rate': [10.0, 1000.0], # default: auto
#          'n_iter': [250, 2000], # default: 1000
#          'n_iter_without_progress': [0, 9], # default: 300 # checked every 50 iterations
#          'min_grad_norm': [1e-09, 0.1], # default: 1e-07
#          'metric': [0, 2], # 'minkowski','manhattan', default: 'euclidean'
#          'init': [0, 1], # ‘random’, ‘pca’
#          'angle': [0.2, 0.8], # FALSE, TRUE
#          # 'preprocess': [0, 5], # not neccessary, data are scaled
#     }
#     params = fun.get_params()
#     return fun, params, hyperpars


# def umap(self) -> (object, dict, dict):
#     '''
#
#      - - - DESCRIPTION - - -
#
#
#     - - - PARAMETERS - - -
#     Uniform Manifold Approximation and Projection
#         Finds a low dimensional embedding of the data that approximates
#         an underlying manifold.
#         Parameters
#         ----------
#         n_neighbors: float (optional, default 15)
#             The size of local neighborhood (in terms of number of neighboring
#             sample points) used for manifold approximation. Larger values
#             result in more global views of the manifold, while smaller
#             values result in more local data being preserved. In general
#             values should be in the range 2 to 100.
#         n_components: int (optional, default 2)
#             The dimension of the space to embed into. This defaults to 2 to
#             provide easy visualization, but can reasonably be set to any
#             integer value in the range 2 to 100.
#         metric: string or function (optional, default 'euclidean')
#             The metric to use to compute distances in high dimensional space.
#             If a string is passed it must match a valid predefined metric. If
#             a general metric is required a function that takes two 1d arrays and
#             returns a float can be provided. For performance purposes it is
#             required that this be a numba jit'd function. Valid string metrics
#             include:
#                 * euclidean
#                 * manhattan
#                 * chebyshev
#                 * minkowski
#                 * canberra
#                 * braycurtis
#                 * mahalanobis
#                 * wminkowski
#                 * seuclidean
#                 * cosine
#                 * correlation
#                 * haversine
#                 * hamming
#                 * jaccard
#                 * dice
#                 * russelrao
#                 * kulsinski
#                 * ll_dirichlet
#                 * hellinger
#                 * rogerstanimoto
#                 * sokalmichener
#                 * sokalsneath
#                 * yule
#             Metrics that take arguments (such as minkowski, mahalanobis etc.)
#             can have arguments passed via the metric_kwds dictionary. At this
#             time care must be taken and dictionary elements must be ordered
#             appropriately; this will hopefully be fixed in the future.
#         n_epochs: int (optional, default None)
#             The number of training epochs to be used in optimizing the
#             low dimensional embedding. Larger values result in more accurate
#             embeddings. If None is specified a value will be selected based on
#             the size of the input dataset (200 for large datasets, 500 for small).
#         learning_rate: float (optional, default 1.0)
#             The initial learning rate for the embedding optimization.
#         init: string (optional, default 'spectral')
#             How to initialize the low dimensional embedding. Options are:
#                 * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
#                 * 'random': assign initial embedding positions at random.
#                 * A numpy array of initial embedding positions.
#         min_dist: float (optional, default 0.1)
#             The effective minimum distance between embedded points. Smaller values
#             will result in a more clustered/clumped embedding where nearby points
#             on the manifold are drawn closer together, while larger values will
#             result on a more even dispersal of points. The value should be set
#             relative to the ``spread`` value, which determines the scale at which
#             embedded points will be spread out.
#         spread: float (optional, default 1.0)
#             The effective scale of embedded points. In combination with ``min_dist``
#             this determines how clustered/clumped the embedded points are.
#         low_memory: bool (optional, default True)
#             For some datasets the nearest neighbor computation can consume a lot of
#             memory. If you find that UMAP is failing due to memory constraints
#             consider setting this option to True. This approach is more
#             computationally expensive, but avoids excessive memory use.
#         set_op_mix_ratio: float (optional, default 1.0)
#             Interpolate between (fuzzy) union and intersection as the set operation
#             used to combine local fuzzy simplicial sets to obtain a global fuzzy
#             simplicial sets. Both fuzzy set operations use the product t-norm.
#             The value of this parameter should be between 0.0 and 1.0; a value of
#             1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
#             intersection.
#         local_connectivity: int (optional, default 1)
#             The local connectivity required -- i.e. the number of nearest
#             neighbors that should be assumed to be connected at a local level.
#             The higher this value the more connected the manifold becomes
#             locally. In practice this should be not more than the local intrinsic
#             dimension of the manifold.
#         repulsion_strength: float (optional, default 1.0)
#             Weighting applied to negative samples in low dimensional embedding
#             optimization. Values higher than one will result in greater weight
#             being given to negative samples.
#         negative_sample_rate: int (optional, default 5)
#             The number of negative samples to select per positive sample
#             in the optimization process. Increasing this value will result
#             in greater repulsive force being applied, greater optimization
#             cost, but slightly more accuracy.
#         transform_queue_size: float (optional, default 4.0)
#             For transform operations (embedding new points using a trained model_
#             this will control how aggressively to search for nearest neighbors.
#             Larger values will result in slower performance but more accurate
#             nearest neighbor evaluation.
#         a: float (optional, default None)
#             More specific parameters controlling the embedding. If None these
#             values are set automatically as determined by ``min_dist`` and
#             ``spread``.
#         b: float (optional, default None)
#             More specific parameters controlling the embedding. If None these
#             values are set automatically as determined by ``min_dist`` and
#             ``spread``.
#         random_state: int, RandomState instance or None, optional (default: None)
#             If int, random_state is the seed used by the random number generator;
#             If RandomState instance, random_state is the random number generator;
#             If None, the random number generator is the RandomState instance used
#             by `np.random`.
#         metric_kwds: dict (optional, default None)
#             Arguments to pass on to the metric, such as the ``p`` value for
#             Minkowski distance. If None then no arguments are passed on.
#         angular_rp_forest: bool (optional, default False)
#             Whether to use an angular random projection forest to initialise
#             the approximate nearest neighbor search. This can be faster, but is
#             mostly on useful for metric that use an angular style distance such
#             as cosine, correlation etc. In the case of those metrics angular forests
#             will be chosen automatically.
#         target_n_neighbors: int (optional, default -1)
#             The number of nearest neighbors to use to construct the target simplcial
#             set. If set to -1 use the ``n_neighbors`` value.
#         target_metric: string or callable (optional, default 'categorical')
#             The metric used to measure distance for a target array is using supervised
#             dimension reduction. By default this is 'categorical' which will measure
#             distance in terms of whether categories match or are different. Furthermore,
#             if semi-supervised is required target values of -1 will be trated as
#             unlabelled under the 'categorical' metric. If the target array takes
#             continuous values (e.g. for a regression problem) then metric of 'l1'
#             or 'l2' is probably more appropriate.
#         target_metric_kwds: dict (optional, default None)
#             Keyword argument to pass to the target metric when performing
#             supervised dimension reduction. If None then no arguments are passed on.
#         target_weight: float (optional, default 0.5)
#             weighting factor between data topology and target topology. A value of
#             0.0 weights predominantly on data, a value of 1.0 places a strong emphasis on
#             target. The default of 0.5 balances the weighting equally between data and
#             target.
#         transform_seed: int (optional, default 42)
#             Random seed used for the stochastic aspects of the transform operation.
#             This ensures consistency in transform operations.
#         verbose: bool (optional, default False)
#             Controls verbosity of logging.
#         tqdm_kwds: dict (optional, defaul None)
#             Key word arguments to be used by the tqdm progress bar.
#         unique: bool (optional, default False)
#             Controls if the rows of your data should be uniqued before being
#             embedded.  If you have more duplicates than you have n_neighbour
#             you can have the identical data points lying in different regions of
#             your space.  It also violates the definition of a metric.
#             For to map from internal structures back to your data use the variable
#             _unique_inverse_.
#         densmap: bool (optional, default False)
#             Specifies whether the density-augmented objective of densMAP
#             should be used for optimization. Turning on this option generates
#             an embedding where the local densities are encouraged to be correlated
#             with those in the original space. Parameters below with the prefix 'dens'
#             further control the behavior of this extension.
#         dens_lambda: float (optional, default 2.0)
#             Controls the regularization weight of the density correlation term
#             in densMAP. Higher values prioritize density preservation over the
#             UMAP objective, and vice versa for values closer to zero. Setting this
#             parameter to zero is equivalent to running the original UMAP algorithm.
#         dens_frac: float (optional, default 0.3)
#             Controls the fraction of epochs (between 0 and 1) where the
#             density-augmented objective is used in densMAP. The first
#             (1 - dens_frac) fraction of epochs optimize the original UMAP objective
#             before introducing the density correlation term.
#         dens_var_shift: float (optional, default 0.1)
#             A small constant added to the variance of local radii in the
#             embedding when calculating the density correlation objective to
#             prevent numerical instability from dividing by a small number
#         output_dens: float (optional, default False)
#             Determines whether the local radii of the final embedding (an inverse
#             measure of local density) are computed and returned in addition to
#             the embedding. If set to True, local radii of the original data
#             are also included in the output for comparison; the output is a tuple
#             (embedding, original local radii, embedding local radii). This option
#             can also be used when densmap=False to calculate the densities for
#             UMAP embeddings.
#         disconnection_distance: float (optional, default np.inf or maximal value for bounded distances)
#             Disconnect any vertices of distance greater than or equal to disconnection_distance when approximating the
#             manifold via our k-nn graph. This is particularly useful in the case that you have a bounded metric.  The
#             UMAP assumption that we have a connected manifold can be problematic when you have points that are maximally
#             different from all the rest of your data.  The connected manifold assumption will make such points have perfect
#             similarity to a random set of other points.  Too many such points will artificially connect your space.
#         precomputed_knn: tuple (optional, default (None,None,None))
#             If the k-nearest neighbors of each point has already been calculated you
#             can pass them in here to save computation time. The number of nearest
#             neighbors in the precomputed_knn must be greater or equal to the
#             n_neighbors parameter. This should be a tuple containing the output
#             of the nearest_neighbors() function or attributes from a previously fit
#             UMAP object; (knn_indices, knn_dists,knn_search_index).
#
#     - - - INFORMATION - - -
#     https://umap-learn.readthedocs.io/en/latest/basic_usage.html
#     '''


# fun = UMAP(a=None,
#            angular_rp_forest=False,
#            b=None,
#            force_approximation_algorithm=False,
#            init='spectral',
#            learning_rate=1.0,
#            local_connectivity=1.0,
#            low_memory=False,
#            metric='euclidean',
#            metric_kwds=None,
#            min_dist=0.1,
#            n_components=None,
#            n_epochs=None,
#            n_neighbors=15,
#            negative_sample_rate=5,
#            output_metric='euclidean',
#            output_metric_kwds=None,
#            random_state=42,
#            repulsion_strength=1.0,
#            set_op_mix_ratio=1.0,
#            spread=1.0,
#            target_metric='categorical',
#            target_metric_kwds=None,
#            target_n_neighbors=-1,
#            target_weight=0.5,
#            transform_queue_size=4.0,
#            transform_seed=42,
#            unique=False,
#            verbose=False
#            )
# hyperpars = {
#              'n_neighbors': [2, 100], # default: 30
#              'min_dist': [0.01, 1.0], # default: auto
#              'metric': [0, 2], # 'minkowski','manhattan', default: 'euclidean'
#
#              # 'min_grad_norm': [1e-09, 0.1], # default: 1e-07
#              # 'n_iter': [250, 2000], # default: 1000
#              # 'n_iter_without_progress': [0, 9], # default: 300 # checked every 50 iterations
#              # 'init': [0, 1], # ‘random’, ‘pca’
#              # 'angle': [0.2, 0.8], # FALSE, TRUE
#              #
#              # 'preprocess': [0, 5], # not neccessary, data are scaled
#             }
# params = fun.get_params()
# return fun, params, hyperpars


# def dictionary_learning(self) -> (object, dict, dict):
#     '''
#     NOT USED: slower version of minibatch_dictionary_learning
#     - - - INFORMATION - - -
#     https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
#     '''
#     fun = DictionaryLearning(n_components=None,
#                              alpha=,
#                              max_iter=1000,
#                              tol=1e-08,
#                              fit_algorithm='lars',
#                              transform_algorithm='omp',
#                              transform_n_nonzero_coefs=None,
#                              transform_alpha=None,
#                              n_jobs=None,
#                              code_init=None,
#                              dict_init=None,
#                              verbose=False,
#                              split_sign=False,
#                              random_state=None,
#                              positive_code=False,
#                              positive_dict=False,
#                              transform_max_iter=1000)
#     string = str(' alpha='+str(alpha) )
#     self.printout(fun_name='dictionary-learning', n=n, string=string)
#     return fun.fit_transform(self.data), string



# def dictionary_learning_online(self, X) -> (object, dict, dict):
#     '''
#     # NOT USED
#     - - - DESCRIPTION - - -
#     Solves a dictionary learning matrix factorization problem online.
#     Finds the best dictionary and the corresponding sparse code for approximating the data matrix X by solving:
#     (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
#     (U,V) with || V_k ||_2 = 1 for all  0 <= k < n_components
#     where V is the dictionary and U is the sparse code. ||.||_Fro stands for the Frobenius norm and ||.||_1,1
#     stands for the entry-wise matrix norm which is the sum of the absolute values of all the entries in the matrix.
#     This is accomplished by repeatedly iterating over mini-batches by slicing the input data.
#     '''
#     fun = dict_learning_online(X, n_components=None,
#                                alpha=1,
#                                n_iter=100,
#                                return_code=True,
#                                dict_init=None,
#                                callback=None,
#                                batch_size=3,
#                                verbose=False,
#                                shuffle=True,
#                                n_jobs=None,
#                                method='lars',
#                                iter_offset=0,
#                                random_state=None,
#                                return_inner_stats=False,
#                                inner_stats=None,
#                                return_n_iter=False,
#                                positive_dict=False,
#                                positive_code=False,
#                                method_max_iter=1000)
#     string = str(' default parameters' )
#     self.printout(fun_name='dictionary-learning-online', n=n, string=string)
#     return fun.fit_transform(self.data), string



# def latent_ditrichlet_allocation(self):
#         '''
#         !!! TEXTVERARBEITUNG
#         - - - DESCRIPTION - - -
#         It is one of the most popular topic modeling methods. Each document is made up of various words, and
#         each topic also has various words belonging to it. The aim of LDA is to find topics a document
#         belongs to, based on the words in it.
#         ... is a generative probabilistic model for collections of discrete dataset such as text corpora.
#         It is also a topic model that is used for discovering abstract topics from a collection of documents.
#         The graphical model of LDA is a three-level generative model.
#         In natural language processing, the latent Dirichlet allocation (LDA) is a generative statistical model
#         that allows sets of observations to be explained by unobserved groups that explain why some parts of the
#         data are similar. For example, if observations are words collected into documents, it posits that each
#         document is a mixture of a small number of topics and that each word's presence is attributable to one
#         of the document's topic discovery for example: dogs = dog, spaniel, beagle...
#         USE: text topic discovery, genetics allele identification
#
#         - - - PARAMETERS - - -
#         n_components : int, default=10
#             Number of topics.
#             .. versionchanged:: 0.19
#                 ``n_topics`` was renamed to ``n_components``
#         doc_topic_prior : float, default=None
#             Prior of document topic distribution `theta`. If the value is None,
#             defaults to `1 / n_components`.
#             In [1]_, this is called `alpha`.
#         topic_word_prior : float, default=None
#             Prior of topic word distribution `beta`. If the value is None, defaults
#             to `1 / n_components`.
#             In [1]_, this is called `eta`.
#         learning_method : {'batch', 'online'}, default='batch'
#             Method used to update `_component`. Only used in :meth:`fit` method.
#             In general, if the data size is large, the online update will be much
#             faster than the batch update.
#             Valid options::
#                 'batch': Batch variational Bayes method. Use all training data in
#                     each EM update.
#                     Old `components_` will be overwritten in each iteration.
#                 'online': Online variational Bayes method. In each EM update, use
#                     mini-batch of training data to update the ``components_``
#                     variable incrementally. The learning rate is controlled by the
#                     ``learning_decay`` and the ``learning_offset`` parameters.
#             .. versionchanged:: 0.20
#                 The default learning method is now ``"batch"``.
#         learning_decay : float, default=0.7
#             It is a parameter that control learning rate in the online learning
#             method. The value should be set between (0.5, 1.0] to guarantee
#             asymptotic convergence. When the value is 0.0 and batch_size is
#             ``n_samples``, the update method is same as batch learning. In the
#             literature, this is called kappa.
#         learning_offset : float, default=10.0
#             A (positive) parameter that downweights early iterations in online
#             learning.  It should be greater than 1.0. In the literature, this is
#             called tau_0.
#         max_iter : int, default=10
#             The maximum number of passes over the training data (aka epochs).
#             It only impacts the behavior in the :meth:`fit` method, and not the
#             :meth:`partial_fit` method.
#         batch_size : int, default=128
#             Number of documents to use in each EM iteration. Only used in online
#             learning.
#         evaluate_every : int, default=-1
#             How often to evaluate perplexity. Only used in `fit` method.
#             set it to 0 or negative number to not evaluate perplexity in
#             training at all. Evaluating perplexity can help you check convergence
#             in training process, but it will also increase total training time.
#             Evaluating perplexity in every iteration might increase training time
#             up to two-fold.
#         total_samples : int, default=1e6
#             Total number of documents. Only used in the :meth:`partial_fit` method.
#         perp_tol : float, default=1e-1
#             Perplexity tolerance in batch learning. Only used when
#             ``evaluate_every`` is greater than 0.
#         mean_change_tol : float, default=1e-3
#             Stopping tolerance for updating document topic distribution in E-step.
#         max_doc_update_iter : int, default=100
#             Max number of iterations for updating document topic distribution in
#             the E-step.
#         n_jobs : int, default=None
#             The number of jobs to use in the E-step.
#             ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#             ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#             for more details.
#         verbose : int, default=0
#             Verbosity level.
#         random_state : int, RandomState instance or None, default=None
#             Pass an int for reproducible results across multiple function calls.
#             See :term:`Glossary <random_state>`.
#
#         - - - INFORMATION - - -
#         https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
#         '''
#         data = self.positive_scale(self.data)
#         fun = LatentDirichletAllocation(n_components=None,
#                                         doc_topic_prior=None,
#                                         topic_word_prior=None,
#                                         learning_method='batch',
#                                         learning_decay=0.7,
#                                         learning_offset=10.0,
#                                         max_iter=10,
#                                         batch_size=128,
#                                         evaluate_every=- 1,
#                                         total_samples=1000000.0,
#                                         perp_tol=0.1,
#                                         mean_change_tol=0.001,
#                                         max_doc_update_iter=100,
#                                         n_jobs=None,
#                                         verbose=0,
#                                         random_state=None)
#         string = str(' data scaled + positive, default parameters' )
#         self.printout(fun_name='latent-dirichlet-allocation', n=n, string=string)
#         return fun.fit_transform(data), string


# def mds(self, n, metric=True) -> (object, dict, dict):
#         '''
#         !!!!!!!! NOT USED, TAKES TOO LONG, R functions used instead
#
#         - - - DESCRIPTION - - -
#         Multidimensional scaling (MDS) seeks a low-dimensional representation of the data in which the distances
#         respect well the distances in the original high-dimensional space.
#         In general, MDS is a technique used for analyzing similarity or dissimilarity data. It attempts to model
#         similarity or dissimilarity data as distances in a geometric spaces. The data can be ratings of similarity
#         between objects, interaction frequencies of molecules, or trade indices between countries.
#         There exists two types of MDS algorithm: metric and non metric. In the scikit-learn, the class MDS implements
#         both. In Metric MDS, the input similarity matrix arises from a metric (and thus respects the triangular inequality),
#         the distances between output two points are then set to be as close as possible to the similarity or dissimilarity data.
#         In the non-metric version, the algorithms will try to preserve the order of the distances, and hence seek for a
#         monotonic relationship between the distances in the embedded space and the similarities/dissimilarities.
#
#         - - - PARAMETERS - - -
#         n_components int, default=2
#             Number of dimensions in which to immerse the dissimilarities.
#         metric bool, default=True
#             If True, perform metric MDS; otherwise, perform nonmetric MDS.
#         n_init int, default=4
#             Number of times the SMACOF algorithm will be run with different initializations.
#             The final results will be the best output of the runs, determined by the run with the smallest final stress.
#         max_iter int, default=300
#             Maximum number of iterations of the SMACOF algorithm for a single run.
#         verbose int, default=0
#             Level of verbosity.
#         eps float, default=1e-3
#             Relative tolerance with respect to stress at which to declare convergence.
#         n_jobs int, default=None
#             The number of jobs to use for the computation. If multiple initializations are used (n_init),
#             each run of the algorithm is computed in parallel.
#             None means 1 unless in a joblib. parallel_backend context. -1 means using all processors.
#         random_state int, RandomState instance or None, default=None
#             Determines the random number generator used to initialize the centers. Pass an int for reproducible results across
#             multiple function calls. See Glossary.
#         dissimilarity{‘euclidean’, ‘precomputed’}, default=’euclidean’
#             ‘euclidean’: Pairwise Euclidean distances between points in the dataset.
#             ‘precomputed’: Pre-computed dissimilarities are passed directly to fit and fit_transform.
#
#         - - - INFORMATION - - -
#         https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
#
#         USE: no idea. it takes, too long >10min -> use R functions
#         '''
#         fun = MDS(n_components=None,
#                   metric=metric,
#                   n_init=4,
#                   max_iter=300,
#                   verbose=0,
#                   eps=0.001,
#                   n_jobs=None,
#                   random_state=None,
#                   dissimilarity='euclidean')
#         string = str(' default parameters' )
#         self.printout(fun_name='multidimensional-scaling', n=n, string=string)
#         return fun.fit_transform(self.data), string



# def som(self, n, n_=10, m_=10) -> (object, dict, dict):
#         '''
#          - - - DESCRIPTION - - -
#         - - - PARAMETERS - - -
#         - - - INFORMATION - - -
#         Self-organizing maps use the unsupervised learning to create a map or a mask for
#         the input data. They provide an elegant solution for large or difficult to interpret
#         data sets. Because of this high adaptivity, they found application in many fields and
#         are in general mostly used for classification. Initially, Kohonen used them for speech
#         recognition, but today they are also used in Bibliographic classification, Image browsing
#         systems and Image classification, Medical Diagnosis, Data compression and so on.
#         MORE: https://pypi.org/project/sklearn-som/
#         m_ = 10
#         n_ = 10 # n_ x m_ segments = 100
#         n = ndim
#         '''
#         fun = SOM(m=m_, n=n_, dim=n)
#         string = str(' n_=' + str(n_) + ' m_=' + str(m_) )
#         self.printout(fun_name='som', n=n, string=string)
#         return fun.fit_transform(self.data), string


# def quadratic_discriminant_analysis(self) -> (object, dict, dict):
#     '''
#     A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data
#     and using Bayes’ rule. The model fits a Gaussian density to each class.
#     MORE: https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda
#
#     :return:
#     '''
#     fun = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
#     print('QuadraticDiscriminantAnalysis n_components:', n)
#     return fun.fit_transform(self.data)



