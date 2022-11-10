from sklearn.decomposition import NMF, FastICA, FactorAnalysis, TruncatedSVD, PCA, \
    IncrementalPCA, SparsePCA, MiniBatchSparsePCA
from sklearn.manifold import MDS

from .py_crca import Crca
from .py_rpcag import rpca
from .py_sammon.py_sammon import Sammon

# TODO: specify returns

class Dimred_functions_python:
    '''
    dimensionality reduction functions from python sklearn.decomposition.
    Those and other functions were tested with several datasets in two steps:
    1) start with parameter ranges:
    - we performed a hyperparameter optimization with full range of values and
    tested for accuracy and speed.
    2) update 20221031:
    - we reduced the hyperparameter range, updated the default hyperparameters and/or
    ereased the hyperparameter from the list (not optimized during hyperparameter optimization).
    The later was the case for most hyperparameters, they had no or only marginal effects on loss
    and/or made the dim reduction very slow when set to a certain value.
    In this case we set a dummy variable.
    '''

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols


    def py_factor_analysis(self)-> (object, dict, dict):
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
             tol=0.01,
             copy=True,
             max_iter=50, # updated: default=1000,
             noise_variance_init=None, # custom noise variances for each feature.
             svd_method='randomized',  # ‘lapack’, default: ‘randomized’
             iterated_power=0, # updated: default = 3
             random_state=42
        )

        hyperpars = {
        # 1) start with parameter ranges:
        #      'max_iter': [5,1000], # maximum iterations
        #      'tol': [0.001, 0.1], # tolerance for expectation–maximization algorithm (log-likelihood)
        #      'svd_method': [0,1], # randomized best
        #      'iterated_power': [0, 10], # default: 0: best; only for: 'randomized'

        # 2) update 20221031: loss: 0.875; speed: ok; reduce as much as possible
            'tol': [0.05, 0.1],  # tolerance for expectation–maximization algorithm (log-likelihood)
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def py_fast_ica(self)-> (object, dict, dict):
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
              algorithm='parallel',
              whiten=True, # !!! not needed but if not set to True: n_components will be set to n_features
              # XXX functional form of the G function used in the approximation to neg-entropy
              fun='cube',  # 20221031 updated from default: 'logcosh', exp and cube are faster
              fun_args=None,
              max_iter=200,
              tol=0.0001,
              w_init=None, # custom: The mixing matrix to be used to initialize the algorithm.
              random_state=42
        )
        hyperpars = {
            # 1) start with parameter ranges:
            #  'algorithm': [0, 1], # fastica algorithm: ‘parallel’ (default), ‘deflation’
            #  'fun': [0, 2], # 'logcosh','exp','cube'
            #  'max_iter': [5, 1000], # maximum iterations
            # 'tol': [1e-8, 0.1], # Tolerance on update at each iteration.
            # 2) update 20221031: loss: 0.875; speed: ok; reduce as much as possible
            'whiten': True # dummy variable
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def py_non_negative_matrix_factorization(self) -> (object, dict, dict):
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
             'nndsvda': NNDSVD with zeros filled with the average of X (better when sparsity is not desired) # removed
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
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF
        '''
        fun = NMF(
              n_components=None,
              init=None, # None: ‘nndsvd’ if n_components <= min(n_samples, n_features), otherwise random. ‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’
              solver='cd', # Numerical solver 'cd' oordinate descent ,'mu' multuplicative update
              beta_loss='frobenius', # distance between X and W*H ‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’ (kl slow)
              tol=0.05, # updated from: default: 0.0001,
              max_iter=200,
              random_state=42,
              alpha=1,
              l1_ratio=0.0,
              verbose=0,
              shuffle=False
              )
        hyperpars = {
            ## 1) start with parameter ranges:
             # 'init': [0, 3], # ‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’
             # 'solver': [0, 1], # ‘cd’, ‘mu’
             # 'beta_loss': [0, 1], # ‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’
             # 'tol': [1e-6, 0.1], #
             # 'max_iter': [5, 1000], #
             # 'alpha': [0.1, 10], # from v1.0 on substituted by alpha_w, alpha_h
             # 'l1_ratio': [0.0, 1.0], # For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

            ## 2) update 20221031: loss: 0.956; speed: fast; 52steps! reduce as much as possible
                'init': [0, 2],  #
                'alpha': [0.1, 10],  #
                'l1_ratio': [0.0, 1.0],  #
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def py_pca(self) -> (object, dict, dict):
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
             # 1) start with parameter ranges: the hyperparameters have no effect on loss
             # 'empty_py': [0]
             # 'svd_solver': [0, 2], # no effect on loss ‘full’, ‘arpack’, ‘randomized’, default=’auto’
             # 'tol': [0.0, 100.0], # no effect on loss !arpack only [0.0, infinity]
             # 'iterated_power': [0, 100], # no effect on loss !randomized only

             # 2) update 20221031:
             'whiten': False # dummy variables
             }
        params = fun.get_params()
        return fun, params, hyperpars


    def py_pca_incremental(self)-> (object, dict, dict):
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
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
        '''
        fun = IncrementalPCA(
             n_components=None,
             whiten=False,
             copy=True,
             batch_size=None # 5 * n_features, provides a balance between accuracy and memory use.
        )
        hyperpars = {
                    'whiten': False # dummy variable
        }
        params = fun.get_params()
        return fun, params, hyperpars



    def py_pca_sparse(self) -> (object, dict, dict):
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
            alpha = 0.1, # default: 1, 0.1 much better // controls sparsity, higher: more sparse
            ridge_alpha=0.01, # L2 Regularization, adds “squared magnitude” of coefficient as penalty term to the loss function
            max_iter=200, # updated: default: 1000, 200 faster
            tol=0.05, # updated: default: 1e-08, 0.05 much faster and similar results
            method='cd', # updated: default: 'lars', cd much faster
            n_jobs=None, # loadings for warm restart scenarios
            U_init=None, # loadings for warm restart scenarios
            V_init=None, # loadings for warm restart scenarios
            verbose=False,
            random_state=42
        )
        hyperpars = {
        # 1) start with parameter ranges:
        #      'alpha': [0.0001, 1], # no effect, remove and set to default
        #      'ridge_alpha': [0.0001, 10], # no effect on loss, remove and set to default
        #      'tol': [0.001, 0.1], # no effect on loss but higher values much faster
        #      'max_iter': [5,1000], # smaller values are much faster
        #      'method': [0,1], # no effect on loss ‘lars’, ‘cd’

        # 2) update 20221031: loss: 0.991 speed: ok, reduce hyperparameters as much as possible
            'random_state': 42 # dummy variable
        }
        params = fun.get_params()
        return fun, params, hyperpars


    def py_pca_sparse_mini_batch(self) -> (object, dict, dict):
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
             n_iter=0, # default: 100
             callback=None,
             batch_size=3,
             verbose=False,
             shuffle=True,
             n_jobs=None,
             method='cd',  # default: 'lars', cd much faster
             random_state=42
        )
        hyperpars = {
            # 1) start with parameter ranges:
            #  'alpha': [0.1, 10], # no effect on loss, remove and set to default
            #  'ridge_alpha': [0.01, 10], # no effect on loss, remove and set to default
            #  'n_iter': [0,1000], # effect on speed and loss, as smaller as better
            #  'method': [0,1], # no effect on loss, but cd much faster

            # 2) update 20221031:  loss: 0.991 speed: ok, reduce carefull
            'random_state': 42 # dummy variable
        }
        params = fun.get_params()
        return fun, params, hyperpars


    def py_truncated_svd(self) -> (object, dict, dict):
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
            algorithm='arpack', # update default: 'randomized', arpack much faster
            n_iter=5, # default 5
            random_state=42,
            tol=0.07 # update default: 0.0 but 0.07 fastest option, no effect on loss
        )
        hyperpars = {
            # 1) start with parameter ranges:
            # 'algorithm': [0, 1], # no effect on loss, arpack much faster
            # 'n_iter': [5, 1000],# no effect on loss, as smaller as faster
            # 'tol': [0, 0.1],  #  only for arpack, no effect on loss

            # 2) update 20221031: loss: 0.991 speed: fast, reduce as much as possible
             'random_state': 42,  # dummy variable
        }
        params = fun.get_params()
        return fun, params, hyperpars


    # TODO: is currently being implemented
    # def py_mds(self) -> (object, dict, dict):
    #     '''
    #     !!!!!!!! NOT USED, TAKES TOO LONG, R functions used instead
    #
    #     - - - DESCRIPTION - - -
    #     Multidimensional scaling (MDS) seeks a low-dimensional representation of the data in which the distances
    #     respect well the distances in the original high-dimensional space.
    #     In general, MDS is a technique used for analyzing similarity or dissimilarity data. It attempts to model
    #     similarity or dissimilarity data as distances in a geometric spaces. The data can be ratings of similarity
    #     between objects, interaction frequencies of molecules, or trade indices between countries.
    #     There exists two types of MDS algorithm: metric and non metric. In the scikit-learn, the class MDS implements
    #     both. In Metric MDS, the input similarity matrix arises from a metric (and thus respects the triangular inequality),
    #     the distances between output two points are then set to be as close as possible to the similarity or dissimilarity data.
    #     In the non-metric version, the algorithms will try to preserve the order of the distances, and hence seek for a
    #     monotonic relationship between the distances in the embedded space and the similarities/dissimilarities.
    #
    #     - - - PARAMETERS - - -
    #     n_components int, default=2
    #         Number of dimensions in which to immerse the dissimilarities.
    #     metric bool, default=True
    #         If True, perform metric MDS; otherwise, perform nonmetric MDS.
    #     n_init int, default=4
    #         Number of times the SMACOF algorithm will be run with different initializations.
    #         The final results will be the best output of the runs, determined by the run with the smallest final stress.
    #     max_iter int, default=300
    #         Maximum number of iterations of the SMACOF algorithm for a single run.
    #     verbose int, default=0
    #         Level of verbosity.
    #     eps float, default=1e-3
    #         Relative tolerance with respect to stress at which to declare convergence.
    #     n_jobs int, default=None
    #         The number of jobs to use for the computation. If multiple initializations are used (n_init),
    #         each run of the algorithm is computed in parallel.
    #         None means 1 unless in a joblib. parallel_backend context. -1 means using all processors.
    #     random_state int, RandomState instance or None, default=None
    #         Determines the random number generator used to initialize the centers. Pass an int for reproducible results across
    #         multiple function calls. See Glossary.
    #     dissimilarity{‘euclidean’, ‘precomputed’}, default=’euclidean’
    #         ‘euclidean’: Pairwise Euclidean distances between points in the dataset.
    #         ‘precomputed’: Pre-computed dissimilarities are passed directly to fit and fit_transform.
    #
    #     - - - INFORMATION - - -
    #     https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
    #
    #     USE: no idea. it takes, too long >10min -> use R functions
    #     '''
    #     fun = MDS(n_components=None,
    #               metric=True,
    #               n_init=4,
    #               max_iter=300,
    #               verbose=0,
    #               eps=0.001,
    #               n_jobs=None,
    #               random_state=None,
    #               dissimilarity='euclidean'
    #     )
    #
    #     hyperpars = {
    #         # 1) start with parameter ranges:
    #                 'metric': [True,False],
    #                 'n_init': [1,10],
    #                 'max_iter': [5,500],
    #                 'eps': [0.00001,0.5],
    #         # 2) update 20221031:
    #     }
    #     params = fun.get_params()
    #     return fun, params, hyperpars



    def py_crca(self) -> (object, dict, dict):
        '''
        Curvilinear Component Analysis implementation for Python.
        Implemented from:
        https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py
        Author : Felipe Augusto Machado

        - - - DESCRIPTION - - -
        This code was implemented using the following article https://ieeexplore.ieee.org/document/554199
        and the book "Nonlinear Dimensionality Reduction" by Michel Verleysen, John A. R. Lee.
        The Algorithm
        Curvilinear Component Analysis (CCA) is a Non-linear Dimensionality Reduction technic, basead on the
        distance betweens the points: it tries to create a new space, with a small dimension, which the distance
        betweens points are equal in the original space (if this distance is smaller than Lambda, one of the
        parameters of the algorithm).

        - - - INFORMATION - - -
        https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py

        :return:
        '''

        fun = Crca(
            n_components = None,
            lmbd = 1,
            alpha = 1,
            max_iter = 5,
            tol = 1e-6
        )

        hyperpars = {
            # 1) start with parameter ranges (results are from R implementation, which runs much slower)
            #      'alpha': [0.1, 10.0], # no effect on loss
            #      'lambda': [0.1, 10.0],  # no effect on loss, small values (default) faster
            #      'max_iter': [1, 5], # smaller values better on loss and speed (high values take hours)
            #      'tol': [1e-08, 0.1], # no effect on loss, small values (default) faster
            'lmbd': 1
        }
        params = fun.get_params()
        return fun, params, hyperpars

    # TODO: is currently being implemented
    # def py_sammon(self) -> (object, dict, dict):
    #     '''
    #
    #     - - - DESCRIPTION - - -
    #
    #     - - - PARAMETERS - - -
    #
    #     - - - INFORMATION - - -
    #
    #     '''
    #     fun = Sammon(
    #         n_components=None,
    #         max_iter = 500,
    #         inputdist = 'raw',
    #         maxhalves = 20,
    #         tolfun = 1e-9,
    #         init = 'default'
    #     )
    #
    #     hyperpars = {
    #         # 1) start with parameter ranges:
    #                 'max_iter': [5,500],
    #                 'inputdist': ['raw',],
    #                 'maxhalves': [20,],
    #                 'tolfun': [1e-9,],
    #                 'init': ['default',]
    #         # 2) update 20221031:
    #     }
    #     params = fun.get_params()
    #     return fun, params, hyperpars

    # TODO: is currently being implemented
    # def py_rpca(self) -> (object, dict, dict):
    #     '''
    #     Curvilinear Component Analysis implementation for Python.
    #     Implemented from:
    #     https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py
    #     Author : Felipe Augusto Machado
    #
    #     - - - DESCRIPTION - - -
    #     This code was implemented using the following article https://ieeexplore.ieee.org/document/554199
    #     and the book "Nonlinear Dimensionality Reduction" by Michel Verleysen, John A. R. Lee.
    #     The Algorithm
    #     Curvilinear Component Analysis (CCA) is a Non-linear Dimensionality Reduction technic, basead on the
    #     distance betweens the points: it tries to create a new space, with a small dimension, which the distance
    #     betweens points are equal in the original space (if this distance is smaller than Lambda, one of the
    #     parameters of the algorithm).
    #
    #     - - - INFORMATION - - -
    #     https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py
    #
    #     :return:
    #     '''
    #
    #     fun = rpca()
    #
    #     hyperpars = {
    #         # 1) start with parameter ranges (results are from R implementation, which runs much slower)
    #         #      'alpha': [0.1, 10.0], # no effect on loss
    #         #      'lambda': [0.1, 10.0],  # no effect on loss, small values (default) faster
    #         #      'max_iter': [1, 5], # smaller values better on loss and speed (high values take hours)
    #         #      'tol': [1e-08, 0.1], # no effect on loss, small values (default) faster
    #         'lmbd': 1
    #     }
    #     params = fun.get_params()
    #     return fun, params, hyperpars
