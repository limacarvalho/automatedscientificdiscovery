from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA, SparsePCA
from .py_fun_crca import Crca
from dim_reduction.Py_funs_dimred.py_fun_sammon import Sammon

# TODo reference papers?
class Dimred_functions_python:
    '''
    wrappers for dimensionality reduction functions mainly from python sklearn.decomposition.
    The wrapper contains information from the indicted resources and some documentation of
    the optimization of the hyperparameter tuning and comments.
    The wrapper returns:
    1) function calls with updated default hyperparameters and comments.
    2) Hyperparameters with the fun.get_params() 
    3) hyperparameter ranges and comments of test results on hyperparameter optimization:
        we have performed extensive testing of hyperparameter importance on quality and speed.
        - all_hp == True: start conditions with evaluating most available hyperparameters and a 
            broad numerical range or range of options for each hyperparameter.
            Then we evaluated accuracy and speed and updated the hyperparameters.
        - all_hp == False: updated hyperparameters.
    In case there are no hyperparameters to tune we set a dummy variable in order to run the script seamless.
    '''
    def __init__(self, nrows: int, ncols: int, all_hp: bool=False):
        '''
        :param nrows: int,
            number of rows, important to set the limits for some hyperparmeters
        :param ncols: int,
            number of columns, important to set the limits for some hyperparmeters
        :param all_hp: bool, default=False
            use the full range of hyperparameters (True) or the optimized range (False)
        '''
        self.nrows = nrows
        self.ncols = ncols
        self.all_hp = all_hp


    # ---------------------------------------------------------------------
    def py_pca(self) -> (object, dict, dict):
        '''
         - - - DESCRIPTION - - -
        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space. The input data is centered
        but not scaled for each feature before applying the SVD.
        It uses the LAPACK implementation of the full SVD or a randomized truncated
        SVD by the method of Halko et al. 2009, depending on the shape of the input
        data and the number of components to extract.
        It can also use the scipy.sparse.linalg ARPACK implementation of the
        truncated SVD.
        Notice that this class does not support sparse input. See
        :class:`TruncatedSVD` for an alternative with sparse data.
        Read more in the :ref:`User Guide <PCA>`.

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

        :return: object, dict, dict
            object: function call
            dict: function hyperparameters (fun.get_params())
            dict: hyperparameter ranges for hyperparameter optimization

        - - - REPOSITORY - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        '''
        fun = PCA(
              n_components=None,
              copy=True,
              whiten=False,
              svd_solver='arpack', # updated from 'auto'
              tol=0.01, # default: 0
              iterated_power=1, # default: 'auto'
              random_state=42
        )
        if self.all_hp:
            hyperpars = {
                 # 1) start with parameter ranges:
                 'svd_solver': [0, 2], # # no effect on loss, arpack much faster
                 'tol': [0.0, 100.0], # # no effect on loss, lower values seem faster
                 'iterated_power': [0, 100], # no effect on loss, as higher as slower
            }
        else:
            hyperpars = {
                 # 2) updated hyperparameters
                 'whiten': False # dummy variables
             }
        params = fun.get_params()
        return fun, params, hyperpars


    # ---------------------------------------------------------------------
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

        :return: object, dict, dict
            object: function call
            dict: function hyperparameters (fun.get_params())
            dict: dummy variable instead of hyperparameter ranges

        - - - REPOSITORY - - -
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


    # ---------------------------------------------------------------------
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

        :return: object, dict, dict,
            object: function call
            dict: function hyperparameters (fun.get_params())
            dict: hyperparameter ranges for hyperparameter optimization

        - - - REPOSITORY - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
        '''
        fun = SparsePCA(
            n_components=None,
            alpha = 0.1, # default: 1, 0.1 much better // controls sparsity, higher: more sparse
            ridge_alpha=0.01, # L2 Regularization, adds “squared magnitude” of coefficient as penalty term to the loss function
            max_iter=200, # updated: default: 1000, 200 faster
            tol=0.05, # updated: default: 1e-08, 0.05 much faster and similar results
            method='cd', # updated: default: 'lars', cd much faster and seems equally accurate
            n_jobs=None,
            U_init=None, # array, loadings for warm restart scenarios
            V_init=None, # array, loadings for warm restart scenarios
            verbose=False,
            random_state=42
        )
        if self.all_hp:
            # 1) start with broad parameter ranges
            hyperpars = {
                 'alpha': [0.0001, 1], # smaller values are better and much faster
                 'ridge_alpha': [0.0001, 10], # higher values seem better but slower
                 'tol': [0.001, 0.1], # no effect on loss but higher values much faster, skip
                 'max_iter': [5,1000], # seems to have no effect on loss, lower values are faster, skip
                 'method': [0,1], # seems to have no effect on loss, lower values are faster
            }
        else:
            # updated hyperparameter ranges
            hyperpars = {
                'alpha': [0.0001, 0.1],
                'ridge_alpha': [0.1, 10]
        }
        params = fun.get_params()
        return fun, params, hyperpars


    # ---------------------------------------------------------------------
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

        :return object, dict, dict:
            object: function call
            dict: function hyperparameters (fun.get_params())
            dict: hyperparameter ranges for hyperparameter optimization

        - - - REPOSITORY - - -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#
        '''
        fun = TruncatedSVD(
            n_components=None,
            algorithm='arpack', # update default: 'randomized', arpack much faster
            n_iter=5, # no effect on loss but on speed
            random_state=42,
            tol=0.2 # update default: 0.0
        )

        if self.all_hp:
            # 1) start with broad parameter ranges
            hyperpars = {
                'algorithm': [0, 1], # no effect on loss, arpack much faster
                'n_iter': [5, 1000], # no effect on loss, huge effect on speed, as smaller as faster
                'tol': [0.0, 0.5],  #  no effect on loss, higher values seem faster
            }
        else:
            # 2) updated hyperparameters
            hyperpars = {
                'random_state': 42,  # dummy variable
        }
        params = fun.get_params()
        return fun, params, hyperpars



    # ---------------------------------------------------------------------
    def py_crca(self) -> (object, dict, dict):
        '''
        Curvilinear Component Analysis implementation for Python.
        Implemented from:
        https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py
        Author : Felipe Augusto Machado

        We also tested a R-implementation which run much slower.

        - - - DESCRIPTION - - -
        This code was implemented using the following article https://ieeexplore.ieee.org/document/554199
        and the book "Nonlinear Dimensionality Reduction" by Michel Verleysen, John A. R. Lee.
        The Algorithm
        Curvilinear Component Analysis (CCA) is a Non-linear Dimensionality Reduction technic, basead on the
        distance betweens the points: it tries to create a new space, with a small dimension, which the distance
        betweens points are equal in the original space (if this distance is smaller than Lambda, one of the
        parameters of the algorithm).

        Parameters
        ----------
        n_components : int
            The new dimension.
        lmbda : float
            Distance limit to update points. It decreases over time : lambda(t) = lambda/(t+1).
        alpha : float
            Learning rate. It decreases over time : alpha(t) = alpha/(t+1)
        max_iter : int (default = 10)
            Number of iterations. Each iteration run all points in 'data_high'.
        tol : float (default = 1e-4)
            Tolerance for the stopping criteria.

        :return: object, dict, dict
            object: function call
            dict: function hyperparameters (fun.get_params())
            dict: hyperparameter ranges

        - - - REPOSITORY - - -
        https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py
        '''
        fun = Crca(
            n_components = None,
            lmbda = 1, # min 0.1
            alpha = 0.2, # min 0.001
            max_iter = 10,
            tol = 1e-4
        )

        if self.all_hp:
            # 1) start with parameter ranges:
            hyperpars = {
                 'lmbda': [1, 100],  # each dataset seems to have a sweat spot, leave it like it is
                 'alpha': [0.001, 1], # each dataset seems to have a sweat spot, leave it like it is
                 'max_iter': [5, 20], # no effect on loss and speed, convergence is reached after few iterations. set to 10
                 'tol': [1e-12, 0.5], # preferred values: 0.1...0.5, no effects on speed
            }
        else:
            # 2) updated hyperparameters
            hyperpars = {
                'lmbda': [1, 100],
                'alpha': [0.001, 1],
                'tol': [0.1, 0.5]
        }
        params = fun.get_params()
        return fun, params, hyperpars


    # ---------------------------------------------------------------------
    def py_sammon(self) -> (object, dict, dict):
        '''
        - - - INFORMATION - - -
        Perform Sammon mapping on dataset x
        y = sammon(x) applies the Sammon nonlinear mapping procedure on
        multivariate data x, where each row represents a pattern and each column
        represents a feature.  On completion, y contains the corresponding
        co-ordinates of each point on the map.  By default, a two-dimensional
        map is created.  Note if x contains any duplicated rows, SAMMON will
        fail (ungracefully).
        [y,E] = sammon(x) also returns the value of the cost function in E (i.e.
        the stress of the mapping).
        An N-dimensional output map is generated by y = sammon(x,n) .
        A set of optimisation options can be specified using optional

        - - - PARAMETERS - - -
            arguments, y = sammon(x,n,[OPTS]):
           max_iter       - maximum number of iterations
           tolfun         - relative tolerance on objective function
           maxhalves      - maximum number of step halvings
           input          - {'raw','distance'} if set to 'distance', X is
                            interpreted as a matrix of pairwise distances.
           display        - 0 to 2. 0 least verbose, 2 max verbose.
           init           - {'pca', 'cmdscale', random', 'default'}
                            default is 'pca' if input is 'raw',
                            'msdcale' if input is 'distance'
        The default options are retrieved by calling sammon(x) with no
        parameters.
        File        : sammon.py
        Date        : 18 April 2014
        Authors     : Tom J. Pollard (tom.pollard.11@ucl.ac.uk)
                    : Ported from MATLAB implementation by
                      Gavin C. Cawley and Nicola L. C. Talbot
        Description : Simple python implementation of Sammon's non-linear
                      mapping algorithm [1].
        References  : [1] Sammon, John W. Jr., "A Nonlinear Mapping for Data
                      Structure Analysis", IEEE Transactions on Computers,
                      vol. C-18, no. 5, pp 401-409, May 1969.
        Copyright   : (c) Dr Gavin C. Cawley, November 2007.
        This program is free software; you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation; either version 2 of the License, or
        (at your option) any later version.
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
        You should have received a copy of the GNU General Public License
        along with this program; if not, write to the Free Software
        Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

         :return: object, dict, dict
            object: function call
            dict: function hyperparameters (fun.get_params())
            dict: hyperparameter ranges for hyperparameter optimization

        - - - REPOSITORY - - -
        https://github.com/tompollard/sammon

        '''
        fun = Sammon(
            n_components=None,
            max_iter = 100, # usually convergence is reached after 1 or a few iterations
            inputdist = 'raw', # default, otherwise distances need to be provided
            maxhalves = 20,
            tolfun = 1e-4,
            init = 'default' # best results with pca (default if inputdist=raw)
        )

        if self.all_hp:
            # 1) start with broad parameter ranges
            hyperpars = {
                'max_iter': [1,1000], # seems to have no effect on loss, no difference on speed suggesting
                                      # that convergence is reached quickly
                'tolfun': [1e-12, 0.2], # as higher as faster and better
            }
        else:
            # 2) updated hyperparameters
            hyperpars = {
                'tolfun': [0.01, 0.2],
        }
        params = fun.get_params()
        return fun, params, hyperpars






