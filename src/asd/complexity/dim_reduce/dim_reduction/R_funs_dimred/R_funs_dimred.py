
class Dimred_functions_R:
    '''
    https://github.com/kisungyou/Rdimtools/tree/master/R
    Dont optimize if hyperparameter has no effect on loss and default option seems fast.
    In this case we set a dummy variable string: 'empty': ''.
    '''
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols


    def funR_adr(self):
        '''
        - - - DESCRIPTION - - -
        Adaptive Dimension Reduction (ADR) iteratively finds the best subspace to perform data clustering.
        It can be regarded as one of remedies for clustering in high dimensional space. Eigenvectors of a
        between-cluster scatter matrix are used as basis of projection.

        - - - PARAMETERS - - -
            maxiter
                maximum number of iterations (default: 100).
            abstol
                absolute tolerance stopping criterion (default: 1e-8).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_ADR.html
        '''
        hyperpars = {
                # 1) start with parameter ranges:
                #     'maxiter': [10, 1000], # as higher as slower (exponentionally)
                #     'abstol': [1e-12, 0.1], # no effect on loss
                # 2) update 20221031: loss: 0.99 speed: very slow, reduce as much as possible
                    'empty': '' # default pars are ok
        }
        return 'r_adr', 'Rfun', hyperpars # R fun is just a filler



    def funR_lmds(self):
        '''
        - - - DESCRIPTION - - -
        Landmark Multidimensional Scaling is a variant of Classical Multidimensional Scaling in that it first finds a
        low-dimensional embedding using a small portion of given dataset and graft the others in a manner to preserve
        as much pairwise distance from all the other data points to landmark points as possible.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        npoints
            the number of landmark points to be drawn
            default: npoints=max(nrow(as.matrix(rdata))/5, ndim + 1)

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_LMDS.html
        '''
        # npoints_max = int(min(self.nrows/2, 100)) # should be max: ndim + 1
        ##
        hyperpars = {
                    # 1) start with parameter ranges:
                    # 'npoints': [5, npoints_max],  # as higher as better max=ndim+1
                    # 2) update 20221031: loss: 0.987 speed: ok, npoints needs optimization
                    # 'npoints'  # default (max=ndim+1) is best option
                    'empty': ''
                    }
        return 'r_lmds', 'Rfun', hyperpars # R fun is just a filler



    def funR_mds(self):
        '''
        - - - DESCRIPTION - - -
        (Classical) Multidimensional Scaling performs a classical Multidimensional Scaling (MDS) using Rcpp and
        Rcpp Armadillo package to achieve faster performance than cmdscale.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_MDS.html
        '''
        hyperpars = {'empty': ''}
        return 'r_mds', 'Rfun', hyperpars # R fun is just a filler



    def funR_npca(self):
        '''
        - - - DESCRIPTION - - -
        Nonnegative Principal Component Analysis, Nonnegative Principal Component Analysis (NPCA) is
        a variant of PCA where projection vectors - or, basis for learned subspace - contain no
        negative values.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        maxiter
            maximum number of iterations (default: 100).
        reltol
            relative tolerance stopping criterion (default: 1e-4).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_NPCA.html
        '''
        hyperpars = {
            # 1) start with parameter ranges:
            #         'maxiter': [5, 1000],  # no speed difference
            #         'reltol': [1e-8, 0.1],  # lower values are better, no speed difference
            # 2) update 20221031: loss: 0.945 speed: slow, reduce as much as possible
                # default parameters look best
                'empty': ''
        }
        return 'r_npca', 'Rfun', hyperpars # R fun is just a filler



    def funR_olpp(self):
        '''
        - - - DESCRIPTION - - -
        Orthogonal Locality Preserving Projection (OLPP) is a variant of do.lpp, which extracts orthogonal
        basis functions to reconstruct the data in a more intuitive fashion. It adopts PCA as preprocessing step
        and uses only one eigenvector at each iteration in that it might incur warning messages for solving
        near-singular system of linear equations.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k), c("enn",radius),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            and c("proportion",ratio). Default is c("proportion",0.1), connecting about 1/10 of nearest data points
            among all data points. See also aux.graphnbd for more details.
        symmetric
            either "intersect" or "union" is supported. Default is "union". See also aux.graphnbd for more details.
        t
            bandwidth for heat kernel in (0,∞)

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_OLPP.html
        '''
        params_fun = 'Rfun: type=c("knn",5), '
        hyperpars = {
            # 1) start with parameter ranges:
            #         'type': [0, 5],  # choose the two best options (knn5, proportion 0.1) and check again
            #         'symmetric': [0, 1],  # union is better
            #         't': [0.1, 100],  # no effect on loss or speed
            # 2) update 20221031: loss: 0.945 speed: slow, reduce as much as possible
            # 'type': [0, 1],  # no effect, set to knn5
            'empty': ''
        }
        return 'r_olpp', params_fun, hyperpars



    def funR_ppca(self):
        '''
        - - - DESCRIPTION - - -
        Probabilistic PCA (PPCA) is a probabilistic framework to explain the well-known PCA model. Using the
        conjugacy of normal model, we compute MLE for values explicitly derived in the paper. Note that unlike PCA
        where loadings are directly used for projection, PPCA uses 𝑊𝑀−1 as projection matrix, as it is relevant to
        the error model. Also, for high-dimensional problem, it is possible that MLE can have negative values if sample
        covariance given the data is rank-deficient.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_PPCA.html
        '''
        hyperpars = {'empty':  ''}
        return 'r_ppca', 'Rfun', hyperpars # R fun is just a filler



    def funR_rndproj(self):
        '''
        - - - DESCRIPTION - - -
        Random Projection is a linear dimensionality reduction method based on random projection technique,
        featured by the celebrated Johnson–Lindenstrauss lemma.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        type
            a type of random projection, one of "gaussian","achlioptas" or "sparse".
            default: "gaussian"
        s   max(sqrt(ncol(X)), 3.0) double
            a tuning parameter for determining values in projection matrix. While default is to use 𝑚𝑎𝑥(𝑙𝑜𝑔𝑝√,3)
            needs to be > 3.0

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_RNDPROJ.html
        '''
        params_fun = 'Rfun: type="achlioptas", '
        hyperpars = {
            # 1) start with parameter ranges:
            #         'type': [0, 2], # "achlioptas" seems best, reset default
            #         's': [3.0, max(self.ncols-1, 3.0)], # lower values around default are best, no effect on speed
            # 2) update 20221031: loss: 0.951 speed: slow, reduce as much as possible
            'empty': ''
        }
        return 'r_rndproj', params_fun, hyperpars



    def funR_rpcag(self):
        '''
         - - - DESCRIPTION - - -
        Robust Principal Component Analysis via Geometric Median, This function robustifies the traditional PCA
        via an idea of geometric median. To describe, the given data is first split into k subsets for each sample
        covariance is attained. According to the paper, the median covariance is computed under Frobenius norm and
        projection is extracted from the largest eigenvectors.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        k int
            the number of subsets for X to be divided. default = 5

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_RPCAG.html
        '''
        kmax = max(self.nrows/10, 2)
        hyperpars = {
            # 1) start with parameter ranges:
            # 'k': [2, max(kmax, 2)] # doesnt seem to have much effect, try again with default
            # 2) update 20221031: loss: 0.994 speed: slow
                'k': [2, max(kmax, 2)]
        }
        return 'r_rpcag', 'Rfun', hyperpars # R fun is just a filler



    def funR_udp(self):
        '''
        - - - DESCRIPTION - - -
        Unsupervised Discriminant Projection (UDP) aims finding projection that balances local and
        global scatter. Even though the name contains the word Discriminant, this algorithm is
        unsupervised. The term there reflects its algorithmic tactic to discriminate distance points
        not in the neighborhood of each data point. It performs PCA as intermittent preprocessing for
        rank singularity issue. Authors clearly mentioned that it is inspired by Locality Preserving
        Projection, which minimizes the local scatter only.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k),
            c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1), connecting
            about 1/10 of nearest data points among all data points. See also aux.graphnbd for more details.
        preprocess
            an additional option for preprocessing the data. Default is "center".
            See also aux.preprocess for more details.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_UDP.html
        '''
        params_fun = 'Rfun: type=c("enn",1), '
        hyperpars = {
            # 1) start with parameter ranges:
            # 'type': [0, 2],  # "enn",1 seems best, set to default
            # 2) update 20221031: loss: 0.991 speed: slow
            'empty': '',
        }
        return 'r_udp', params_fun, hyperpars # R fun is just a filler



    '''
    ###################   METHODS Nonlinear Embedding
    '''

    def funR_cisomap(self):
        '''
        - - - DESCRIPTION - - -
        Conformal Isomap(C-Isomap) is a variant of a celebrated method of Isomap. It aims at, rather than preserving full
        isometry, maintaining infinitestimal angles - conformality - in that it alters geodesic distance to reflect scale
        information.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1), connecting about
            1/10 of nearest data points among all data points. See also aux.graphnbd for more details.
        symmetric       Default is "union".
            one of "intersect", "union" or "asymmetric" is supported.
        weight boolean  default: True
            TRUE to perform Isomap on weighted graph, or FALSE otherwise.
        preprocess
            an additional option for preprocessing the data. Default is "center".

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_CISOMAP.html
        '''
        hyperpars = {
            # 1) start with parameter ranges:
            # 'type': [0, 5],  # "proportion",0.1 (default) seems better, no effect on speed
            # 'symmetric': [0, 2],  # union (default) seems better, no effect on speed
            # 'weight': [0, 1],  # 'TRUE' (default) seems better, no effect on speed

            # 2) update 20221031:  loss: 0.959 speed: slow, reduce as much as possible
                # !! most important is the symmetric== union (default), set type and weight to default
            'empty': ''
        }
        return 'r_cisomap', 'Rfun', hyperpars # R fun is just a filler



    # def funR_crca(self):
    #     '''
    #     - - - DESCRIPTION - - -
    #     Curvilinear Component Analysis (CRCA) is a type of self-organizing algorithms for manifold learning. Like MDS, it
    #     aims at minimizing a cost function (Stress) based on pairwise proximity. Parameter lambda is a heaviside function
    #     for penalizing distance pair of embedded data, and alpha controls learning rate similar to that of subgradient
    #     method in that at each iteration alpha/t the gradient is weighted by
    #
    #     - - - PARAMETERS - - -
    #     ndim
    #         an integer-valued target dimension.
    #     lambda      default: 1.0 double
    #         threshold value.
    #     alpha       default: 1.0 double
    #         initial value for updating.
    #     maxiter     default: 1000 int
    #         maximum number of iterations allowed.
    #     tolerance   default: 1e-06
    #         stopping criterion for maximum absolute discrepancy between two distance matrices.
    #
    #     - - - INFORMATION - - -
    #     https://kisungyou.com/Rdimtools/reference/nonlinear_CRCA.html
    #     '''
    #     params_fun = 'Rfun: maxiter=5, '
    #     hyperpars = {
    #             # 1) start with parameter ranges:
    #             #      'alpha': [0.1, 10.0], # no effect on loss
    #             #      'lambda': [0.1, 10.0],  # no effect on loss, small values (default) faster
    #             #      'maxiter': [1, 5], # smaller values better on loss and speed (high values take hours)
    #             #      'tolerance': [1e-08, 0.1], # no effect on loss, small values (default) faster
    #             # 2) update 20221031: loss: 0.991 speed: super slow, reset maxiter and remove all hyperparameters
    #                  'empty': ''
    #     }
    #     return 'r_crca', params_fun, hyperpars # R fun is just a filler


    # def funR_crda(self):
    #     '''
    #     - - - DESCRIPTION - - -
    #     Curvilinear Distance Analysis (CRDA) is a variant of Curvilinear Component Analysis in that the input pairwise
    #     distance is altered by curvilinear distance on a data manifold. Like in Isomap, it first generates neighborhood
    #     graph and finds shortest path on a constructed graph so that the shortest-path length plays as an approximate
    #     geodesic distance on nonlinear manifolds.
    #
    #     - - - PARAMETERS - - -
    #     ndim
    #         an integer-valued target dimension.
    #     type
    #         a vector of neighborhood graph construction. Following types are supported; c("knn",k), c("enn",radius),
    #         -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
    #         and c("proportion",ratio). Default is c("proportion",0.1), connecting about 1/10 of nearest data points
    #         among all data points.
    #     symmetric   Default is "union".
    #         one of "intersect", "union" or "asymmetric" is supported.
    #     weight      default: True
    #         TRUE to perform CRDA on weighted graph, or FALSE otherwise.
    #     lambda      default: 1.0 double
    #         threshold value.
    #     alpha       default: 1.0 double
    #         initial value for updating.
    #     maxiter     default: 1000
    #         maximum number of iterations allowed.
    #     tolerance   default: 1e-06
    #         stopping criterion for maximum absolute discrepancy between two distance matrices.
    #
    #     - - - INFORMATION - - -
    #     https://kisungyou.com/Rdimtools/reference/nonlinear_CRDA.html
    #     '''
    #     params_fun = 'Rfun: maxiter=5, '
    #     hyperpars = {
    #                 # 1) start with parameter ranges:
    #                 #  'type': [0, 5], # no effect, default a bit faster
    #                 #  'symmetric': [0, 2], # no effect on loss and speed
    #                 #  'weight': [0, 1], # no effect on loss and speed
    #                 #  'lambda': [0.1, 10.0], # most important, smaller values around default: 1.0 are good
    #                 #  'alpha': [0.1, 10.0], # no effect on loss, small values (default) faster
    #                 #  'maxiter': [1, 5], # needs to be as low as possible, otherwise it takes hours
    #                 #  'tolerance': [1e-08, 0.1],  # no effect on loss, small values (default) faster
    #             # 2) update 20221031:  loss: 0.991 speed: good but slowest function!
    #                 # 'maxiter': [5, 200], # no effect, set maxiter to 5
    #             'empty': ''
    #     }
    #     return 'r_crda', params_fun, hyperpars # R fun is just a filler



    def funR_fastmap(self):
        '''
        - - - DESCRIPTION - - -
        do.fastmap is an implementation of FastMap algorithm. Though it shares similarities with MDS, it is innately a
        nonlinear method that makes an iterative update for the projection information using pairwise distance information.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "null".
            preprocess = "null", "center", "scale", "cscale", "decorrelate", "whiten" Default: "null"

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_FastMap.html
        '''
        hyperpars = {'empty':  ''}
        return 'r_fastmap', 'Rfun', hyperpars # R fun is just a filler


    def funR_ispe(self):
        '''
        - - - DESCRIPTION - - -
        The isometric SPE (ISPE) adopts the idea of approximating geodesic distance on embedded
        manifold when two data points are close enough. It introduces the concept of cutoff where
        the learning process is only applied to the pair of data points whose original proximity
        is small enough to be considered as mutually local whose distance should be close to
        geodesic distance.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        proximity
            a function for constructing proximity matrix from original data dimension.
            default: function(x) { dist(x, method = "euclidean") }
        C   default: 50 int
            the number of cycles to be run; after each cycle, learning parameter
        S   default: 50 int
            the number of updates for each cycle.
        lambda  default: 1.0 float
            initial learning parameter.
        drate   default: 0.9 float
            multiplier for lambda at each cycle; should be a positive real number in (0,1).
        cutoff  cutoff = 1 double
            cutoff threshold value

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_ISPE.html
        '''
        params_fun = 'Rfun: drate=0.5, proximity=function(x){dist(x,method="minkowski",p=3)}, '
        hyperpars = {
                    # 1) start with parameter ranges:
                    # 'proximity': [0, 2],  # minkowski is the best
                    # 'C': [5, 500],  # no effect on loss, as lower as faster
                    # 'S': [5, 500],  #no effect on loss, as lower as faster
                    # 'drate': [0.001, 0.999],  # no effect on loss and speed
                    # 'cutoff': [0.0, 10.0],  # no effect on loss and speed
                    # 'lambda': [0.0, 10.0], # no effect on loss and speed

                    # 2) update 20221031: loss: 0.992 speed: super slow, remove unneccesary parameters
                    # 'lambda': [0.0, 10.0], # no effect and very slow, remove lambda
                    'empty': ''
                    }
        return 'r_ispe', params_fun, hyperpars # R fun is just a filler


    def funR_lamp(self):
        '''
        - - - DESCRIPTION - - -
        Local Affine Mulditimensional Projection (LAMP) can be considered as a nonlinear method even though each
        datum is projected using locally estimated affine mapping. It first finds a low-dimensional embedding
        for control points and then locates the rest data using affine mapping. We use 𝑛√n number of data as
        controls and Stochastic Neighborhood Embedding is applied as an initial projection of control set.
        Note that this belongs to the method for visualization so projection onto 𝐑2 is suggested for use.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_LAMP.html
        '''
        hyperpars = {'empty':  ''}
        return 'r_lamp', 'Rfun', hyperpars # R fun is just a filler


    def funR_plp(self):
        '''
        - - - DESCRIPTION - - -
        is an implementation of Piecewise Laplacian-based Projection (PLP) that adopts two-stage reduction
        scheme with local approximation.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1), connecting about
            1/10 of nearest data points among all data points. See also aux.graphnbd for more details.

        - - - ERRORS - - -
        covid and spectra: arg: type=c("proportion",0.1) or type=c("enn",1)
            Error in eigvecs[, 2:(ndim + 1)]: subscript out of bounds

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_PLP.html
        '''
        hyperpars = {'empty': ''}
        return 'r_plp', 'Rfun', hyperpars # R fun is just a filler



    def funR_sammon(self):
        '''
        - - - DESCRIPTION - - -
        do.sammon is an implementation for Sammon mapping, one of the earliest dimension reduction
        techniques that aims to find low-dimensional embedding that preserves pairwise distance
        structure in high-dimensional data space.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "null".
            preprocess = "null", "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        initialize default: pca
            "random" or "pca"; the former performs fast random projection (see also do.rndproj)
            and the latter performs standard PCA (see also do.pca).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_SAMMON.html
        '''
        hyperpars = { # 'initialize': [0,1] # default pca, is the best
                     'empty': ''
                    }
        return 'r_sammon', 'Rfun', hyperpars # R fun is just a filler


    def funR_spe(self):
        '''
        - - - DESCRIPTION - - -
        One of drawbacks for Multidimensional Scaling or Sammon mapping is that they have quadratic
        computational complexity with respect to the number of data. Stochastic Proximity Embedding
        (SPE) adopts stochastic update rule in that its computational speed is much improved. It
        performs C number of cycles, where for each cycle, it randomly selects two data points and
        updates their locations correspondingly S times. After each cycle, learning parameter λ is
        multiplied by drate, becoming smaller in magnitude.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        proximity
            a function for constructing proximity matrix from original data dimension.
            function(x) { dist(x, method = "euclidean") },
        C 50
            the number of cycles to be run; after each cycle, learning parameter
        S 50
            the number of updates for each cycle.
        lambda 1.0
            initial learning parameter.
        drate 0.9
            multiplier for lambda at each cycle; should be a positive real number in (0,1).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_SPE.html
        '''
        hyperpars = {
                    # 1) start with parameter ranges:
                    # 'proximity': [0, 2], # "euclidean" (default) seems the best
                    #  'C': [5, 500], # no effect on loss, as lower as faster, default 50 is good
                    #  'S': [5, 500], # no effect on loss, as lower as faster, default 50 is good
                    #  'lambda': [0.0, 10.0], # as lower as better and faster, default 1 is good
                    #  'drate': [0.0, 1.0], # no effect on loss and speed

                    # 2) update 20221031: loss: 0.993 speed: slow,
                         # default parameters seem to be the best
                        'empty': ''
                    }
        return 'r_spe', 'Rfun', hyperpars # R fun is just a filler

