import pyper as pr
r = pr.R(use_pandas = True)
r('library(Rdimtools)')
r('Sys.setenv(LANGUAGE="en")')


class class_dimred_functions_R:
    '''
    404 error on kisungyou website when clicking on source.
        address changed, codes are here: https://github.com/kisungyou/Rdimtools/tree/master/R
    '''
    def __init__(self, nrows, ncols):
        self.nrows = nrows #data.shape[0]
        self.ncols = ncols #data.shape[1]


    def funR_adr(self):
        '''
        - - - DESCRIPTION - - -
        Adaptive Dimension Reduction (ADR) iteratively finds the best subspace to perform data clustering.
        It can be regarded as one of remedies for clustering in high dimensional space. Eigenvectors of a
        between-cluster scatter matrix are used as basis of projection.

        - - - PARAMETERS - - -
            maxiter
                maximum number of iterations (default: 100).
                -I tested 100 and 1000 with Covid (190rows), 1000 is much slower, same result.
            abstol
                absolute tolerance stopping criterion (default: 1e-8).
                -I tested -8 and -12 with Covid (190rows), -12 a bit faster, same result.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_ADR.html
        '''
        hyperpars = {'maxiter': [10, 1000], # covid: 0.02 effect, 10 -> 1000 10x slower
                     'abstol': [1e-12, 0.1], # covid: 0.02 effect,
                    }
        return 'adr', 'Rfun', hyperpars # R fun is just a filler



    def funR_asi(self):
        '''
        - - - DESCRIPTION - - -
        Adaptive Subspace Iteration (ASI) iteratively finds the best subspace to perform data clustering.
        It can be regarded as one of remedies for clustering in high dimensional space. Eigenvectors of a
        within-cluster scatter matrix are used as basis of projection.

        - - - PARAMETERS - - -
            maxiter
                maximum number of iterations (default: 100).
                -I tested 100 and 1000 with Covid (190rows), 1000 is much slower and inacurate.
            abstol
                absolute tolerance stopping criterion (default: 1e-8).
                -I tested -8 and -12 with Covid (190rows), -12 a bit faster, same result.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_ASI.html
        '''
        hyperpars = {'maxiter': [5, 1000], #
                     'abstol': [1e-12, 0.1], #
                    }
        return 'asi', 'Rfun', hyperpars # R fun is just a filler



    def funR_elpp2(self):
        '''
        - - - DESCRIPTION - - -
        Enhanced Locality Preserving Projection proposed in 2013 (ELPP2) is built upon a
        parameter-free philosophy from PFLPP. It further aims to exclude its projection to
        be uncorrelated in the sense that the scatter matrix is placed in a generalized
        eigenvalue problem.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_ELPP2.html
        '''
        hyperpars = {'empty': [0]
                    }
        return 'elpp2', 'Rfun', hyperpars # R fun is just a filler



    def funR_extlpp(self):
        '''
        - - - DESCRIPTION - - -
        Extended Locality Preserving Projection (EXTLPP) is an unsupervised dimension reduction
        algorithm with a bit of flavor in adopting discriminative idea by nature. It raises a question
        on the data points at moderate distance in that a Z-shaped function is introduced in defining
        similarity derived from Euclidean distance.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        numk
            the number of neighboring points for k-nn graph construction.
            default: max(ceiling(nrow(X)/10), 2), /50 is faster and equaly accurate
            !! please introduce the numk integer as follows in the arg_string: numk=my_nk (5,10,20...)

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_EXTLPP.html
        '''
        neighbors_max = int(min(self.nrows/2, 100))
        ##
        hyperpars = {'numk': [2, neighbors_max] # default: max(ceiling(nrow(as.matrix(rdata))/50),2)
                    }
        return 'extlpp', 'Rfun', hyperpars # R fun is just a filler



    def funR_ldakm(self):
        '''
        - - - DESCRIPTION - - -
        is an unsupervised subspace discovery method that combines linear discriminant analysis (LDA) and
        K-means algorithm. It tries to build an adaptive framework that selects the most discriminative subspace.
        It iteratively applies two methods in that the clustering process is integrated with the subspace selection,
        and continuously updates its discrimative basis. From its formulation with respect to generalized eigenvalue
        problem, it can be considered as generalization of Adaptive Subspace Iteration (ASI) and Adaptive Dimension
        Reduction (ADR).

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        maxiter     default: 10
            maximum number of iterations allowed.
        abstol      default: 0.001
            stopping criterion for incremental change in projection matrix.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_LDAKM.html
        '''
        hyperpars = {'maxiter': [5, 1000], # default: 10
                     'abstol': [1e-12, 0.1], # default: 0.001
                    }
        return 'ldakm', 'Rfun', hyperpars # R fun is just a filler



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
            - I tested 5 and 50, 5 is more accurate and 2x slower

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_LMDS.html
        '''
        npoints_max = int(min(self.nrows/2, 100)) # should be: ndim + 1
        ##
        hyperpars = {'npoints': [5, npoints_max], #
                    }
        return 'lmds', 'Rfun', hyperpars # R fun is just a filler



    def funR_lpp(self):
        '''
        - - - DESCRIPTION - - -
        Locality Preserving Projection  is a linear approximation to Laplacian Eigenmaps.
        More precisely, it aims at finding a linear approximation to the eigenfunctions
        of the Laplace-Beltrami operator on the graph-approximated data manifold.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. c("knn",k), c("enn",radius), default: c("proportion",0.1)
            Our package supports three ways of defining nearest neighborhood. First is knn, which
            finds k nearest points and flag them as neighbors. Second is enn-epsilon nearest neighbor
            that connects all the data poinst within a certain radius. Finally, proportion flag is
            to connect proportion-amount of data points sequentially from the nearest to farthest.
            c("proportion",0.1), connecting about 1/10 of nearest data points.
            See also aux.graphnbd for more details.
            -enn with radius 1 works much better than the others
        symmetric
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
            In many graph setting, it starts from dealing with undirected graphs. NN search, however,
            does not necessarily guarantee if symmetric connectivity would appear or not. There are
            two easy options for symmetrization; intersect for connecting two nodes if both of them
            are nearest neighbors of each other and union for only either of them to be present.
            See also aux.graphnbd for more details.
            -"intersect", "asymmetric" are 2x faster and similar accuate
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        t = 1 double
            kernel bandwidth in (0.0,∞)
        lambda double
            regularization parameter for kernel matrix in [0.0,∞)

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_LLP.html
        '''
        hyperpars = {'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                     't': [0.1, 100.0],
                    }
        return 'lpp', 'Rfun', hyperpars # R fun is just a filler



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
        hyperpars = {'empty': [0],
                    }
        return 'mds', 'Rfun', hyperpars # R fun is just a filler



    def funR_nonpp(self):
        '''
        - - - DESCRIPTION - - -
        Nonnegative Orthogonal Neighborhood Preserving Projections, Nonnegative Orthogonal Neighborhood Preserving
        Projections (NONPP) is a variant of ONPP where projection vectors - or, basis for learned subspace - contain
        no negative values.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k), c("enn",radius),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            and c("proportion",ratio). Default is c("proportion",0.1), connecting about 1/10 of nearest data points
            among all data points. See also aux.graphnbd for more details.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "null", "center", "scale", "cscale", "decorrelate", "whiten" Default: null
         maxiter     default: 1000
            maximum number of iterations allowed.
        reltol  default: 1e-05

        - - - ERROR - - -
        spectra, covid;  arg: type=c("proportion",ratio), type=c("enn",radius)
            Error in if (dnormval > 1e-10) {: missing value where TRUE/FALSE needed

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_NONPP.html
        !!! best performer on 31 feature dataset. slow: 30sec
        '''
        hyperpars = {
                     'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'maxiter': [5, 1000], # default: 100
                     'reltol': [1e-9, 0.1], # default: 1e-4
                    }
        return 'nonpp', 'Rfun', hyperpars # R fun is just a filler


    def funR_npca(self):
        '''
        - - - DESCRIPTION - - -
        Nonnegative Principal Component Analysis, Nonnegative Principal Component Analysis (NPCA) is
        a variant of PCA where projection vectors - or, basis for learned subspace - contain no negative values.

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
        hyperpars = {'maxiter': [5, 1000], # default: 100
                     'reltol': [1e-8, 0.1], # default: 1e-4
                    }
        return 'npca', 'Rfun', hyperpars # R fun is just a filler



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
        hyperpars = {'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 1], # "intersect" or "union"
                     't': [0.1, 100],
                    }
        return 'olpp', 'Rfun', hyperpars # R fun is just a filler



    def funR_pflpp(self):
        '''
         - - - DESCRIPTION - - -
        Parameter-Free Locality Preserving Projection, Conventional LPP is known to suffer from sensitivity upon
        choice of parameters, especially in building neighborhood information. Parameter-Free LPP (PFLPP) takes an
        alternative step to use normalized Pearson correlation, taking an average of such similarity as a threshold
        to decide which points are neighbors of a given datum.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_PFLPP.html
        '''
        hyperpars = {'empty': [0],
                    }
        return 'pflpp', 'Rfun', hyperpars # R fun is just a filler



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
        hyperpars = {'empty': [0],
                    }
        return 'ppca', 'Rfun', hyperpars # R fun is just a filler



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
        s   max(sqrt(ncol(X)), 3.0) double
            a tuning parameter for determining values in projection matrix. While default is to use 𝑚𝑎𝑥(𝑙𝑜𝑔𝑝√,3)
            needs to be > 3.0

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_RNDPROJ.html
        '''
        hyperpars = {'type': [0, 2], # "gaussian","achlioptas" or "sparse"
                     's': [3.0, max(self.ncols-1, 3.0)], # default: max(sqrt(ncol(X)), 3)
                    }
        return 'rndproj', 'Rfun', hyperpars # R fun is just a filler



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
                     'k': [2, max(kmax, 2)],
                    }
        return 'rpcag', 'Rfun', hyperpars # R fun is just a filler



    def funR_sdlpp(self):
        '''
        - - - DESCRIPTION - - -
        Sample-Dependent Locality Preserving Projection, Many variants of Locality Preserving Projection are
        contingent on graph construction schemes in that they sometimes return a range of heterogeneous results when
        parameters are controlled to cover a wide range of values. This algorithm takes an approach called
        sample-dependent construction of graph connectivity in that it tries to discover intrinsic structures of data
        solely based on data.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        t = 1.0
            kernel bandwidth in (0.0,∞)

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_SDLPP.html
        '''
        hyperpars = {
                     't': [0.1, 100.0],
                    }
        return 'sdlpp', 'Rfun', hyperpars # R fun is just a filler



    def funR_spp(self):
        '''
        - - - DESCRIPTION - - -
        Sparsity Preserving Projection, Sparsity Preserving Projection (SPP) is an unsupervised linear dimension
        reduction technique. It aims to preserve high-dimensional structure in a sparse manner to find projections that
        keeps such sparsely-connected pattern in the low-dimensional space. Note that we used CVXR for convenient
        computation, which may lead to slower execution once used for large dataset.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        reltol
            relative tolerance stopping criterion (default: 1e-4).

        - - - ERROR - - -
        covid, spectra no args.
            Error in result$getValue(si): attempt to apply non-function

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_SPP.html
        '''
        hyperpars = {
                     'reltol': [1e-8, 0.1], # default: 1e-4
                    }
        return 'spp', 'Rfun', hyperpars # R fun is just a filler



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
        hyperpars = {'type': [0, 5], # # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                    }
        return 'udp', 'Rfun', hyperpars # R fun is just a filler



    '''
    ###################   METHODS Nonlinear Embedding
    '''
    def funR_bmds(self):
        '''
        - - - DESCRIPTION - - -
        Bayesian Multidimensional Scaling:
        A Bayesian formulation of classical Multidimensional Scaling is presented. Even though this method is based on
        MCMC sampling, we only return maximum a posterior (MAP) estimate that maximizes the posterior distribution.
        Due to its nature without any special tuning, increasing mc.iter requires much computation. A note on the
        method is that this algorithm does not return an explicit form of projection matrix so it's classified in
        our package as a nonlinear method. Also, automatic dimension selection is not supported for simplicity as
        well as consistency with other methods in the package.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        par.a  5 double
            hyperparameter for conjugate prior on variance term, i.e., 𝜎2∼𝐼𝐺(𝑎,𝑏)
        par.alpha 0.5 double
            hyperparameter for conjugate prior on diagonal term, i.e., 𝜆𝑗∼𝐼𝐺(𝛼,𝛽𝑗).
            Note that 𝛽𝑗 is chosen appropriately as in paper.
        par.step 1 double
            stepsize for random-walk, which is standard deviation of Gaussian proposal.
        mc.iter 50 int
            the number of MCMC iterations.
        print.progress FALSE
            a logical; TRUE to show iterations, FALSE otherwise (default: FALSE).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_BMDS.html
        '''
        hyperpars = {
                     'par.a': [1, 10], # default: 5
                     'par.alpha': [0.01, 10], # default: 0.5
                     'par.step': [1, 10], # default: 1.0
                     'mc.iter': [5, 1000], # default: 50
                    }
        return 'bmds', 'Rfun', hyperpars # R fun is just a filler



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
        symmetric
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
        weight boolean
            TRUE to perform Isomap on weighted graph, or FALSE otherwise.
        preprocess
            an additional option for preprocessing the data. Default is "center".

        - - - ERROR - - -
        covid: type=c("proportion",0.1), symmetric="union"

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_CISOMAP.html
        '''
        hyperpars = {'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                     'weight': [0, 1], # 'FALSE','TRUE'
                    }
        return 'cisomap', 'Rfun', hyperpars # R fun is just a filler



    def funR_crca(self):
        '''
        - - - DESCRIPTION - - -
        Curvilinear Component Analysis (CRCA) is a type of self-organizing algorithms for manifold learning. Like MDS, it
        aims at minimizing a cost function (Stress) based on pairwise proximity. Parameter lambda is a heaviside function
        for penalizing distance pair of embedded data, and alpha controls learning rate similar to that of subgradient
        method in that at each iteration alpha/t the gradient is weighted by

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        lambda      default: 1.0 double
            threshold value.
        alpha       default: 1.0 double
            initial value for updating.
        maxiter     default: 1000 int
            maximum number of iterations allowed.
        tolerance   default: 1e-06
            stopping criterion for maximum absolute discrepancy between two distance matrices.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_CRCA.html
        '''
        hyperpars = {
                     'alpha': [0.1, 10.0],
                     'lambda': [0.1, 10.0],
                     'maxiter': [5, 1000],
                     'tolerance': [1e-08, 0.1],
                    }
        return 'crca', 'Rfun', hyperpars # R fun is just a filler



    def funR_crda(self):
        '''
        - - - DESCRIPTION - - -
        Curvilinear Distance Analysis (CRDA) is a variant of Curvilinear Component Analysis in that the input pairwise
        distance is altered by curvilinear distance on a data manifold. Like in Isomap, it first generates neighborhood
        graph and finds shortest path on a constructed graph so that the shortest-path length plays as an approximate
        geodesic distance on nonlinear manifolds.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k), c("enn",radius),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            and c("proportion",ratio). Default is c("proportion",0.1), connecting about 1/10 of nearest data points
            among all data points.
        symmetric
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
        weight
            TRUE to perform CRDA on weighted graph, or FALSE otherwise.
        lambda      default: 1.0 double
            threshold value.
        alpha       default: 1.0 double
            initial value for updating.
        maxiter     default: 1000
            maximum number of iterations allowed.
        tolerance   default: 1e-06
            stopping criterion for maximum absolute discrepancy between two distance matrices.

        ERRORS
        covid - 50% of otimizations better with large datasets?
        "Error in aux.shortestpath(wD): * aux.shortestpath : input 'dist' should be either (n*n)
         matrix or 'dist' class object."
        https://rdrr.io/cran/Rdimtools/src/R/auxiliary_shortestpath.R

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_CRDA.html
        '''
        hyperpars = {
                     'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                     'weight': [0, 1], # TRUE, FALSE
                     'lambda': [0.1, 10.0], # default: 1.0
                     'alpha': [0.1, 10.0], # default: 1.0
                     'maxiter': [5, 1000], # default: 1000
                     'tolerance': [1e-08, 0.1], # default: 1e-06
                    }
        return 'crda', 'Rfun', hyperpars # R fun is just a filler



    def funR_dm(self):
        '''
        - - - DESCRIPTION - - -
        discovers low-dimensional manifold structure embedded in high-dimensional data space using Diffusion Maps (DM).
        It exploits diffusion process and distances in data space to find equivalent representations in low-dimensional
        space.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            an additional option for preprocessing the data. Default is "null". See also aux.preprocess for more details.
        bandwidth float
            a scaling parameter for diffusion kernel. Default is 1 and should be a nonnegative real number.
        timescale float   if multiscale==FALSE
            a target scale whose value represents behavior of heat kernels at time t. Default is 1 and should be a
            positive real number.
        multiscale
            logical; FALSE is to use the fixed timescale value, TRUE to ignore the given value.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_DM.html
        '''
        hyperpars = {
                     'bandwidth': [0.0, 10.0], # K = exp(-((as.matrix(dist(pX)))^2)/bandwidth)  default: 1
                     'timescale': [0.0, 10.0], # lambda_t = (svdA$d[2:(ndim+1)]^timescale) default: 1
                     'muliscale': [0,1], # FALSE, TRUE
                    }
        return 'dm', 'Rfun', hyperpars # R fun is just a filler



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
        hyperpars = {
                     'empty': [0],
                    }
        return 'fastmap', 'Rfun', hyperpars # R fun is just a filler



    def funR_idmap(self):
        '''
        - - - DESCRIPTION - - -
        Interactive Document Map originates from text analysis to generate maps of documents by placing similar
        documents in the same neighborhood. After defining pairwise distance with cosine similarity, authors
        asserted to use either NNP or FastMap as an engine behind.
        preprocess = c("null", "center", "scale", "cscale", "whiten", "decorrelate"),
        engine = c("NNP", "FastMap")

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
         preprocess
            additional option for preprocessing the data. Default is "null".
            preprocess = "null", "center", "scale", "cscale", "decorrelate", "whiten" Default: "null"
        engine
            either NNP or FastMap. engine = c("NNP", "FastMap")

        - - - ERRORS - - -
        covid and spectra, arg: engine=FastMap
            Error in if (dab2 > (sqrt(123 * .Machine$double.eps))) {: missing value where TRUE/FALSE needed
            seems to be a generic error of this line, skip engine == Fastmap

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_IDMAP.html
        '''
        hyperpars = {
                     'empty': [0],
                     #'engine': [0, 1], # "NNP", "FastMap"
                    }
        return 'idmap', 'Rfun', hyperpars # R fun is just a filler


    def funR_iltsa(self):
        '''
        - - - DESCRIPTION - - -
        Conventional LTSA method relies on PCA for approximating local tangent spaces. Improved
        LTSA (ILTSA) provides a remedy that can efficiently recover the geometric structure of
        data manifolds even when data are sparse or non-uniformly distributed.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type
            a vector of neighborhood graph construction. Following types are supported; c("knn",k),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1), connecting about
            1/10 of nearest data points among all data points.
        symmetric
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        t
            heat kernel bandwidth parameter in (0,∞)

        - - - ERRORS - - -
        covid (192,41)
        default parameters:
            Error in eigs_real_sym(A, nrow(A), k, which, sigma, opts, mattype = \"sym_matrix\", : 'k' must satisfy 0 < k < nrow(A)
            I think that has something to do with the ndim to big or too small
            https://rdrr.io/cran/RSpectra/src/R/10_eigs_real_gen.R

        type=c("enn",radius), symmetric="intersect"
            Error in do.iltsa(as.matrix(rdata), ndim = 25, type = c(\"enn\", 1), symmetric = c(\"intersect\")):
            * do.iltsa : select the larger neighborhood.
            -> spectral dataset, tokamak h and ohm: same error

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_ILTSA.html
        '''
        hyperpars = {
                     'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                     't': [0.1, 100.0], # default: 10.0
                    }
        return 'iltsa', 'Rfun', hyperpars # R fun is just a filler



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
        hyperpars = {'proximity': [0, 2], # "euclidean", "manhattan", "minkowski"
                     'C': [5, 500], # default: 50
                     'S': [5, 500], # default: 50
                     'drate': [0.001, 0.999], # default: 0.9
                     'cutoff': [0.0, 10.0], # default: 1.0
                    }
        return 'ispe', 'Rfun', hyperpars # R fun is just a filler


    def funR_keca(self):
        '''
        SLOW METHOD
        - - - DESCRIPTION - - -
        Kernel Entropy Component Analysis(KECA) is a kernel method of dimensionality reduction.
        Unlike Kernel PCA(do.kpca), it utilizes eigenbasis of kernel matrix 𝐾 in accordance with
        indices of largest Renyi quadratic entropy in which entropy for 𝑗-th eigenpair is defined
        j-th eigenvector of an uncentered kernel matrix 𝐾

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        kernel
            a vector containing name of a kernel and corresponding parameters.
            See also aux.kernelcov for complete description of Kernel Trick.
            kernel=c("gaussian",5), kernel=c("laplacian",1), kernel=c("histintx")
            more kernels: https://kisungyou.com/Rdimtools/reference/aux_kernelcov.html
        preprocess
            additional option for preprocessing the data. Default is "null".
            preprocess = "null", "center", "scale", "cscale", "decorrelate", "whiten" Default: "null"

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_KECA.html

        - - - ERROR
        Error in aux.kernelcov(pX, ktype): * aux.kernelcov : 'histintx' can be used for data with positive values only.
        '''
        hyperpars = {'kernel': [0, 3], # "gaussian", "laplacian", "histintx", "chisq", "spline"
                    }
        return 'keca', 'Rfun', hyperpars # R fun is just a filler



    def funR_lamp(self):
        '''
ifl        - - - DESCRIPTION - - -
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
        hyperpars = {
                     'empty': [0],
                    }
        return 'lamp', 'Rfun', hyperpars # R fun is just a filler



    def funR_lapeig(self):
        '''
        - - - DESCRIPTION - - -
        do.lapeig performs Laplacian Eigenmaps (LE) to discover low-dimensional manifold embedded in
        high-dimensional data space using graph laplacians. This is a classic algorithm employing spectral
        graph theory.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        type        c("proportion", 0.1),
            a vector of neighborhood graph construction. Following types are supported; c("knn",k), c("enn",radius),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            and c("proportion",ratio). Default is c("proportion",0.1), connecting about 1/10 of nearest data points
            among all data points. See also aux.graphnbd for more details.
        symmetric
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
            See also aux.graphnbd for more details.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        weighted boolean
            TRUE for weighted graph laplacian and FALSE for combinatorial laplacian where connectivity is
            represented as 1 or 0 only.
        kernelscale double
            kernel scale parameter. Default value is 1.0.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_LAPEIG.html
        '''
        hyperpars = {'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                     'weighted': [0,1], # FALSE, TRUE
                     'kernelscale': [0,10], # default=1.0 used: W=exp((-(nbdstruct$mask*nbdstruct$dist)^2)/kernelscale)
                    }
        return 'lapeig', 'Rfun', hyperpars # R fun is just a filler



    def funR_lisomap(self):
        '''
        !!! ERRORS !!!! not used due to too many errors
        - - - DESCRIPTION - - -
        Landmark Isomap is a variant of Isomap in that it first finds a low-dimensional embedding using
        a small portion of given dataset and graft the others in a manner to preserve as much pairwise
        distance from all the other data points to landmark points as possible.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        ltype   c("random", "MaxMin"),
            on how to select landmark points, either "random" or "MaxMin".
        npoints   npoints = max(nrow(X)/5, ndim + 1),
            the number of landmark points to be drawn.
        preprocess
            additional option for preprocessing the data. Default is "center".
            preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
        type   c("proportion", 0.1)
            a vector of neighborhood graph construction. Following types are supported; c("knn",k),
            -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
            c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1), connecting
            about 1/10 of nearest data points among all data points. See also aux.graphnbd for more details.
        symmetric
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
            See also aux.graphnbd for more details.
        weight
            TRUE (default) to perform Landmark Isomap on weighted graph, or FALSE otherwise.

        - - - ERRORS - - -
        covid and spectral ds with arg: npoints=10
            Error in do.lisomap(as.matrix(rdata), ndim = 17, npoints = 10): * do.lisomap : the number of landmark points should be [ndim+1,#(total data points)/2].
            code: n points: !is.numeric(npoints) || (npoints<=ndim) || (npoints>(nrow(X)/2+1)) || is.na(npoints)||is.infinite(npoints)

            Error in h(simpleError(msg, call)): error in evaluating the argument 'x' in selecting a method for function 'diag': NA/NaN argument

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_LISOMAP.html
        '''
        npoints_max = int(min(self.data.shape[0]/2, 100)) # should be: ndim + 1
        hyperpars = {
                     'ltype': [0, 1], # c("random", "MaxMin")
                     'npoints': [5, npoints_max], # default: max(nrow(X)/5, ndim + 1)
                     'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                     'weight': [0,1], # FALSE, TRUE
                    }
        return 'lisomap', 'Rfun', hyperpars # R fun is just a filler



    def funR_nnp(self):
        '''
        - - - DESCRIPTION - - -
        Nearest Neighbor Projection is an iterative method for visualizing high-dimensional dataset in that
        a data is sequentially located in the low-dimensional space by maintaining the triangular distance
        spread of target data with its two nearest neighbors in the high-dimensional space. We extended
        the original method to be applied for arbitrarily low-dimensional space. Due the generalization,
        we opted for a global optimization method of Differential Evolution (DEoptim) within in that it
        may add computational burden to certain degrees.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        preprocess
            additional option for preprocessing the data. Default is "null".
            preprocess = "null","center", "scale", "cscale", "decorrelate", "whiten" Default: center

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_NNP.html
        '''
        hyperpars = {
                     'empty': [0],
                    }
        return 'nnp', 'Rfun', hyperpars # R fun is just a filler


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
        hyperpars = {
                     'empty': [0],
                    }
        return 'plp', 'Rfun', hyperpars # R fun is just a filler



    def funR_ree(self):
        '''
        - - - DESCRIPTION - - -
        Robust Euclidean Embedding (REE) is an embedding procedure exploiting robustness of ℓ1
        cost function. In our implementation, we adopted a generalized version with weight matrix to be
        applied as well. Its original paper introduced a subgradient algorithm to overcome
        memory-intensive nature of original semidefinite programming formulation.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        W
            an (𝑛×𝑛) weight matrix. Default is uniform weight of 1s.
        preprocess
            an additional option for preprocessing the data. Default is "null".
        initc   1.0
            initial c value for subgradient iterating stepsize, 𝑐/𝑖√
        dmethod  c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"),
            a type of distance measure.
        maxiter 100
            maximum number of iterations for subgradient descent method.
        abstol  0.001
            stopping criterion for subgradient descent method.

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_REE.html
        '''
        hyperpars = {
                     'initc': [0.1, 10.0], # default: 1
                     'dmethod': [0, 5], # c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")
                     'maxiter': [5, 1000], #
                     'abstol': [1e-05, 0.1], #
                    }
        return 'ree', 'Rfun', hyperpars # R fun is just a filler


    def funR_rpca(self):
        '''
        - - - DESCRIPTION - - -
        Robust PCA (RPCA) is not like other methods in this package as finding explicit
        low-dimensional embedding with reduced number of columns. Rather, it is more of
        a decomposition method of data matrix 𝑋, possibly noisy, into low-rank and sparse
        matrices by solving the following, minimize‖𝐿‖∗+𝜆‖𝑆‖1𝑠.𝑡.𝐿+𝑆=𝑋 minimize where 𝐿 is a
        low-rank matrix, 𝑆 is a sparse matrix and ‖⋅‖∗ denotes nuclear norm, i.e., sum of
        singular values. Therefore, it should be considered as preprocessing procedure of
        denoising. Note that after RPCA is applied, 𝐿 should be used as kind of a new data
        matrix for any manifold learning scheme to be applied.

        - - - PARAMETERS - - -
        mu      1.0 double
            an augmented Lagrangian parameter
        lambda  sqrt(1/(max(dim(X)))) double
            parameter for the sparsity term ‖𝑆‖1
            efault value is given accordingly to the referred paper.
        maxiter int
            maximum number of iterations (default: 100).
        abstol float
            absolute tolerance stopping criterion (default: 1e-8).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_RPCA.html
        '''
        lambda_max = 1/max(self.nrows, self.ncols)
        hyperpars = {
                     'mu': [0, 2.0], # its a gues
                     'lambda': [0, lambda_max], #  sqrt(1/(max(dim(X))))
                     'maxiter': [5, 1000], # default 100
                     'abstol': [1e-10, 0.1], # default: 1e-8
                    }
        return 'rpca', 'Rfun', hyperpars # R fun is just a filler



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
        initialize default:
            "random" or "pca"; the former performs fast random projection (see also do.rndproj)
            and the latter performs standard PCA (see also do.pca).

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_SAMMON.html
        '''
        hyperpars = {
                     'empty': [0],
                    }
        return 'sammon', 'Rfun', hyperpars # R fun is just a filler


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
        hyperpars = {'proximity': [0, 2], # "euclidean", "manhattan", "minkowski"
                     'C': [5, 500], # default: 50
                     'S': [5, 500], # default: 50
                     'lambda': [0.0, 10.0], # default: 1.0
                     'drate': [0.0, 1.0], # default: 0.9
                    }
        return 'spe', 'Rfun', hyperpars # R fun is just a filler



    def funR_spmds(self):
        '''
        - - - DESCRIPTION - - -
        transfers the classical multidimensional scaling problem into the data spectral domain using
        Laplace-Beltrami operator. Its flexibility to use subsamples and spectral interpolation of
        non-reference data enables relatively efficient computation for large-scale data.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.
        neigs   max(2, nrow(X)/10),
            number of eigenvectors to be used as spectral dimension.
        ratio 0.1
            percentage of subsamples as reference points.
        preprocess
            an additional option for preprocessing the data. Default is "null". S
            ee also aux.preprocess for more details.
        type c("proportion", 0.1)
            a vector of neighborhood graph construction. Following types are supported;
            c("knn",k), c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1),
            connecting about 1/10 of nearest data points among all data points.
            -> enn epsilon nearest neighbor - that connects all the data poinst within a certain radius
        symmetric c("union", "intersect", "asymmetric")
            one of "intersect", "union" or "asymmetric" is supported. Default is "union".
            See also aux.graphnbd for more details.

        - - - ERRORS - - -
        there are several errors but somehow it works a few times, so doesnt matter
        covid and spectra:
            Error in solve.default(LHS, RHS): system is computationally singular: reciprocal condition number = 4.18156e-18
            somewhere there is a division which leads to such small number, rare erroe, doesnt matter.

        covid:
            Error in array(1, c(m_s, 1)) %*% x2 - 2 * t(X_ind) %*% X_ind: non-conformable arrays

        - - - INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_SPMDS.html
        '''
        hyperpars = {'neigs': [2, int(self.nrows/2)], # max(2, nrow(X)/10)
                     'ratio': [0.01, 0.99], # max(2, nrow(X)/10)
                     'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
                     'symmetric': [0, 2], # "intersect", "union" or "asymmetric"
                    }
        return 'spmds', 'Rfun', hyperpars # R fun is just a filler





'''
#############################################################################################
#############################################################################################
####################################### SLOW METHODS ########################################
#############################################################################################
#############################################################################################
'''
# !!!! only for 1,2,3 dimensions
#     def funR_tsne(self):
#         '''
#         NOT USED, PYTHON FUNCTION INSTEAD
#         !!!!
#         ERROR: if (!is.wholenumber(dims) || dims < 1 || dims > 3) { stop("dims should be either 1, 2 or 3") }
#         !!!!
#         SLOW method and a lot of hyperparameters to calculate
#         - - - DESCRIPTION - - -
#         distributed Stochastic Neighbor Embedding (t-SNE) is a variant of Stochastic Neighbor Embedding
#         (SNE) that mimicks patterns of probability distributinos over pairs of high-dimensional objects
#         on low-dimesional target embedding space by minimizing Kullback-Leibler divergence. While
#         conventional SNE uses gaussian distributions to measure similarity, t-SNE, as its name suggests,
#         exploits a heavy-tailed Student t-distribution.
#
#         - - - PARAMETERS - - -
#         ndim
#             an integer-valued target dimension.
#         perplexity 30
#             desired level of perplexity; ranging [5,50].
#         eta 0.05
#             learning parameter.
#         maxiter 2000
#             maximum number of iterations.
#         jitter  0.3
#             level of white noise added at the beginning.
#         jitterdecay 0.99
#             decay parameter in (0,1). The closer to 0, the faster artificial noise decays.
#         momentum 0.5
#             level of acceleration in learning.
#         pca TRUE
#             whether to use PCA as preliminary step; TRUE for using it, FALSE otherwise.
#         pcascale FALSE
#             a logical; FALSE for using Covariance, TRUE for using Correlation matrix.
#             See also do.pca for more details.
#         symmetric FALSE
#             a logical; FALSE to solve it naively, and TRUE to adopt symmetrization scheme.
#         BHuse TRUE
#             a logical; TRUE to use Barnes-Hut approximation. See Rtsne for more details.
#         BHtheta 0.25
#             speed-accuracy tradeoff. If set as 0.0, it reduces to exact t-SNE.
#
#          - - - INFORMATION - - -
#          https://kisungyou.com/Rdimtools/reference/nonlinear_TSNE.html
#          '''
#         hyperpars = {
#                      'perplexity': [5,50], # default: 30
#                      'eta': [0.0, 1.0], # default: 0.05
#                      'maxiter': [5, 2000], # default: 2000
#                      'jitter': [0.0, 1.0], # default: 0.3
#                      'jitterdecay': [0.0, 1.0], # default: 0.99
#                      'momentum': [0.0, 1.0], # default: 0.5
#                      'pca': [0, 1], # FALSE, TRUE
#                      'pcascale': [0, 1], # FALSE, TRUE
#                      'symmetric': [0, 1], # FALSE, TRUE
#                      'BHuse': [0, 1], # FALSE, TRUE
#                      'BHtheta': [0.0, 1.0], # default: 0.99
#                      
#                      # 'preprocess': [0, 5], # not neccessary, data are scaled
#                     }
#         return 'tsne', 'Rfun', hyperpars # R fun is just a filler

# def funR_cnpe(self, arg_string='preprocess="center", type=c("proportion", 0.1)', ndim=2):
#     '''
#     type = c("proportion", 0.1),  # c("knn",k),c("enn",radius),c("proportion",ratio). connects 1/10 nearest data points
#     preprocess = c("center", "scale", "cscale", "decorrelate", "whiten"),
#     !!! takes very long: 2.5min for unscaled data 1800/31feat
#     '''
#     return self.R_function_exe('cnpe', arg_string, ndim)
#
#
#
# def funR_dve(self, arg_string='type=c("proportion", 0.1)', ndim=2):
#     '''
#     - - - DESCRIPTION - - -
#     Distinguishing Variance Embedding (DVE) is an unsupervised nonlinear manifold learning method. It can be
#     considered as a balancing method between Maximum Variance Unfolding and Laplacian Eigenmaps. The algorithm
#     unfolds the data by maximizing the global variance subject to the locality-preserving constraint. Instead
#     of defining certain kernel, it applies local scaling scheme in that it automatically computes adaptive
#     neighborhood-based kernel bandwidth.
#
#     - - - PARAMETERS - - -
#     ndim
#         an integer-valued target dimension.
#     type
#         a vector of neighborhood graph construction. Following types are supported; c("knn",k), c("enn",radius),
#         -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
#         -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
#         and c("proportion",ratio). Default is c("proportion",0.1), connecting about 1/10 of nearest data points
#         among all data points. See also aux.graphnbd for more details.
#     preprocess
#         an additional option for preprocessing the data. Default is "null". See also aux.preprocess for more details.
#
#     - - - INFORMATION - - -
#     https://kisungyou.com/Rdimtools/reference/nonlinear_DVE.html
#     '''
#     return self.R_function_exe('dve', arg_string, ndim)
#
#
# def funR_isoproj(self, arg_string='type=c("proportion", 0.1), symmetric="union"', ndim=2):
#     '''
#     Isometric Projection is a linear dimensionality reduction algorithm that exploits geodesic distance in original data
#     dimension and mimicks the behavior in the target dimension. Embedded manifold is approximated by graph construction
#     as of ISOMAP. Since it involves singular value decomposition and guesses intrinsic dimension by the number of positive
#     singular values from the decomposition of data matrix, it automatically corrects the target dimension accordingly.
#     symmetric = c("union", "intersect", "asymmetric"),   # controls how nearest neighborhood graph should be symmetrized.
#     preprocess = c("center", "scale", "cscale", "decorrelate", "whiten"),
#
#     !!! takes 75sec for data 1800/31feat
#     '''
#     return self.R_function_exe('isoproj', arg_string, ndim)
#
#
# def funR_kudp(self, arg_string='type=c("proportion", 0.1), bandwidth=1', ndim=2):
#     '''
#     Kernel-Weighted Unsupervised Discriminant Projection (KUDP) is a generalization of UDP where proximity
#     is given by weighted values via heat kernel, whence UDP uses binary connectivity. If bandwidth is +∞,
#     it becomes a standard UDP problem. Like UDP, it also performs PCA preprocessing for rank-deficient case.
#     type = c("proportion", 0.1),  # c("knn",k),c("enn",radius),
#     -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
#     c("proportion",ratio). connects 1/10 nearest data points
#     preprocess = c("center", "scale", "cscale", "decorrelate", "whiten"),
#     bandwidth = 1   # no effect in online example
#     !!! takes 20sec for data 1800/31feat
#     '''
#     return self.R_function_exe('kudp', arg_string, ndim)
#
#
# def funR_lltsa(self, arg_string='type=c("proportion", 0.1), symmetric="union"', ndim=2):
#     '''
#     Linear Local Tangent Space Alignment (LLTSA) is a linear variant of the celebrated LTSA method. It uses the
#     tangent space in the neighborhood for each data point to represent the local geometry. Alignment of those
#     local tangent spaces in the low-dimensional space returns an explicit mapping from the high-dimensional space.
#     type = c("proportion", 0.1),  # c("knn",k),c("enn",radius),c("proportion",ratio). connects 1/10 nearest data points
#     preprocess = c("center", "scale", "cscale", "decorrelate", "whiten"),
#     symmetric = c("union", "intersect", "asymmetric"),   # controls how nearest neighborhood graph should be symmetrized.
#     !!! takes 16sec for data 1800/31feat
#     '''
#     return self.R_function_exe('lltsa', arg_string, ndim)
#
#
# def funR_npe(self, arg_string='preprocess="center", type=c("proportion",0.1), symmetric="union", weight=TRUE, regtype=FALSE, regparam=1',
#              ndim=2):
#     '''
#     performs a linear dimensionality reduction using Neighborhood Preserving Embedding (NPE) proposed by He et al (2005).
#     It can be regarded as a linear approximation to Locally Linear Embedding (LLE). Like LLE, it is possible for the weight
#     matrix being rank deficient. If regtype is set to TRUE with a proper value of regparam, it will perform Tikhonov
#     regularization as designated. When regularization is needed with regtype parameter to be FALSE, it will automatically
#     find a suitable regularization parameter and put penalty for stable computation. See also do.lle for more details.
#     type = c("proportion", 0.1),  # c("knn",k),c("enn",radius),c("proportion",ratio). connects 1/10 nearest data points
#     preprocess = c("center", "scale", "cscale", "decorrelate", "whiten"),
#     symmetric = c("union", "intersect", "asymmetric"),   # controls how nearest neighborhood graph should be symmetrized.
#     weight = TRUE     # TRUE to perform NPE on weighted graph, or FALSE otherwise.
#     regtype = FALSE,  # FALSE for not applying automatic Tikhonov Regularization, or TRUE otherwise.
#     regparam = 1   # a positive real number for Regularization. Default value is 1
#     !!! takes 16sec for data 1800/31feat
#     '''
#     return self.R_function_exe('npe', arg_string, ndim)
#
#
# def funR_phate(self, arg_string='k=5, alpha=10, dtype="sqrt", smacof=TRUE', ndim=2):
#     '''
#     SLOW METHOD
#     - - - DESCRIPTION - - -
#     PHATE is a nonlinear method that is specifically targeted at visualizing high-dimensional data by
#     embedding it on 2- or 3-dimensional space. We offer a native implementation of PHATE solely in
#     R/C++ without interface to python module.
#
#     - - - PARAMETERS - - -
#     ndim
#         an integer-valued target dimension (default: 2).
#     k
#         size of nearest neighborhood (default: 5).
#     alpha  double
#         decay parameter for Gaussian kernel exponent (default: 10).
#     dtype
#         type of potential distance transformation; "log" or "sqrt" (default: "sqrt").
#     smacof
#         a logical; TRUE to use SMACOF for Metric MDS or FALSE to use Classical MDS (default: TRUE).
#     maxiter
#         maximum number of iterations (default: 100).
#     abstol
#         absolute stopping criterion for metric MDS iterations (default: 1e-8).
#
#     - - - INFORMATION - - -
#     https://kisungyou.com/Rdimtools/reference/nonlinear_PHATE.html
#     '''
#     return self.R_function_exe('phate', arg_string, ndim)
#
# # !!! slow 53, very low trust, cont, lmcm
# def funR_llp(self, arg_string='preprocess="center", type=c("proportion", 0.1), symmetric="union", t=1, lambda=1', ndim=2):
#     '''While Principal Component Analysis (PCA) aims at minimizing global estimation error, Local Learning Projection
#     (LLP) approach tries to find the projection with the minimal local estimation error in the sense that each projected
#     datum can be well represented based on ones neighbors. For the kernel part, we only enabled to use a gaussian kernel
#     as suggested from the original paper. The parameter lambda controls possible rank-deficiency of kernel matrix.
#        HYPERPARAMETERS default
#        type = c("proportion", 0.1)  # c("knn",k),c("enn",radius),c("proportion",ratio). connects 1/10 nearest data points
#        symmetric = c("union", "intersect", "asymmetric")
#        preprocess = c("center", "scale", "cscale", "decorrelate", "whiten")
#        t = 1,
#        lambda = 1
#        !!! slow 53, very low trust, cont, lmcm
#     '''
#     return self.R_function_exe('llp', arg_string, ndim)
#
#
#
# def funR_mve(self):
#         '''
#         SLOW METHOD
#         - - - DESCRIPTION - - -
#         Minimum Volume Embedding (MVE) is a nonlinear dimension reduction algorithm that exploits
#         semidefinite programming (SDP), like MVU/SDE. Whereas MVU aims at stretching through all
#         direction by maximizing ∑𝜆𝑖, MVE only opts for unrolling the top eigenspectrum and chooses
#         to shrink left-over spectral dimension. For ease of use, unlike kernel PCA, we only made
#         use of Gaussian kernel for MVE. Note that we adopted Rcsdp package in that when given
#         large-scale dataset, it may result in extremely deteriorated computational performance.
#
#         - - - PARAMETERS - - -
#         ndim
#             an integer-valued target dimension.
#         knn
#             size of 𝑘-nn neighborhood.   ceiling(nrow(X)/10),
#         kwidth  1
#             bandwidth for Gaussian kernel.
#         preprocess
#             additional option for preprocessing the data. Default is "center".
#             preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
#         tol     default: 1e-04
#             stopping criterion for incremental change.
#         maxiter     default: 1000
#             maximum number of iterations allowed.
#
#         - - - INFORMATION - - -
#         https://kisungyou.com/Rdimtools/reference/nonlinear_MVE.html
#         '''
#         knn_max = int(self.data.s)# ceiling(nrow(X)/10)
#         hyperpars = {
#                      'knn': [1, knn_max], # ceiling(nrow(X)/10),
#                      'kwidth': [0.1, 100], # default: 1
#                      'tol': [1e-06, 0.1], # default: 1e-04
#                      'maxiter': [5, 1000], # default: 1000
#                      'init_iterations': [self.init, self.iterations]
#                      
#                     }
#         return 'mve', 'Rfun', hyperpars # R fun is just a filler
#
#
#
# def funR_mvu(self):
#         '''
#         SLOW METHOD
#         - - - DESCRIPTION - - -
#         The method of Maximum Variance Unfolding(MVU), also known as Semidefinite Embedding(SDE) is,
#         as its names suggest, to exploit semidefinite programming in performing nonlinear dimensionality
#         reduction by unfolding neighborhood graph constructed in the original high-dimensional space.
#         Its unfolding generates a gram matrix 𝐾 in that we can choose from either directly finding
#         embeddings ("spectral") or use again Kernel PCA technique ("kpca") to find low-dimensional
#         representations. Note that since do.mvu depends on Rcsdp, we cannot guarantee its computational
#         efficiency when given a large dataset.
#
#         - - - PARAMETERS - - -
#         ndim
#             an integer-valued target dimension.
#         type
#             a vector of neighborhood graph construction. Following types are supported; c("knn",k),
#             -> epsilon nearest neighbor - that connects all the data poinst within a certain radius
#             c("enn",radius), and c("proportion",ratio). Default is c("proportion",0.1), connecting
#             about 1/10 of nearest data points among all data points. See also aux.graphnbd for more details.
#         preprocess
#             additional option for preprocessing the data. Default is "center".
#             preprocess = "center", "scale", "cscale", "decorrelate", "whiten" Default: center
#         projtype    c("spectral", "kpca")
#             type of method for projection; either "spectral" or "kpca" used.
#
#         - - - INFORMATION - - -
#         https://kisungyou.com/Rdimtools/reference/nonlinear_MVU.html
#         '''
#         hyperpars = {
#                      'type': [0, 5], # "knn",5 "knn",50, "enn",1, "enn",2 "proportion",0.1 "proportion",0.3
#                      'projtype': [0, 1], # c("spectral", "kpca")
#                      'init_iterations': [self.init, self.iterations]
#                      
#                     }
#         return 'mvu', 'Rfun', hyperpars # R fun is just a filler
#
# def funR_sne(self):
#         '''
#         SLOW METHOD
#         - - - DESCRIPTION - - -
#         Stochastic Neighbor Embedding (SNE) is a probabilistic approach to mimick distributional description
#         in high-dimensional - possible, nonlinear - subspace on low-dimensional target space. do.sne fully
#         adopts algorithm details in an original paper by Hinton and Roweis (2002).
#
#         - - - PARAMETERS - - -
#         ndim
#             an integer-valued target dimension.
#         perplexity 30
#             desired level of perplexity; ranging [5,50].
#         eta 0.05
#             learning parameter.
#         maxiter 2000
#             maximum number of iterations.
#         jitter 0.3
#             level of white noise added at the beginning.
#         jitterdecay 0.99
#             decay parameter in (0,1), The closer to 0, the faster artificial noise decays.
#         momentum 0.5
#             level of acceleration in learning.
#         pca TRUE
#             whether to use PCA as preliminary step; TRUE for using it, FALSE otherwise.
#         pcascale FALSE
#             a logical; FALSE for using Covariance, TRUE for using Correlation matrix.
#             See also do.pca for more details.
#         symmetric FALSE
#             a logical; FALSE to solve it naively, and TRUE to adopt symmetrization scheme.
#
#         - - - INFORMATION - - -
#         https://kisungyou.com/Rdimtools/reference/nonlinear_SNE.html
#         '''
#         hyperpars = {
#                      'perplexity': [5,50], # default: 30
#                      'eta': [0.0, 1.0], # default: 0.05
#                      'maxiter': [5, 2000], # default: 2000
#                      'jitter': [0.0, 1.0], # default: 0.3
#                      'jitterdecay': [0.0, 1.0], # default: 0.99
#                      'momentum': [0.0, 1.0], # default: 0.5
#                      'pca': [0, 1], # FALSE, TRUE
#                      'pcascale': [0, 1], # FALSE, TRUE
#                      'symmetric': [0, 1], # FALSE, TRUE
#                      'init_iterations': [self.init, self.iterations]
#                      # 'preprocess': [0, 5], # not neccessary, data are scaled
#                     }
#         return 'sne', 'Rfun', hyperpars # R fun is just a filler


    #         # sne
    #         if key == 'perplexity':
    #             arg_string = arg_string + str(key) + '=' + str(int(round(value,0))) + ', '
    #
    #         # sne
    #         if key == 'eta':
    #             arg_string = arg_string + str(key) + '=' + str(value) + ', '
    #
    #         # sne
    #         if key == 'jitter':
    #             arg_string = arg_string + str(key) + '=' + str(value) + ', '
    #
    #         # sne
    #         if key == 'jitterdecay':
    #             arg_string = arg_string + str(key) + '=' + str(value) + ', '
    #
    #         # sne
    #         if key == 'momentum':
    #             arg_string = arg_string + str(key) + '=' + str(value) + ', '
    #
    #         # sne
    #         if key == 'pca':
    #             val = str(['FALSE','TRUE'][int(round(value,0))])
    #             arg_string = arg_string + str(key) + '=' + val + ', '
    #
    #         # sne
    #         if key == 'pcascale':
    #             val = str(['FALSE','TRUE'][int(round(value,0))])
    #             arg_string = arg_string + str(key) + '=' + val + ', '
    #
    #         # sne
    #         if key == 'symmetric':
    #             val = str(['FALSE','TRUE'][int(round(value,0))])
    #             arg_string = arg_string + str(key) + '=' + val + ', '


'''FUNCTIONS WHICH TAKE IMPOSSIBLY LONG TIME'''
# Feature Subset Selection using Expectation-Maximization (FSSEM) takes a wrapper approach to feature
# selection problem. It iterates over optimizing the selection of variables by incrementally including
# each variable that adds the most significant amount of scatter separability from a labeling obtained
# by Gaussian mixture model. This method is quite computation intensive as it pertains to multiple
# fitting of GMM. Setting smaller max.k for each round of EM algorithm as well as target dimension
# ndim would ease the burden.
# 'fssem', # max.k, preprocess, takes >30min for for unscaled data 1800/31feat
#'lpe',

 # def funR_kpca(self):
 #        '''
 #        NOT USEDIMPLEMENTED IN PYTHON
 #        - - - DESCRIPTION - - -
 #        Kernel principal component analysis (KPCA/Kernel PCA) is a nonlinear extension of classical
 #        PCA using techniques called kernel trick, a common method of introducing nonlinearity by
 #        transforming, usually, covariance structure or other gram-type estimate to make it flexible
 #        in Reproducing Kernel Hilbert Space.
 #        - - - PARAMETERS - - -
 #        - - - INFORMATION - - -
 #        '''



'''
#############################################################################################
#############################################################################################
##################   FUNCTIONS WHICH ARE NOT WORKING, mainly complex number output ##########
#############################################################################################
#############################################################################################
'''

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess N, sigma, alpha
# 'lpmip',

# Orthogonal Locality Preserving Projections
# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess
# 'onpp',

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess
# 'lpca2006'

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: preprocess, lambda,
#'crp',  #  not working

# !!! not working !!! something with the datainput does not work here
# hyperparameters: type, preprocess
#'bpca'


'''
#############################################################################################
#############################################################################################
R packages which: A) require label or response matrix (supervised)
                  B) are implemented with python (sklearn etc.)
                  C) require 2 data matrizes
#############################################################################################
#############################################################################################
'''
#'ammc', #label
#'ammm', #label
#'cca', #data1, data2
#'dagdne', #label
#'dne', #label
#'dspp', #label
#'elde', #label
#'eslpp', #label
#'fa', sklern
#'ica', sklearn
#'kmvp', # label
#'lda',  # sklearn
#'lde', # label
#'ldp', # label
#'lea', # sklearn
#'lfda', # label
#'lpfda', #label
#'lqmi', #label
#'lsda', #label
#'lsir', #label-response
#'lspp', #label
#'mfa', # label, multiple factor analysis (MFA) large number of categorical variables
#'mlie', # label
#'mmc', # label
#'mmp', # label
#'mmsd', # label
#'modp', # label
#'msd', # label
#'mvp', # label
#'odp', # label
#'olda', # label
#'opls', # data1, data2
#'pca', # facil, sklearn
#'pls', # data1, data2
#'rlda', # label
#'rsir', # label response
#'sammc', # label
#'save', # label response
#'sda', # label
#'sir', # label response
#'slpe', # label
#'slpp', # label
#'spc', # label response
#'spca', # sklearn
#'ssldp', # label
#'ulda', # label

# non linear
# 'cge', # label
# 'isomap', # sklearn
# 'klde', # label
# 'klfda', # label
# 'klsda', # label
# 'kmfa', # label
# 'kmmc', # label
# 'kqmi', # label
# 'ksda', # label
# 'splapeig', # label



## All methods that
#    A) are not in sklearn
#    B) no need for label
#    C) only one data matrix
# 'adr', # facil
# 'pca', # scaling
# 'asi', # facil
# 'bpca', # facil
# 'bpca', # facil
# 'cnpe', # type, preprocess
# 'crp',  # preprocess, lambda
# 'elpp2', # preprocess
# 'extlpp', # preprocess, numk
# 'fssem', # max.k, preprocess
# 'isoproj', # type, symetric, preprocess
# 'kmvp', # preprocess, bandwidth
# 'kudp', # type, preprocess, bandwidth
# 'ldakm', # preprocess, maxiter = 10, abstol = 0.001
# 'llp', # type, symmetric, preprocess, t, lambda
# 'lltsa', #type, symmetric, preprocess
# 'lmds', #npoints
# 'lpca2006', #type, preprocess
# 'lpe', #preprocess, numk
# 'lpmip', #type, preprocess N, sigma, alpha=0.5
# 'lpp', #type, symmetric, preprocess, t
# 'mds', # facil
# 'nonpp', #type, preprocess
# 'npca', # facil
# 'npe', # type, symmetric, weight, preprocess, regtype, regparam
# 'olpp', # type, symmetric, weight, t
# 'onpp', # type, preprocess
# 'pflpp', # preprocess
# 'ppca', # facil
# 'rndproj', # type, preprocess N, s
# 'rpcag', # k, preprocess
# 'sdlpp', # t, preprocess
# 'spp', # preprocess, reltol
# 'udp', # type, preprocess
