
class Dimred_functions_R:
    '''
    This class contains wrappers for R dimensionality reduction functions.
    thanks to Kisung You for the functions and information in the R-dimtools package.
    repository: https://github.com/kisungyou/Rdimtools/tree/master/R
    we implemented and tested many of these functions and choose the ones that worked best
    for our purposes.
    The wrapper contains information from the indicted resources and some documentation of 
    the optimization of the hyperparameter tuning and comments.
    The wrapper returns:
    1) function id
    2) Hyperparameters dictionary
    In case there are no hyperparameters to tune we set an 'empty' dummy variable in order to 
       run the script seamless.
    '''
    def __init__(self,nrows: int, ncols: int, all_hp: bool):
        self.nrows = nrows
        self.ncols = ncols
        self.all_hp = all_hp


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

        - - - SOURCE / MORE INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_ADR.html
        '''
        if self.all_hp:
            # 1) start with broad parameter ranges
            hyperpars = {
                'maxiter': [5, 1000], # as higher as slower, but lower values not much tested, try lower maxiters
                'abstol': [1e-15, 0.2], # speed depends more on maxiter, higher values preferred
            }
        else:
            # 2) updated hyperparameters
            hyperpars = {
                'maxiter': [2, 100],
                'abstol': [0.01, 0.2]
        }
        return 'r_adr', 'Rfun', hyperpars # R fun is just a filler




    def funR_mds(self):
        '''
        better results than similar method lmds and no optimization needed here
        - - - DESCRIPTION - - -
        (Classical) Multidimensional Scaling performs a classical Multidimensional Scaling (MDS) using Rcpp and
        Rcpp Armadillo package to achieve faster performance than cmdscale.

        - - - PARAMETERS - - -
        ndim
            an integer-valued target dimension.

        - - - SOURCE / MORE INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_MDS.html
        '''
        hyperpars = {'empty': ''}
        return 'r_mds', 'Rfun', hyperpars # R fun is just a filler


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

        - - - SOURCE / MORE INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_PPCA.html
        '''
        hyperpars = {'empty':  ''}
        return 'r_ppca', 'Rfun', hyperpars # R fun is just a filler


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

        - - - SOURCE / MORE INFORMATION - - -
        https://kisungyou.com/Rdimtools/reference/linear_RPCAG.html
        '''
        # one of the best best but one of the slowest methods
        kmax = max(self.nrows/10, 2)
        if self.all_hp:
            # 1) start with broad parameter ranges
            hyperpars = {
            'k': [2, max(kmax, 2)] # each dataset seems to have a sweet spot, no effect on speed, leave it like it is
            }
        else:
            # 2) updataed hyperparameters
            hyperpars = {
                'k': [2, max(kmax, 2)]
        }
        return 'r_rpcag', 'Rfun', hyperpars # R fun is just a filler


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

        - - - SOURCE - - -
        https://kisungyou.com/Rdimtools/reference/nonlinear_ISPE.html
        '''
        if self.all_hp:
            # equal or better than pca for some of the datasets
            # 1) start with broad parameter ranges
            hyperpars = {
                    'proximity': [0, 2],  # euclidean (default) is the best, minkowski the slowest
                    'cutoff': [0.01, 100.0],  # no effect on loss and speed.
                    'C': [1, 500],  # ncycles no effect on loss, as lower as faster, best value: 100
                    'S': [1, 500],  # nupdates per cycle, as lower as faster, best range: 1...100
                    'lambda': [0.01, 100.0],  # initial learning rate, no effect on loss and speed
                    'drate': [0.01, 0.999],  # multiplier for lambda in each cycle, no effect on loss and speed
            }
        else:
            # 2) updated parameters
            hyperpars = {
                    'S': [1, 100],
                    'drate': [0.01, 0.999],
                    }
        return 'r_ispe', 'Rfun', hyperpars # R fun is just a filler