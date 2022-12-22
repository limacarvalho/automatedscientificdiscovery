import numpy as np
import pandas as pd
from typing import Union
from utils_logger import logger
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances as pdist
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


class Crca(BaseEstimator):
    '''
    Curvilinear Component Analysis implementation for Python adapted from the below source.
    We thank author: Felipe Augusto Machado for the implementation of this algorithm.
    source: https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python/blob/master/cca.py

    This code was implemented using the following article https://ieeexplore.ieee.org/document/554199 and the
    book "Nonlinear Dimensionality Reduction" by Michel Verleysen, John A. R. Lee.

    The Algorithm
    Curvilinear Component Analysis (CCA) is a Non-linear Dimensionality Reduction technic, basead on the
    distance betweens the points: it tries to create a new space, with a small dimension, which the distance
    betweens points are equal in the original space (if this distance is smaller than Lambda, one of the
    parameters of the algorithm).
    ''

    we adapted it to the sklearn and our nomenclature and added some sklearn functionality for seamless
    integration into our pipeline.
    '''
    def __init__(self,
                 n_components: Union[int, None],
                 lmbda: Union[float, None],
                 alpha: Union[float, None],
                 max_iter: Union[int, None],
                 tol: Union[float, None]
        ):
        """
        Creates the CCA object.
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
        """
        self.n_components = n_components
        self.lmbda = lmbda
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol


    def _stress(self, dist_y: np.array, dist_x: np.array, lmbda: float):
        """
        Calculates the stress function (quadratic cost function) given the distances in original space (dist_y)
        and the distances in reduced space (dist_x).

        Parameters
        ----------
        dist_y : numpy.array
            Array with distances in original space.
        dist_x : numpy.array
            Array with distances in reduced space.
        lmbda : float
            Distance limit to update points.
        """
        stress = np.mean((dist_y - dist_x) ** 2 * (lmbda > dist_x).astype(int)) # 0/1
        return stress


    def fit_transform(self, data_high: Union[pd.DataFrame, np.array]):
        """
        Method to reduce dimension. Every iteration run all points. The new data
        is stored in attribute 'data_x'.
        Parameters
        ----------
        data_high : numpy.array
            Array with the original data.

        Returns
        -------
        data_x : numpy.array
            New data representation.
        """
        self.data_y = data_high
        n = len(data_high)
        triu = np.triu_indices(n, 1)
        dist_y = pdist(data_high)
        data_x = PCA(self.n_components).fit_transform(data_high)
        stress = np.zeros(self.max_iter)

        for q in range(self.max_iter):
            alpha = max(0.001, self.alpha / (1 + q))
            lmbda = max(0.1, self.lmbda / (1 + q))
            for i in range(n):
                dist_x = cdist(data_x[i].reshape(1, -1), data_x)
                dy = np.delete(dist_y[i], i, 0)
                dx = np.delete(dist_x, i, 1)
                delta_x = (alpha * (lmbda > dx) * (dy - dx) / dx).reshape((-1, 1)) * (
                            data_x[i] - np.delete(data_x, i, 0))
                delta_x = np.insert(delta_x, i, 0, axis=0)
                data_x -= delta_x
            dist_x = pdist(data_x)
            stress[q] = self._stress(dist_y[triu], dist_x[triu], lmbda)
            if stress[q] < self.tol:
                break

        return data_x



# CRDA
# ## MAIN COMPUTATION
#   #   1. ISOMAP-type Curvilinear Distance
#   nbdstruct = aux.graphnbd(X,method="euclidean",
#                            type=nbdtype,symmetric=nbdsymmetric)
#   D     = nbdstruct$dist
#   Dmask = nbdstruct$mask
#   nD    = ncol(D)
#   if (algweight){
#     wD = Dmask*D
#     idnan = is.na(wD)
#     wD[idnan] = 0
#   } else {
#     wD = matrix(as.double(Dmask),nrow=nD)
#   }
#   Xij = aux.shortestpath(wD)

#   #   2. Initialization via PCA
#   Yinit = do.pca(X, ndim=ndim)$Y

#   #   3. vecselector for random-number generation
#   vecselector = as.vector(sample(0:(nrow(X)-1), maxiter, replace=TRUE))

#   #   4. main computation
#   Youtput = method_crca(Xij,Yinit,lambda,alpha,maxiter,tolerance,vecselector)
#
#   #------------------------------------------------------------------------
#   ## RETURN OUTPUT
#   trfinfo = list()
#   trfinfo$type = "null"
#   trfinfo$algtype = "nonlinear"
#   result = list()
#   result$Y = Youtput$Y
#   result$niter = Youtput$niter
#   result$trfinfo = trfinfo
#   return(result)
# }


# ISOMAP
# 4. process : neighborhood selection
#   nbdstruct = aux.graphnbd(pX,method="euclidean",
#                            type=nbdtype,symmetric=nbdsymmetric)
#   D     = nbdstruct$dist
#   Dmask = nbdstruct$mask
#   nD    = ncol(D)
#   # 5. process : nbd binarization
#   if (algweight){
#     wD = Dmask*D
#     idnan = is.na(wD)
#     wD[idnan] = 0
#   } else {
#     wD = matrix(as.double(Dmask),nrow=nD)
#   }
#   # 6. process : shortest path
#   sD = aux.shortestpath(wD)
#
# here comes the pca part


#   # 7. main computation
#   output = method_mdsD(sD);
#   eigvals = rev(output$eigval)
#   eigvecs = output$eigvec[,rev(seq_len(length(eigvals)))]
#
#   # 8. output
#   matS = diag(sqrt(eigvals[1:ndim]))
#   matU = eigvecs[,1:ndim]
#   result = list()
#   result$Y = t(matS %*% t(matU));
#   result$trfinfo = trfinfo
#   return(result)