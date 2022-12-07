# from __future__ import division, print_function
# MIT License
#
# Copyright (c) 2019 Deep Ganguli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO: is currently being implemented
import numpy as np
from sklearn.base import BaseEstimator
from typing import Union


class rpca(BaseEstimator):

    def __init__(self,
                 n_components: Union[int, None],
                 mu = Union[None, float],
                 lmbda = Union[None, float],
                 tol = Union[None, float],
                 max_iter=1000
                 ):

        self.n_components = n_components
        self.mu = mu
        self.lmbda = lmbda
        self.tol = tol
        self.max_iter = max_iter


    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))


    def fit_transform(self, D):

        n1, n2 = D.shape
        S = np.random.rand(n1, n2)

        percent_cols = self.n_components / n2
        D[S < 0.2] = 0

        self.S = np.zeros(D.shape)
        self.Y = np.zeros(D.shape)

        if self.mu:
            self.mu = self.mu
        else:
            self.mu = np.prod(D.shape) / (4 * np.linalg.norm(D, ord=1))

        self.mu_inv = 1 / self.mu

        if self.lmbda:
            self.lmbda = self.lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(D.shape))


        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(D.shape)

        if self.tol:
            _tol = self.tol
        else:
            _tol = 1E-7 * self.frobenius_norm(D)

        # this loop implements the principal component pursuit (PCP) algorithm
        # located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < self.max_iter:
            Lk = self.svd_threshold(D - Sk + self.mu_inv * Yk, self.mu_inv) # this line implements step 3
            Sk = self.shrink(D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda) # this line implements step 4
            Yk = Yk + self.mu * (D - Lk - Sk)   # this line implements step 5
            err = self.frobenius_norm(D - Lk - Sk)
            iter += 1
            # if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
            #     print('iteration: {0}, error: {1}'.format(iter, err))

        # self.L = Lk
        # self.S = Sk
        return Lk # , Sk




    # def plot_fit(self, size=None, tol=0.1, axis_on=True):
    #
    #     n, d = self.D.shape
    #
    #     if size:
    #         nrows, ncols = size
    #     else:
    #         sq = np.ceil(np.sqrt(n))
    #         nrows = int(sq)
    #         ncols = int(sq)
    #
    #     ymin = np.nanmin(self.D)
    #     ymax = np.nanmax(self.D)
    #     print('ymin: {0}, ymax: {1}'.format(ymin, ymax))
    #
    #     numplots = np.min([n, nrows * ncols])
    #     plt.figure()
    #
    #     for n in range(numplots):
    #         plt.subplot(nrows, ncols, n + 1)
    #         plt.ylim((ymin - tol, ymax + tol))
    #         plt.plot(self.L[n, :] + self.S[n, :], 'r')
    #         plt.plot(self.L[n, :], 'b')
    #         if not axis_on:
    #             plt.axis('off')