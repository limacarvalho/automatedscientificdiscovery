from sklearn.manifold import trustworthiness
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback

'''
- T&C and MRREs try to detect what goes wrong in a given embedding, 
- LCMC accounts for things that work well -overall performance of an NLDR 
- The strength of T&C and MRREs is their ability to distinguish two sorts of undesired events. 
'''


def fun_kmax(data_high: np.array) -> int:
    '''
    calculates the kmax value.
    kmax is used to calculate Q-matrix based metrizes such as trustworthiness or continuity.
    I choose 5 percent of number of datapoints (rows) as a good kmax value.
    :param data_high: high dimesnional data
    :return: int of kmax
    '''
    return int(data_high.shape[0] * 0.05)



class Metrics:

    def __init__(self,
                 fun_id: str,
                 data_high: np.array,
                 data_low: np.array,
                 kmax: int = None
                 ):
        '''

        :param fun_id: function identifier (only used for error messages)
        :param data_high: high_dimensional data
        :param data_low: low dimensional data
        :param kmax: kmax if customized, else its 5 percent of the number of rows
        '''
        if fun_id:
            self.fun_id = fun_id
        self.data_high = data_high
        self.data_low  = data_low

        if kmax is not None:
            self.kmax = fun_kmax(self.data_high)
        else:
            self.kmax = kmax



    def adjust_data_size(self):
        '''
        Coranking matrix Q is very slow for huge datasets.
        If has more then 3000 rows, every nth row will be selected.
        Thereefore it is important to know that the coranking matrix and metrizes
        for large datasets > 3000 rows are estimates.
        returns: the number of every which row is selected
        '''
        length = self.data_high.shape[0]
        step   = 2000
        nth_row = max(int(round(length/step,0)), 1) # must be minimum 1
        return nth_row



    def coranking_matrix(self,
                         data_high: np.array,
                         data_low: np.array
                         ) -> (np.array, np.array, np.array):
        '''
        check if this is licenced!
        Generates a co-ranking matrix and arrays of flat rankings for high and low dimensional data.
        :param high_data: DataFrame containing the higher dimensional data.
        :param low_data: DataFrame containing the lower dimensional data.
        :returns: the co-ranking matrix, flattened rankings of low and high data
        '''
        n, m = data_high.shape
        high_distance = distance.squareform(distance.pdist(data_high))
        low_distance  = distance.squareform(distance.pdist(data_low))

        high_ranking  = high_distance.argsort(axis=1).argsort(axis=1)
        low_ranking   = low_distance.argsort(axis=1).argsort(axis=1)

        # avoid divisions by zero (NaN, InF)
        high_ranking_flat = high_ranking.flatten() + 1
        low_ranking_flat  = low_ranking.flatten() + 1

        Q, xedges, yedges = np.histogram2d(high_ranking_flat, # high_ranking.flatten(),
                                           low_ranking_flat,  # low_ranking.flatten(),
                                           bins=n)
        return Q[1:, 1:], high_ranking_flat, low_ranking_flat


    def trustworthiness_(self, data_high: np.array, data_low:np.array, kmax: int) -> float:
        '''
        Quality of dim reduction: parameter
        Expresses to what extent the local structure is retained. eg: false positives
        The trustworthiness is within [0, 1]. It is defined as
        Any unexpected nearest neighbors in the output space are penalised in proportion to their
        rank in the input space. n_neighbors (default) = 5, we use kmax which is 5 percent of n-rows
        --- INFORMATION ---
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
        '''
        trust = trustworthiness(data_high, data_low, n_neighbors=kmax, metric='euclidean')
        return round(trust, 3)


    def lcmc_(self, Q):
        """
        The local continuity meta-criteria measures the number of mild
        intrusions and extrusions. This can be thought of as a measure of the
        number of true postives.
        I just found this reference:
        https://coranking.readthedocs.io/en/latest/_modules/coranking/metrics.html#LCMC
        However, the code is a faster implementation of the original code:

        summation = 0.0
        for k in range(K):
            for l in range(K):
                summation += Q[k, l] # k=rows, l=cols
        lcmc = (K / (1. - n)) + (1. / (n*K)) * summation
        lcmc = 1 / (n*K) * np.sum(li)

        Args:
            Q: the co-ranking matrix to calculate continuity from
            k (int): the number of neighbours to use.
        Returns:
            The LCMC metric for the given K
        """
        n = Q.shape[0]
        li = [Q[k, l] for l in range(self.kmax) for k in range(self.kmax)] # 2 x quicker than loop
        lcmc = round(1 / (n*self.kmax) * np.sum(li), 3)
        return lcmc


    def mean_squared_error_(self, high_data_flat: np.array, low_data_flat: np.array, mse_=True) -> float:
        '''
        calculates the mean_sqauerd error.
        --- INFORMATION ---
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#
        :param mse: if True: se is returned, if False: rmse is returned
        :return: mse or rmse
        '''
        mse = mean_squared_error(y_true=high_data_flat,
                                 y_pred=low_data_flat,
                                 squared=mse_)
        return np.round(mse, 3)


    def mean_absolute_error_(self, high_data_flat: np.array, low_data_flat: np.array) -> float:
        '''
        calculates the mean absolute error
        --- INFORMATION ---
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#
        :param high_data_flat: flattened high data
        :param low_data_flat: flattened low data
        :return: mae
        '''
        mae = mean_absolute_error(y_true = high_data_flat,
                                  y_pred = low_data_flat)
        return np.round(mae, 3)


    def r2_score_(self, high_data_flat: np.array, low_data_flat: np.array) -> float:
        '''
        calculates the r2 score.
         --- INFORMATION ---
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#
        :return: r2 score
        '''
        r2 = r2_score(y_true=high_data_flat, y_pred=low_data_flat)
        return np.round(r2, 3)


    def empty_array(self):
        '''
        empty array for try catch exceptions
        :return:
        '''
        return np.array([[1, 2], [1, 2]])


    def metrix_all(self):
        '''
        here we collect several coranking based metrizes and return them in a dictionary.
        :return:
        '''
        error_messages = '! errors ' + str(self.fun_id) + ' '

        try:
            nth_row = self.adjust_data_size()
        except:
            nth_row = 1
            error_messages = error_messages + 'nth_row '

        try:
            data_high_reduced = self.data_high[::nth_row]
        except:
            data_high_reduced = self.empty_array()
            error_messages = error_messages + 'data_high '

        # kmax based on nth_row, nrows reduced based on shape of reduced dataset
        kmax_ = int(self.kmax / nth_row)
        nrows_reduced = data_high_reduced.shape[0]

        try:
            data_low_reduced  = self.data_low[::nth_row]
        except:
            data_low_reduced = self.empty_array()
            error_messages = error_messages + 'data_low '

        # trustworthiness
        try:
            trust = self.trustworthiness_(data_high_reduced,
                                          data_low_reduced,
                                          kmax=kmax_)
        except:
            trust = 0
            error_messages = error_messages + 'trustworthiness '

        # continuity
        try:
            cont = self.trustworthiness_(data_low_reduced,
                                         data_high_reduced,
                                         kmax=kmax_)
        except:
            cont = 0
            error_messages = error_messages + 'continuity '

        # coranking matrix
        try:
            Q, high_data_flat, low_data_flat = self.coranking_matrix(data_high_reduced,
                                                                     data_low_reduced)
        except:
            Q, high_data_flat, low_data_flat = np.array([[1, 2],[1,2]]), 0, 0
            error_messages = error_messages + 'Q-matrix '

        # lcmc
        try:
            lcmc = self.lcmc_(Q=Q)
        except:
            lcmc = 0
            error_messages = error_messages + 'lcmc '

        # MAE - normalized
        try:
            # mean absolute error, normalized by number of rows, in this way we get
            # a measure between 0...1 which can be compared to the others (R2, trust, continuity...)
            mae_norm = self.mean_absolute_error_(high_data_flat, low_data_flat)
            mae_norm = np.round(1-(mae_norm / nrows_reduced), 3)
        except:
            mae_norm = 0
            error_messages = error_messages + 'mae_norm '

        # R2
        try:
            r2 = self.r2_score_(high_data_flat, low_data_flat)
            r2 = np.round(r2, 3)
        except:
            r2 = 0
            error_messages = error_messages + 'r2'

        # Print Error messages, we collect the error messages, because when one measure fails we
        # usulaly have several failures, and too many error messages.
        if len(error_messages) > 10 + len(self.fun_id): print(error_messages)

        dict_results = {'Q': Q, # np.array of coranking matrix
                        'kmax': self.kmax, # kmax
                        'trust': trust, # trustworthiness
                        'cont': cont, # continuity
                        'lcmc':lcmc, # lcmc
                        'mae_norm': mae_norm, # mean absolute error, normalized by number of rows
                        'r2': r2
                        }
        return dict_results







    # other metrizes which are not used above:

    # MAE
    # try:
    #     mae = self.mean_absolute_error_(high_data_flat, low_data_flat)
    #     mae = np.round(mae, 3)
    # except:
    #     mae = 0
    #     error_messages = error_messages + 'mae '

    # MSE
    # try:
    #     mse = self.mean_squared_error_(high_data_flat, low_data_flat, mse_=True)
    #     mse = np.round(mse, 3)
    # except:
    #     mse = 0
    #     error_messages = error_messages + 'mse '

    # RMSE
    # try:
    #     rmse = self.mean_squared_error_(high_data_flat, low_data_flat, mse_=False)
    #     rmse = np.round(rmse, 3)
    # except:
    #     rmse = 0
    #     error_messages = error_messages + 'rmse '