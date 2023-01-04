
# def timeit_steps(txt, step, start):
#     t = time.time()
#     time_ = round(t - start, 3)
#     msg = txt + step + ': ' + str(time_) + ' '
#     return t, msg


# def fun_kmax(data_high: np.array) -> int:
#     '''
#     calculates the kmax value.
#     kmax is used to calculate Q-matrix based metrizes such as trustworthiness or continuity.
#     We found 5 percent of number of datapoints (rows) as a good kmax value.
#     :param data_high: high dimesnional data
#     :return: int of kmax
#     '''
#     return int(data_high.shape[0] * 0.05)
#
#
# # kmax adjusted to the nth_row parameter
# kmax_ = int(self.kmax / nth_row)
#
# if kmax is not None:
#     self.kmax = fun_kmax(self.data_high)
# else:
#     self.kmax = kmax



# from sklearn.manifold import trustworthiness
# def trustworthiness_(self, data_high: np.array, data_low: np.array, kmax: int) -> float:
#     '''
#     Quality of dim reduction: parameter
#     Expresses to what extent the local structure is retained. eg: false positives
#     The trustworthiness is within [0, 1]. It is defined as
#     Any unexpected nearest neighbors in the output space are penalised in proportion to their
#     rank in the input space. n_neighbors (default) = 5, we use kmax which is 5 percent of n-rows
#
#     --- INFORMATION ---
#     https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
#
#     :param data_high: np.array, containing the higher dimensional data.
#     :param data_low: np.array, containing the lower dimensional data.
#     :param kmax: int, maximum number of neighbors
#     :return: float, trustworthiness value
#     '''
#     trust = trustworthiness(data_high, data_low, n_neighbors=kmax, metric='euclidean')
#     return round(trust, 3)
#
#
# def lcmc_(self, Q: np.array) -> float:
#     '''
#     The local continuity meta-criteria measures the number of mild
#     intrusions and extrusions. This can be thought of as a measure of the
#     number of true positives.
#     reference: https://coranking.readthedocs.io/en/latest/_modules/coranking/metrics.html#LCMC
#     :param Q: np.array, the co-ranking matrix to calculate continuity from
#     :return: float, lcmc value
#     '''
#     n = Q.shape[0]
#     li = [Q[k, l] for l in range(self.kmax) for k in range(self.kmax)]
#     lcmc = round(1 / (n * self.kmax) * np.sum(li), 3)
#     return lcmc
#
#
# def mean_squared_error_(self, high_data_flat: np.array, low_data_flat: np.array, mse_=True) -> float:
#     '''
#      calculates the mean_squared error.
#     --- INFORMATION ---
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#
#     :param high_data_flat: np.array, flattened high data
#     :param low_data_flat: np.array, flattened high data
#     :param mse_: bool, if True: se is returned, if False: rmse is returned
#     :return: float, se or mse
#     '''
#     mse = mean_squared_error(y_true=high_data_flat, y_pred=low_data_flat, squared=mse_)
#     return np.round(mse, 3)

# other metrizes which are not used:
# # trustworthiness, calculation takes very long (617cols, 7800rows = 6sec)
# try:
#     trust = self.trustworthiness_(data_high_reduced,
#                                   data_low_reduced,
#                                   kmax=kmax_)
# except:
#     trust = 0
#     error_messages = error_messages + 'trustworthiness '
# start, msg = timeit_steps(txt=msg, step='trus', start=start)
#
# # continuity calculation takes very long (617cols, 7800rows = 6sec)
# try:
#     cont = self.trustworthiness_(data_low_reduced,
#                                  data_high_reduced,
#                                  kmax=kmax_)
# except:
#     cont = 0
#     error_messages = error_messages + 'continuity '
# start, msg = timeit_steps(txt=msg, step='cont', start=start)

# # lcmc
# try:
#     lcmc = self.lcmc_(Q=Q)
# except:
#     lcmc = 0
#     error_messages = error_messages + 'lcmc '
# start, msg = timeit_steps(txt=msg, step='lcmc', start=start)

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

# # Mean Absolute Error, normalized by range of rows, errorenous calculation
# try:
#     # mean absolute error, normalized by number of rows, possible values are 0.5...1
#     mae_norm_ = self.mean_absolute_error_(high_data_flat, low_data_flat)
#     mae_norm = np.round(1-(mae_norm_ / nrows_reduced), 3)
# except:
#     mae_norm = 0
#     error_messages = error_messages + 'mae_norm '
# start, msg = timeit_steps(txt=msg, step='mae1', start=start)