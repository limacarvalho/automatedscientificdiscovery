import logging
import time

import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error, r2_score
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()


class Metrics:
    """
    collection of functions to asses the relative error of dimensionality reduction.
    High and low dimensional data can be compared with the coranking matrix:
    1) distances among datapoints
    2) ranking of distances (closest is 0, farest is data.shape[0]
    3) rank differences: difference between rankings of high and low data
        - coranking matrix: how many samples of rank k become rank l (count)
        - relative error: 1 - (mean absolute error of rank differences / mean(range rank))
    """

    def __init__(self, fun_id: str, data_high: np.array, data_low: np.array):
        """
        :param fun_id: str,
            function identifier (only used for error messages)
        :param data_high: np.array,
            high_dimensional data
        :param data_low: np.array,
            low dimensional data
        """
        if fun_id:
            self.fun_id = fun_id
        self.data_high = data_high
        self.data_low = data_low

    def adjust_data_size(self) -> int:
        """
        Calculation of the coranking matrix Q is slow for large datasets.
        If there are more than 2000 rows, every nth row will be selected.
        Therefore, Q matrix and quality measurements are estimates for datasets > 2000 rows.
        :returns: int,
            the number of every which row is selected
        """
        length = self.data_high.shape[0]
        step = 2000
        nth_row = max(int(round(length / step, 0)), 1)  # must be minimum 1
        return nth_row

    def coranking_matrix(self, data_high: np.array, data_low: np.array) -> (np.array, np.array, np.array):
        """
        Generates a co-ranking matrix and arrays of rankings for high and low dimensional data.
        Its used to calculate the quality of the dimensionality reduction.
        Short: it summarizes the differences of the rankings between datapoints for high and low
        dimensional data. It can be used for several metrizes.
        please find more details here: Wouter Lueks et al. 2011
        :param data_high: np.array,
            high dimensional dataset.
        :param data_low: np.array,
            low dimensional dataset.
        :returns: (np.array, np.array, np.array),
            np.array:
                the co-ranking matrix
            np.array:
                flattened ranking matrix of high dimensional dataset
            np.array:
                flattened ranking matrix of low dimensional dataset
        """
        n, m = data_high.shape

        # calculate distances between datapoints
        high_distance = distance.squareform(distance.pdist(data_high))
        low_distance = distance.squareform(distance.pdist(data_low))

        # distances are ranked 1...nrows
        high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
        low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

        # flatten the array eg. n-dimensional array to 1D array. + 1 in order
        # to avoid divisions by 0 (ranks and differences will not change).
        high_ranking_flat = high_ranking.flatten() + 1
        low_ranking_flat = low_ranking.flatten() + 1

        # Coranking matrix Q
        Q, _, __ = np.histogram2d(high_ranking_flat, low_ranking_flat, bins=n)
        return Q[1:, 1:], high_ranking_flat, low_ranking_flat

    def relative_error(self, high_data_flat: np.array, low_data_flat: np.array, nrows_reduced: int) -> float:
        """
        calculates the relative error from the mean absolute error.
        Relative error: 1 - (mean absolute error of rank differences / mean(range ranks))

        --- INFORMATION ---
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#
        :param nrows_reduced:
            number of rows of reduced dataset eg: number of ranks
        :param high_data_flat: np.array,
            flattened high data (reduced size)
        :param low_data_flat: np.array,
            flattened low data (reduced size)
        :return: float,
            relative error
        """
        mae = mean_absolute_error(y_true=high_data_flat, y_pred=low_data_flat)
        rel_err = 1 - (mae / np.mean(range(nrows_reduced)))
        return np.round(rel_err, 3)

    def r2_score_(self, high_data_flat: np.array, low_data_flat: np.array) -> float:
        """
        calculates the r2 score.
         --- INFORMATION ---
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#
        :param high_data_flat: np.array,
            flattened high data
        :param low_data_flat: np.array,
            flattened high data
        :return: float,
            R squared value
        """
        r2 = r2_score(y_true=high_data_flat, y_pred=low_data_flat)
        return np.round(r2, 3)

    def empty_array(self) -> np.array:
        """
        empty array for try catch exceptions
        :return: np.array,
            empty array with dummy values
        """
        return np.array([[1, 2], [1, 2]])

    def metrix_all(self) -> dict:
        """
        calculation of many coranking based measurements.
        here we collect several coranking based measurments and return them in a dictionary.
        We testet several loss functions (mean-squared error, trustworthiness etc.) and choose
        the relative error as quality measurement.
        :return: dict,
            with coranking matrix based measurements:
                'Q': Q,            # np.array: coranking matrix
                'rel_err': rel_err, # float: relative error 0...1 (1=perfect dimensionality reduction)
                'r2': r2            # float: r-squared value
        """
        error_messages = "! errors " + str(self.fun_id) + " "

        # nth row for reducing the size of the dataset
        time.time()
        try:
            nth_row = self.adjust_data_size()
        except:
            nth_row = 1
            error_messages = error_messages + "nth_row "

        # reduce the high_data and low_data size to save time
        try:
            data_high_reduced = self.data_high[::nth_row]
        except:
            data_high_reduced = self.empty_array()
            error_messages = error_messages + "data_high "

        try:
            data_low_reduced = self.data_low[::nth_row]
        except:
            data_low_reduced = self.empty_array()
            error_messages = error_messages + "data_low "

        # nrows of reduced dataset
        nrows_reduced = data_high_reduced.shape[0]

        # coranking matrix
        try:
            Q, high_data_flat, low_data_flat = self.coranking_matrix(data_high_reduced, data_low_reduced)
        except:
            Q, high_data_flat, low_data_flat = np.array([[1, 2], [1, 2]]), 0, 0
            error_messages = error_messages + "Q-matrix "

        # Relative error: 1 - (mean absolute error of rank differences / mean(range ranks))
        try:
            rel_err = self.relative_error(high_data_flat, low_data_flat, nrows_reduced)
        except:
            rel_err = 0
            error_messages = error_messages + "rel_err "

        # R2
        try:
            r2 = self.r2_score_(high_data_flat, low_data_flat)
            r2 = np.round(r2, 3)
        except:
            r2 = 0
            error_messages = error_messages + "r2"

        # Print Error messages, we collect the error messages, because when one measure fails we
        # usulaly have several failures, and too many error messages.
        if len(error_messages) > 10 + len(self.fun_id):
            logging.error(error_messages)  # here we dont want the exact description, thats too much information.

        dict_results = {
            "Q": Q,  # coranking matrix: np.array,
            "rel_err": rel_err,  # relative error: float, 0...1 (1=perfect)
            "r2": r2,  # r-squared value: float
        }
        return dict_results
