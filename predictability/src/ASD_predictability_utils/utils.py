import numpy as np
import itertools
import time
import dcor
import ray
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from scipy.special import comb

from tpot import TPOTRegressor

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils_logger import logger

from src.ASD_predictability_utils.tpot_config import regressor_config_dict_ASD

# dict for mapping the scoring input strings to the respective sklearn options
scoring_dict = {
    "r2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
    "neg_mean_absolute_percentage_error": "neg_mean_absolute_percentage_error",
    "RMSE": "neg_root_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error"
}


def get_column_combinations(all_cols, inputs, outputs, targets=[], amount_only=False, return_targets=False):
    # TODO: update return: different if amount_only or if return_targets
    """
    This function creates the list of all column combinations that should be analysed. It can also be used to determine
    the amount of combinations before the run in order to get a feeling for the runtime, e.g.

    If via `amount_only` only the amount of combinations should be returned, this accounts for

        - binom(N, I)*binom(N-I, O) = N!/(I!*O!) for N data columns considered and predicting I-O-connections of I-many
            inputs and O-many targets (outputs)
        - binom(N-T, I)*binom(T, O) if, on top of the above, there are T-many data columns (out of the N) set to be the
            exclusive targets

    :param all_cols: list
        The (sub-)set of columns that should be considered
    :param inputs: int
        The number of inputs per fit. For a 4-2 fit, it is 4.
    :param outputs: int
        The number of outputs per fit. For a 4-2 fit, it is 2.
    :param targets: list, default=[]
        Specify columns that should be treated exclusively as targets.
    :param amount_only: bool, default=False
        Specify whether only the amount of different column combinations should be returned.
    :param return_targets: boolean, default=False
        Specify whether the possible target combinations are returned.
    :return: list
        List of combination tuples. In each tuple, the first `inputs`-many are the inputs (4 in the example above)
        and the remaining `outputs`-many the outputs (2 in the example above). If `amount_only` is set, amount of
        combinations is returned.
    """

    # assertion to check that the numbers can be matched
    #
    # inputs & outputs vs. given columns
    try:
        assert inputs + outputs <= len(all_cols)
    except AssertionError as err:
        logger.error("Assertion failed: More input and output columns specified than there are columns.", exc_info=True)
        raise err
    #
    # outputs vs. targets
    if targets:
        try:
            assert outputs <= len(targets)
        except AssertionError as err:
            logger.error("Assertion failed: More output columns specified than there are in the targets list.",
                         exc_info=True)
            raise err

    # get the combination tuples
    if not amount_only:
        # initialise final list of column combinations
        col_combinations = []

        # if we fix some columns to be considered as targets only, drop these from input list
        if targets:
            all_input_cols = [x for x in all_cols if x not in targets]
        else:
            all_input_cols = all_cols

        # first, draw possible input tuples
        input_combinations = list(itertools.combinations(all_input_cols, inputs))
        # now go through all possible input combinations
        for i in input_combinations:
            if targets:
                # if there are pre-defined targets, use these as possible targets
                curr_output_cols = targets
            else:
                # otherwise, use remaining columns
                curr_output_cols = [o for o in all_input_cols if o not in i]
            # now draw from that list all currently possible output combinations
            output_combinations = list(itertools.combinations(curr_output_cols, outputs))
            # append all currently possible output combinations to the current input columns and save in final list
            for oc in output_combinations:
                col_combinations.append(i + (*oc,))
        if not return_targets:
            return col_combinations
        else:
            return col_combinations, output_combinations

    # only get the amount of combination tuples according to the formulae in the docstring description
    else:
        if targets:
            amount_combinations = comb(len(all_cols) - len(targets), inputs, exact=True) * \
                                  comb(len(targets), outputs, exact=True)
        else:
            amount_combinations = comb(len(all_cols), inputs, exact=True) * \
                                  comb(len(all_cols) - inputs, outputs, exact=True)
        return amount_combinations


def data_prep_split(data, input_cols, output_cols, random_state_split):
    """
    Performs split of the data-dataframe into inputs and outputs and then applies the train-test-split.
    :param data: pandas DataFrame
        Dataframe containing all the data.
    :param input_cols: int
        The number of input columns for the fit. For a 4-1 fit, input_cols = 4.
    :param output_cols: int
        The number of target columns for the fit. For a 4-1 fit, output_cols = 1.
    :param random_state_split: int, default=1
        Specifies shuffling during `sklearn.model_selection`'s `train_test_split`. Set to a specific integer value for
        reproducibility.
    :return: list, list, list, list
        Lists containing the train input values, test input values, train target values and test target values.
    """
    # get x and y value(s)
    curr_x = np.array(data[input_cols])
    curr_y = np.array(data[output_cols])

    # train test split
    curr_X_train, curr_X_test, curr_y_train, curr_y_test = train_test_split(curr_x, curr_y, test_size=.3, shuffle=True,
                                                                            random_state=random_state_split
                                                                            )

    return curr_X_train, curr_X_test, curr_y_train, curr_y_test


def rae(true, predicted):
    """
    Compute the relative absolute error for the predicticted values `predicted` of the true values `true`.
    :param true: list
        List containing the true values.
    :param predicted: list
        List containing the predicted values.
    :return: float
        The relative absolute error of the predicted and true values.
    """

    numerator = np.sum(np.abs(predicted - true))
    denominator = np.sum(np.abs(np.mean(true) - true))

    return numerator / denominator


@ray.remote(num_returns=2)
def parallel_pred_step_MLP(data, curr_tuple, input_cols, hidden_layers, alphas, scaling,
                           max_iter, scoring, verbose, n_jobs, counter_tuples, len_data_tuples, random_state_split):
    """
    The explicit step of predictability prediction for one combination tuple if the MLP is chosen as method. It performs
    several reference fits (linear, power law (if possible), mean) along a GridSearchCV of an MLPRegressor with
    different hidden layer sizes `hidden_layers`, alpha values `alpha` and preprocessing with / without usage of the
    StandardScaler. The scoring can be adjusted via `scoring`.
    :param data: pandas DataFrame (stored by Ray)
        The dataframe containing all relevant data
    :param curr_tuple: tuple
        The tuple containing the current column combination.
    :param input_cols: int
        Integer specifying the number of input columns. The first `input_cols`-many columns of the `curr_tuple` tuple
        correspond to the input columns.
    :param hidden_layers: list
        Specifies choices for sklearn's `hidden_layer_sizes` during CV fit. The predictability routine uses
        [(12,), (50,), (70, 5,)] as default; also used here if unspecified.
    :param alphas: list
        Specifies choices for sklearn's `alpha` during CV fit. The predictability routine uses [0.001, 0.0001, 0.00001]
        as default; also used here if unspecified.
    :param scaling: str
        Specifies usage of `sklearn.preprocessing`'s `StandardScaler` before fit. Default is "test", can be set to
        "yes" or "no". If "test" (or unspecified or anything but "yes" or "no"), `StandardScaler` becomes part of
        fitting pipeline and benefit of usage / skipping will be evaluated.
    :param max_iter: int
        Specifies maximum number of iterations during fit. Corresponds to `max_iter` within `sklearn.model_selection`'s
        `GridSearchCV`. The predictability routine uses 10000 as default.
    :param scoring: str
        Scoring for the CV fit. Choices are (mapping automatically to the respective `sklearn.metrics` metric):

            - "r2": `r2`,
            - "MAPE": `neg_mean_absolute_percentage_error`,
            - "neg_mean_absolute_percentage_error": `neg_mean_absolute_percentage_error`,
            - "RMSE": `neg_root_mean_squared_error`,
            - "neg_root_mean_squared_error": `neg_root_mean_squared_error`,
            - "MAE": `neg_mean_absolute_error`,
            - "neg_mean_absolute_error": `neg_mean_absolute_error`

    :param verbose: int
        Specifies the verbosity level. Corresponds to `verbose` within `sklearn.model_selection`'s `GridSearchCV`.
    :param n_jobs: int
        Specifies number of jobs to run in parallel. Choose -1 to use all processors. Corresponds to `n_jobs` within
        `sklearn.model_selection`'s `GridSearchCV`.
    :param counter_tuples: int
        Number of current combination tuple within the overall list. Used during the predictability routine to count
        the process of running through the list of combination tuples.
    :param len_data_tuples: int
        Overall amount of different combination tuples. Used during the predictability routine to count the process of
        running through the list of combination tuples.
    :param random_state_split: int
        Specifies shuffling during `sklearn.model_selection`'s `train_test_split`. Set to a specific integer value for
        reproducibility.
    :return: dict, dict
        First dict contains all evaluation metrics, the second one all data (train, test, predict
        values, GridSearch parameters, CV scores). Both are nested dictionaries, where the outermost key corresponds to
        the current combination tuple ("input column 1", ..., "input column `input_cols`", "target column 1", ...,
        "target column `output_cols`").

        The inner dictionaries are then composed of the following keys with the corresponding values:

        metric dict:

            - "TYPE r2",
            - "TYPE RMSE",
            - "TYPE RMSE/std",
            - "TYPE MAPE",
            - "TYPE rae",
            - "TYPE dcor"

            for TYPE in ["MLP", "linear", "mean", "pow. law"]
            (all `None` for "pow. law" if no power law fit performed)

         data dict:

            - "X_train",
            - "X_test",
            - "y_train",
            - "y_test",
            - "y_test_pred",
            - "y_test_pred_linear",
            - "y_test_pred_pl", (if power law fit performed)
            - "y_test_pred_mean",
            - "GridSearchParams",
            - "scores"
    """

    # to measure the current tuple's analysis time
    curr_start = time.time()

    logger.info("Analysing " + str(curr_tuple) + " now.")

    # get current inputs and outputs
    curr_inputs = list(curr_tuple[:input_cols])
    curr_outputs = list(curr_tuple[input_cols:])

    # reduce data to current columns and drop NAs
    curr_data = data[curr_inputs + curr_outputs].dropna()

    # do data preparations and train-test-split
    curr_X_train, curr_X_test, curr_y_train, curr_y_test = data_prep_split(curr_data, curr_inputs, curr_outputs,
                                                                           random_state_split)

    # compute standard deviation of curr_y_test for later scaling of the RMSE
    curr_y_test_std = np.std(curr_y_test)

    curr_y_train = curr_y_train.ravel()

    #
    # y-mean "prediction"
    #
    curr_y_train_mean = np.mean(curr_y_train)
    curr_y_test_pred_mean = curr_y_train_mean * np.ones(len(curr_X_test))
    # metrics
    curr_mean_r2 = r2_score(curr_y_test, curr_y_test_pred_mean)
    curr_mean_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_mean, squared=False)
    curr_mean_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_mean)
    curr_mean_rae = rae(curr_y_test, curr_y_test_pred_mean)
    curr_mean_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_mean)

    #
    # linear regression
    #
    lin_reg = LinearRegression().fit(curr_X_train, curr_y_train)
    curr_y_test_pred_linear = lin_reg.predict(curr_X_test)
    # metrics
    curr_lin_r2 = r2_score(curr_y_test, curr_y_test_pred_linear)
    curr_lin_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_linear, squared=False)
    curr_lin_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_linear)
    curr_lin_rae = rae(curr_y_test, curr_y_test_pred_linear)
    curr_lin_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_linear)

    #
    # power law fit
    #
    if ((curr_X_train > 0).all().all()) and ((curr_X_test > 0).all().all()) and ((curr_y_train > 0).all().all()) and (
            (curr_y_test > 0).all().all()):
        do_pl_fit = True
    else:
        do_pl_fit = False

    if do_pl_fit:
        # log-transform values
        curr_X_train_log = np.log(curr_X_train)
        curr_X_test_log = np.log(curr_X_test)
        curr_y_train_log = np.log(curr_y_train)

        # pow. law fit
        pl_fit = LinearRegression().fit(curr_X_train_log, curr_y_train_log)
        curr_y_test_pred_pl = pl_fit.predict(curr_X_test_log)
        curr_y_test_pred_pl = np.exp(curr_y_test_pred_pl)

        # respective metrics
        curr_pl_r2 = r2_score(curr_y_test, curr_y_test_pred_pl)
        curr_pl_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_pl, squared=False)
        curr_pl_rmse_std = curr_pl_rmse / curr_y_test_std
        curr_pl_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_pl)
        curr_pl_rae = rae(curr_y_test, curr_y_test_pred_pl)
        curr_pl_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_pl)

    # to allow for uniform dicts and later metric dataframes, fill metrics also if no pow. law fit was done.
    else:
        curr_pl_r2 = None
        curr_pl_rmse = None
        curr_pl_rmse_std = None
        curr_pl_mape = None
        curr_pl_rae = None
        curr_pl_dcor = None

    #
    # MLP regression
    #

    # list of hidden layer sizes for GridSearch
    if hidden_layers is None:
        hidden_layers = [(12,),
                         (50,),
                         (70, 5,),
                         # (40,18,3,)
                         ]
    # list of alpha values for GridSearch
    if alphas is None:
        alphas = [0.001,
                  0.0001,
                  0.00001
                  ]

    # via pipeline (with and without scaler, or use both for testing / finding best)
    if scaling == "yes":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(max_iter=max_iter))
        ])
        pipe_params = [
            {'mlp__hidden_layer_sizes': hidden_layers,
             'mlp__alpha': alphas}
        ]
        clf = GridSearchCV(pipe,
                           param_grid=pipe_params,
                           cv=3,
                           scoring=scoring,
                           return_train_score=True,
                           verbose=verbose,
                           n_jobs=n_jobs
                           )
    elif scaling == "no":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(max_iter=max_iter))
        ])
        pipe_params = [{'scaler': ['passthrough'],
                        'mlp__hidden_layer_sizes': hidden_layers,
                        'mlp__alpha': alphas}]
        clf = GridSearchCV(pipe,
                           param_grid=pipe_params,
                           cv=3,
                           scoring=scoring,
                           return_train_score=True,
                           verbose=verbose,
                           n_jobs=n_jobs
                           )
    else:
        if scaling != "test":
            logger.error("The specified usage of scaling ('", scaling, "') is not an allowed option. Allowed options "
                                                                       "are 'yes', 'no', 'test'. Using 'test' now.",
                         exc_info=True)

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(max_iter=max_iter))
        ])
        pipe_params = [{'scaler': ['passthrough'],
                        'mlp__hidden_layer_sizes': hidden_layers,
                        'mlp__alpha': alphas},
                       {'mlp__hidden_layer_sizes': hidden_layers,
                        'mlp__alpha': alphas}]
        clf = GridSearchCV(pipe,
                           param_grid=pipe_params,
                           cv=3,
                           scoring=scoring,
                           return_train_score=True,
                           verbose=verbose,
                           n_jobs=n_jobs
                           )

    clf.fit(curr_X_train, curr_y_train)
    curr_best_params = clf.best_params_
    curr_y_train_pred = clf.predict(curr_X_train)
    curr_y_test_pred = clf.predict(curr_X_test)

    # metrics
    curr_mlp_r2 = r2_score(curr_y_test, curr_y_test_pred)
    curr_mlp_rmse = mean_squared_error(curr_y_test, curr_y_test_pred, squared=False)
    curr_mlp_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred)
    curr_mlp_rae = rae(curr_y_test, curr_y_test_pred)
    curr_mlp_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred)

    # log results
    logger.info(f'{curr_tuple}: \n Train R2 score: {r2_score(curr_y_train, clf.predict(curr_X_train))} \n '
                f'Test R2 score: {r2_score(curr_y_test, curr_y_test_pred)}')

    # save metrics into dict
    curr_metric_dict = {curr_tuple: {"MLP r2": curr_mlp_r2, "linear r2": curr_lin_r2,
                                     "pow. law r2": curr_pl_r2, "mean r2": curr_mean_r2,
                                     "MLP RMSE": curr_mlp_rmse, "linear RMSE": curr_lin_rmse,
                                     "pow. law RMSE": curr_pl_rmse, "mean RMSE": curr_mean_rmse,
                                     "MLP RMSE/std": curr_mlp_rmse / curr_y_test_std,
                                     "linear RMSE/std": curr_lin_rmse / curr_y_test_std,
                                     "pow. law RMSE/std": curr_pl_rmse_std,
                                     "mean RMSE/std": curr_mean_rmse / curr_y_test_std,
                                     "MLP MAPE": curr_mlp_mape, "linear MAPE": curr_lin_mape,
                                     "pow. law MAPE": curr_pl_mape, "mean MAPE": curr_mean_mape,
                                     "MLP rae": curr_mlp_rae, "linear rae": curr_lin_rae,
                                     "pow. law rae": curr_pl_rae, "mean rae": curr_mean_rae,
                                     "MLP dcor": curr_mlp_dcor, "linear dcor": curr_lin_dcor,
                                     "pow. law dcor": curr_pl_dcor, "mean dcor": curr_mean_dcor,
                                     }
                        }

    # save values into dict
    if do_pl_fit:
        curr_data_dict = {curr_tuple: {"X_train": curr_X_train, "X_test": curr_X_test,
                                       "y_train": curr_y_train, "y_test": curr_y_test,
                                       "y_test_pred": curr_y_test_pred, "y_test_pred_linear": curr_y_test_pred_linear,
                                       "y_test_pred_pl": curr_y_test_pred_pl, "y_test_pred_mean": curr_y_test_pred_mean,
                                       "y_train_pred": curr_y_train_pred,
                                       "GridSearchParams": curr_best_params, "scores": clf.cv_results_
                                       }
                          }
    else:
        curr_data_dict = {curr_tuple: {"X_train": curr_X_train, "X_test": curr_X_test,
                                       "y_train": curr_y_train, "y_test": curr_y_test,
                                       "y_test_pred": curr_y_test_pred, "y_test_pred_linear": curr_y_test_pred_linear,
                                       "y_test_pred_mean": curr_y_test_pred_mean,
                                       "y_train_pred": curr_y_train_pred,
                                       "GridSearchParams": curr_best_params, "scores": clf.cv_results_
                                       }
                          }

    logger.info("The analysis of this tuple took " + str(round(time.time() - curr_start, 2)) + "s.")
    # for logging the progress of the analysis
    counter_tuples += 1
    logger.info("-----" + str(counter_tuples) + "/" + str(len_data_tuples) + "-----")

    return curr_metric_dict, curr_data_dict


@ray.remote(num_returns=2)
def parallel_pred_step_kNN(data, curr_tuple, input_cols, scaling,
                           scoring, verbose, n_jobs, counter_tuples, len_data_tuples, random_state_split,
                           greedy=False):
    """
    The explicit step of predictability prediction for one combination tuple if kNN is chosen as method. It performs
    several reference fits (linear, power law (if possible), mean) along a GridSearchCV of a kNN-regressor with
    different k-values (3, 8, 15) and preprocessing with / without usage of the StandardScaler. The scoring can be
    adjusted via `scoring`.
    :param data: pandas DataFrame (stored by Ray)
        The dataframe containing all relevant data
    :param curr_tuple: tuple
        The tuple containing the current column combination.
    :param input_cols: int
        Integer specifying the number of input columns. The first `input_cols`-many columns of the `curr_tuple` tuple
        correspond to the input columns.
    :param scaling: bool
        Specifies usage of `sklearn.preprocessing`'s `StandardScaler` before fit. Can be set to `True`, `False` or
        `None`. If `None` / kept unspecified, `StandardScaler` becomes part of fitting pipeline and benefit of usage /
        skipping will be evaluated. The predictability routine uses `True` as default.
    :param scoring: str
        Specifies usage of `sklearn.preprocessing`'s `StandardScaler` before fit. Default is "test", can be set to
        "yes" or "no". If "test" (or unspecified or anything but "yes" or "no"), `StandardScaler` becomes part of
        fitting pipeline and benefit of usage / skipping will be evaluated.
    :param verbose: int
        Specifies the verbosity level. Corresponds to `verbose` within `sklearn.model_selection`'s `GridSearchCV`.
    :param n_jobs: int
        Specifies number of jobs to run in parallel. Choose -1 to use all processors. Corresponds to `n_jobs` within
        `sklearn.model_selection`'s `GridSearchCV`.
    :param counter_tuples: int
        Number of current combination tuple within the overall list. Used during the predictability routine to count
        the process of running through the list of combination tuples.
    :param len_data_tuples: int
        Overall amount of different combination tuples. Used during the predictability routine to count the process of
        running through the list of combination tuples.
    :param random_state_split: int
        Specifies shuffling during `sklearn.model_selection`'s `train_test_split`. Set to a specific integer value for
        reproducibility.
    :param greedy: boolean, default=False
        Specifies whether the function is run by the greedy predictability routine and then also includes the targets
        in the metric dict
    :return: dict, dict
        First dict contains all evaluation metrics, the second one all data (train, test, predict
        values, GridSearch parameters, CV scores). Both are nested dictionaries, where the outermost key corresponds to
        the current combination tuple ("input column 1", ..., "input column `input_cols`", "target column 1", ...,
        "target column `output_cols`").

        The inner dictionaries are then composed of the following keys with the corresponding values:

        metric dict:

            - "TYPE r2",
            - "TYPE RMSE",
            - "TYPE RMSE/std",
            - "TYPE MAPE",
            - "TYPE rae",
            - "TYPE dcor"

            for TYPE in ["kNN", "linear", "mean", "pow. law"]
            (all `None` for "pow. law" if no power law fit performed)

         data dict:

            - "X_train",
            - "X_test",
            - "y_train",
            - "y_test",
            - "y_test_pred",
            - "y_test_pred_linear",
            - "y_test_pred_pl", (if power law fit performed)
            - "y_test_pred_mean",
            - "GridSearchParams",
            - "scores"
    """
    # if we want to measure the current tuple's analysis time
    curr_start = time.time()

    logger.info("Analysing " + str(curr_tuple) + " now.")

    # get current inputs and outputs
    curr_inputs = list(curr_tuple[:input_cols])
    curr_outputs = list(curr_tuple[input_cols:])

    # reduce data to current columns and drop NAs
    curr_data = data[curr_inputs + curr_outputs].dropna()

    # do data preparations and train-test-split
    curr_X_train, curr_X_test, curr_y_train, curr_y_test = data_prep_split(curr_data, curr_inputs, curr_outputs,
                                                                           random_state_split)

    # compute standard deviation of curr_y_test for later scaling of the RMSE
    curr_y_test_std = np.std(curr_y_test)

    # curr_y_train = curr_y_train.ravel()

    #
    # y-mean "prediction"
    #
    curr_y_train_mean = np.mean(curr_y_train, axis=0)
    curr_y_test_pred_mean = np.outer(np.ones(len(curr_X_test)), curr_y_train_mean)
    # metrics
    curr_mean_r2 = r2_score(curr_y_test, curr_y_test_pred_mean)
    curr_mean_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_mean, squared=False)
    curr_mean_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_mean)
    curr_mean_rae = rae(curr_y_test, curr_y_test_pred_mean)
    curr_mean_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_mean)

    #
    # linear regression
    #
    lin_reg = LinearRegression().fit(curr_X_train, curr_y_train)
    curr_y_test_pred_linear = lin_reg.predict(curr_X_test)
    # metrics
    curr_lin_r2 = r2_score(curr_y_test, curr_y_test_pred_linear)
    curr_lin_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_linear, squared=False)
    curr_lin_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_linear)
    curr_lin_rae = rae(curr_y_test, curr_y_test_pred_linear)
    curr_lin_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_linear)

    #
    # power law fit
    #
    if ((curr_X_train > 0).all().all()) and ((curr_X_test > 0).all().all()) and ((curr_y_train > 0).all().all()) and (
            (curr_y_test > 0).all().all()):
        do_pl_fit = True
    else:
        do_pl_fit = False

    if do_pl_fit:

        curr_X_train_log = np.log(curr_X_train)
        curr_X_test_log = np.log(curr_X_test)
        curr_y_train_log = np.log(curr_y_train)

        pl_fit = LinearRegression().fit(curr_X_train_log, curr_y_train_log)
        curr_y_test_pred_pl = pl_fit.predict(curr_X_test_log)
        curr_y_test_pred_pl = np.exp(curr_y_test_pred_pl)

        # metrics
        curr_pl_r2 = r2_score(curr_y_test, curr_y_test_pred_pl)
        curr_pl_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_pl, squared=False)
        curr_pl_rmse_std = curr_pl_rmse / curr_y_test_std
        curr_pl_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_pl)
        curr_pl_rae = rae(curr_y_test, curr_y_test_pred_pl)
        curr_pl_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_pl)

    # to allow for uniform dicts and later metric dataframes:
    else:
        curr_pl_r2 = None
        curr_pl_rmse = None
        curr_pl_rmse_std = None
        curr_pl_mape = None
        curr_pl_rae = None
        curr_pl_dcor = None

    #
    # kNN regression
    #
    # list of k neighbour values for GridSearch
    k_list = [3, 8, 15]

    # via pipeline (with and without scaler)
    if scaling == "yes":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])
        pipe_params = [
            {'knn__n_neighbors': k_list}
        ]

        clf = GridSearchCV(pipe,
                           param_distributions=pipe_params,
                           cv=3,
                           scoring=scoring,
                           return_train_score=True,
                           verbose=verbose,
                           n_jobs=n_jobs
                           )
    elif scaling == "no":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])
        pipe_params = [{'scaler': ['passthrough'],
                        'knn__n_neighbors': k_list}]
        clf = GridSearchCV(pipe,
                           param_grid=pipe_params,
                           cv=3,
                           scoring=scoring,
                           return_train_score=True,
                           verbose=verbose,
                           n_jobs=n_jobs
                           )
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])
        pipe_params = [{'scaler': ['passthrough'],
                        'knn__n_neighbors': k_list},
                       {'knn__n_neighbors': k_list}]
        clf = GridSearchCV(pipe,
                           param_grid=pipe_params,
                           cv=3,
                           scoring=scoring,
                           return_train_score=True,
                           verbose=verbose,
                           n_jobs=n_jobs
                           )

    clf.fit(curr_X_train, curr_y_train)
    curr_best_params = clf.best_params_
    curr_y_train_pred = clf.predict(curr_X_train)
    curr_y_test_pred = clf.predict(curr_X_test)

    # metrics
    curr_knn_r2 = r2_score(curr_y_test, curr_y_test_pred)
    curr_knn_rmse = mean_squared_error(curr_y_test, curr_y_test_pred, squared=False)
    curr_knn_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred)
    curr_knn_rae = rae(curr_y_test, curr_y_test_pred)
    curr_knn_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred)

    # log results
    # TODO: scores are not printed to logfile!
    logger.info(f'{curr_tuple}: \n Train R2 score: {r2_score(curr_y_train, clf.predict(curr_X_train))} \n '
                f'Test R2 score: {r2_score(curr_y_test, curr_y_test_pred)}')

    # save metrics into dict
    if not greedy:
        curr_metric_dict = {curr_tuple:
                                {"kNN r2": curr_knn_r2, "linear r2": curr_lin_r2,
                                 "pow. law r2": curr_pl_r2, "mean r2": curr_mean_r2,
                                 "kNN RMSE": curr_knn_rmse, "linear RMSE": curr_lin_rmse,
                                 "pow. law RMSE": curr_pl_rmse, "mean RMSE": curr_mean_rmse,
                                 "kNN RMSE/std": curr_knn_rmse / curr_y_test_std,
                                 "linear RMSE/std": curr_lin_rmse / curr_y_test_std,
                                 "pow. law RMSE/std": curr_pl_rmse_std, "mean RMSE/std": curr_mean_rmse / curr_y_test_std,
                                 "kNN MAPE": curr_knn_mape, "linear MAPE": curr_lin_mape,
                                 "pow. law MAPE": curr_pl_mape, "mean MAPE": curr_mean_mape,
                                 "kNN rae": curr_knn_rae, "linear rae": curr_lin_rae,
                                 "pow. law rae": curr_pl_rae, "mean rae": curr_mean_rae,
                                 "kNN dcor": curr_knn_dcor, "linear dcor": curr_lin_dcor,
                                 "pow. law dcor": curr_pl_dcor, "mean dcor": curr_mean_dcor,
                                 }
                            }
    else:
        curr_metric_dict = {curr_tuple:
                                {"kNN r2": curr_knn_r2, "linear r2": curr_lin_r2,
                                 "pow. law r2": curr_pl_r2, "mean r2": curr_mean_r2,
                                 "kNN RMSE": curr_knn_rmse, "linear RMSE": curr_lin_rmse,
                                 "pow. law RMSE": curr_pl_rmse, "mean RMSE": curr_mean_rmse,
                                 "kNN RMSE/std": curr_knn_rmse / curr_y_test_std,
                                 "linear RMSE/std": curr_lin_rmse / curr_y_test_std,
                                 "pow. law RMSE/std": curr_pl_rmse_std,
                                 "mean RMSE/std": curr_mean_rmse / curr_y_test_std,
                                 "kNN MAPE": curr_knn_mape, "linear MAPE": curr_lin_mape,
                                 "pow. law MAPE": curr_pl_mape, "mean MAPE": curr_mean_mape,
                                 "kNN rae": curr_knn_rae, "linear rae": curr_lin_rae,
                                 "pow. law rae": curr_pl_rae, "mean rae": curr_mean_rae,
                                 "kNN dcor": curr_knn_dcor, "linear dcor": curr_lin_dcor,
                                 "pow. law dcor": curr_pl_dcor, "mean dcor": curr_mean_dcor,
                                 "target": curr_tuple[input_cols:]
                                 }
                            }

    # save values into dict
    if do_pl_fit:
        curr_data_dict = {curr_tuple: {"X_train": curr_X_train, "X_test": curr_X_test,
                                       "y_train": curr_y_train, "y_test": curr_y_test,
                                       "y_train_pred": curr_y_train_pred,
                                       "y_test_pred": curr_y_test_pred, "y_test_pred_linear": curr_y_test_pred_linear,
                                       "y_test_pred_mean": curr_y_test_pred_mean, "y_test_pred_pl": curr_y_test_pred_pl,
                                       "GridSearchParams": curr_best_params, "scores": clf.cv_results_
                                       }
                          }
    else:
        curr_data_dict = {curr_tuple: {"X_train": curr_X_train, "X_test": curr_X_test,
                                       "y_train": curr_y_train, "y_test": curr_y_test,
                                       "y_train_pred": curr_y_train_pred,
                                       "y_test_pred": curr_y_test_pred, "y_test_pred_linear": curr_y_test_pred_linear,
                                       "y_test_pred_mean": curr_y_test_pred_mean,
                                       "GridSearchParams": curr_best_params, "scores": clf.cv_results_
                                       }
                          }

    logger.info("The analysis of this tuple took " + str(round(time.time() - curr_start, 2)) + "s.")
    # for logging the progress of the analysis
    counter_tuples += 1
    logger.info("-----" + str(counter_tuples) + "/" + str(len_data_tuples) + "-----")

    return curr_metric_dict, curr_data_dict

def refinement_step_tpot(data_dict,
                         curr_tuple,
                         time_left_for_this_task,
                         per_run_time_limit,
                         n_jobs
                         ):
    # get current inputs and outputs
    # curr_inputs = list(curr_tuple[:input_cols])
    # curr_outputs = list(curr_tuple[input_cols:])

    # reduce data to current columns and drop NAs
    # curr_data = data[curr_inputs + curr_outputs].dropna()

    # do data preparations and train-test-split
    curr_X_train, curr_X_test, curr_y_train, curr_y_test = data_dict[curr_tuple]["X_train"], \
                                                           data_dict[curr_tuple]["X_test"], \
                                                           data_dict[curr_tuple]["y_train"], \
                                                           data_dict[curr_tuple]["y_test"].ravel()

    tpot = TPOTRegressor(generations=100,
                         population_size=100,
                         # preprocessing=any_preprocessing('pre'),
                         scoring="neg_mean_squared_error",
                         config_dict=regressor_config_dict_ASD,  # "TPOT light",
                         verbosity=1,
                         max_time_mins=np.ceil(time_left_for_this_task / 60),
                         # TODO: check whether time_left_for_this_task makes sense here
                         # early_stop=3,
                         n_jobs=n_jobs
                         # BUT should be a parameter according to https://github.com/hyperopt/hyperopt-sklearn/blob/a84603c5232e01a9edda961d9f8aba6ea0f3038a/hpsklearn/estimator/estimator.py
                         )
    tpot.fit(curr_X_train, curr_y_train)

    curr_y_train_pred = tpot.predict(curr_X_train)
    curr_y_test_pred = tpot.predict(curr_X_test)
    logger.info(f'{curr_tuple}: \n Train R2 score: {r2_score(curr_y_train, curr_y_train_pred)} \n '
                f'Test R2 score: {r2_score(curr_y_test, curr_y_test_pred)}')

    # metrics
    curr_knn_r2 = r2_score(curr_y_test, curr_y_test_pred)
    curr_knn_rmse = mean_squared_error(curr_y_test, curr_y_test_pred, squared=False)
    curr_knn_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred)
    curr_knn_rae = rae(curr_y_test, curr_y_test_pred)
    curr_knn_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred)

    # data / info about ensemble models
    '''
    ensemble_models_dict = automl.show_models()
    curr_reduced_ensemble_models_dict = {}
    for key in ensemble_models_dict.keys():
        rank = ensemble_models_dict[key]["rank"]
        ensemble_weight = ensemble_models_dict[key]["ensemble_weight"]
        sklearn_regressor = ensemble_models_dict[key]["sklearn_regressor"]
        curr_reduced_ensemble_models_dict[key] = {"rank": rank, "ensemble_weight": ensemble_weight,
                                                  "sklearn_regressor": sklearn_regressor}
    '''

    # save metrics into dict
    curr_metric_dict = {curr_tuple: {"r2": curr_knn_r2,
                                     "RMSE": curr_knn_rmse,
                                     "MAPE": curr_knn_mape,
                                     "rae": curr_knn_rae,
                                     "dcor": curr_knn_dcor
                                     }
                        }
    curr_data_dict = {curr_tuple: {"X_train": curr_X_train, "X_test": curr_X_test,
                                   "y_train": curr_y_train, "y_test": curr_y_test,
                                   "y_train_pred": curr_y_train_pred, "y_test_pred": curr_y_test_pred,
                                   "ensemble": tpot.fitted_pipeline_,
                                   "pareto_pipelines": tpot.pareto_front_fitted_pipelines_,
                                   "all_individuals": tpot.evaluated_individuals_,  # TODO: maybe not necessary
                                   }
                      }

    return curr_metric_dict, curr_data_dict


@ray.remote(num_returns=2)
def parallel_refinement_step_tpot(data_dict,
                                  curr_tuple,
                                  time_left_for_this_task,
                                  per_run_time_limit,
                                  n_jobs
                                  ):
    curr_metric_dict, curr_data_dict = refinement_step_tpot(data_dict=data_dict,
                                                            curr_tuple=curr_tuple,
                                                            time_left_for_this_task=time_left_for_this_task,
                                                            per_run_time_limit=per_run_time_limit,
                                                            n_jobs=n_jobs
                                                            )

    return curr_metric_dict, curr_data_dict


def plot_result(input_datas_dict, plot_comb, plot_along=[]):
    """

    :param input_datas_dict: dict
        Dictionary from predictability routine containing all the data lists.
    :param plot_comb: tuple
        The combination tuple whose results shall be plotted.
    :param plot_along: list
        Allows for specifying further prediction methods to be plotted along the kNN/MLP ones. Possible choices are
        (subsets of) ["linear", "mean", "pl"]
    :return: fig
        The plotting figure.
    """
    #
    # TODO: include argument to differ between plotting initial and refined results.
    #

    # make dict a dataframe, name columns appropriately and compute error of kNN prediction
    results_df = pd.DataFrame(
        [input_datas_dict[plot_comb]["y_test_pred"].flatten(),
         input_datas_dict[plot_comb]["y_test"].flatten()]).transpose()
    results_df.columns = ["pred", "true"]
    results_df["error"] = results_df["pred"] - results_df["true"]

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.7, 0.15, 0.15],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}]],
        horizontal_spacing=.15
    )

    #
    # plot results
    #

    # for plotting other than kNN predictions as well, go through additional methods
    if plot_along:
        for comparison in plot_along:
            # even if power law is chosen as additional method, some tuples may not have been fitted via power law
            # due to non-positive values:
            if (comparison == "pl") and ("y_test_pred_pl" not in input_datas_dict[plot_comb].keys()):
                logger.error("no power law fit performed, some columns did not include positive values only",
                             exc_info=True)
            else:
                # load data, compute error
                results_df["pred_" + comparison] = input_datas_dict[plot_comb]["y_test_pred_" + comparison]
                results_df["error_" + comparison] = results_df["pred_" + comparison] - results_df["true"]
                # add plot
                fig.add_trace(go.Scatter(
                    x=results_df["true"],
                    y=results_df["pred_" + comparison],
                    mode="markers",
                    name=comparison + " preds vs trues",
                    opacity=.8
                ),
                    row=1, col=1
                )
    # add refined TPOT plot
    fig.add_trace(go.Scatter(
        x=results_df["true"],
        y=results_df["pred"],
        mode="markers",
        name="ref. preds vs trues",
        opacity=.8
    ),
        row=1, col=1
    )

    #
    # local plot of errors
    #

    # for plotting other than kNN predictions as well, go through additional methods
    if plot_along:
        for comparison in plot_along:
            # even if power law is chosen as additional method, some tuples may not have been fitted via power law
            # due to non-positive values:
            if (comparison == "pl") and ("y_test_pred_pl" not in input_datas_dict[plot_comb].keys()):
                continue
            else:
                fig.add_trace(
                    go.Scatter(
                        x=results_df["error_" + comparison],
                        y=results_df["pred_" + comparison],
                        mode="markers",
                        name="pred. error " + comparison,
                        opacity=.6
                    ),
                    row=1, col=2
                )
    # add refined TPOT plot
    fig.add_trace(
        go.Scatter(
            x=results_df["error"],
            y=results_df["pred"],
            mode="markers",
            marker_color="Maroon",
            name="pred. error ref.",
            opacity=.6
        ),
        row=1, col=2
    )

    #
    # histogram of errors
    #

    # for plotting other than kNN predictions as well, go through additional methods
    if plot_along:
        for comparison in plot_along:
            # even if power law is chosen as additional method, some tuples may not have been fitted via power law
            # due to non-positive values:
            if (comparison == "pl") and ("y_test_pred_pl" not in input_datas_dict[plot_comb].keys()):
                continue
            else:
                fig.add_trace(
                    go.Histogram(
                        y=results_df["error_" + comparison],
                        nbinsx=int(np.floor(len(results_df["pred"]) / 10)),
                        name="pred. error " + comparison
                    ),
                    row=1, col=3
                )
    # add refined TPOT plot
    fig.add_trace(
        go.Histogram(
            y=results_df["error"],
            nbinsx=int(np.floor(len(results_df["pred"]) / 10)),
            name="pred. error ref."
        ),
        row=1, col=3
    )

    # set title
    title = str(plot_comb)

    # set layout, axis labels etc.
    fig.update_layout(
        title=title,
        width=950,
        height=555,
        xaxis=dict(title="true"),
        yaxis=dict(title="pred"),
        xaxis2=dict(title="pred-true"),
        yaxis2=dict(title="pred",
                    matches="y"),
        xaxis3=dict(title="freq"),
        yaxis3=dict(title="pred-true")
    )

    fig.show()
