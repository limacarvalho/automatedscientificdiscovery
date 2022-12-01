import pandas as pd
import time
import ray
import numpy as np

from src.ASD_predictability_utils.utils import get_column_combinations, parallel_pred_step_MLP, \
    parallel_pred_step_kNN, refinement_step_autosklearn, parallel_refinement_step_autosklearn, scoring_dict, \
    refinement_step_hyperopt, parallel_refinement_step_hyperopt, refinement_step_tpot, parallel_refinement_step_tpot

from utils_logger import logger


def predictability(data, input_cols=1, output_cols=1, col_set=None, primkey_cols=None, targets=None,
                   method="kNN", hidden_layers=None, alphas=None, scoring="r2", scaling="test",
                   max_iter=10000, n_jobs=-1, verbose=1,
                   random_state_split=1):
    """
    The main predictability routine. The routine runs over all column combinations and uses the method `method` to
    determine the predictability of the `output_cols`-many target columns given the `input_cols`-many input values.
    Running through the set of all the column combinations is done using Ray.
    The run's parameters can be modified according to the following list.
    :param data: pandas DataFrame
        dataframe containing all the necessary data in its columns.
    :param input_cols: int
        The number of input columns for the fit. For a 4-1 fit, input_cols = 4.
    :param output_cols: int
        The number of target columns for the fit. For a 4-1 fit, output_cols = 1.
    :param col_set: list, default=None
        The (sub-)set of columns that should be considered. Default `None` corresponds to considering all
        columns (except for `primkey_cols`, if set).
    :param primkey_cols: list, default=None
        The subset of columns corresponding to primary keys. These will neither be used as inputs nor outputs.
    :param targets: list, default=None
        The subset of columns that should be treated exclusively as targets.
        `len(targets) >= output_cols`
    :param method: str, default="kNN"
        The method used within the predictability routine. Default is kNN (k-nearest-neighbours), possible
        other choice is MLP (Multi-Layer Perceptron).
    :param hidden_layers: list, default=[(12,), (50,), (70, 5,)]
        If method="MLP". Specifies choices for sklearn's `hidden_layer_sizes` during CV fit.
        If not specified, [(12,), (50,), (70, 5,)] is used.
    :param alphas: list, default=[0.001, 0.0001, 0.00001]
        If method="MLP". Specifies choices for sklearn's `alpha` during CV fit.
        If not specified, [0.001, 0.0001, 0.00001] is used.
    :param scoring: str
        Scoring for the CV fit. Choices are (mapping automatically to the respective `sklearn.metrics` metric):

            - "r2": `r2`,
            - "MAPE": `neg_mean_absolute_percentage_error`,
            - "neg_mean_absolute_percentage_error": `neg_mean_absolute_percentage_error`,
            - "RMSE": `neg_root_mean_squared_error`,
            - "neg_root_mean_squared_error": `neg_root_mean_squared_error`,
            - "MAE": `neg_mean_absolute_error`,
            - "neg_mean_absolute_error": `neg_mean_absolute_error`

    :param scaling: str, default="test"
        Specifies usage of `sklearn.preprocessing`'s `StandardScaler` before fit. Default is "test", can be set to
        "yes" or "no". If "test", `StandardScaler` becomes part of fitting pipeline and benefit of usage / skipping
        will be evaluated.
    :param max_iter: int, default=10000
        If `method="MLP"`. Specifies maximum number of iterations during fit. Corresponds to `max_iter` within
        `sklearn.model_selection`'s `GridSearchCV`.
    :param n_jobs: int, default=-1
        Specifies number of jobs to run in parallel. Choose -1 to use all processors. Corresponds to `n_jobs` within
        `sklearn.model_selection`'s `GridSearchCV`.
    :param verbose: int, default=1
        Specifies the verbosity level. Corresponds to `verbose` within `sklearn.model_selection`'s `GridSearchCV`.
    :param random_state_split: int, default=1
        Specifies shuffling during `sklearn.model_selection`'s `train_test_split`. Set to a specific integer value for
        reproducibility.
    :return: dict, dict
        First dict contains all evaluation metrics, the second one all data (train, test, predicted
        values, GridSearch parameters, CV scores). Both are nested dictionaries, where the outermost keys correspond to
        the respective combination tuples ("input column 1", ..., "input column `input_cols`", "target column 1", ...,
        "target column `output_cols`").

        For each combination, the inner dictionaries are then composed of the following keys with the corresponding
        values:

        metric dict:

            - "TYPE r2", float
            - "TYPE RMSE", float
            - "TYPE RMSE/std", float
            - "TYPE MAPE", float
            - "TYPE rae", float
            - "TYPE dcor", float

            for TYPE in ["kNN", "linear", "mean", "pow. law"]
            (all `None` for "pow. law" if no power law fit performed; included for uniform outputs.)

        data dict:

            - "X_train", numpy.ndarray of shape (train_data_points, input_cols)
            - "X_test", numpy.ndarray of shape (test_data_points, input_cols)
            - "y_train", numpy.ndarray of shape (train_data_points, output_cols)
            - "y_test", numpy.ndarray of shape (test_data_points, output_cols)
            - "y_train_pred", numpy.ndarray of shape (train_data_points, output_cols)
            - "y_test_pred", numpy.ndarray of shape (test_data_points, output_cols)
            - "y_test_pred_linear", numpy.ndarray of shape (test_data_points, output_cols)
            - "y_test_pred_mean", numpy.ndarray of shape (test_data_points, output_cols)
            - "y_test_pred_pl", numpy.ndarray of shape (test_data_points, output_cols), (if power law fit performed)
            - "GridSearchParams", dict corresponding to `best_params_'-dict of `sklearn.model_selection`'s `GridSearchCV'
            - "scores", dict corresponding to `cv_results_'-dict of `sklearn.model_selection`'s `GridSearchCV'
    """

    if targets is None:
        targets = []
    if primkey_cols is None:
        primkey_cols = []
    scoring = scoring_dict[scoring]

    # to measure the overall time
    start = time.time()

    # if primary keys are fed in, data columns should not contain these
    data_cols = [col for col in data.columns.to_list() if col not in primkey_cols]

    # if set of columns that should be considered is fed in, use it
    if col_set is not None:
        data_cols = list(set(col_set))

    # get the list of possible combination tuples of input and output columns
    data_tuples = get_column_combinations(data_cols, input_cols, output_cols, targets)

    # for logging the progress of the analysis
    # TODO: fix the counting – needs to be adapted for parallel Ray usage
    counter_tuples = 0

    # initialise lists for results
    metrics_list = []
    datas_list = []

    # put data into ray for speed-up
    data_id = ray.put(data)

    # go through all tuples
    for curr_tuple in data_tuples:
        if method == "MLP":
            curr_metrics, curr_datas = parallel_pred_step_MLP.remote(data_id, curr_tuple, input_cols, hidden_layers,
                                                                     alphas,
                                                                     scaling,
                                                                     max_iter, scoring, verbose, n_jobs, counter_tuples,
                                                                     len(data_tuples),
                                                                     random_state_split)
        elif method == "kNN":
            curr_metrics, curr_datas = parallel_pred_step_kNN.remote(data_id, curr_tuple, input_cols,
                                                                     scaling, scoring, verbose, n_jobs, counter_tuples,
                                                                     len(data_tuples),
                                                                     random_state_split)
        else:
            logger.error("The specified method '", method, "' is not an allowed option. Allowed options are 'kNN' and "
                                                           "'MLP'; or keep unspecified.", exc_info=True)

        metrics_list.append(curr_metrics)
        datas_list.append(curr_datas)

    # let Ray collect the results
    metrics_list = ray.get(metrics_list)
    datas_list = ray.get(datas_list)

    # save results in respective dicts
    metric_dict = dict((key, d[key]) for d in metrics_list for key in d)
    data_dict = dict((key, d[key]) for d in datas_list for key in d)

    logger.info("The whole run took " + str(round(time.time() - start, 2)) + "s.")

    return metric_dict, data_dict


def tuple_selection(all_metrics, n_best=None):
    """
    Routine to select which of the previously analysed combination tuples will be sent into the next step of a refined
    analysis.
    :param all_metrics: dict
        Metrics dictionary that was the output of a predictability run
    :param n_best: int
        Specifies how many combination tuples should be selected according to the `tuple_selection` logic.
    :return: list
        List of the `n_best`-many combination tuples that shall be further investigated.
    """

    # first sort predictability results by r2-score of kNN regressor
    metrics_df = pd.DataFrame.from_dict(all_metrics).transpose().sort_values(by="kNN r2", ascending=False)

    # for first setup, just use best 10%, 20 max – if not explicitly specified via argument n_best
    initial_number = len(metrics_df)
    if not n_best:
        limited_number = np.floor(0.1 * initial_number)
        if limited_number == 0:
            limited_number = 1
        elif limited_number > 20:
            limited_number = 20
    else:
        limited_number = n_best

    metrics_df = metrics_df.iloc[:limited_number]

    best_tuples = metrics_df.index.tolist()

    return best_tuples


'''
@ray.remote(num_returns=2)
def main_parallel_refinement_step(data_dict, curr_tuple,
                                  data_name,
                                  time_left_for_this_task,
                                  per_run_time_limit,
                                  n_jobs):
    metr, dat = parallel_refinement_step.remote(data_dict, curr_tuple,
                                                data_name,
                                                time_left_for_this_task,
                                                per_run_time_limit,
                                                n_jobs)
    return metr, dat
'''


# @ray.remote(num_returns=2)
def refine_predictability(best_tuples, data_dict, n_jobs=-1, data_name=None, time_left_for_this_task=120,
                          per_run_time_limit=30, use_ray=False, package="hyperopt"):
    """

    :param best_tuples:
    :param data_dict:
    :param n_jobs:
    :param data_name:
    :param time_left_for_this_task:
    :param per_run_time_limit:
    :param use_ray:
    :param package:
    :return:  dict, dict
        First dict contains all evaluation metrics, the second one all data (train, test, predicted
        values, `TPOT' / `sklearn' `Pipeline' etc. ). Both are nested dictionaries, where the outermost keys correspond
        to the respective combination tuples ("input column 1", ..., "input column `input_cols`", "target column 1",
        ..., "target column `output_cols`").

        For each combination, the inner dictionaries are then composed of the following keys with the corresponding
        values:

        metric dict:

            - "r2", float
            - "RMSE", float
            - "MAPE", float
            - "rae", float
            - "dcor", float

        data dict:

            - "X_train", numpy.ndarray of shape (train_data_points, input_cols)
            - "X_test", numpy.ndarray of shape (test_data_points, input_cols)
            - "y_train", numpy.ndarray of shape (train_data_points, output_cols)
            - "y_test", numpy.ndarray of shape (test_data_points, output_cols)
            - "y_train_pred", numpy.ndarray of shape (train_data_points, output_cols)
            - "y_test_pred", numpy.ndarray of shape (test_data_points, output_cols)
            - "ensemble", `sklearn' `Pipeline' object
            - "pareto_pipelines", dict directly from `TPOT': 'the key is the string representation of the pipeline and
            the value is the corresponding pipeline fitted on the entire training dataset'
            - "all_individuals", dict directly from `TPOT': 'the key is the string representation of the pipeline and
            the value is a tuple containing (# of steps in pipeline, accuracy metric for the pipeline)'
    """

    # data_df = pd.DataFrame.from_dict(data_dict).transpose()
    # data_df = data_df.loc[best_tuples]

    metrics_list = []
    datas_list = []

    # put data into ray for speed-up
    if use_ray:
        data_dict_id = ray.put(data_dict)

    for curr_tuple in best_tuples:
        if use_ray:
            if package == "autosklearn":
                curr_metrics, curr_datas = parallel_refinement_step_autosklearn.remote(data_dict=data_dict_id,
                                                                                       curr_tuple=curr_tuple,
                                                                                       data_name=data_name,
                                                                                       time_left_for_this_task=time_left_for_this_task,
                                                                                       per_run_time_limit=per_run_time_limit,
                                                                                       n_jobs=n_jobs)
                # curr_metrics = ray.get(curr_metrics)
                # curr_datas = ray.get(curr_datas)
            elif package == "hyperopt":
                curr_metrics, curr_datas = parallel_refinement_step_hyperopt.remote(data_dict=data_dict_id,
                                                                                    curr_tuple=curr_tuple,
                                                                                    time_left_for_this_task=time_left_for_this_task,
                                                                                    per_run_time_limit=per_run_time_limit,
                                                                                    # n_jobs=n_jobs
                                                                                    )
            elif package == "tpot":
                curr_metrics, curr_datas = parallel_refinement_step_tpot.remote(data_dict=data_dict_id,
                                                                                curr_tuple=curr_tuple,
                                                                                time_left_for_this_task=time_left_for_this_task,
                                                                                per_run_time_limit=per_run_time_limit,
                                                                                n_jobs=n_jobs
                                                                                )
            else:
                logger.error("The specified package '", package, "' is not an allowed option. Allowed options are "
                                                                 "'autosklearn' or 'hyperopt'.", exc_info=True)
        else:
            if package == "autosklearn":
                curr_metrics, curr_datas = refinement_step_autosklearn(data_dict=data_dict, curr_tuple=curr_tuple,
                                                                       data_name=data_name,
                                                                       time_left_for_this_task=time_left_for_this_task,
                                                                       per_run_time_limit=per_run_time_limit,
                                                                       n_jobs=n_jobs)
            elif package == "hyperopt":
                curr_metrics, curr_datas = refinement_step_hyperopt(data_dict=data_dict,
                                                                    curr_tuple=curr_tuple,
                                                                    time_left_for_this_task=time_left_for_this_task,
                                                                    per_run_time_limit=per_run_time_limit,
                                                                    # n_jobs=n_jobs
                                                                    )
            elif package == "tpot":
                curr_metrics, curr_datas = refinement_step_tpot(data_dict=data_dict,
                                                                curr_tuple=curr_tuple,
                                                                time_left_for_this_task=time_left_for_this_task,
                                                                per_run_time_limit=per_run_time_limit,
                                                                n_jobs=n_jobs
                                                                )
            else:
                logger.error("The specified package '", package, "' is not an allowed option. Allowed options are "
                                                                 "'autosklearn', 'hyperopt' or 'tpot'.", exc_info=True)
        metrics_list.append(curr_metrics)
        datas_list.append(curr_datas)

    if use_ray:
        # let Ray collect the results
        metrics_list = ray.get(metrics_list)
        datas_list = ray.get(datas_list)

    # save results in respective dicts
    refined_metric_dict = dict((key, d[key]) for d in metrics_list for key in d)
    refined_data_dict = dict((key, d[key]) for d in datas_list for key in d)

    return refined_metric_dict, refined_data_dict


if __name__ == "__main__":

    # some command line way for running the predictability routine

    # ask for data file
    data_file = input("Which data shall be analysed?")
    data = pd.read_csv(data_file)

    input_cols = int(input("How many input columns shall be considered? (For a 4-2 fit, e.g., insert 4.)"))

    output_cols = int(input("How many output columns shall be considered? (For a 4-2 fit, e.g., insert 2.)"))

    print("List of available columns:")
    print(str(data.columns.to_list()))

    num_prim_keys = int(input("How many primary keys should be chosen?"))
    primkey_cols = []
    for pk in range(0, num_prim_keys):
        curr_input = input("Please enter primary key no. " + str(pk + 1) + ". (The order is irrelevant.)")
        primkey_cols.append(curr_input)

    define_cols = input("Do you want to specify the input and target columns to be considered? [y/n]")
    all_cols = data.columns.to_list()
    if define_cols == "y":
        for col in primkey_cols:
            all_cols.delete(col)
        print("List of available input columns:")
        print(str(all_cols))
        limit_inputs = input("Do you want to reduce this set of considered columns? [y/n]")
        if limit_inputs == "y":
            col_set = []
            new_col = input("Enter a column that shall be considered. Enter STOP once you entered all.")
            while new_col != "STOP":
                col_set.append(new_col)
                new_col = input("Enter a column that shall be considered. Enter STOP once you entered all.")
        elif limit_inputs == "n":
            col_set = all_cols
        else:
            print("Input wasn't [y/n], assume no.")
            col_set = all_cols

        define_targets = input("Do you want to define a set of columns ONLY considered as targets? [y/n]")
        if define_targets == "y":
            target_set = []
            new_col = input("Enter a target column that shall be considered. Enter STOP once you entered all.")
            while new_col != "STOP":
                target_set.append(new_col)
                new_col = input("Enter a target column that shall be considered. Enter STOP once you entered all.")
        elif define_targets == "n":
            target_set = []
        else:
            print("Input wasn't [y/n], assume no.")
            target_set = []
    else:
        col_set = all_cols
        target_set = []

    specify_method = input(
        "Do you want to specify the running method? If not, parallel kNN with RMSE scoring will be used. [y/n]")
    if specify_method == "y":
        method = input("Which method should be applied? [kNN/MLP]")
        while method not in ["kNN", "MLP"]:
            method = input("Input was neither kNN nor MLP, please enter again.")
        print(scoring_dict.keys())
        scoring = input("Which of these scorings should be used?")
        while scoring not in list(scoring_dict.keys()):
            print(scoring_dict.keys())
            scoring = input("Which of these scorings should be used?")
        n_jobs = int(input("How many CPUs shall be used? Enter -1 for all."))
    else:
        method = "kNN"
        scoring = "RMSE"
        n_jobs = -1

    print(col_set)
    print(target_set)

    metrics, datas = predictability(data,
                                    input_cols=input_cols, output_cols=output_cols, col_set=col_set,
                                    primkey_cols=primkey_cols, targets=target_set,
                                    method=method, scoring=scoring,  # scaling=True,
                                    n_jobs=n_jobs, verbose=1,  # bayes_optimisation=False,
                                    random_state_split=1)
    for key in metrics.keys():
        print(f'{key}: r2-score = {metrics[key]["kNN r2"]}')

    best_tuples = tuple_selection(metrics, n_best=10)

    ref_metrics, ref_datas = refine_predictability(
        best_tuples, datas, n_jobs=-1, data_name=None, time_left_for_this_task=60,
        per_run_time_limit=30, use_ray=True
    )
    for key in ref_metrics.keys():
        print(
            f'{key}: r2-score refined = {ref_metrics[key]["kNN r2"]} \t r2_score initially = {metrics[key]["kNN r2"]}')
