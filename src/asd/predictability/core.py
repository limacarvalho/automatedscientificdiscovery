import logging
import time

import pandas as pd
import ray
from utils_logger import LoggerSetup

from .utils import (get_column_combinations, parallel_pred_step_kNN,
                    parallel_pred_step_MLP, parallel_refinement_step,
                    refinement_step, scoring_dict, tuple_selection)

# Initialize logging object (Singleton class) if not already
LoggerSetup()


def run_predictability(
    data,
    input_cols=1,
    output_cols=1,
    col_set=None,
    primkey_cols=None,
    targets=None,
    method="kNN",
    scoring="r2",
    scaling="test",
    hidden_layers=None,
    alphas=None,
    max_iter=10000,
    greedy=False,
    refined_n_best=1,
    n_jobs=-1,
    verbose=1,
    random_state_split=1,
):
    """
    The main predictability routine. The routine runs over all column combinations and uses the method `method` to
    determine the predictability of the ``output_cols``-many target columns given the ``input_cols``-many input values.
    Running through the set of all the column combinations is done using Ray.
    The run's parameters can be modified according to the following list.
    :param data: pandas DataFrame
        dataframe containing all the necessary data in its columns.
    :param input_cols: int
        The number of input columns for the fit. For a 4-1 fit, input_cols = 4.
    :param output_cols: int
        The number of target columns for the fit. For a 4-1 fit, output_cols = 1.
    :param col_set: list, default=None
        The (sub-)set of columns that should be considered. Default ``None`` corresponds to considering all
        columns (except for ``primkey_cols``, if set).
    :param primkey_cols: list, default=None
        The subset of columns corresponding to primary keys. These will neither be used as inputs nor outputs.
    :param targets: list, default=None
        The subset of columns that should be treated exclusively as targets.
        ``len(targets) >= output_cols``
    :param method: str, default="kNN"
        The method used within the predictability routine. Default is kNN (k-nearest-neighbours), possible
        other choice is MLP (Multi-Layer Perceptron).
    :param hidden_layers: list, default=[(12,), (50,), (70, 5,)]
        If ``method="MLP"``. Specifies choices for ``sklearn``'s ``hidden_layer_sizes`` during CV fit.
        If not specified, [(12,), (50,), (70, 5,)] is used.
    :param alphas: list, default=[0.001, 0.0001, 0.00001]
        If ``method="MLP"``. Specifies choices for ``sklearn``'s ``alpha`` during CV fit.
        If not specified, [0.001, 0.0001, 0.00001] is used.
    :param scoring: str
        Scoring for the CV fit. Choices are (mapping automatically to the respective ``sklearn.metrics`` metric):

            - "r2": ``r2``,
            - "MAPE": ``neg_mean_absolute_percentage_error``,
            - "neg_mean_absolute_percentage_error": ``neg_mean_absolute_percentage_error``,
            - "RMSE": ``neg_root_mean_squared_error``,
            - "neg_root_mean_squared_error": ``neg_root_mean_squared_error``,
            - "MAE": ``neg_mean_absolute_error``,
            - "neg_mean_absolute_error": ``neg_mean_absolute_error``

    :param scaling: str, default="test"
        Specifies usage of ``sklearn.preprocessing``'s ``StandardScaler`` before fit. Default is "test", can be set to
        "yes" or "no". If "test", ``StandardScaler`` becomes part of fitting pipeline and benefit of usage / skipping
        will be evaluated.
    :param max_iter: int, default=10000
        If ``method="MLP"``. Specifies maximum number of iterations during fit. Corresponds to ``max_iter`` within
        ``sklearn.model_selection``'s ``GridSearchCV``.
    :param greedy: boolean, default=False
        Sets whether all combinations of input columns are checked or, if ``True``, a greedy algorithm is run instead.
        The greedy algorithm iteratively determines the best inputs, starting with one input. If ``input_cols``>1, this
        best input is then fixed in the input tuple combination and the next iteration finds the best partner for a
        2-``output_cols`` fit. This goes on until the greedily best input tuple combination for a
        ``input_cols``-``output_cols`` fit is found.
        Note that this is done for all possible choices of targets.
    :param refined_n_best:  int, default=1
        If non-zero, the ``refined_predictability`` routine will be triggered subsequently. Sets the number of how many
        of the best results will go into the ``refined_predictability`` routine.
    :param n_jobs: int, default=-1
        Specifies number of jobs to run in parallel. Choose -1 to use all processors. Corresponds to ``n_jobs`` within
        ``sklearn.model_selection``'s ``GridSearchCV``.
    :param verbose: int, default=1
        Specifies the verbosity level. Corresponds to ``verbose`` within ``sklearn.model_selection``'s ``GridSearchCV``.
    :param random_state_split: int, default=1
        Specifies shuffling during ``sklearn.model_selection``'s ``train_test_split``. Set to a specific integer value for
        reproducibility.
    :return: dict, dict
        First dict contains all evaluation metrics, the second one all data (train, test, predicted
        values, GridSearch parameters, CV scores). Both are nested dictionaries, where the outermost keys correspond to
        the respective combination tuples ("input column 1", ..., "input column ``input_cols``", "target column 1", ...,
        "target column ``output_cols``"). Note that for ``greedy=True``, the lower-dimensional tuples (which
        successively built up the final best tuple) are filled up with leading empty entries to have the same length
        ``input_cols+output_cols`` as the final tuple.

        For each combination, the inner dictionaries are then composed of the following keys with the corresponding
        values if ``refined=False``:

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
            - "GridSearchParams", dict corresponding to ``best_params_``-dict of ``sklearn.model_selection``'s
            ``GridSearchCV``
            - "scores", dict corresponding to ``cv_results_``-dict of ``sklearn.model_selection``'s ``GridSearchCV``

        If ``refined_n_best``>0, both dictionaries include a further key "refined_metrics" / "refined_datas" for the
        ``refined_n_best``-many best results of the ``run_predictability``-run. Their values are yet again
        dictionaries, corresponding to the respective return dictionaries of the ``refine_predictability`` routine.
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

    # for logging the progress of the analysis
    counter_tuples = 0

    # initialise lists for results
    metrics_list = []
    datas_list = []

    # put data into ray for speed-up
    data_id = ray.put(data)

    # run the routine over all available input-output-combinations if greedy is not set
    if not greedy:
        # get the list of possible combination tuples of input and output columns
        data_tuples = get_column_combinations(data_cols, input_cols, output_cols, targets)
        # go through all tuples
        for curr_tuple in data_tuples:
            if method == "MLP":
                curr_metrics, curr_datas, counter_tuples = parallel_pred_step_MLP.remote(
                    data_id,
                    curr_tuple,
                    input_cols,
                    hidden_layers,
                    alphas,
                    scaling,
                    max_iter,
                    scoring,
                    verbose,
                    n_jobs,
                    counter_tuples,
                    len(data_tuples),
                    random_state_split,
                )
            elif method == "kNN":
                curr_metrics, curr_datas, counter_tuples = parallel_pred_step_kNN.remote(
                    data_id,
                    curr_tuple,
                    input_cols,
                    scaling,
                    scoring,
                    verbose,
                    n_jobs,
                    counter_tuples,
                    len(data_tuples),
                    random_state_split,
                )
            else:
                logging.error(
                    f"The specified method '{method}' is not an allowed option. Allowed options are 'kNN' "
                    "and 'MLP'; or keep unspecified.",
                )

            metrics_list.append(curr_metrics)
            datas_list.append(curr_datas)

        # let Ray collect the results
        metrics_list = ray.get(metrics_list)
        datas_list = ray.get(datas_list)

    # if greedy is set, run the routine first on all targets and one input, then iteratively add one input to the
    # respective best previous inputs per target choice
    else:
        initial_metrics_list = []
        initial_datas_list = []
        # get the initial tuples, consisting of one input and output_cols-many targets
        # also get the list of possible target choices to allow for iteratively scanning through all those
        initial_tuples, target_choices = get_column_combinations(
            data_cols, 1, output_cols, targets, return_targets=True
        )

        # get the overall number of combinations that will be analysed (for the counter)
        num_greedy_combinations = int(
            len(targets) * ((input_cols) * (len(data_cols) - len(targets)) - 0.5 * input_cols * (input_cols - 1))
        )
        # # # #
        # do the initial predictability step, with 1 input column
        #
        for curr_couple in initial_tuples:
            if method == "kNN":
                curr_metrics, curr_datas, counter_tuples = parallel_pred_step_kNN.remote(
                    data_id,
                    curr_couple,
                    1,
                    scaling,
                    scoring,
                    verbose,
                    n_jobs,
                    counter_tuples,
                    num_greedy_combinations,
                    random_state_split,
                    greedy=True,
                )
            elif method == "MLP":
                curr_metrics, curr_datas, counter_tuples = parallel_pred_step_MLP.remote(
                    data_id,
                    curr_couple,
                    1,
                    hidden_layers,
                    alphas,
                    scaling,
                    max_iter,
                    scoring,
                    verbose,
                    n_jobs,
                    counter_tuples,
                    num_greedy_combinations,
                    random_state_split,
                    greedy=True,
                )
            else:
                logging.error(
                    f"The specified method '{method}' is not an allowed option. Allowed options are 'kNN'  and 'MLP'; or keep unspecified."
                )

            # add target to dataframe to later find best input per target
            # curr_metrics["target"] = curr_couple[-output_cols:]
            initial_metrics_list.append(curr_metrics)
            initial_datas_list.append(curr_datas)
        initial_metrics_list = ray.get(initial_metrics_list)
        initial_datas_list = ray.get(initial_datas_list)
        initial_metrics = pd.DataFrame.from_dict(
            dict((key, d[key]) for d in initial_metrics_list for key in d)
        ).transpose()
        initial_datas = pd.DataFrame.from_dict(dict((key, d[key]) for d in initial_datas_list for key in d)).transpose()
        # get the best 1-input "combination" per target choice and save those and their metrics
        best_initial_choices = []
        for curr_target in target_choices:
            if "kNN r2" in initial_metrics.columns:
                curr_best_initial_choice = (
                    initial_metrics.loc[initial_metrics["target"] == curr_target]
                    .sort_values(by="kNN r2", ascending=False)
                    .iloc[0]
                    .name
                )
                best_initial_choices.append(curr_best_initial_choice)
                curr_best_initial_choice_metrics = (
                    initial_metrics.loc[curr_best_initial_choice].drop(columns="target").to_dict()
                )
                curr_best_initial_choice_datas = (
                    initial_datas.loc[curr_best_initial_choice].drop(columns="target").to_dict()
                )
            elif "MLP r2" in initial_metrics.columns:
                curr_best_initial_choice = (
                    initial_metrics.loc[initial_metrics["target"] == curr_target]
                    .sort_values(by="MLP r2", ascending=False)
                    .iloc[0]
                    .name
                )
                best_initial_choices.append(curr_best_initial_choice)
                curr_best_initial_choice_metrics = (
                    initial_metrics.loc[curr_best_initial_choice].drop(columns="target").to_dict()
                )
                curr_best_initial_choice_datas = (
                    initial_datas.loc[curr_best_initial_choice].drop(columns="target").to_dict()
                )

            # append empty entries to be in line with higher dim. inputs later
            choice_as_saved = (input_cols - 1) * (" ",) + curr_best_initial_choice
            metrics_list.append({choice_as_saved: curr_best_initial_choice_metrics})
            datas_list.append({choice_as_saved: curr_best_initial_choice_datas})

        # dict that collects the best choices
        best_choices = {}
        best_choices[1] = best_initial_choices

        # # # #
        # now do as many iterations until input_cols-many input columns are in the greedily best column combination
        # each step adds a further "best" input column to the input column combination
        #
        nth_iter = 2
        while nth_iter <= input_cols:
            curr_metrics_list = []
            curr_datas_list = []
            for curr_best_former_choice in best_choices[nth_iter - 1]:
                poss_added_input_cols = [
                    col for col in data_cols if ((col not in targets) and (col not in curr_best_former_choice))
                ]

                curr_data_tuples = [
                    (added_input_col,) + curr_best_former_choice for added_input_col in poss_added_input_cols
                ]

                for curr_tuple in curr_data_tuples:
                    if method == "kNN":
                        curr_metrics, curr_datas, counter_tuples = parallel_pred_step_kNN.remote(
                            data_id,
                            curr_tuple,
                            nth_iter,
                            scaling,
                            scoring,
                            verbose,
                            n_jobs,
                            counter_tuples,
                            num_greedy_combinations,
                            random_state_split,
                            greedy=True,
                        )
                    elif method == "MLP":
                        curr_metrics, curr_datas, counter_tuples = parallel_pred_step_MLP.remote(
                            data_id,
                            curr_tuple,
                            nth_iter,
                            hidden_layers,
                            alphas,
                            scaling,
                            max_iter,
                            scoring,
                            verbose,
                            n_jobs,
                            counter_tuples,
                            num_greedy_combinations,
                            random_state_split,
                            greedy=True,
                        )
                    else:
                        logging.error(
                            f"The specified method '{method}' is not an allowed option. Allowed options are 'kNN'  and 'MLP'; or keep unspecified."
                        )

                    # add target to dataframe to later find best input per target
                    curr_metrics_list.append(curr_metrics)
                    curr_datas_list.append(curr_datas)

            curr_metrics_list = ray.get(curr_metrics_list)
            curr_datas_list = ray.get(curr_datas_list)

            curr_metrics = pd.DataFrame.from_dict(
                dict((key, d[key]) for d in curr_metrics_list for key in d)
            ).transpose()
            curr_datas = pd.DataFrame.from_dict(dict((key, d[key]) for d in curr_datas_list for key in d)).transpose()

            best_curr_choices = []
            for curr_target in target_choices:
                if "kNN r2" in curr_metrics.columns:
                    curr_best_choice = (
                        curr_metrics.loc[curr_metrics["target"] == curr_target]
                        .sort_values(by="kNN r2", ascending=False)
                        .iloc[0]
                        .name
                    )
                    best_curr_choices.append(curr_best_choice)
                    curr_best_choice_metrics = curr_metrics.loc[curr_best_choice].drop(columns="target").to_dict()
                    curr_best_choice_datas = curr_datas.loc[curr_best_choice].drop(columns="target").to_dict()
                elif "MLP r2" in curr_metrics.columns:
                    curr_best_choice = (
                        curr_metrics.loc[curr_metrics["target"] == curr_target]
                        .sort_values(by="MLP r2", ascending=False)
                        .iloc[0]
                        .name
                    )
                    best_curr_choices.append(curr_best_choice)
                    curr_best_choice_metrics = curr_metrics.loc[curr_best_choice].drop(columns="target").to_dict()
                    curr_best_choice_datas = curr_datas.loc[curr_best_choice].drop(columns="target").to_dict()

                # append empty entries to be in line with higher dim. inputs later
                choice_as_saved = (input_cols - nth_iter) * (" ",) + curr_best_choice
                metrics_list.append({choice_as_saved: curr_best_choice_metrics})
                datas_list.append({choice_as_saved: curr_best_choice_datas})

            best_choices[nth_iter] = best_curr_choices
            nth_iter += 1

    # save results in respective dicts
    metric_dict = dict((key, d[key]) for d in metrics_list for key in d)
    data_dict = dict((key, d[key]) for d in datas_list for key in d)

    # run refinement routine afterwards, if desired
    if refined_n_best == 0:
        logging.info(f"The whole run took {str(round(time.time() - start, 2))}s.")

        return metric_dict, data_dict
    else:
        # first get the best results of the main routine
        curr_best_tuples = tuple_selection(metric_dict, refined_n_best)
        # then run the refine routine
        refined_metric_dict, refined_data_dict = refine_predictability(
            best_tuples=curr_best_tuples, data_dict=data_dict, n_jobs=n_jobs
        )
        # attach the results to the respective tuples' result dictionaries
        for curr_best_tuple in curr_best_tuples:
            metric_dict[curr_best_tuple]["refined_metrics"] = refined_metric_dict[curr_best_tuple]
            data_dict[curr_best_tuple]["refined_datas"] = refined_data_dict[curr_best_tuple]

        logging.info(f"The whole run took {str(round(time.time() - start, 2))}s.")

        return metric_dict, data_dict


def refine_predictability(
    best_tuples, data_dict, time_left_for_this_task=120, use_ray=True, generations=100, population_size=100, n_jobs=-1
):
    """
    The refined predictability routine. It can be run after the initial ``run_predictability`` routine and after having
    chosen a list of ``best_tuples`` (output of the ``utils`` function ``tuple_selection``) that should be further
    analysed. It is started automatically after the ``run_predictability`` routine if ``refined_n_best`` > 0 is set t
    here. The routine runs ``TPOT`` (https://github.com/EpistasisLab/tpot) on a custom ``config_dict`` and for
    ``time_left_for_this_task`` seconds. ``TPOT``'s ``generations`` and ``population_size`` can be further specified.
    :param best_tuples: list
        A list containing the to be further analysed tuples.
    :param data_dict: dict
        A dictionary containing all the necessary data; should be the output of a ``run_predictability`` run.
    :param time_left_for_this_task: float
        Time in seconds that specifies for how long the routine should run.
    :param use_ray: boolean, default=True
        Specifies whether the routine should run in parallel, using ``Ray``.
    :param generations: int
        Corresponds to ``TPOT``'s ``generations``: Number of iterations to run the pipeline optimization process.
    :param population_size: int
        Corresponds to ``TPOT``'s ``population_size``: Number of individuals to retain in the GP population every
        generation.
    :param n_jobs: int, default=-1
        Specifies number of jobs to run in parallel. Choose -1 to use all processors. Corresponds to
        ``TPOTRegressor``'s ``n_jobs``.
    :return:  dict, dict
        First dict contains all evaluation metrics, the second one all data (train, test, predicted
        values, ``TPOT`` / ``sklearn`` ``Pipeline`` etc. ). Both are nested dictionaries, where the outermost keys
        correspond to the respective combination tuples ("input column 1", ..., "input column ``input_cols``", "target
        column 1", ..., "target column ``output_cols``").

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
            - "pareto_pipelines", dict directly from ``TPOT``: 'the key is the string representation of the pipeline and
            the value is the corresponding pipeline fitted on the entire training dataset'
            - "all_individuals", dict directly from ``TPOT``: 'the key is the string representation of the pipeline and
            the value is a tuple containing (# of steps in pipeline, accuracy metric for the pipeline)'
    """

    # initialise lists for results
    metrics_list = []
    datas_list = []

    # put data into ray for speed-up
    if use_ray:
        data_dict_id = ray.put(data_dict)

    # go through all the previously defined best tuples that should be considered and run (parallel) refinement steps
    for curr_tuple in best_tuples:
        if use_ray:
            curr_metrics, curr_datas = parallel_refinement_step.remote(
                data_dict=data_dict_id,
                curr_tuple=curr_tuple,
                time_left_for_this_task=time_left_for_this_task,
                generations=generations,
                population_size=population_size,
                n_jobs=n_jobs,
            )
        else:
            curr_metrics, curr_datas = refinement_step(
                data_dict=data_dict,
                curr_tuple=curr_tuple,
                time_left_for_this_task=time_left_for_this_task,
                generations=generations,
                population_size=population_size,
                n_jobs=n_jobs,
            )
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
    # NOTE this does not yet contain every available step

    # ask for data file
    data_file = input("Which data shall be analysed?")
    data_df = pd.read_csv(data_file)

    input_cols_ = int(input("How many input columns shall be considered? (For a 4-2 fit, e.g., insert 4.)"))

    output_cols_ = int(input("How many output columns shall be considered? (For a 4-2 fit, e.g., insert 2.)"))

    print("List of available columns:")
    print(str(data_df.columns.to_list()))

    num_prim_keys = int(input("How many primary keys should be chosen?"))
    primkey_cols_ = []
    for pk in range(0, num_prim_keys):
        curr_input = input("Please enter primary key no. " + str(pk + 1) + ". (The order is irrelevant.)")
        primkey_cols_.append(curr_input)

    define_cols = input("Do you want to specify the input and target columns to be considered? [y/n]")
    all_cols = data_df.columns.to_list()
    if define_cols == "y":
        for col_ in primkey_cols_:
            all_cols.delete(col_)
        print("List of available input columns:")
        print(str(all_cols))
        limit_inputs = input("Do you want to reduce this set of considered columns? [y/n]")
        if limit_inputs == "y":
            col_set_ = []
            new_col = input("Enter a column that shall be considered. Enter STOP once you entered all.")
            while new_col != "STOP":
                col_set_.append(new_col)
                new_col = input("Enter a column that shall be considered. Enter STOP once you entered all.")
        elif limit_inputs == "n":
            col_set_ = all_cols
        else:
            print("Input wasn't [y/n], assume no.")
            col_set_ = all_cols

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
        col_set_ = all_cols
        target_set = []

    specify_method = input(
        "Do you want to specify the running method? If not, parallel kNN with RMSE scoring will be used. [y/n]"
    )
    if specify_method == "y":
        method_ = input("Which method should be applied? [kNN/MLP]")
        while method_ not in ["kNN", "MLP"]:
            method_ = input("Input was neither kNN nor MLP, please enter again.")
        print(scoring_dict.keys())
        scoring_ = input("Which of these scorings should be used?")
        while scoring_ not in list(scoring_dict.keys()):
            print(scoring_dict.keys())
            scoring_ = input("Which of these scorings should be used?")
        n_jobs_ = int(input("How many CPUs shall be used? Enter -1 for all."))
    else:
        method_ = "kNN"
        scoring_ = "RMSE"
        n_jobs_ = -1

    print(col_set_)
    print(target_set)

    metrics, datas = run_predictability(
        data_df,
        input_cols=input_cols_,
        output_cols=output_cols_,
        col_set=col_set_,
        primkey_cols=primkey_cols_,
        targets=target_set,
        method=method_,
        scoring=scoring_,
        n_jobs=n_jobs_,
        verbose=1,
        random_state_split=1,
    )
    for key_ in metrics.keys():
        print(f'{key_}: r2-score = {metrics[key_]["kNN r2"]}')

    best_tuples_ = tuple_selection(metrics, n_best=10)

    ref_metrics, ref_datas = refine_predictability(
        best_tuples_, datas, n_jobs=-1, time_left_for_this_task=60, use_ray=True
    )
    for key_ in ref_metrics.keys():
        print(
            f'{key_}: r2-score refined = {ref_metrics[key_]["kNN r2"]} \t '
            f'r2_score initially = {metrics[key_]["kNN r2"]}'
        )
