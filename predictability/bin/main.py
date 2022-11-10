import pandas as pd
import time
import ray
import numpy as np

from src.ASD_predictability_utils.utils import get_column_combinations, parallel_pred_step_MLP, \
    parallel_pred_step_kNN, parallel_refinement_step

# map scoring to possible options
scoring_dict = {
    "r2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
    "neg_mean_absolute_percentage_error": "neg_mean_absolute_percentage_error",
    "RMSE": "neg_root_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error"
}


def predictability(data, input_cols=1, output_cols=1, col_set=None, primkey_cols=None, targets=None,
                   method="kNN", hidden_layers=None, alphas=None, scoring="r2", scaling=True,
                   max_iter=10000, n_jobs=-1, verbose=1,  # bayes_optimisation=False,
                   random_state_split=1):

    if targets is None:
        targets = []
    if primkey_cols is None:
        primkey_cols = []
    scoring = scoring_dict[scoring]

    # if we want to measure the overall time
    start = time.time()

    # initialise the dictionary that is going to save the metrics per tuple
    metric_dict = {}

    # dict to save x-/y-train/-test and predicted values for subsequent plotting
    data_dict = {}

    # if primary keys are fed in, data columns should not contain these
    data_cols = [col for col in data.columns.to_list() if col not in primkey_cols]

    # if set of columns that should be considered is fed in, use this
    if col_set is not None:
        data_cols = list(set(col_set))

    # get the list of tuples of input and output columns
    data_tuples = get_column_combinations(data_cols, input_cols, output_cols, targets)

    # for printing the progress of the analysis
    counter_tuples = 0

    # go through all tuples
    metrics_list = []
    datas_list = []
    for curr_tuple in data_tuples:
        '''
        if bayes_optimisation:
             curr_metrics, curr_datas = parallel_pred_step_bayes.remote(data, curr_tuple, input_cols, hidden_layers, alphas, scaling,
                                                              max_iter, scoring, verbose, n_jobs, counter_tuples, data_tuples,
                                                              random_state_split)
         else:
         '''
        if method == "MLP":
            curr_metrics, curr_datas = parallel_pred_step_MLP.remote(data, curr_tuple, input_cols, hidden_layers,
                                                                     alphas,
                                                                     scaling,
                                                                     max_iter, scoring, verbose, n_jobs, counter_tuples,
                                                                     data_tuples,
                                                                     random_state_split)
        elif method == "kNN":
            curr_metrics, curr_datas = parallel_pred_step_kNN.remote(data, curr_tuple, input_cols,
                                                                     scaling, scoring, verbose, n_jobs, counter_tuples,
                                                                     data_tuples,
                                                                     random_state_split)
        metrics_list.append(curr_metrics)
        datas_list.append(curr_datas)

    # let Ray collect the results
    metrics_list = ray.get(metrics_list)
    datas_list = ray.get(datas_list)

    # save results in respective dicts
    metric_dict = dict((key, d[key]) for d in metrics_list for key in d)
    data_dict = dict((key, d[key]) for d in datas_list for key in d)

    print("The whole run took " + str(round(time.time() - start, 2)) + "s.")

    return metric_dict, data_dict


def tuple_selection(all_metrics, n_best=None):
    """
    Routine to select which of the previously analysed combination tuples will be sent into the next step of a refined
    analysis.
    :param n_best:
    :param all_metrics: metrics dictionary that was the output of the predictability-run
    :return:
    """

    metrics_df = pd.DataFrame.from_dict(all_metrics).transpose().sort_values(by="kNN r2", ascending=False)
    initial_number = len(metrics_df)

    # for first setup, just use best 10%, 20 max – if not set via argument
    if not n_best:
        limited_number = np.floor(0.1*initial_number)
        if limited_number == 0:
            limited_number = 1
        elif limited_number > 20:
            limited_number = 20
    else:
        limited_number = n_best

    metrics_df = metrics_df.iloc[:limited_number]

    best_tuples = metrics_df.index.tolist()

    return best_tuples


def refine_predictability(best_tuples, data_dict, n_jobs=-1, data_name=None, time_left_for_this_task=120,
                          per_run_time_limit=30):
    """
    Routine to run a refined analysis on the previously obtained best results of the predictability routine
    :param best_tuples:
    :param data_dict:
    :return:
    """

    #data_df = pd.DataFrame.from_dict(data_dict).transpose()
    #data_df = data_df.loc[best_tuples]

    metrics_list = []
    datas_list = []

    for curr_tuple in best_tuples:
        '''curr_metrics, curr_datas = parallel_refinement_step.remote(data_dict=data_dict, curr_tuple=curr_tuple,
                                                                   data_name=data_name,
                                                                   time_left_for_this_task=time_left_for_this_task,
                                                                   per_run_time_limit=per_run_time_limit,
                                                                   n_jobs=n_jobs)'''
        curr_metrics, curr_datas = parallel_refinement_step(data_dict=data_dict, curr_tuple=curr_tuple,
                                                                   data_name=data_name,
                                                                   time_left_for_this_task=time_left_for_this_task,
                                                                   per_run_time_limit=per_run_time_limit,
                                                                   n_jobs=n_jobs)
        metrics_list.append(curr_metrics)
        datas_list.append(curr_datas)

    # let Ray collect the results
    #metrics_list = ray.get(metrics_list)
    #datas_list = ray.get(datas_list)

    # save results in respective dicts
    refined_metric_dict = dict((key, d[key]) for d in metrics_list for key in d)
    refined_data_dict = dict((key, d[key]) for d in datas_list for key in d)

    return refined_metric_dict, refined_data_dict


if __name__ == "__main__":

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
