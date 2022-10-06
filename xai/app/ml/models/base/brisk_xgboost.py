
from utils import dasker
from pprint import pprint


import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
import pickle




import optuna
import joblib
from dask.distributed import Client

import dask_optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)



NUM_BOOST_ROUND = 150
N_TRIALS = 600
pred_type = "regression"

TEMP_PATH = "/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/ml/models/saved/temp/xgboost/"
MODEL_SAVE_PATH = "/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/ml/models/saved/base/xgboost/"

DROP_LIST = ['location']
N_SPLITS = 3 # number of folds


import pandas as pd
def get_dataset():

    df = pd.read_csv('/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/test/data/20220319_covid_merge_processed.csv', sep=",")

    X = df[df.columns[df.columns!='y']]

    X = X.drop(DROP_LIST, axis = 1)

    y = df[df.columns[df.columns=='y']]

    return X, y



def __objective(trial):

    df_X, df_y = get_dataset()

    # X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25)

    param = {
        # "objective": "reg:squarederror",
        "objective": "count:poisson",
        # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }


    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


    kf = KFold(n_splits=N_SPLITS)

    avg_r2_train_list = []
    err_test_list = []

    for train_index, test_index in kf.split(df_X, df_y):
        X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
        y_train, y_test = df_y.loc[train_index], df_y.loc[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(param, dtrain, num_boost_round = NUM_BOOST_ROUND, verbose_eval = 1)

        preds = bst.predict(dtest)
        pred_labels = np.rint(preds)
        err_test = sklearn.metrics.mean_squared_error(y_test, pred_labels)

        #print(f'{err_test}')

        err_test_list.append(err_test)

    mean_err = np.mean(err_test_list)


    with open(TEMP_PATH+"{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(bst, fout)

    return mean_err



def discover_model():

    print(f'Starting train for trials:{ N_TRIALS} with boosted rounds:{NUM_BOOST_ROUND}')

    storage = dask_optuna.DaskStorage()
    study = optuna.create_study(storage=storage, direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())


    with joblib.parallel_backend("dask"):
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)

    print("Number of trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Number of trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    # Load the best model.
    with open(TEMP_PATH+"{}.pickle".format(study.best_trial.number), "rb") as fin:
        best_model = pickle.load(fin)

    ### eval the model first
    # best_model.eval()

    return best_model



def build_final_model(trail_params, df_X, df_y):
    
    best_param = {
        #"objective": "reg:squarederror",
        "objective": "count:poisson",
        "booster": trail_params["booster"],
        "lambda": trail_params["lambda"],
        "alpha": trail_params["alpha"],
    }

    if trail_params["booster"] == "gbtree" or trail_params["booster"] == "dart": 
            best_param["max_depth"] = trail_params["max_depth"]
            best_param["eta"] =  trail_params["eta"]
            best_param["gamma"] = trail_params["gamma"]
            best_param["grow_policy"] = trail_params["grow_policy"]
        

    if trail_params["booster"] == "dart":
            best_param['sample_type'] = trail_params["sample_type"]
            best_param['normalize_type'] = trail_params["normalize_type"]
            best_param['rate_drop'] = trail_params["rate_drop"]
            best_param['skip_drop'] = trail_params["skip_drop"]
        

    X_base, X_val, y_base, y_val = train_test_split(df_X, df_y, random_state=0, shuffle=True)

    dtrain = xgb.DMatrix(X_base, label=y_base)
    dtest = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(best_param, dtrain, num_boost_round = NUM_BOOST_ROUND)

    preds = bst.predict(dtest)
    pred_labels = np.rint(preds)
    err = sklearn.metrics.mean_squared_error(y_val, pred_labels)

    print(f'final error: {err}')

    return bst



def fetch_model():

    status = True

    client = dasker.get_dask_client()
    print(f"Dask dashboard is available at {client.dashboard_link}")

    best_model = discover_model()

    return best_model


def __fetch_model(save):

    print(f'Starting train for trials:{ N_TRIALS} with boosted rounds:{NUM_BOOST_ROUND}')

    # with Client() as client:
    client = dasker.get_dask_client()
    print(f"Dask dashboard is available at {client.dashboard_link}")

    storage = dask_optuna.DaskStorage()
    study = optuna.create_study(storage=storage, direction="minimize")

    with joblib.parallel_backend("dask"):
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)

    print("Best params:")
    pprint(study.best_params)

    print("Number of trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df_X, df_y = get_dataset()

    best_model = build_final_model(trial.params, df_X, df_y)

    if save is True:
        model_path = MODEL_SAVE_PATH + 'xgboost.json'

        print(model_path)

        best_model.save_model(model_path)

    return best_model