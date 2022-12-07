import numpy as np
import pandas as pd
from utils import dasker, config as util_config
from dask_ml.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import fnmatch


from keras import models, layers
from keras.optimizers import Adam, RMSprop
from keras.backend import clear_session
from sklearn.metrics import mean_squared_error

import warnings

import optuna
import dask_optuna
# from tensorflow.keras.optimizers import RMSprop
from ml.preprocess.data import SingletonDataSet

# from sklearn.model_selection import StratifiedKFold, KFold

import joblib


# BATCHSIZE = 128
N_LAYERS = 5
N_NEURONS = 256
EPOCHS = 300
N_TRIALS = 400
no_of_workers = 4

pred_type = "regression"

MODEL_SAVE_PATH = "/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/ml/models/saved/base/brisk_neural_net/"

DROP_LIST = ['location']



import pandas as pd

def get_dataset():
    df = pd.read_csv('/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/test/data/20220319_covid_merge_processed.csv', sep=",")

    X = df[df.columns[df.columns!='y']]

    X = X.drop(DROP_LIST, axis = 1)

    y = df[df.columns[df.columns=='y']]

    return X, y

    


def build_shallow_model(trial, input_dim, lr, loss, pred_type):

    model = models.Sequential()
    # Input layer

    n_layers = trial.suggest_int("n_layers", 1, N_LAYERS)
    hidden_units = []

    # model.add(layers.Dense(units = input_dim, name = "input_layer", kernel_initializer='normal', activation='relu'))

    model.add(layers.BatchNormalization())

    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 1, N_NEURONS)
        # hidden_units.append(n_units)
        # Input layer
        model.add(layers.Dense(n_units, name = "dense_hidden_"+str(i), kernel_initializer='normal', activation='relu'))
    
    # Output layer
    if pred_type == 'classification':
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(1, activation='relu'))

    model.add(layers.Dense(1, activation='softplus'))

    # Compile a model
    model.compile(loss=loss, optimizer=Adam(lr), metrics=['accuracy'])
    return model






def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    scores = []

    df_X, df_y = get_dataset()

    df_X = StandardScaler().fit_transform(df_X)

    X_base, X_val, y_base, y_val = train_test_split(df_X, df_y, random_state=0, shuffle=True)
    
    input_shape = X_base.shape[1]

    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)

    bs = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 512, 1024, 2048])

    loss = 'mean_squared_error'

    model = build_shallow_model(trial, input_shape, learning_rate, loss, 'regression')

    model.fit(
            X_base,
            y_base,
            validation_data=(X_val, y_val),
            shuffle=True,
            batch_size=bs,
            epochs=EPOCHS,
            verbose=False,
        )
    
    y_pred = model.predict(X_val)

    err = np.round_(mean_squared_error(y_val, y_pred), decimals=2, out=None)

    return err





def build_final_model(params):
    
    df_X, df_y = get_dataset()

    df_X = StandardScaler().fit_transform(df_X)

    X_base, X_val, y_base, y_val = train_test_split(df_X, df_y, random_state=0, shuffle=True)
    
    input_shape = X_base.shape[1]
    
    loss = 'mean_squared_error'

    learning_rate = params['learning_rate']

    batch_size = params['batch_size']

    n_layers = params['n_layers']

    units = []
    for param in params:
        if fnmatch.fnmatch(param, 'n_units*'):
            # print(param)
            units.append(param)  
        

    best_model = models.Sequential()
    # Input layer

    # best_model.add(layers.Dense(units = input_shape, name = "input_layer", kernel_initializer='normal', activation='relu'))

    # best_model.add(layers.BatchNormalization())

    for i in range(n_layers):
        best_model.add(layers.Dense(params[units[i]], name = "dense_hidden_"+str(i), kernel_initializer='normal', activation='relu'))

    best_model.add(layers.Dense(1, activation='softplus'))

    # Compile a model
    best_model.compile(loss=loss, optimizer=Adam(learning_rate), metrics=['accuracy'])

    best_model.fit(
            X_base,
            y_base,
            shuffle=True,
            batch_size=batch_size,
            epochs=EPOCHS,
            verbose=False,
        )


    y_pred = best_model.predict(X_val)

    err = np.round_(mean_squared_error(y_val, y_pred), decimals=2, out=None)

    print(err)

    return best_model





def fetch_model(save):

    # with Client() as client:
    client = dasker.get_dask_client()
    print(f"Dask dashboard is available at {client.dashboard_link}")

    storage = dask_optuna.DaskStorage()
    study = optuna.create_study(storage=storage, direction="minimize")

    with joblib.parallel_backend("dask"):
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)

    print("Number of trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))    


    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    best_model = build_final_model(trial.params)

    if save == True:
        best_model.save(MODEL_SAVE_PATH)

    return best_model
