
from ml.models.base.tabular_dataset import TabularDataset
# from ml.models.base.brisk_model_lightning import TabularDataset
# from ml.preprocess import data
from utils import dasker, helper
import pickle

import optuna
import dask_optuna
import joblib

# from torch.utils.data import random_split, DataLoader, Dataset
from dask_ml.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
# from torchmetrics import Accuracy, MeanSquaredError
from torch import nn
import torch.optim as optim

import pandas as pd
import numpy as np

import os
import glob


N_LAYERS = 8
MAX_N_NEURONS = 256
EPOCHS = 300
N_TRIALS = 1000
RANDOM_STATE = 432
TEMP_PATH = "/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/ml/models/saved/temp/"
MODEL_SAVE_PATH = "/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/ml/models/saved/base/brisk_nn_pytorch/"

DROP_LIST = ['location']
STD_DEMON = 3
NUM_RERUNS = 5



def get_dataset():
    df = pd.read_csv('/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/test/data/20220319_covid_merge_processed.csv', sep=",")

    X = df[df.columns[df.columns!='y']]

    X = X.drop(DROP_LIST, axis = 1)

    y = df[df.columns[df.columns=='y']]

    return X, y



def get_split_dataset():
    X, y = get_dataset()
    X_scalar = StandardScaler().fit_transform(X)     
    return train_test_split(X_scalar, y, random_state=RANDOM_STATE, test_size=0.33)



def define_model(trial, input_dim):

    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, N_LAYERS)
    layers = []

    in_features = input_dim

    # layers.append(nn.BatchNorm1d(num_features = in_features))
    # layers.append(nn.ReLU())
    
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 1, MAX_N_NEURONS)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_uniform("dropout_l{}".format(i), 0.1, 0.5)
        # layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.ReLU())
    # layers.append(torch.nn.Flatten(0, 1))
    # layers.append(nn.LogSoftmax(dim=1))

    model = nn.Sequential(*layers)

    return model


# Train and evaluate the accuracy of neural network with the addition of pruning mechanism
def train_and_evaluate(param, model, trial, df_X, df_y):
    
    X_base, X_val, y_base, y_val = train_test_split(df_X, df_y, random_state=RANDOM_STATE, 
                                        shuffle=True, test_size=0.2)

    # train_data, val_data = train_test_split(df_, test_size = 0.2, random_state = 42)
    base, val = TabularDataset(X_base, y_base), TabularDataset(X_val, y_val)

    base_dataloader = torch.utils.data.DataLoader(base, batch_size=param['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=param['batch_size'])

    criterion = nn.MSELoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

#    if use_cuda:

#            model = model.cuda()
#            criterion = criterion.cuda()

    for epoch_num in range(EPOCHS):

            total_err_train = 0
            total_loss_train = 0

            for base_x, base_y in base_dataloader:

                # train_label = train_label.to(device)
                # train_input = train_input.to(device)
                # train_y = train_y.reshape(train_y.shape[0], )
                output = model(base_x.float())
                
                # print(output.shape)
                # print(f'y: {train_y.shape}')

                batch_loss = criterion(output, base_y)
                # print(batch_loss)
                total_loss_train += batch_loss.item()
                
                # acc = (output.argmax(dim=1) == train_label).sum().item()

                #err = np.round_(mean_squared_error(output, train_y), decimals=2, out=None)
                # total_err_train += err

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_err_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_x, val_y in val_dataloader:

                    # val_label = val_label.to(device)
                    # val_input = val_input.to(device)
                    # val_y = val_y.reshape(val_y.shape[0], )
                    output = model(val_x.float())

                    batch_loss = criterion(output, val_y)
                    total_loss_val += batch_loss.item()
                    
                    # acc = (output.argmax(dim=1) == val_label).sum().item()
                    # err = np.round_(mean_squared_error(output, val_y), decimals=2, out=None)
                    #total_acc_val += err
            
            accuracy = total_loss_val
            
            # Add prune mechanism
            trial.report(accuracy, epoch_num)

#            if trial.should_prune():
#                raise optuna.exceptions.TrialPruned()

    return accuracy




def objective(trial):

     params = {
          'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
          'optimizer': trial.suggest_categorical("optimizer", ["Adam", "Adamax", 'NAdam']),
          'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 512, 1024, 2048])
          }

     train_X, test_x, train_y, test_y= get_split_dataset()

     input_shape = train_X.shape[1]

     model = define_model(trial, input_shape)

     accuracy = train_and_evaluate(params, model, trial, train_X, train_y)

     with open(TEMP_PATH+"{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)

     return accuracy
      

def discover_model():

    print(f'Starting train for trials:{ N_TRIALS} with epochs:{EPOCHS}')

    storage = dask_optuna.DaskStorage()
    study = optuna.create_study(storage=storage, direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())


    with joblib.parallel_backend("dask"):
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)

    print("Number of trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    # Load the best model.
    with open(TEMP_PATH+"{}.pickle".format(study.best_trial.number), "rb") as fin:
        best_model = pickle.load(fin)

    ### eval the model first
    best_model.eval()

    return best_model


    # save the model
    



def get_rerun_status(model):

    train_X, test_X, train_y, test_y= get_split_dataset()

    X_test_tensor = helper.df_to_tensor(test_X)

    #print(X_test.shape)
    # print(X_test_tensor.shape)


    pred = model(X_test_tensor)

    pred = helper.torch_tensor_to_numpy(pred)
    

    test_y = test_y.values
    
    STD_DEMON = 3
    ul = test_y.mean() + test_y.std()/STD_DEMON
    ll = test_y.mean() - test_y.std()/STD_DEMON

    mean_pred = pred.mean()

    if (mean_pred < ul) & (mean_pred > ll):
        return False

    else: 
        return True




def save_model(model):

    print(f"Model saved at:{MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH+"best_pytorch.pickle", "wb") as fout:
        pickle.dump(model, fout)
    
    files = glob.glob(TEMP_PATH+'/*')
    for f in files:
        os.remove(f)



def fetch_model():

    status = True

    client = dasker.get_dask_client()
    print(f"Dask dashboard is available at {client.dashboard_link}")


    # results are acceptable, no need to rerun.
    for run_num in range(1, NUM_RERUNS):
        if status is True:
            print("########## Re-running the discovery process ###############")
            best_model = discover_model()
            status = get_rerun_status(best_model)
        else:        
            save_model(best_model)
            break            

    return best_model
      