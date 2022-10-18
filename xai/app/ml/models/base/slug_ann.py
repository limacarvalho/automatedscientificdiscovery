
from ml.models.base.tabular_dataset import TabularDataset
# from ml.models.base.brisk_model_lightning import TabularDataset
# from ml.preprocess import data
from utils import dasker, helper


import optuna
import dask_optuna
import joblib
import pickle

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score



import torch
# from torchmetrics import Accuracy, MeanSquaredError
from torch import nn
import torch.optim as optim

import pandas as pd
import numpy as np

import os
import glob


from utils import dasker, helper, config
import optuna



optuna.logging.set_verbosity(optuna.logging.WARNING)



class SlugANN:
    def __init__(self) -> None:
        self.n_layers = 8
        self.max_neurons_per_layer = 256
        self.epochs = 300
        self.n_trials = 100
        self.random_state = 0


        self.temp_path = config.main_dir + "/ml/models/saved/temp/ann"
        self.model_save_path = config.main_dir + "/ml/models/saved/base/ann/slug"


        # self.pred_class = "regression"
                    
        # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ] check "Learning Task Parameters" section at https://xgboost.readthedocs.io/en/stable/parameter.html

        # self.loss = "count:poisson" # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]

        self.acceptance_threshold = 0.8
        self.best_fit = None


        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None




    # Load pickled models
    def load_model(self, file_name):
        with open(file_name, "rb") as fin:
            model = pickle.load(fin)

        return model


    def save_model(self, model, file_name):
        with open(file_name, "wb") as fout:
            pickle.dump(model, fout)


    def __create_slug_ann_model__(self, trial, input_dim):

        # We optimize the number of layers, hidden untis and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, self.n_layers)
        layers = []

        in_features = input_dim

        # layers.append(nn.BatchNorm1d(num_features = in_features))
        # layers.append(nn.ReLU())
        
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 1, self.max_neurons_per_layer)
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



    def __objective__(self, trial):

        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "Adamax", 'NAdam']),
            'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 512, 1024, 2048])
            }

        # train_X, test_x, train_y, test_y= get_split_dataset()

        input_shape = self.X_train.shape[1]
    
        model = self.__create_slug_ann_model__(trial, input_shape)

        accuracy = self.train_and_evaluate(params, model, trial)


        file_name =  self.temp_path + "/brisk_ann_{}.pickle".format(trial.number)
        
        # save model in temp folder
        self.save_model(model, file_name)


#        with open(TEMP_PATH+"{}.pickle".format(trial.number), "wb") as fout:
#            pickle.dump(model, fout)

        return accuracy
      


    # Train and evaluate the accuracy of neural network with the addition of pruning mechanism
    def train_and_evaluate(self, param, model, trial):
        
        #X_base, X_val, y_base, y_val = train_test_split(df_X, df_y, random_state=RANDOM_STATE, shuffle=True, test_size=0.2)

        # train_data, val_data = train_test_split(df_, test_size = 0.2, random_state = 42)
        train, val = TabularDataset(self.X_train, self.y_train), TabularDataset(self.X_val, self.y_val)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=param['batch_size'], shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=param['batch_size'])

        criterion = nn.MSELoss()
        optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    #    if use_cuda:

    #            model = model.cuda()
    #            criterion = criterion.cuda()

        for epoch_num in range(self.epochs):

                total_err_train = 0
                total_loss_train = 0

                for X_train, y_train in train_dataloader:

                    # train_label = train_label.to(device)
                    # train_input = train_input.to(device)
                    # train_y = train_y.reshape(train_y.shape[0], )
                    output = model(X_train.float())
                    
                    # print(output.shape)
                    # print(f'y: {train_y.shape}')

                    batch_loss = criterion(output, y_train)
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

                    for X_val, y_val in val_dataloader:

                        # val_label = val_label.to(device)
                        # val_input = val_input.to(device)
                        # val_y = val_y.reshape(val_y.shape[0], )
                        output = model(X_val.float())

                        batch_loss = criterion(output, y_val)
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



    def __discover_model__(self):

        print(f'Starting train for trials:{self.n_trials} with epochs:{self.epochs}')

        storage = dask_optuna.DaskStorage()
        study = optuna.create_study(storage=storage, direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())


        with joblib.parallel_backend("dask"):
            study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1)

        print("Number of trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial


        # load model from temp folder
        file_name =   "/brisk_ann_{}.pickle".format(study.best_trial.number)
        best_model = self.load_model(self.temp_path + file_name)

        ### eval the model first
        best_model.eval()


        # save it to permanent folder
        print(f"Model saved at:{self.model_save_path + file_name}")
        self.save_model(best_model, self.model_save_path + file_name)

        config.clear_temp_folder(self.temp_path)


        return best_model


    def fetch_model(self):

        client = dasker.get_dask_client()
        print(f"Dask dashboard is available at {client.dashboard_link}")

        self.best_fit = self.__discover_model__()

        return self.best_fit





    ### return score on test dataset
    def get_model_score(self, score_func=None):

        metirc_scores = []
        model_files = glob.glob(self.model_save_path+'/*')
        

        model = self.load_model(model_files[0])    


        X_test_tensor = helper.df_to_tensor(self.X_test)
        pred = model(X_test_tensor)
        pred = helper.torch_tensor_to_numpy(pred)

        
        if score_func is None:                
            if self.pred_class == 'regression':
                metirc_score = r2_score(pred, self.y_test)

            else:
                metirc_score = f1_score(pred, self.y_test)
        else:
            for func in score_func:
                metirc_score = func(pred, self.y_test)
                metirc_scores.append(metirc_score)

        return metirc_scores
      