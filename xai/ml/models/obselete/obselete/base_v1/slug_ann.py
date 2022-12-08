
from ml.models.base.tabular_dataset import TabularDataset


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


from utils import helper, config, logger
import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)

### change the logger type to save info logs
customlogger = logger.logging.getLogger('console_info')



class SlugANN:
    def __init__(self) -> None:
        self.n_layers = 8
        self.max_neurons_per_layer = 256
        self.epochs = 300
        self.n_trials = 100
        self.random_state = 0


        self.temp_path = config.main_dir + config.project_name + "/tmp/ann/slug"
        self.model_save_path = config.main_dir + config.project_name  + "/base/ann/slug"
        
        self.pred_class = "regression"        
        
        self.best_fit = None
        self.score = None        
        
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

        self.pred_train = None
        self.pred_test = None        




    # Load pickled models
    def __load_model__(self, file_name):
        with open(file_name, "rb") as fin:
            model = pickle.load(fin)

        return model


    
    def __save_model__(self, model, file_name):
        with open(file_name, "wb") as fout:
            pickle.dump(model, fout)


            
    def __load_best_fit__(self):        
        if self.best_fit is None:            
            model_files = glob.glob(self.model_save_path+'/*')
            if len(model_files) == 0:
                return None
            else:
                self.best_fit = self.__load_model__(model_files[0])      

            

    def __create_slug_ann_model__(self, trial, input_dim):

        # We optimize the number of layers, hidden untis and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, self.n_layers)
        layers = []

        in_features = input_dim

        
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 1, self.max_neurons_per_layer)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))            
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


        input_shape = self.X_train.shape[1]
    
        model = self.__create_slug_ann_model__(trial, input_shape)

        accuracy = self.train_and_evaluate(params, model, trial)

        file_name =  self.temp_path + "/slug_ann_{}.pickle".format(trial.number)
        
        # save model in temp folder
        self.__save_model__(model, file_name)


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

                # total_err = 0
                
                total_loss_train = 0
                total_loss_val = 0

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
                
                val_loss = total_loss_val
                
                # print(f'val loss {val_loss}')
                # Add prune mechanism
                trial.report(val_loss, epoch_num)

    #            if trial.should_prune():
    #                raise optuna.exceptions.TrialPruned()

        return val_loss



    def __discover_model__(self):

        customlogger.info('slug ann: Starting train for trials:%d with epochs: %d', self.n_trials, self.epochs)

        customlogger.info('slug ann: Cleared previous models in the model save path')        
        config.clear_temp_folder(self.model_save_path)        
        
        storage = dask_optuna.DaskStorage()
        study = optuna.create_study(storage=storage, direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())


        with joblib.parallel_backend("dask"):
            study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1)

            
        customlogger.info('slug xgboost: Number of trials: %d', len(study.trials))
                   
        customlogger.info('Best trial: %s', study.best_trial.number)
        trial = study.best_trial


        # load model from temp folder
        file_name =   "/slug_ann_{}.pickle".format(study.best_trial.number)
        best_model = self.__load_model__(self.temp_path + file_name)

        ### eval the model first
        best_model.eval()


        # save it to permanent folder
        customlogger.info('slug ann: Model saved at %s', self.model_save_path + file_name)
        self.__save_model__(best_model, self.model_save_path + file_name)

        config.clear_temp_folder(self.temp_path)
        return best_model



    ### perform hyper-parameter search on xgboost model
    def fetch_model(self, retrain = True):
    
        if retrain:                
            self.best_fit = self.__discover_model__()
        else:
            self.__load_best_fit__()
            if self.best_fit is None:
                customlogger.info("slug ann: no saved models found, please rerun the 'fetch_model' first.")
                return None

            
        self.get_model_score()                
        return self.best_fit    
    



    ### return score on test dataset
    def get_model_score(self, score_func=None, persist_pred=True):
        
        self.__load_best_fit__()
                        
        if self.best_fit is None:
            customlogger.info("xgboost: no saved models found, please rerun the 'fetch_model' first.")
            return None
        
        X_train_tensor = helper.df_to_tensor(self.X_train)
        X__tensor = helper.df_to_tensor(self.X_train)
        X_test_tensor = helper.df_to_tensor(self.X_test)

        pred_train = self.best_fit(X_train_tensor)
        pred_train = helper.torch_tensor_to_numpy(pred_train)
        pred_train = pred_train.reshape(pred_train.shape[0], )

        
        pred_test = self.best_fit(X_test_tensor)
        pred_test = helper.torch_tensor_to_numpy(pred_test)
        pred_test = pred_test.reshape(pred_test.shape[0], )
        

        if persist_pred:
            self.pred_train = pred_train
            self.pred_test = pred_test
        
        
        if score_func is None:                
            if self.pred_class == 'regression':
                metirc_score_train = r2_score(pred_train, self.y_train)
                metirc_score_test = r2_score(pred_test, self.y_test)
            else:
                metirc_score_train = f1_score(pred_train, self.y_train)
                metirc_score_test = f1_score(pred_test, self.y_test)                
        else:
            metirc_score_train = score_func(pred_train, self.y_train)
            metirc_score_test = score_func(pred_test, self.y_test)            

        self.score = [metirc_score_train, metirc_score_test]
        return self.score


          
    def get_predictions(self, model):

        X_train = pd.concat([self.X_train, self.X_val])
        y_train = pd.concat([self.y_train, self.y_val])
        
        X_train_tensor = helper.df_to_tensor(X_train)
        X_test_tensor = helper.df_to_tensor(self.X_test)

        pred_train = model(X_train_tensor)
        pred_train = helper.torch_tensor_to_numpy(pred_train)
        pred_train = pred_train.reshape(pred_train.shape[0], )        

        pred_test = model(X_test_tensor)
        pred_test = helper.torch_tensor_to_numpy(pred_test)
        pred_test = pred_test.reshape(pred_test.shape[0], )        

        return pred_train, pred_test, y_train, self.y_test        
    
        
    ### predict 
    def predict(self, df_X):                
        
        self.__load_best_fit__()        
        if self.best_fit is None:
            customlogger.info("xgboost: no saved models found, please rerun the 'fetch_model' first.")            
            return None
            
        X_test_tensor = helper.df_to_tensor(df_X)
        pred = self.best_fit(X_test_tensor)
        pred = helper.torch_tensor_to_numpy(pred)
        pred = pred.reshape(pred.shape[0], )
        return pred
