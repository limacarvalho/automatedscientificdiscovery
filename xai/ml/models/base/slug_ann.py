from .base_model import BaseModel
from ml.models import common
from utils import config, helper
from utils.asd_logging import logger as  customlogger

from ml.models.base.tabular_dataset import TabularDataset

import pickle

# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import torch
from torch import nn
import torch.optim as optim
import optuna

import pandas as pd







optuna.logging.set_verbosity(optuna.logging.WARNING)

### change the logger type to save info logs
# customlogger = logger.logging.getLogger('console_info')



class SlugANN(BaseModel):
    def __init__(self,
                    name,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    timeout=None,
                ) -> None:

        self.pred_class = "regression"        

        self.n_layers = 8
        self.max_neurons_per_layer = 256
        self.epochs = 300
        self.n_trials = 100
        self.random_state = 0

        self.score_func = None

        self.timeout = timeout

        self.model_file_name = name        
        
        super().__init__( X_train, X_test, y_train, y_test)




            

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

        weighted_score = self.train_and_evaluate(params, model, trial)


        file_name =  self.temp_path + "/" + self.model_file_name + '_' + str(trial.number) +'.pickle'      
        self.__save_model__(model, file_name)

        return weighted_score

      

    # Train and evaluate the accuracy of neural network with the addition of pruning mechanism
    def train_and_evaluate(self, param, model, trial):
        
        #X_base, X_val, y_base, y_val = train_test_split(df_X, df_y, random_state=RANDOM_STATE, shuffle=True, test_size=0.2)

        # train_data, val_data = train_test_split(df_, test_size = 0.2, random_state = 42)
        train, test = TabularDataset(self.X_train, self.y_train, self.pred_class), TabularDataset(self.X_test, self.y_test, self.pred_class)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=param['batch_size'], shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=param['batch_size'])

        criterion = nn.MSELoss()
        optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])


        for epoch_num in range(self.epochs):

                loss_train = 0
                loss_test = 0

                for X_train, y_train in train_dataloader:

                    # train_label = train_label.to(device)
                    # train_input = train_input.to(device)
                    y_train = y_train.reshape(y_train.shape[0], 1)
                    output = model(X_train.float())
                    
                    # print(output - y_train) 
                    # print(y_train[0:5]) 

                    #y_train = y_train.reshape(y_train.shape[0], )
#                    y_train = y_train.values

#                    print(output)
#                    print(y_test)


                    batch_loss = criterion(output, y_train)
                    # print(batch_loss)
                    loss_train += batch_loss.item()
                    
                    # acc = (output.argmax(dim=1) == train_label).sum().item()

                    #err = np.round_(mean_squared_error(output, train_y), decimals=2, out=None)
                    # total_err_train += err

                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                

                with torch.no_grad():

                    for X_test, y_test in test_dataloader:

                        # val_label = val_label.to(device)
                        # val_input = val_input.to(device)
                        # val_y = val_y.reshape(val_y.shape[0], )
                        output = model(X_test.float())

                        # y_test = y_test.reshape(-1, 1)
                        y_test = y_test.reshape(y_test.shape[0], 1)
                        #y_test = y_test.values

                        batch_loss = criterion(output, y_test)
                        loss_test += batch_loss.item()
                        
                        # acc = (output.argmax(dim=1) == val_label).sum().item()
                        # err = np.round_(mean_squared_error(output, val_y), decimals=2, out=None)
                        #total_acc_val += err
                
                accuracy = loss_test
                
                # print(loss_test)
                # print(loss_train)

                weighted_score = common.get_weighted_score(loss_train, loss_test, self.pred_class, self.score_func)      

                

                # Add prune mechanism
                trial.report(weighted_score, epoch_num)


        # customlogger.info( self.model_file_name + ': weighted score: %f', weighted_score)
        return weighted_score



    def __predict__(self, model):

        X_train_tensor = helper.df_to_tensor(self.X_train)        
        pred_train = model(X_train_tensor)
        pred_train = helper.torch_tensor_to_numpy(pred_train)
        pred_train = pred_train.reshape(pred_train.shape[0], )        

        X_test_tensor = helper.df_to_tensor(self.X_test)
        pred_test = model(X_test_tensor)
        pred_test = helper.torch_tensor_to_numpy(pred_test)
        pred_test = pred_test.reshape(pred_test.shape[0], )        

        return pred_train, pred_test



    def __discover_model__(self):

        customlogger.info(self.model_file_name + ': Starting train for trials:%d with epochs: %d', self.n_trials, self.epochs)

        
        study = optuna.create_study(study_name=config.create_study_name(),
                                    direction="minimize",
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.MedianPruner()
                                    )


        study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1)

            
        customlogger.info(self.model_file_name +': Number of trials: %d', len(study.trials))
                   
        customlogger.info('Best trial: %s', study.best_trial.number)

        # load model from temp folder        
        file_name = self.temp_path + "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'
        best_model = self.__load_model__(file_name)

        ### eval the model first
        customlogger.info(best_model.eval())

        pred_train, pred_test = self.__predict__(best_model)
        
        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test, score_func=self.score_func)
        
        self.score = [metirc_score_train, metirc_score_test]
        customlogger.info('  test r2 score: %s', metirc_score_test)

        # save it to permanent folder
        customlogger.info( self.model_file_name + ': Model saved at %s', self.model_save_path + file_name)
        file_name = self.model_save_path + "/" + self.model_file_name + '_' + str(study.best_trial.number) +'.pickle'        
        self.__save_model__(best_model, file_name)

        config.clear_temp_folder(self.temp_path)

        return best_model



    ### perform hyper-parameter search on xgboost model
    def fetch_model(self,  score_func=None, threshold=None):

        ### It is important to call this function here so that the directory structure can be built in the remote cluster rather on the local machine    
        super().__create_dir_structure__(self.model_file_name)

    
        self.best_fit = self.__discover_model__()
        self.model = self.best_fit

        self.load_score(score_func, threshold)
        
        return self.best_fit 
    



    ### return score on test dataset
    def load_score(self, score_func=None, threshold=None):
        
        if self.model is None:
            customlogger.info(self.model_file_name + "No trained models found, pls rerun 'fetch_models'")
            return None

        if self.X_train is None:
            customlogger.info(self.model_file_name + "No train/test dataset found, pls explicity set the parameters.")
            return None

        pred_train, pred_test = self.__predict__(self.model)

        metirc_score_train, metirc_score_test, weighted_score = self.__get_model_score__(pred_train, pred_test)

        if threshold:
            if metirc_score_test > threshold:
                self.best_fit = self.model
                self.score = [metirc_score_train, metirc_score_test]
            else:
                self.best_fit = None
                self.score = []
        else:
            self.best_fit = self.model
            self.score = [metirc_score_train, metirc_score_test]


        customlogger.info(self.model_file_name + ": scores loaded: " + str(self.score) )            
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
        
        if self.best_fit is None:
            customlogger.info(self.model_file_name + "no model attached as per your selection threshold. Lower the threshold in the 'load_score' function.")  
            return None        
        
        if self.X_train is None:
            customlogger.info(self.model_file_name + "No train/test dataset found, pls explicity set the parameters.")
            return None

            
        X_test_tensor = helper.df_to_tensor(df_X)
        pred = self.best_fit(X_test_tensor)
        pred = helper.torch_tensor_to_numpy(pred)
        pred = pred.reshape(pred.shape[0], )
        return pred
