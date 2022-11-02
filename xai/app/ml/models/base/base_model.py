
import pickle
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from ml.models import common


# Python program showing
# abstract base class work
 
from abc import ABC, abstractmethod



class BaseModel(ABC):
    def __init__(self, X_train, X_test, y_train, y_test, temp_path, model_save_path, model_file_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.temp_path = temp_path
        self.model_save_path = model_save_path
        self.model_file_name = model_file_name
        self.pred_train = None
        self.pred_test = None

        self.model = None        
        self.score = None


    @abstractmethod
    def __objective__(self, trial):
        raise NotImplementedError


    @abstractmethod
    def __discover_model__(self):
        raise NotImplementedError


    @abstractmethod    
    def fetch_model(self):
        raise NotImplementedError


    @abstractmethod    
    def predict(self, df_X):
        raise NotImplementedError


    def empty_media(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
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


    
    def __load_saved_models__(self):
        model_files = glob.glob(self.model_save_path+'/*')    
        self.models = []

        for file in model_files:
            model = self.__load_model__(file)
            self.models.append(model)

        return len(self.models)


    
    def __get_model_score__(self, pred_train, pred_test, score_func=None):

        if self.pred_class == 'regression':

            if score_func is None:
                err_test = r2_score(self.y_test, pred_test)
                err_train = r2_score(self.y_train, pred_train)

                err_weighted = common.get_weighted_score(err_train, err_test, pred_class=self.pred_class)
            else:
                err_test = score_func(self.y_test, pred_test)
                err_train = score_func(self.y_train, pred_train)                

                ### to keep track of overfitting
                r2_err_test = r2_score(self.y_test, pred_test)
                r2_err_train = r2_score(self.y_train, pred_train)

                ### we need r2 score for regression problems
            err_weighted = common.get_weighted_score(err_train, err_test, pred_class=self.pred_class)

        else:
            if score_func is None:
                err_test = f1_score(self.y_test, pred_test)
                err_train = f1_score(self.y_train, pred_train)

                err_weighted = common.get_weighted_score(err_train, err_test, pred_class=self.pred_class)
            else:
                err_test = score_func(self.y_test, pred_test)
                err_train = score_func(self.y_train, pred_train)                
            
                ### to keep track of overfitting, we need f1-score for classification problems
                f1_err_test = f1_score(self.y_test, pred_test)
                f1_err_train = f1_score(self.y_train, pred_train)
                err_weighted = common.get_weighted_score(f1_err_train, f1_err_test, pred_class=self.pred_class)

                            
        return err_train, err_test, err_weighted
