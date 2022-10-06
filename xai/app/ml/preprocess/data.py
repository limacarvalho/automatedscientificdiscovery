
# from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from utils.logger import logging
import traceback



class SingletonDataSet(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = object.__new__(cls)
        return cls.__instance


    def __init__(self):
        pass
    
    def load_data(self, df, itr, fdr, pred_type):


        self.itr = itr
        self.fdr = fdr
        self.pred_type = pred_type

        self.X = df[df.columns[df.columns!='y']]
        self.y = df[df.columns[df.columns=='y']]

        

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = configdata.TESTSET, random_state=42)

        # self.X = self.X.to_numpy()
        # self.y = self.y.values.ravel()
        self.class_weights = None

        if pred_type == 'classification':
            self.class_weights = compute_class_weight(class_weight = 'balanced',
                                                    classes = np.unique(self.y),
                                                    y = self.y.values.ravel())
        


    def get_data(self):
        return self.X, self.y


#    def get_train_testset(self):
#        return self.X_train, self.X_test, self.y_train, self.y_test

    
    def get_weights(self):
        return self.class_weights



# Examples 
### f_name = '20220319_covid_merge_processed' , sep = ','
### f_name = '' , sep = ','

def get_dataset(f_name, sep):

    # f_name = "20220319_covid_merge_processed.csv"

    try:
        f_path = configdata.TEMP_PATH + f_name

        # df = pd.read_csv(f_path, sep=",")
        df = pd.read_csv(f_path, sep=sep)


        X = df[df.columns[df.columns!='y']]
        y = df[df.columns[df.columns=='y']]

        return X, y
    except Exception as e:
        logging.error(traceback.format_exc())
        return None




class ConfigData():
    TESTSET: float = 0.33 # in percents between 0.0 and 1.0
    TEMP_PATH = "/mnt/c/Users/rwmas/GitHub/xai/xai_api/app/test/data/"
    class Config:
        case_sensitive = True

configdata = ConfigData()
