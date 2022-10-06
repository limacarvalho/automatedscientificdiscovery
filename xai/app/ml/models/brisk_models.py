from ..preprocess.data import SingletonDataSet
from .custom_cross_val import CustomCrossVal
from sklearn.model_selection import StratifiedKFold, KFold
from ml.models.base import BriskXGBoost, BriskNL
from app.ml.models.base.brisk_neural_net import run_all


from utils import  logging
# from dask_ml.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import os


class BriskModels:
    def __init__(self, client ) -> None:

        if client is None:
            print("cluster object is needed")
            logging.exception("cluster object is needed")
            os._exit()

        self.client = client
        self.cv = None
    
        #self.dataset = SingletonDataSet()
        #self.X, self.y = self.dataset.get_data()
        #self.weights = self.dataset.get_weights()


        #if self.dataset.pred_type == 'classification':
        #    self.cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        #else:
        #    self.cv = KFold(n_splits=5, random_state=None, shuffle=False)
        
        

    def __fit_xgboost(self):

        # self.y = self.y.to_numpy()
        # self.ddf = from_pandas(self.X, npartitions=10)

        briskxgboost = BriskXGBoost(self.client, self.cv, logger = logging, pred_type = self.dataset.pred_type)

        res = briskxgboost.fit(self.X, self.y)



    def __fit_shalow_nn(self):

        brisknl = BriskNL(self.client, self.cv, logger = logging)
            
        res = brisknl.fit()



    def fit_models(self):

        # self.__fit_xgboost()
        # self.__fit_shalow_nn()
        run_all()


#        shallownl = ShallowNL()
#        shallownl.fit(self.X, self.y, self.weights, 'classification')
        







