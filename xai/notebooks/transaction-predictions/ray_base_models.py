import sys; sys.path.insert(0, '../..') # add parent folder path where lib folder is
import ray
import time
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


from utils import helper, config, rayer, kaggle_dataset_helper
from ml.models.base.v2.brisk_xgboost import BriskXGBoost
from ml.models.base.v2.brisk_bagging import BriskBagging
from ml.models.base.v2.slug_xgboost import SlugXGBoost
from ml.models.base.v2.slug_knn import SlugKNN
from ml.models.base.v2.slug_rf import SlugRF

from ml.models import common




@ray.remote #(num_returns=2)
def worker(base_model, X_train_id, X_test_id, y_train_id, y_test_id):     
    base_model.fetch_model(X_train_id, X_test_id, y_train_id, y_test_id)
    return base_model



def main():
    
    rayer.get_global_cluster(num_cpus=4)
    ds_train, ds_test = kaggle_dataset_helper.get_transaction_predictions_dataset()
    ds_train = common.label_encode(ds_train)
    ds_test = common.label_encode(ds_test)

    ds_train = ds_train.fillna(-1)
    ds_test = ds_test.fillna(-1)

    df_X = ds_train.loc[:, ds_train.columns != 'target']
    df_y = ds_train['target']

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=config.rand_state)

    
    ss = StandardScaler()
    X_train_scalar = pd.DataFrame(ss.fit_transform(X_train), columns = X_train.columns)
    X_test_scalar = pd.DataFrame(ss.fit_transform(X_test), columns = X_test.columns)

    
    X_train_id = ray.put(X_train_scalar)
    y_train_id = ray.put(y_train)
    X_test_id = ray.put(X_test_scalar)
    y_test_id = ray.put(y_test)

    
    brisk_xgb1 = BriskXGBoost('brisk_xgb1')
    brisk_xgb1.boosted_round = 100
    brisk_xgb1.n_trials = 500

    brisk_xgb2 = BriskXGBoost('brisk_xgb2')
    brisk_xgb2.boosted_round = 10
    brisk_xgb2.n_trials = 10

    base_models = [brisk_xgb1, brisk_xgb2]

    brisk_bagging_1 = BriskBagging('brisk_bagging_1')
    brisk_bagging_1.boosted_round = 100
    brisk_bagging_1.n_trials = 50

    brisk_bagging_2 = BriskBagging('brisk_bagging_2')
    brisk_bagging_2.boosted_round = 100
    brisk_bagging_2.n_trials = 50

    base_models_bagging = [brisk_bagging_1, brisk_bagging_2]

    slug_xgb1 = SlugXGBoost('slug_xgb1')
    slug_xgb1.boosted_round = 100
    slug_xgb1.n_trials = 500

    slug_xgb2 = SlugXGBoost('slug_xgb2')
    slug_xgb2.boosted_round = 100
    slug_xgb2.n_trials = 500

    base_models_slug = [slug_xgb1, slug_xgb2]

    
    #slug_ann_1 = SlugANN('slug_ann_1', X_train, X_test, y_train, y_test)
    #slug_ann_1.epochs = 50
    #slug_ann_1.n_trials = 50

    #slug_ann_2 = SlugANN('slug_ann_2', X_train, X_test, y_train, y_test)
    #slug_ann_2.epochs = 50
    #slug_ann_2.n_trials = 50

    #base_models_ann = [slug_ann_1, slug_ann_2]


    slug_rf_1 = SlugRF('slug_rf_1')
    slug_rf_1.max_n_estimators = 1000
    slug_rf_1.n_trials = 50

    slug_rf_2 = SlugRF('slug_rf_2')
    slug_rf_2.max_n_estimators = 1000
    slug_rf_2.n_trials = 50

    base_models_rf = [slug_rf_1, slug_rf_2]


    slug_knn_1 = SlugKNN('slug_knn_1')
    slug_knn_1.n_neighbors = 100
    slug_knn_1.n_trials = 500

    slug_knn_2 = SlugKNN('slug_knn_2')
    slug_knn_2.n_neighbors = 100
    slug_knn_1.n_trials = 500

    base_models_knn = [slug_knn_1, slug_knn_2]

    model_results = ray.get([worker.remote(base_model, X_train_id, X_test_id, y_train_id, y_test_id) for base_model in base_models_bagging])
    
    
    

if __name__ == '__main__':
    main()