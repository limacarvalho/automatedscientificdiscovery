# from xgboost import XGBRegressor  # change import
from dask_ml.xgboost import XGBRegressor, XGBClassifier

class ConfigXGBoostBrisk():
    MAX_DEPTH = 5
    LEARNING_RATE = 0.1
    N_ESTIMATORS = 500
    OBJECTIVE='binary:logistic'
    N_JOBS = 1
    NTHREAD = None
    GAMMA = 0
    MIN_CHILD_WEIGHT = 1
    MAX_DELTA_STEP = 0
    SUBSAMPLE = 1
    COLSAMPLE_BYTREE = 1
    COLSAMPLE_BYLEVEL = 1
    COLSAMPLE_BYNODE = 1
    REG_ALPHA = 0
    REG_LAMBDA = 1

    class Config:
        case_sensitive = True

configxgboostbrisk = ConfigXGBoostBrisk()




