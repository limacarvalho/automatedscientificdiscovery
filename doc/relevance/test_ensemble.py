




from sklearn.model_selection import train_test_split

from asd.relevance.utils import helper, rayer
from asd.relevance.ml.models.ensemble import Ensemble

from sklearn.metrics import make_scorer
from sklearn import metrics

from asd.relevance.ml.models import common
import dataset_handler
import pandas as pd
import time

# list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']



def test_case_1():

    df = dataset_handler.get_tokamat_dataset()
    df = df.reset_index()

    df = common.label_encode(df)
    df = df.fillna(-1)

    potential_targets = ['WTOT', 'WTH', 'PLTH']
    df_y = df[potential_targets[0]]

    df_X = df[df.columns[~df.columns.isin(potential_targets)]]
    df_X = df_X.drop(['TOK_ID', 'LCUPDATE', 'DATE', 'NEL', 'ENBI'], axis = 1)
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33)

    r2_scoring = make_scorer(score_func=metrics.r2_score, greater_is_better=True)
    
    ens_mdl = Ensemble(   
                                xgb_objective='count:poisson',  # ["reg:squarederror", "count:poisson", "binary:logistic",  "binary:hinge" ]
                                lgbm_objective='poisson',    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
                                pred_class='regression',
                                score_func=r2_scoring,
                                metric_func=metrics.r2_score,
                                list_base_models=['slugrf'],
                                n_trials=100,
                                boosted_round=100,
                                max_depth=30,
                                rf_n_estimators=1500,
                                bagging_estimators=100,
                                n_neighbors=30,
                                cv_splits=3,
                                ensemble_bagging_estimators=50,
                                ensemble_n_trials=50,
                                timeout=None
                 )

    
    ens_mdl.fetch_models(X_train, X_test, y_train, y_test)

    print('########### Base Model Scores ###################')
    print(ens_mdl.base_model_scores)

    print('############# Ensemble Score #################')
    print(ens_mdl.scores)





def main():

    # rayer.get_global_cluster()
    rayer.get_local_cluster()
    test_case_1()

if __name__ == '__main__':
    main()