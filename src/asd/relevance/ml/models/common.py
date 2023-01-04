import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def get_weighted_score(err_train, err_test, pred_class, score_func=None, penalty=4):
    
    ### make the train/test difference 4 times more important than test score
    if pred_class == 'regression':
        #### only valid for mean_square_error
        return np.sqrt((err_train - err_test)**2) * penalty + err_test
    else:
        return np.sqrt((err_train - err_test)**2) + 4 * (1 - err_test)



def label_encode(df):
    
    df_tmp = df.copy()
    le = preprocessing.LabelEncoder()
    for col in df_tmp:
        if df_tmp[col].dtype == object:
            #df_train[col] = df_train[col].astype('str')
            df_tmp[col] = le.fit_transform(df_tmp[col])
    return df_tmp





# Load the state of base/ensemble model(s)
def load_state(file_name):
    with open(file_name, "rb") as fin:
        model = pickle.load(fin)
    return model


# save the state of base/ensemble model(s)
def save_state(state, file_name):
    with open(file_name, "wb") as fout:
        pickle.dump(state, fout)
