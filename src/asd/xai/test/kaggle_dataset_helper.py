
from utils import config
import pandas as pd


def get_house_prices_dataset():
    df_train = pd.read_csv('/mnt/c/Users/rwmas/GitHub/data/house-prices/train.csv', sep = ',')
    df_test = pd.read_csv('/mnt/c/Users/rwmas/GitHub/data/house-prices/test.csv', sep = ',')
        
    # df_train = df_train.rename(columns={"SalePrice": "y"})

    return df_train, df_test



def get_transaction_predictions_dataset():
    df_train = pd.read_csv('/mnt/c/Users/rwmas/GitHub/data/transaction-predictions/train.csv', sep = ',')
    df_test = pd.read_csv('/mnt/c/Users/rwmas/GitHub/data/transaction-predictions/test.csv', sep = ',')

    return df_train, df_test