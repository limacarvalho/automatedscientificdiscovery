
import numpy as np
import pandas as pd
from pathlib import Path
import datatable as dt
from .utilities_asd_team  import convert_datatypes


def get_covid_dataset():
    df = pd.read_csv('./covid/20220319_covid_merge_processed.csv', sep=",")
    X = df[df.columns[df.columns!='y']]
    # X = X.drop(DROP_LIST, axis = 1)
    y = df[df.columns[df.columns=='y']]
    return X, y



def get_tokamat_dataset():
    
    ## read data table
    base_path = Path(__file__).parent
    print(base_path)
    # HDB5.2.3_nans.csv,
    path = ('/mnt/c/Users/rwmas/GitHub/data/tokamak/HDB5.2.3_no_nans.csv')#.resolve()
    data = dt.fread(path).to_pandas()

    ## important: convert datatypes first!
    data = convert_datatypes(data)
    # print('success:', data.shape, '\n', data.dtypes)
    return data