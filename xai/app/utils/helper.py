import torch
import numpy as np
import pandas as pd


def get_covid_dataset():
    df = pd.read_csv('/home/wasif/python-asd/xai/app/test/data/20220319_covid_merge_processed.csv', sep=",")
    X = df[df.columns[df.columns!='y']]
    # X = X.drop(DROP_LIST, axis = 1)
    y = df[df.columns[df.columns=='y']]
    return X, y


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    # device = get_device()
    # return torch.from_numpy(df.values).float().to(device)
    return torch.from_numpy(df.values).float()


def tensor_to_df(tensor):
    # device = get_device()
    # return torch.from_numpy(df.values).float().to(device)
    return torch.from_numpy(df.values).float()    


def torch_tensor_to_numpy(tensor_array):
    return tensor_array.cpu().detach().numpy()


def check_if_all_same(list_of_elem, item):
    """ Check if all elements in list are same and matches
     the given item"""
    result = True
    for elem in list_of_elem:
        if elem != item:
            return False
    return result


def custom(train_score, test_score):
    return np.round((train_score * test_score) / (train_score + test_score), 2)*2  