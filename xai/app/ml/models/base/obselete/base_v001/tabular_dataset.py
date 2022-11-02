from torch.utils.data import Dataset
import torch
import pandas as pd
# id_cols = ['Player_Id']
# features = ['Deposit_Method', 'Country', 'Gender', 'Dim_Alert_Type_Desc', 'Deposit_Amount', 'Deposit_Quantity', 'Age']
# label_col = 'VIP'


class TabularDataset(Dataset):
    def __init__(self, path, features, label_col):
        self.path = path
        self.features = features
        self.label_col = label_col
        self.df = pd.read_csv(path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        num_array = self.df[self.features].iloc[idx].values
        num_array = torch.tensor(num_array, dtype = torch.float32)
        label_array = self.df[self.label_col].iloc[idx]
        label_array = torch.tensor(label_array, dtype = torch.long)
        return num_array, label_array
