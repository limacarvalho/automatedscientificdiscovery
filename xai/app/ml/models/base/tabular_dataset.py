from torch.utils.data import Dataset
import torch




class TabularDataset(Dataset):
    def __init__(self, features, label_col, pred_class):
        # self.path = path
        self.features = features
        self.label_col = label_col
        self.pred_class = pred_class
        # self.df = pd.read_csv(path)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        num_array = self.features.iloc[idx].values
        num_array = torch.tensor(num_array, dtype = torch.float32)
        label_array = self.label_col.iloc[idx]
        if self.pred_class == 'regression':
            label_array = torch.tensor(label_array, dtype = torch.float32)
        else:
            label_array = torch.tensor(label_array, dtype = torch.long)
        return num_array, label_array
