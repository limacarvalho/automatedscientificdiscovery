
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):        
        num_array = torch.tensor(self.x.iloc[index], dtype=torch.float32)        
        label_array = torch.tensor(self.y.iloc[index], dtype=torch.float32)
        return num_array, label_array

    def __len__(self):
        return len(self.x)