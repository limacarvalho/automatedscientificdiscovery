from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from ml.models.base.tabular_dataset import TabularDataset
from torch.utils.data import DataLoader
import torch
from torchmetrics import Accuracy, MeanSquaredError
from torch import nn




class BriskDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size):
        #Define required parameters here
        super().__init__()
        self.X = X
        self.y = y
        self.TEST_SIZE = 0.3
        self.VAL_SIZE = 0.2
        self.BATCH_SIZE = batch_size
        self.SEED = 42
        # self.train_dataset = None
        # self.test_dataset = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=self.SEED, test_size=self.TEST_SIZE)

        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, random_state=self.SEED,
                                                                                                 shuffle=True, test_size=self.TEST_SIZE)
    
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, random_state=self.SEED,
                                                                                                 shuffle=True, test_size=self.VAL_SIZE)


        
    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        # generate indices: instead of the actual data we pass in integers instead
        # print(self.test_split)

        # print(X_train.shape)

        if stage == "fit":
            self.train_dataset = TabularDataset(self.X_train, self.y_train)
            self.val_dataset = TabularDataset(self.X_val, self.y_val)
            # print('fit')


        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = TabularDataset(self.X_test, self.y_test)
            # print('test')
            

        if stage == "predict":
            self.pred_dataset = TabularDataset(self.X_test, self.y_test)
            # print('predict')
        


    def train_dataloader(self):
        # Return DataLoader for Training Data here
        # print('train_loader')        
        return DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=False)


    def val_dataloader(self):
        # Return DataLoader for Testing Data here
        # print('val_dataloader')
        dl = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        return dl

        
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        # print('test_dataloader')
        return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)


    def predict_dataloader(self):
        # print('predict_dataloader')
        return DataLoader(self.pred_dataset, batch_size=self.BATCH_SIZE)






class BriskModel(pl.LightningModule): 
    def __init__(self, input_dim = 40): 
        super(BriskModel, self).__init__() 

        # Defining learning rate
        self.lr = 0.004
        #self.batch_size = batch_size
        #self.X = X
        #self.y = y
        #hidden_size = X.shape[0]

        # Defining our model architecture
        self.layer_0 = nn.BatchNorm1d(num_features = input_dim)
        # self.layer_0 = nn.Linear(hidden_size, 200)
        self.layer_1 = nn.Linear(in_features = input_dim, out_features = 112)
        self.layer_2 = nn.Linear(112, 36)
        self.layer_3 = nn.Linear(36, 192)
        self.layer_final = nn.Linear(192, 1)
        self.relu = nn.ReLU()

        self.flatten = torch.nn.Flatten(0, 1)

        self.val_mse = MeanSquaredError()
        self.train_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.predictions = []


    def forward(self, x):

        x = self.relu(self.layer_0(x))
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.flatten(self.layer_final(x))
        return x
    
    def configure_optimizers(self):
          # Defining and returning the optimizer for our model
        # with the defines parameters
        #return torch.optim.SGD(self.parameters(), lr = self.lr) 
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, batch_idx): 
        # Defining training steps for our mode
        x, y = train_batch        
        y = y.reshape(y.shape[0], )

        pred = self.forward(x)        

        loss = nn.MSELoss()(pred, y)

        self.train_mse.update(pred, y)
        return loss



    def on_train_end(self) -> None:
        print(f'training ended: MSE {self.train_mse.compute()}')
        return super().on_train_end()



    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(y.shape[0], )
        pred = self.forward(x)
        
        loss = nn.MSELoss()(pred, y)

        self.val_mse.update(pred, y)
        return loss

    

    def on_validation_end(self) -> None:
        print(f'validation ended: MSE {self.val_mse.compute()}')
        return super().on_validation_end()


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.reshape(y.shape[0], )
        pred = self.forward(x)
        loss = nn.MSELoss()(pred, y)

        self.test_mse.update(pred, y)

        list_np = pred.cpu().detach().numpy()

        self.predictions.extend(list_np)
        return loss



    def on_test_end(self) -> None:
        print(f'test ended: MSE {self.test_mse.compute()}')
        return super().on_test_end()

    def on_predict_end(self) -> None:        
        return super().on_predict_end()

