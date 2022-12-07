

from .config import deepknockoffsettings
from . import selection
from . import parameters
import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs


class DeepKnockoffFilterSim:
    def __init__(self):
        # Set the parameters for training deep knockoffs
        self.pars = dict()
        # Number of epochs
        self.pars['epochs'] = deepknockoffsettings.EPOCHS
        # Number of iterations over the full data per epoch
        self.pars['epoch_length'] = deepknockoffsettings.EPOCH_LENGTH
        # Data type, either "continuous" or "binary"
    
        # Size of the test set
        self.pars['test_size']  = deepknockoffsettings.TEST_SIZE
        # Learning rate
        self.pars['lr'] = deepknockoffsettings.LR
        # When to decrease learning rate (unused when equal to number of epochs)
        self.pars['lr_milestones'] = deepknockoffsettings.LR_MILESTONE
        # Kernel widths for the MMD measure (uniform weights)
        self.pars['alphas'] = deepknockoffsettings.ALPHAS


    def sim_deepknockoffs(self, param, itr):

        np.random.seed(deepknockoffsettings.SEED)
        model = 'mstudent'

        df = param['df']
        fdr = param['fdr']
        
        X = df[df.columns[df.columns!='y']]
        y = df[df.columns[df.columns=='y']]
        X = X.to_numpy()
        y = y.to_numpy()

        # Dimensions of the data
        n, p = df.shape
        self.pars['p'] = p
        # Batch size
        self.pars['batch_size'] = int(0.5*n)

        self.pars['family'] = 'continuous'

        self.pars['dim_h'] = int(10*p)

        # Load the default hyperparameters for this model
        training_params = parameters.GetTrainingHyperParams(model)

        # Penalty for the MMD distance
        self.pars['GAMMA'] = training_params['GAMMA']
        # Penalty encouraging second-order knockoffs
        self.pars['LAMBDA'] = training_params['LAMBDA']
        # Decorrelation penalty hyperparameter
        self.pars['DELTA'] = training_params['DELTA']

        # Target pairwise correlations between variables and knockoffs

        # Compute the empirical covariance matrix of the training data
        SigmaHat = np.cov(X, rowvar=False)

        # Initialize generator of second-order knockoffs
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X, 0), method="sdp")

        # Measure pairwise second-order knockoff correlations 
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
        self.pars['target_corr'] = corr_g

        # Where the machine is stored
        #checkpoint_name = "tmp/" + model

        # Initialize the machine
        machine = KnockoffMachine(self.pars)

        # Load the machine
        # machine.load(checkpoint_name)

        # Generate deep knockoffs
        Xk_m = machine.generate(X)
        # Compute importance statistics
        
        test_params = parameters.GetFDRTestParams(model)
        W_m  = selection.lasso_stats(X, Xk_m, y, alpha=test_params["elasticnet_alpha"], scale=False)

        
