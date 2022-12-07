
from this import d
from .knockoff_sim import KnockoffFilterSim
# from .knockoff_sim import knockoff_sim as ks

import pandas as pd
import sys
import numpy as np
import pandas as pd
import os
from knockoff_filter import config
from utils import config as util_config




####
# Flags of whether each feature was rejected
from knockpy.knockoff_filter import KnockoffFilter


# Create a random covariance matrix for X
import knockpy
import warnings
from utils import dasker



def main():

    n = 300 # number of data points
    p = 500  # number of features
    Sigma = knockpy.dgp.AR1(p=p, rho=0.5) # Stationary AR1 process with correlation 0.5

    # Sample X
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=(n,))

    # Create random sparse coefficients
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=0.1)
    y = np.dot(X, beta) + np.random.randn(n)

    df = pd.DataFrame(X)
    df['y']= y


    param = {'df': df, 'Sigma': Sigma, 'ksampler': 'gaussian', 'fstat': 'lasso', 'fdr': 0.1}

    client = dasker.get_dask_client()

    print(f'Local dask client created: localhost:{util_config.dask_local_scheduler_port}')
    print(client)

    ks = KnockoffFilterSim(client)

    print(f'running knockoff filter with iteration { config.itr} and seed {config.seed}')
    ks.sim_knockoffs(param, config.itr, rand=config.seed)

    power = np.dot(ks.rejections, beta != 0) / (beta != 0).sum()
    fdp = np.around(100*np.dot(ks.rejections, beta==0) / ks.rejections.sum())
    print(f"The knockoff filter has discovered {100*power}% of the non-nulls with a FDP of {fdp}%")

    # client.shutdown()


    



if __name__ == "__main__":
    main()