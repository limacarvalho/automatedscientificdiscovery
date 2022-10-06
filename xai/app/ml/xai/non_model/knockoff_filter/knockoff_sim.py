import os
import logging
from knockpy.knockoff_filter import KnockoffFilter
import pandas as pd
import numpy as np
import dask
import warnings
import knockpy
from .config import knockoffsettings


def simmulate_obs(param, index):

    np.random.seed(knockoffsettings.SEED)

    df = param['df']
    # Sigma = param['Sigma']
    Sigma = None
    ksampler = param['ksampler']
    fstat = param['fstat']
    fdr = param['fdr']



    X = df[df.columns[df.columns!='y']]
    y = df[df.columns[df.columns=='y']]
    X = X.to_numpy()
    y = y.to_numpy()
    
    
    kfilter = KnockoffFilter(
        ksampler = ksampler,
        fstat = fstat,
    )

    rejections = kfilter.forward(
        X = X,
        y = y,
        Sigma = Sigma,
        fdr = fdr
    )
    
    i = [index] * len(rejections)
    return list(zip(rejections, kfilter.W, i))



def simmulate(param, index):

    np.random.seed(knockoffsettings.SEED)
    df = param['df']
    Sigma = None
    ksampler = param['ksampler']
    fstat = param['fstat']
    fdr = param['fdr']
    
    X = df[df.columns[df.columns!='y']]
    y = df[df.columns[df.columns=='y']]
    X = X.to_numpy()
    y = y.values.ravel()

    print( 'itr: ' + str(index) + ', ksampler:' + str(ksampler) + ', fstat:' + str(fstat) )

    if ksampler != 'metro':
        kfilter = KnockoffFilter(
            ksampler = ksampler,
            fstat = fstat,
        )

        rejections = kfilter.forward(
            X = X,
            y = y,
            Sigma = Sigma,
            fdr = fdr
        )

    else:
        n, p = X.shape
        U = np.zeros((p, p))
        for xcoord in range(p):
            for offset in [-2, 1, 0, 1, 2]:
                ycoord = min(max(0, xcoord + offset), p-1)
                U[xcoord, ycoord] = 1

        warnings.filterwarnings("ignore")
        metrosampler = knockpy.metro.MetropolizedKnockoffSampler(
            log_likelihood, X=X, undir_graph=U
        )
        Xk = metrosampler.sample_knockoffs()        
        kfilter = KnockoffFilter(ksampler=metrosampler, 
                                       fstat=fstat)
        rejections = kfilter.forward(
                X = X,
                Xk = Xk,
                y = y,
                fdr = fdr)
        
    i = [index] * len(rejections)
    return list(zip(rejections, kfilter.W, i))

class KnockoffFilterSim:
### when running in local mode, it is recommended not to change the n_threads and n_workers as Dask will automatically identify this number based on the current CPU load
    def __init__(self, cluster ):

        if cluster is None:
            print("cluster object is needed")
            logging.exception("cluster object is needed")
            os._exit()

        self.client = cluster
        self.columns = ['rejections', 'W', 'i']
        self.df_results = pd.DataFrame(columns = self.columns)
        self.rejections = []
        

    def sim_knockoffs(self, param, itr):

        lazy_results = []
#        for i in range(1, itr+1):
#            temp = []
#            lazy_results.extend(temp)
#            for fstat in knockoffsettings.FSTATS:
#                param['fstat'] = fstat
#                temp_temp = dask.delayed(simmulate)(param, fstat, i)
#                temp.append(temp_temp)
                # param['fstat'] = fstat
#                    res = dask.delayed(simmulate)(param, i)
#                    lazy_results.append(res)

        for ksampler in knockoffsettings.KSAMPLER:
            param['ksampler'] = ksampler
            for fstat in knockoffsettings.FSTATS:
                for i in range(1, itr+1):
                    param['fstat'] = fstat                    
                    res = dask.delayed(simmulate)(param, i)
                    lazy_results.append(res)

        results = dask.compute(*lazy_results, scheduler='distributed')

        self.rejections = results

        df_results = pd.DataFrame(columns = self.columns)

        for result in results:
            df_res = pd.DataFrame(result, columns = self.columns)
            df_results = pd.concat([df_results, df_res])
        self.df_results = df_results
        
        self.rejections = df_results.groupby([df_results.index])['rejections'].agg(lambda x: pd.Series.mode(x)[0])



    def sim_metro_knockoffs(self, param, itr):
        # Undirected graph
        U = np.zeros((p, p))
        for xcoord in range(p):
            for offset in [-2, 1, 0, 1, 2]:
                ycoord = min(max(0, xcoord + offset), p-1)
                U[xcoord, ycoord] = 1

        warnings.filterwarnings("ignore")
        metrosampler = knockpy.metro.MetropolizedKnockoffSampler(
            log_likelihood, X=X_metro, undir_graph=U
        )
        Xk = metrosampler.sample_knockoffs()
                

    

##################### helper functions ########################
def log_likelihood(X):
    # An arbitrary (unnormalized) log-likelihood function
    n, p = X.shape
    rhos = np.random.randn(p)
    return np.sum(X[:, 0:-1]*rhos[0:-1]*np.abs(X[:, 1:]))




    
    def sim_knockoffs_obs(self, param, itr):

        func = dask.delayed(simmulate)

        lazy_results = []
        #for i in range(1, itr+1):
            # param['fstat'] = fstat
        for fstat in ['dlasso', 'ridge', 'deeppink', 'randomforest']:
            # param['fstat'] = fstat
            res = dask.delayed(simmulate)(param, fstat, 1)
            lazy_results.append(res)

        results = dask.compute(*lazy_results, scheduler='distributed')

        self.rejections = results

        df_results = pd.DataFrame(columns = self.columns)

        for result in results:
            df_res = pd.DataFrame(result, columns = self.columns)
            df_results = pd.concat([df_results, df_res])
        self.df_results = df_results
        
        self.rejections = df_results.groupby([df_results.index])['rejections'].agg(lambda x: pd.Series.mode(x)[0])
