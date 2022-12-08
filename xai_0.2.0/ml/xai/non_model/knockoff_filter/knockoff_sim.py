import os
import logging
from knockpy.knockoff_filter import KnockoffFilter
import pandas as pd
import numpy as np
import ray
import warnings
import knockpy
from .config import knockoffsettings
from utils.asd_logging import logger as  customlogger



def __simmulate__(param, index, X, y):

    np.random.seed(knockoffsettings.SEED)

    Sigma = None
    ksampler = param['ksampler']
    fstat = param['fstat']
    fdr = param['fdr']
    
    #X = df_X
    #y = df_y
    #X = X.to_numpy()
    #y = y.values.ravel()

    print( 'itr: ' + str(index) + ', ksampler:' + str(ksampler) + ', fstat:' + str(fstat) )
    
    if ksampler == 'metro':
        customlogger.error('knockoff: metro sampler is not implemented')
        raise ValueError('knockoff: metro sampler is not implemented')    

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
    rejections_series = pd.Series(rejections)
    kfilter_W_series = pd.Series(kfilter.W)
    kfilter_W_series = kfilter_W_series.abs()
    
    df_res = pd.DataFrame()
    df_res = pd.concat([rejections_series, kfilter_W_series, pd.Series(i)], axis=1)
    
    return df_res.values


@ray.remote(num_returns=1)
def __worker__(simmulate, param, index, X, y):
    res = simmulate(param, index, X, y)
    return res




def simulate_knockoffs(itr, df_X, df_y, fdr=None):

    columns = ['rejections', 'W', 'i']
    df_results = pd.DataFrame(columns = columns)
    rejections = []    
    lazy_results = []
    
    X = df_X.to_numpy()
    y = df_y.values.ravel()

    
    param = {}

    if fdr is None:
        param['fdr'] = knockoffsettings.FDR
    else:
        param['fdr'] = fdr


    X_ref = ray.put(X)
    y_ref = ray.put(y)
    


    for ksampler in knockoffsettings.KSAMPLER:
        param['ksampler'] = ksampler
        for fstat in knockoffsettings.FSTATS:
            for i in range(1, itr+1):
                param['fstat'] = fstat                    
                lazy_results.append(__worker__.remote(__simmulate__, param, i, X_ref, y_ref))



    results = ray.get(lazy_results)
    
    df_results = pd.DataFrame(columns=columns)

    for result in results:
        df_res = pd.DataFrame(result, columns=columns)
        df_results = pd.concat([df_results, df_res])
    df_results = df_results

    rejections = df_results.groupby([df_results.index])['rejections'].agg(lambda x: pd.Series.mode(x)[0])
    ws = df_results.groupby([df_results.index])['W'].agg(lambda x: pd.Series.mean(x))
    # print(ws)
    
    df_final = pd.DataFrame()
    df_final = pd.concat([rejections, ws, pd.Series(df_X.columns.values)], axis=1)

    df_final.columns = ['rejections', 'W', 'attr']
    
    del X_ref
    del y_ref
    
    return df_final.sort_values(['W'], ascending=False)
    
