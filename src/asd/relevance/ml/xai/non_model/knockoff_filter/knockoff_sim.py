import logging

import numpy as np
import pandas as pd
import ray
from knockpy.knockoff_filter import KnockoffFilter
from utils_logger import LoggerSetup

from .config import knockoffsettings

# Initialize logging object (Singleton class) if not already
LoggerSetup()

list_fstats = ["lasso", "ridge", "randomforest"]


def __simmulate__(fdr, fstat, itr, X, y):
    np.random.seed(knockoffsettings.SEED)

    Sigma = None

    print("itr: " + str(itr) + ", ksampler: 'gaussian',  fstat:" + str(fstat))

    kfilter = KnockoffFilter(
        ksampler="gaussian",
        fstat=fstat,
    )

    rejections = kfilter.forward(X=X, y=y, Sigma=Sigma, fdr=fdr)

    i = [itr] * len(rejections)
    rejections_series = pd.Series(rejections)
    kfilter_W_series = pd.Series(kfilter.W)
    kfilter_W_series = kfilter_W_series.abs()

    df_res = pd.DataFrame()
    df_res = pd.concat([rejections_series, kfilter_W_series, pd.Series(i)], axis=1)

    return df_res.values


@ray.remote(num_returns=1)
def __worker__(fdr, fstat, itr, X_ref, y_ref):
    res = __simmulate__(fdr, fstat, itr, X_ref, y_ref)
    return res


def simulate_knockoffs(fdr, fstats, itr, df_X, df_y):
    """
    Knockoff frame is the procedure to perform variable selection while controling false discovery rate (fdr). This is to estimate and eliminate
    bad variables. See the details here:https://candes.su.domains/publications/downloads/MX_Knockoffs.pdf
    :param fdr (int): False Discovery Rate.
    :param fstats (list): methods to calculate fstats, i.e., ['lasso', 'ridge', 'randomforest']
    :param df_x (pd.DataFrame): input X
    :param df_x (pd.DataFrame): target column
    :return pd.Series: list of varaibles sorted as per their importance, zeros equate to no importance at all.
    """
    columns = ["rejections", "W", "i"]
    df_results = pd.DataFrame(columns=columns)
    rejections = []
    lazy_results = []

    X = df_X.to_numpy()
    y = df_y.values.ravel()

    X_ref = ray.put(X)
    y_ref = ray.put(y)

    for fstat in fstats:
        for i in range(1, itr + 1):
            lazy_results.append(__worker__.remote(fdr, fstat, i, X_ref, y_ref))

    results = ray.get(lazy_results)

    df_results = pd.DataFrame(columns=columns)

    for result in results:
        df_res = pd.DataFrame(result, columns=columns)
        df_results = pd.concat([df_results, df_res])
    df_results = df_results

    rejections = df_results.groupby([df_results.index])["rejections"].agg(lambda x: pd.Series.mode(x)[0])
    ws = df_results.groupby([df_results.index])["W"].agg(lambda x: pd.Series.mean(x))
    # print(ws)

    df_final = pd.DataFrame()
    df_final = pd.concat([rejections, ws, pd.Series(df_X.columns.values)], axis=1)

    df_final.columns = ["rejections", "W", "cols"]

    df_final["W"] = np.where(df_final["rejections"] == 1, 0, df_final["W"])

    df_final = df_final[["W", "cols"]]

    df_scores_ranked = pd.DataFrame()
    for col in df_final:
        if col != "cols":
            df_scores_ranked[col] = df_final[col].rank(na_option="bottom", ascending=True, method="max", pct=False)
            df_scores_ranked.replace(to_replace=df_scores_ranked.min(), value=0, inplace=True)

    res = df_scores_ranked.mode(axis=1, numeric_only=True).mean(axis=1)
    res.index = df_final["cols"]

    res = res.sort_values(ascending=False)

    del X_ref
    del y_ref

    return res
