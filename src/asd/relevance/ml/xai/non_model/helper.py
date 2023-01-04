import pandas as pd
import numpy as np


def consolidate_rejection_results(results, columns):
    df_results = pd.DataFrame(columns = columns)

    for result in results:
        df_res = pd.DataFrame(result, columns = columns)
        df_results = pd.concat([df_results, df_res])


    df_results = df_results
    
    rejections = df_results.groupby([df_results.index])['rejections'].agg(lambda x: pd.Series.mode(x)[0])

    return rejections

