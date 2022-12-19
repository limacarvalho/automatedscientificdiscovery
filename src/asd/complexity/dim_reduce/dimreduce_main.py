import pandas as pd
from typing import Union
import numpy as np
import random
from utils_logger import logger, empty_existing_logfile
# create new logfile
empty_existing_logfile()
from steps.main_steps import Dimension_reduce_main


def complexity(
        data_high: Union[pd.DataFrame, np.array],
        data_id: Union[str, None],
        columns: Union[list, None],
        cutoff_loss: Union[float, None],
        functions: Union[list, None],
        docu: bool=False # TODO: remove
    ) -> (int, list, pd.DataFrame):
    '''
    main function for dimensionality reduction.
    :param docu: bool
        asd developer team only, if steps are documented or not
    :param data_high: Union[pd.DataFrame, np.array],
        high dimensional dataset
    :param data_id: Union[str, None],
        identifier for dataset, default: random generated 6 char string
    :param columns: Union[list, None],
        customer can provide names of columns to be used, default: all columns will be used
    :param cutoff_loss: Union[float, None],
        cutoff for loss function (dimensionality reduction quality).
        0...1  with 1 beeing perfect dim reduction. default: 0.99
        more details: helper_metrix -> metrics_dimred ->
    :param functions: Union[list, None],
        list with dim reduction functions to be used.
        if include only specific functions provide them as strings in a list: ['py_pca','r_crda'...]
        if you want to exclude functions add a '!' upfront, '!py_pca' excludes pca from functions.
        default: 'all_functions' uses all functions in our portfolio.
        Note: function and available functions names check dim_reduction -> dimred_main.py

    :return: int, list, pd.DataFrame
        int:
            intrinsic dimension of the dataset
        list:
            list of dictionaries with best results form the final step (best functions, full data).
            'Q': Q,                 # np.array: coranking matrix
            'rel_err': rel_err,     # float: relative error 0...1 (1=perfect dimensionality reduction)
            'r2': r2                # float: r-squared value
            'step': step            # str: step of the dim reduction process
            'fun_id': fun_id        # str: function identifier, example: 'py_pca'
            'time': seconds         # float: time for dimensionality reduction in seconds
            'rows_cols': 'rows_cols' # str: 'nrows_ncolumns' of high dimensional dataset
            'dim_low': dim_low      # int: dimension (ncolumns)of low dimensional dataset
            'hyperparameters': 'hyperparameters'   # str: string of hyperparameters 'hyperparameter=value'
            'data_lowdim': data_lowdim   # np.array: low dimensional data
        pd.DataFrame:
            documentation of the dim reduction process. empty if docu=False

    '''
    # create data_id: random 6 char string, if data identifier is not provided by customer
    if not data_id:
        data_id = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(6))

    # if column names are provided by customer
    if columns and isinstance(data_high, pd.DataFrame):
        data_high = data_high[columns]
    elif not columns and isinstance(data_high, pd.DataFrame):
        data_high = data_high[data_high.columns]
    else:
        data_high = data_high

    # if no functions are provided by customer, use all functions
    if not functions:
        functions = ['all_functions']

    # if loss cutoff is provided by customer, 0.99 usually gives very good quality
    if not cutoff_loss:
        cutoff_loss = 0.99

    # dictionary with data and custom parameters
    params = {
      'data_high': data_high,
      'data_id': data_id,
      'dimred_functions': functions,
      'cutoff_loss': cutoff_loss
    }

    Main = Dimension_reduce_main(params, docu=docu)
    intrinsic_dimension, ncols, best_results, df_summary = Main.main()

    logger.info(msg=(data_id + ' shape: '+str(data_high.shape)+' n numeric columns: '+str(ncols))
                     +' INTRINSIC DIMENSION: '+str(intrinsic_dimension))
    logger.info(msg=('SUMMARY \n', df_summary))

    return intrinsic_dimension, best_results, df_summary
