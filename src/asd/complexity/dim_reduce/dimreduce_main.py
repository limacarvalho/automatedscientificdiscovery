import pandas as pd
from typing import Union
import numpy as np
from utils_logger import logger, empty_existing_logfile
# create new logfile
empty_existing_logfile()
from steps.main import Dimension_reduce_main


def intrinsic_dimension(
        data_high: Union[pd.DataFrame, np.array],
        data_id: Union[str, None],
        columns: Union[list, None],
        cutoff_loss: Union[float, None],
        functions: Union[list, None],
        docu: bool=False # TODO: remove
    ) -> (int, list, pd.DataFrame):
    '''
    main function for dim reduction.
    :param docu:
    :param data_high: Union[pd.DataFrame, np.array], high dimesnional data
    :param data_id: Union[str, None], identifier data
    :param columns: Union[list, None], columns to include
    :param cutoff_loss: Union[float, None], cutoff for loss of dim reduction quality control
        0.99 only works with > 0.9 otherwise the program will stop, ???
    :param functions: Union[list, None], list with dim reduction functions to use.
        if include only specific functions provide them as strings in a list: ['py_pca','r_crda'...]
        if you want to exclude functions add a '!' upfront, '!py_pca' excludes pca from functions.
        default: 'all_functions' uses all functions in our portfolio on a small dataset
        Note: function and available functions names check dim_reduction -> dimred_call_functions.py

    other parameters: specified in params dictionary below
        Union[int, str] target_dim: target dimension for optimization of dimred function hyperparameters
            default: 'auto' calculates the target dim on a dataset with less rows.

        'size_reduce': Union [int, str],
            default: 'auto' or percent of rows used for calcualating target dim and hyperparameter optimization.
            if 100: use full dataset

    :return: int, intrinsic dimesnion of dataset
    '''
    # if data identifier is provided by customer
    if not data_id:
        data_id = 'data'

    # if columns is provided by customer
    if columns and isinstance(data_high, pd.DataFrame):
        data_high = data_high[columns]
    elif not columns and isinstance(data_high, pd.DataFrame):
        data_high = data_high[data_high.columns]
    else:
        data_high = data_high

    # if functions is provided by customer
    if not functions:
        functions = ['all_functions']

    # if cutoff is provided by customer
    # 0.99 results in a 1 percent error wehn tested on benchmark datasets.
    # on predictions (rf, catboost) we saw both +0.04 increase or -0.01 in accuracy
    if not cutoff_loss:
        cutoff_loss = 0.99

    params = {
      'data_high': data_high,
      'data_id': data_id,
      'dimred_functions': functions,
      'cutoff_loss': cutoff_loss,
      'target_dim': 'auto',
      'size_reduce': 'auto',
    }

    Main = Dimension_reduce_main(params, docu=docu) # TODO: remove the docu
    intrinsic_dimension, best_results, df_summary = Main.main()

    logger.info(msg=('VOILA! LE NOTRE DIMENSION INTRINSIQUE: ', str(intrinsic_dimension)))
    logger.info(msg=('SUMMARY \n', df_summary))

    return intrinsic_dimension, best_results, df_summary
