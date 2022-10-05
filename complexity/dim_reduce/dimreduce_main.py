from pathlib import Path
import pandas as pd
import datatable as dt
from typing import Union
import traceback
import numpy as np
from helper_data.global_vars import *
from dimension_tools.dimension_suite.dim_reduce.steps.main import Dimension_reduce_main

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import warnings
warnings.filterwarnings('ignore')


def intrinsic_dimension(
        data_high: Union[pd.DataFrame(), np.array],
        data_id: Union[str, None],
        columns: Union[list, None],
        cutoff_loss: Union[float, None],
        functions: Union[list, None]
        ) -> (int, list, pd.DataFrame()):
    '''
    calculates the intrinsic dimension of the dataset.
    custom data are provided in a dictionary (params)
    :param Union data_high: dataframe or np.array with high dimesnional data
    :param Union[str, None] data_id: identifier data
    :param Union[list, None] columns: columns to include
    :param Union[float, None] cutoff_loss: cutoff for loss of dim reduction quality control
        0.99 only works with > 0.9 otherwise the program will stop, ???
    :param Union[list, None] functions: list with dim reduction functions to use.
        if include only specific functions provide them as strings in a list: ['py_pca','r_crda'...]
        if you want to exclude functions add a '!' upfront, '!py_pca' excludes pca from functions.
        default: 'all_functions' uses all functions in our portfolio on a small dataset
        Note: function and available functions names check dim_reduction -> dimred_call_functions.py

    other parameters: specified in params dictionary below
        Union[int, str] target_dim: target dimension for optimization of dimred function hyperparameters
            default: 'auto' calculates the target dim on a dataset with less rows.

        Union [int, str] 'size_reduce':
            default: 'auto' or percent of rows used for calcualating target dim and hyperparameter optimization.
            if 100: use full dataset

    :return: intrinsic dimesnion of dataset
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
      'path_r_errors': globalvar_path_dir_r_errors,
      'data_high': data_high,
      'data_id': data_id,
      'dimred_functions': functions,
      'cutoff_loss': cutoff_loss,
      'target_dim': 'auto',
      'size_reduce': 'auto',
    }

    pkg = Dimension_reduce_main(params)
    intrinsic_dimension, best_results, df_summary = pkg.main()
    return intrinsic_dimension, best_results, df_summary





from dimension_tools.datasets import tokamak_modes, dataset_ansurII, dataset_spectra

data, target, data_id = dataset_ansurII(sexo='female')
data = data.select_dtypes(exclude=['object'])
intrinsic_dim, best_results, df_summary = intrinsic_dimension(
    data_high=data,
    data_id=data_id,
    columns=None,
    cutoff_loss=0.99,
    functions=['all_functions']
)
print('VOILA! LE NOTRE DIMENSION INTRINSIQUE: ', intrinsic_dim)
print(df_summary)
path = globalvar_path_dir_results
df_summary.to_csv(path + data_id + '_results_dimred.csv', index=False)

for i in best_results:
    data_low = pd.DataFrame(i['data_lowdim'])
    fun_id = i['fun_id']
    dim = i['dim_low']
    loss = i[globalvar_loss_function]
    data_low.to_csv(
        path + '/data_dimlow/'
        + data_id + '_'
        + fun_id + '_'
        + str(dim) + '_'
        + str(loss) + '.csv',
        index=False
        )


# TODO: hyperparameter optimization optimization, early stop, redue hps, reduce range and reduce steps
# TODO: logging with a common logging style and save to a common place
# TODO: docstrings with a common format
# TODO: remove prints


#
def dataset_covid():
    df_id = 'covid_'
    base_path = Path(__file__).parent
    path = (base_path / "./datasets/20220319_covid_merge.csv").resolve()
    df = dt.fread(path).to_pandas()
    return df, df_id

################ Datasets
## covid
# data, data_id = dataset_covid() # dataset_covid_merge()
# data = data[data.columns[1:]] # for covid_merge
## ansur
# data, target, data_id = dataset_ansurII(sexo='male')
## spectra
# data, data_id = dataset_spectra(dims=6)
## tokamak
# data, data_id = tokamak_modes(mode_id='h_mode', verbose=1)
# data, data_id = tokamak_modes(mode_id='ohm_mode', verbose=1)
# data, data_id = tokamak_modes(mode_id='ri_mode', verbose=1)
# data = data.select_dtypes(exclude=['object'])

## this is the comand running the function
## run 'all_functions' for all functions unchecked in dimred_call_functions
# id, df_summary = intrinsic_dimension(
#                     data_high=data,
#                     data_id=data_id,
#                     columns=None,
#                     cutoff_loss=0.99,
#                     functions=['all_functions']
#                     )
# print('VOILA! LE NOTRE DIMENSION INTRINISIQUE: ', id)
# print(df_summary)
# path = globalvar_path_dir_results
# df_summary.to_csv(path + data_id + '_results_dimred.csv', index=False)

### Loop through datasets
# for fun in [tokamak_modes(mode_id='ohm_mode', verbose=1),
#             tokamak_modes(mode_id='ri_mode', verbose=1),
#             dataset_spectra(dims=6)
#             ]:
#     data, data_id = fun
#     data = data.select_dtypes(exclude=['object'])
#
#     id, df_summary = intrinsic_dimension(
#         data_high=data,
#         data_id=data_id,
#         columns=None,
#         cutoff_loss=0.99,
#         functions=['all_functions']
#     )
#     print('VOILA! LE NOTRE DIMENSION INTRINISIQUE: ', id)
#     print(df_summary)
#     path = globalvar_path_dir_results
#     df_summary.to_csv(path + data_id + '_results_dimred.csv', index=False)

### Benchmark datasets
# _dict_truth = {
    # "M1_Sphere": (10, 11, "10D sphere linearly embedded"),
    # "M2_Affine_3to5": (3, 5, "Affine space"),
    # "M3_Nonlinear_4to6": (4, 6, "Concentrated figure, mistakable with a 3D one"), #*
    # "M4_Nonlinear": (4, 8, "Nonlinear manifold"),
    # "M5b_Helix2d": (2, 3, "2D helix"),
    # "M6_Nonlinear": (6, 36, "Nonlinear manifold"),
    # "M7_Roll": (2, 3, "Swiss Roll"),
    # "M9_Affine": (20, 20, "Affine space"),
    # "M10a_Cubic": (10, 11, "10D hypercube"),
    # "M10b_Cubic": (17, 18, "17D hypercube"),
    # "M10c_Cubic": (24, 25, "24D hypercube"),
    # "M10d_Cubic": (70, 71, "70D hypercube"),
    # "M11_Moebius": (2, 3, "Möebius band 10-times twisted"),
    # "M12_Norm": (20, 20, "Isotropic multivariate Gaussian"),
    #"M13b_Spiral": (1, 13, "1D helix curve"),
#}


# for key, item in _dict_truth.items():
#     data = skdim.datasets.BenchmarkManifolds().generate(name=key, dim=item[1], d=item[0])
#     data_id = str(key) + '_95'
#     params = parameters(data, data_id=data_id)
#     pkg = Dimension_reduce_main(params)
#     pkg.main()
#     path = globalvar_path_dir_results
#     df_results.to_csv(path + self.data_id + '_results_dimred.csv', index=False)
#     self.hynova.to_csv(path + self.data_id + '_importance_hyperparameters.csv', index=False)
#     self.hp_progress.to_csv(path + self.data_id + '_progress_hyperparameters.csv', index=False)