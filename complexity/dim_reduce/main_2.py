from pathlib import Path
import pandas as pd
import datatable as dt
from typing import Union
import warnings
from os import path, remove
from helper_data.helper_data import class_preprocess_data, timit_
from helper_data.global_vars import *
from steps.III_full_data import class_dimreduce_full_data
from steps.II_multiprocessing import multi_optimization
from steps.I_target_dimension import class_target_dim
from asd_logging import logger, empty_existing_logfile
# from dimension_tools.dimension_suite.extra.hyperparameter_analysis import main_analysis_hyperparameters
# from dimension_tools.dimension_suite.extra.plots.plot_hyperparameters import class_plot_hyperparameters
# from dimension_tools.dimension_suite.extra.plots.plot_q_coranking import class_figure_q_plots
# from dimension_tools.dimension_suite.extra.dataframe_summary import class_dataframe_summary

# Delete any existing log file content from previous executions 
empty_existing_logfile()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore')


class class_dim_reduce_pkg:
    '''

    '''

    def __init__(self, params: dict):
        '''
        :param params: dictionary with user defined parameters
        '''
        self.params = params
        self.loss_fun = globalvar_loss_function
        self.data_high = params.get('data_customer')
        self.data_id = params.get('data_id')
        self.target_dim = params.get('target_dim')
        self.dimred_functions = params.get('dimred_functions')
        self.size_reduce = params.get('size_reduce')
        self.cutoff_loss = params.get('cutoff_loss')
        self.params['ndims'] = []
        self.params['step'] = '_'
        logger.info(f"{globalstring_info} + FUNCTIONS: , {self.dimred_functions}")


    # def df_plot_summary(self, results_step: list, new_df: bool =False):
    #     '''
    #     Builds a results dataframe with all important results.
    #     During the build up phase we actualize it and plot after each step.
    #     After deploy we will keep all results in a list and plot and save everything
    #     after the last step.
    #     :param results_step: list of dictionaries with results to be included
    #     :param new_df: if new DataFrame needs to be created (only for first instance)
    #     :return: the dataframe will be build up as self.df_summary
    #     '''
    #     if new_df:
    #         self.df_summary = pd.DataFrame()
    #
    #     # df summary
    #     class_df_summary = class_dataframe_summary(self.params)
    #     self.df_summary =  class_df_summary.make_dataframe(results_step, self.df_summary)
    #
    #     # q_plots and linegraph with losses
    #     q_plot = class_figure_q_plots(self.data_id, self.cutoff_loss)
    #     q_plot.figure_q_plots_(results_step, self.params['step'])



    @timit_(fun_id='main')
    def main(self):
        '''
        main funtion to execute the sequence of data-preprocessing,
        finding the target dimension, finding the target functions and finding
        the intrinic dimension of the full dataset.
        :return: intrinsic dimension
        '''
        class_preprocess = class_preprocess_data()

        '''
        Reduce size and scale data
        '''
        data_small = class_preprocess.reduce_file_size(self.data_high, self.size_reduce)
        data_small_scale, status_preprocess = class_preprocess.preprocess_scaling(data_small)


        '''
        1st step: find best dimension: small data
        '''
        class_s1 = class_target_dim(data_small_scale, data_id=self.data_id, cutoff_loss=self.cutoff_loss)
        results_step1 = class_s1.getter_target_dimension()

        ## INTERNAL USE: SUMMARY, PLOTS
        # append results to summary dataframe
        self.params['step'] = globalstring_step1
        self.params['status_preprocess'] = status_preprocess
        #self.df_plot_summary(results_step1['results'], new_df=True)
        #print(results_step1['results'])


        '''
        2nd step: find best dim reduce functions and hyperparameters: small data
                  keep only the best results
        '''
        self.step = globalstring_step2
        target_dim = results_step1['target_dim']
        results_step2 = multi_optimization(self.dimred_functions,
                                           data_small_scale,
                                           target_dim,
                                           self.cutoff_loss
                                           )

        ## INTERNAL USE:
        # analysis of hyperparameters, plots, all_results: all dim reductions with different hyperparameters
        # main_analysis_hyperparameters(results_step2['all_results'],
        #                               self.cutoff_loss,
        #                               self.data_id
        #                               )

        # LIST OF DICTS OF BEST FUNCTIONS
        lods_best_functions = []
        for dict_obj in results_step2['best_results']:
            logger.info(f"{dict_obj['fun_id']}\n{dict_obj}")
            if dict_obj[self.loss_fun] >= self.cutoff_loss:
                lods_best_functions.append(dict_obj)
        # case something failes in the hp optimization step and nothing is returned we continue with only pca
        if not lods_best_functions:
            lods_best_functions = [results_step1[-1]]


        ## INTERNAL USE: SUMMARY, DF, PLOTS
        # append best results to summary dataframe
        self.params['step'] = globalstring_step2
        #self.df_plot_summary(results_step2['best_results'])


        '''
        Scale full size data
        '''
        # if data_high is small enough it will not reduced, thus data_small == data high
        # and we can save the data reduction and scaling step
        if self.data_high.shape[1] == data_small.shape[1]:
            data_scale = data_small_scale
        else:
            data_scale, status_preprocess = class_preprocess.preprocess_scaling(self.data_high)


        '''
        3rd step: scaling and finetune dimension and find best functions: full dataset
        '''
        class_s3 = class_dimreduce_full_data()
        results_step3 = class_s3.dim_reduction_best_functions(lods_best_functions,
                                                              data_scale,
                                                              target_dim
                                                             )
        ## SUMMARY, PLOTS
        # append results to summary dataframe
        self.params['step'] = globalstring_step3
        self.params['status_preprocess'] = status_preprocess
        #self.df_plot_summary(results_step3['results'])

        return results_step3['intrinsic_dimension'] #, self.df_summary


def intrinsic_dimension(df: pd.DataFrame,
                        data_id: Union[str,None],
                        columns: Union[list,None],
                        cutoff_loss: Union[float,None],
                        functions: Union[list,None] ) -> int:
    '''
    calculates the intrinsic dimension of the dataset.
    :param df: dataframe or np.array with high dimesnional data
    :param data_id: identifier data
    :param columns: columns to include
    :param cutoff_loss: cutoff for loss of dim reduction quality control
    :param functions: list with dim reduction functions to use. default
                    default: 'all_functions' tests all functions
                    if include only specific functions provide them as strings in a list:
                    ['py_pca','r_crda'...]
                    if exclude functions add a '!' upfront, '!py_pca' excludes pca from functions
    :return: intrinsic dimesnion of dataset
    '''
    # if data identifier is provided by customer
    if data_id:
        id = data_id
    else:
        id = 'data'

    # if columns is provided by customer
    if columns:
        cols = columns
    else:
        cols = df.columns

    # if functions is provided by customer
    if functions:
        funs = functions
    else:
        funs = ['all_functions']

    # if cutoff is provided by customer
    if cutoff_loss:
        cutoff = cutoff_loss
    else:
        cutoff = 0.99

    params = {'path_r_errors': globalvar_path_dir_r_errors,
              'data_customer': df[cols],
              'data_id': id,
              'target_dim': 'auto',  # 'auto'
              'dimred_functions': funs, # 'all_functions'
              'size_reduce': 'auto',  # automatic: nrows = <40k ==1; nrows <200k ==5, nrows >200k == 5+nrows^0,25
              'cutoff_loss': cutoff,  # 0.99 only works with > 0.9 otherwise the program will stop, ???
              }

    pkg = class_dim_reduce_pkg(params)
    intrinsic_dimension = pkg.main()
    return intrinsic_dimension

def dataset_covid():
    df_id = 'covid_'
    base_path = Path(__file__).parent
    path = (base_path / "./datasets/20220319_covid_merge.csv").resolve()
    df = dt.fread(path).to_pandas()
    return df, df_id


# covid dataset
data, data_id = dataset_covid() # dataset_covid_merge()
data = data[data.columns[1:]] # for covid_merge


# this is the comand running the function
# run 'all_functions' for all functions unchecked in dimred_call_functions
# id, df_summary = intrinsic_dimension(data, data_id, columns=None, cutoff_loss=0.99, functions=['all_functions'])
id = intrinsic_dimension(data, data_id, columns=None, cutoff_loss=0.99, functions=['py_pca']) #'all_functions'
logger.info(f"+++ This is the intrinsic dimension: {id} +++")
logger.info(f"VOILA! LE NOTRE DIMENSION INTRINISIQUE: {id}")
logger.info(f"\n\n\n++++++ EXECUTION FINISHED ++++++\n\n\n")

# print(df_summary)
# path = globalvar_path_dir_results
# df_summary.to_csv(path + data_id + '_results_dimred.csv', index=False)



# data, target, data_id = dataset_ansurII()

# data, data_id = dataset_spectra(dims=6)

# data, data_id = tokamak_modes(mode_id='h_mode', verbose=1)
# data, data_id = tokamak_modes(mode_id='ohm_mode', verbose=1)
# data, data_id = tokamak_modes(mode_id='ri_mode', verbose=1)

# data, data_id = dataset_spectra(dims=9)
# params = parameters(data, data_id=data_id)
# pkg = class_dim_reduce_pkg(params)
# pkg.main()

# _dict_truth = {
#             # "M1_Sphere": (10, 11, "10D sphere linearly embedded"),
#             # "M2_Affine_3to5": (3, 5, "Affine space"),
#             # "M3_Nonlinear_4to6": (4, 6, "Concentrated figure, mistakable with a 3D one"), #*
#             # "M4_Nonlinear": (4, 8, "Nonlinear manifold"),
#             "M5b_Helix2d": (2, 3, "2D helix"),
#             # "M6_Nonlinear": (6, 36, "Nonlinear manifold"),
#             "M7_Roll": (2, 3, "Swiss Roll"),
#             # "M9_Affine": (20, 20, "Affine space"),
#             # "M10a_Cubic": (10, 11, "10D hypercube"),
#             # "M10b_Cubic": (17, 18, "17D hypercube"),
#             # "M10c_Cubic": (24, 25, "24D hypercube"),
#             # "M10d_Cubic": (70, 71, "70D hypercube"),
#             # "M11_Moebius": (2, 3, "Möebius band 10-times twisted"),
#             # "M12_Norm": (20, 20, "Isotropic multivariate Gaussian"),
#             "M13b_Spiral": (1, 13, "1D helix curve"),
#        }

# for key, item in _dict_truth.items():
#     data = skdim.datasets.BenchmarkManifolds().generate(name=key, dim=item[1], d=item[0])
#     data_id = str(key) + '_95'
#     params = parameters(data, data_id=data_id)
#     pkg = class_dim_reduce_pkg(params)
#     pkg.main()



#     path = globalvar_path_dir_results
#     df_results.to_csv(path + self.data_id + '_results_dimred.csv', index=False)
#     self.hynova.to_csv(path + self.data_id + '_importance_hyperparameters.csv', index=False)
#     self.hp_progress.to_csv(path + self.data_id + '_progress_hyperparameters.csv', index=False)