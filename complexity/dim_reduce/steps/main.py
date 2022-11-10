from .III_full_data import Full_data
from .II_functions_multiprocess import multi_optimization
from .I_target_dimension import Target_dimension
from helper_data.utils_data import Preprocess_data, timit_
from helper_data.global_vars import *
from utils_logger import logger
import pandas as pd
from operator import itemgetter



class Dimension_reduce_main:
    '''
    main class for dimensionality reduction.
    Contains the main function whch runs a sequence of subfubnctions.
    Also contains functions for analysis of the steps (extra package)
    '''

    def __init__(self, params: dict, docu=False):
        '''
        :param params: dictionary with user defined parameters and data
        '''
        self.params = params
        self.loss_fun = globalvar_loss_function
        self.data_high = params.get('data_high')
        self.data_id = params.get('data_id')
        self.target_dim = params.get('target_dim')
        self.dimred_functions = params.get('dimred_functions')
        self.size_reduce = params.get('size_reduce')
        self.cutoff_loss = params.get('cutoff_loss')
        self.params['ndims'] = []
        self.params['step'] = '_'
        self.df_summary = pd.DataFrame()
        self.docu = docu
        logger.info(msg=('FUNCTIONS: ', self.dimred_functions))


    def best_functions_step2(self,
                             results_step1: list,
                             results_step2: list
                             ) -> list:
        '''
        find fuctions with best dimensionality reuction results on our small dataset.
        In order to reduce the computational time we coose the n: best and fastest functions.
        In case there are no functions with the requiered quality, we use the best PCA
        result from step 1. In theory this should be the same results for pca in step 2.

        :param list results_step1: list of dictionaries with results from step 1
            the dictionary contains all results and data corresponding to the dim reduction
            with pca and the target dimension.
        :param list results_step2: list of dictionaries with results from step 2.
            each dictionary contains all results and data corresponding to the best result
            of one dimred fucnction of the hyperparameter optimization step.
        :return: list of dictionaries with best results
            each dictionary contains all results and data corresponding to the best result
            of one dimred fucnction of the hyperparameter optimization step ONLY in case
            the quality is > cutoff_loss.
        '''

        n_funs = min(5, len(results_step2))
        try:
            tmp = [dic for dic in results_step2 if dic['step'] == globalstring_step2]
            lods_best_functions = sorted(tmp, key=lambda x: (-x[self.loss_fun], x['time']))[:n_funs]
        except:
            # something failes in the hyperparameter optimization step and nothing is returned. we
            # continue with only with pca which performed well at the target dimension in step 1.
            logger.error(msg='', exc_info=True)
            lods_best_functions = [results_step1[-1]]

        # TODO: remove
        # lods_best_functions = []
        # for dict_result in results_step2:
        #     try:
        #         # if dict_result[self.loss_fun] >= self.cutoff_loss:
        #         #     lods_best_functions.append(dict_result)
        #     except:
        #         # there is something weird with the dictionary, erase it and avoid problems
        #         # during plotting and data summmary.
        #         results_step2.remove(dict_result)
        #         logger.error(msg='', exc_info=True)
        #         continue
        # something failes in the hyperparameter optimization step and nothing is returned. we
        # continue with only with pca which performed well at the target dimension in step 1.

        if not lods_best_functions:
            lods_best_functions = [results_step1[-1]]
        return lods_best_functions


    @timit_(fun_id='main ')
    def main(self) -> (int, dict, pd.DataFrame):
        '''
        main funtion to find the intrinsic dimensionality of a dataset (full dataset).
        Executes the the following steps:
        1) data-preprocessing: scaling and make a small dataset with reduced number of rows.
           the following steps carry out many hundret dimensionality reductions, we create this
           dataset to to speed up the process.
        2) target dimension: calculates the target dimension with PCA (very fast and very good)
           with small data. This dimension is an estimate of what might be the intrinsic dimension
           of the full dataset. Its used as dimension for step 3 and starting point for step 4.
        3) hyperprameter tuning: optimize hyperparametrs of R and Python dim reduction
           functions with the small dataset and the target dimension.
        4) intrinsic dimension full data set: reduce dimension of full dataset with target dim and
           the best functions (with optimized hyperparameters) from step 3.
           this step includes a finetuning of the target dimension.
        We acess the quality of each dim reduction with the mae_norm of the coranking matrix Q.
        The quality is suficient if mae_nor >= cutoff_loss we find that 0.99 is a good cutoff.

        :return: int, intrinsic dimension
        '''
        # TODO ONLY INTERNAL: SUMMARY DF, PLOTS
        ###############################################################################
        '''
        Only if documentation wanted. Its a bit ugly but does the job.
        ! make sure external packages are availbale
        '''
        if self.docu:
            from analysis.hyperparameter_analysis import main_analysis_hyperparameters
            from dimension_tools.dimension_suite.extra.plots.plot_q_coranking import Figure_q_plots
            from analysis.dataframe_results_dimred import Dataframe_results_dimred
            def df_plot_summary(results_step: list) -> pd.DataFrame(): # , new_df: bool = False
                '''
                Builds a results dataframe and plotting of data.
                The dataframe is actualized and plots are saved after each step.
                After deploy we will keep all results in a list and plot and save everything
                after the last step.
                :param list results_step: list of dictionaries with results to be included
                :param bool new_df: if new DataFrame needs to be created (only for first instance)
                :return: pd.Dataframe() the dataframe will be build up as self.df_summary
                '''
                Summary = Dataframe_results_dimred(self.params)
                self.df_summary = Summary.make_dataframe(results_step, self.df_summary)
                # q_plots and linegraph with losses
                q_plot = Figure_q_plots(self.data_id, self.cutoff_loss)
                q_plot.figure_q_plots_(results_step, self.params['step'])



        ###############################################################################
        '''
        scale data
        '''
        Preprocess = Preprocess_data()
        data_scale, status_preprocess = Preprocess.preprocess_scaling(self.data_high)

        '''
        reduce data size
        '''
        data_small = Preprocess.reduce_file_size(data_scale, self.size_reduce)
        self.params['nrows'] = data_small.shape[0]
        self.params['ncols'] = data_small.shape[1]
        self.params['status_preprocess'] = status_preprocess

        '''
        1st: target dimension with small data
        '''
        self.params['step'] = globalstring_step1
        Target_dim = Target_dimension(
            data_high=data_scale,
            data_high_small=data_small,
            data_id=self.data_id,
            cutoff_loss=self.cutoff_loss
        )
        results_step1 = Target_dim.getter_target_dimension(self.target_dim)

        ## TODO ONLY INTERNAL: SUMMARY DF, PLOTS
        if self.docu:
            df_plot_summary(results_step1['results']) # , new_df=True)
            logger.info(msg=('1st step \n' +  str(self.df_summary)))

        '''
        2nd: best dim reduce functions and hyperparameters with small data
        '''
        self.step = globalstring_step2
        target_dim = results_step1['target_dim']

        results_step2 = multi_optimization(
            functions = self.dimred_functions,
            data_high = data_small,
            dim_low = target_dim,
            cutoff_loss = self.cutoff_loss
        )

        # LIST OF DICTS OF BEST FUNCTIONS
        lods_best_functions = self.best_functions_step2(
            results_step1=results_step1['results'],
            results_step2=results_step2['best_results']
        )

        ## TODO ONLY INTERNAL: SUMMARY DF, PLOTS SUMMARY, HYPERPARAMETER ANALYSIS
        if self.docu:
            self.params['step'] = globalstring_step2
            # analysis of hp tuning steps
            main_analysis_hyperparameters(
                params = self.params,
                lods_results = results_step2['all_results'],
                cutoff_loss = self.cutoff_loss,
                data_id = self.data_id
            )
            df_plot_summary(results_step2['best_results'])
            logger.info(msg=('2nd step \n' + str(self.df_summary)))

        '''
        3rd: intrinsic dimension full dataset
        '''
        logger.info('full data')
        results_step3 = Full_data(cutoff_loss=self.cutoff_loss).dim_reduction_best_functions(
            lods_best_functions=lods_best_functions,
            data_high=data_scale,
            dim_low=target_dim
        )

        # intrinsic dimension
        intrinsic_dim = results_step3['intrinsic_dimension']
        best_results_final = results_step3['best_results']

        ## TODO plotting takes too long? check and modify
        ## TODO ONLY INTERNAL: SUMMARY DF, PLOTS
        if self.docu:
            self.params['step'] = globalstring_step3
            self.params['nrows'] = data_scale.shape[0]
            self.params['ncols'] = data_scale.shape[1]
            df_plot_summary(results_step3['results'])

        return intrinsic_dim, best_results_final, self.df_summary
