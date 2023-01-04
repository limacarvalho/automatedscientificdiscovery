from .III_full_data import dimreduce_full_data
from .II_functions_multiprocess import ray_get_optimization
from .I_target_dimension import Target_dimension
from helper_data.utils_data import Preprocess_data, timit_
from helper_data.global_vars import *
from utils_logger import logger
import pandas as pd



class Dimension_reduce_main:
    '''
    main class for dimensionality reduction.
    Also contains functions for analysis and documentation (extra package, not provided here)
    '''

    def __init__(self, params: dict, docu=False):
        '''
        :param params: dictionary with user/default defined parameters and data
        '''
        self.params = params
        self.loss_fun = globalvar_loss_function
        self.data_high = params.get('data_high')
        self.data_id = params.get('data_id')
        self.dimred_functions = params.get('dimred_functions')
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
        filter for functions/hyperparameters with best dimensionality reduction results on reduced dataset.
        In order to reduce the computational time we coose the n: best and fastest functions.
        In case there are no functions with the requiered quality, we use the best result from step 1.

        :param list results_step1: list,
            list of dictionaries with results from each step of target dimension search step
            the dictionary contains all results and data corresponding to the dim reduction
            with pca and the target dimension.
        :param list results_step2: list,
            list of dictionaries with best results for each function (step 2).
            each dictionary contains all results and data corresponding to the best result
            of one dimred fucnction of the hyperparameter optimization step.
        :return: list
            list of dictionaries with best results for each function (step 1 or 2).
            each dictionary contains all results and data corresponding to the best result
            of one dimred fucnction of the hyperparameter optimization step ONLY in case
            the quality is > cutoff_loss.
        '''
        lods_best_functions = []
        for dict_result in results_step2:
            try:
                if dict_result[self.loss_fun] >= self.cutoff_loss:
                    lods_best_functions.append(dict_result)
            except:
                # there is something weird with the dictionary, erase it and avoid problems in next steps
                results_step2.remove(dict_result)
                logger.error(msg='', exc_info=True)
                continue

        # something failes in the hyperparameter optimization step and nothing is returned. we
        # continue with only with pca which performed well at the target dimension in step 1.
        if not lods_best_functions:
            lods_best_functions = [results_step1[-1]]
        return lods_best_functions


    @timit_(fun_id='main ')
    def main(self) -> (int, dict, pd.DataFrame):
        '''
        main function running all steps to find the intrinsic dimensionality of the dataset
        Executes the the following steps:
        1) data-preprocessing:
            scaling and generating a reduced size dataset with smaller number of rows (speed).
        2) target dimension:
            calculates the target dimension with PCA (very fast and very good quality) with the reduced data.
            Its used as dimension for step 3 and starting point for step 4.
        3) hyperprameter tuning:
            optimize hyperparametrs of R and Python dim reduction functions (reduced dataset, target dimension).
        4) intrinsic dimension full data set:
            reduce dimension of full dataset starting at target dim (1) and the optimized functions (3).
            The target dimension will be finetuned in this step.

        :return: int, int, list, pd.DataFrame
            int:
                intrinsic dimension of the dataset
            int:
                number of columns after scaling
                might be reduced due to NaN/Inf columns that were filtered out
            list:
                list of dictionaries with best results form the final step (best functions, full data).
                'Q': Q,                 # np.array: coranking matrix
                'rel_err': rel_err,     # float: relative error 0...1 (1=perfect dimensionality reduction)
                'r2': r2                # float: r-squared value
                'step': step            # str: step of the dim reduction process
                'fun_id': fun_id        # str: function identifier, example: 'py_pca'
                'time': seconds         # float: time for dimensionality reduction in seconds
                'rows_cols': 'rows_cols' # str: 'nrows_ncolumns' of high dimensional dataset
                'dim_low': dim_low       # int: dimension (ncolumns)of low dimensional dataset
                'hyperparameters': 'hyperparameters'   # str: string of hyperparameters 'hyperparameter=value'
                'data_lowdim': data_lowdim   # np.array: low dimensional data
            pd.DataFrame:
                documentation of the dim reduction process. empty if docu=False
        '''


        '''
        ONLY FOR DEVELOPER TEAM: SUMMARY DF, PLOTS
        ! make sure external packages are availbale
        '''
        if self.docu:
            from analysis.hyperparameter_analysis import main_analysis_hyperparameters
            from dimension_tools.dimension_suite.extra.plots.plot_q_coranking import Figure_q_plots
            from analysis.dataframe_results_dimred import Dataframe_results_dimred
            #
            def df_plot_summary(results_step: list) -> pd.DataFrame():
                '''
                Main function for documentation (plots, results dataframes) of each step of the process.
                The dataframe is actualized and plots are saved after each step.
                TODO: After deploy we will keep all results in a list and plot and save everything after the last step.
                :param list results_step:
                    list of dictionaries with results which will be documented
                :return: pd.Dataframe,
                    the dataframe will be updated in each step, new results will be addaed as rows
                '''
                Summary = Dataframe_results_dimred(self.params)
                self.df_summary = Summary.update_dataframe(results_step, self.df_summary)
                # q_plots and linegraph with losses
                q_plot = Figure_q_plots(self.data_id, self.cutoff_loss)
                q_plot.figure_q_plots_(results_step, self.params['step'])

        ###############################################################################
        logger.info(msg='#############  '+self.data_id+'  ##############')
        '''
        standart scale data
        '''
        Preprocess = Preprocess_data()
        data_scale, status_preprocess = Preprocess.preprocess_scaling(self.data_high)


        '''
        reduce data size (for faster target dimension search and hyperparameter optimization)
        '''
        data_small = Preprocess.reduce_file_size(data_scale)
        self.params['nrows'] = data_small.shape[0]
        self.params['ncols'] = data_small.shape[1]
        self.params['status_preprocess'] = status_preprocess


        '''
        1st: target dimension with reduced dataset or provided by the customer
        '''
        self.params['step'] = globalstring_step1
        Target_dim = Target_dimension(
            data_high=data_scale,
            data_high_small=data_small,
            data_id=self.data_id,
            cutoff_loss=self.cutoff_loss
        )

        # calculates target dimension or adjusts loss cutoff in case target dimension is provided by the customer
        results_step1, self.cutoff_loss = Target_dim.main_target_dimension()
        target_dim = results_step1['target_dim']
        if self.docu:
            df_plot_summary(results_step1['results'])
            logger.info(msg=('1st step \n' +  str(self.df_summary)))


        '''
        2nd: best dim reduce functions and hyperparameters with small data
        '''
        self.step = globalstring_step2
        results_step2 = ray_get_optimization(
            functions = self.dimred_functions,
            data_high = data_small,
            dim_low = target_dim,
            cutoff_loss = self.cutoff_loss
        )

        # LIST OF DICTS OF BEST FUNCTIONS
        lods_best_functions = self.best_functions_step2(results_step1['results'], results_step2['best_results'])

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
        results_step3 = dimreduce_full_data(cutoff_loss=self.cutoff_loss).intrinsic_dimension_full_data(
            lods_best_functions=lods_best_functions,
            data_high=data_scale,
            dim_low=target_dim
        )

        # intrinsic dimension
        intrinsic_dim = results_step3['intrinsic_dimension']
        best_results_final = results_step3['best_results']

        if self.docu:
            self.params['step'] = globalstring_step3
            self.params['nrows'] = data_scale.shape[0]
            self.params['ncols'] = data_scale.shape[1]
            df_plot_summary(results_step3['results'])

        return intrinsic_dim, data_scale.shape[1], best_results_final, self.df_summary
