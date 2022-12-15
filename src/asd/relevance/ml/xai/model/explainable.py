import ray
import numpy as np
import pandas as pd


from torch import nn

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
# from xgboost import XGBClassifier, XGBRegressor
from xgboost.sklearn import XGBRegressor, XGBClassifier

from captum.attr import IntegratedGradients, GradientShap, DeepLift, NoiseTunnel
import shap

from asd.relevance.utils import helper, config
from asd.relevance.utils.asd_logging import logger as  customlogger

list_xai_algos = ['ig', 'shap', 'gradientshap', 'knockoffs']



@ray.remote(num_returns=1)
def __get_gs_attr__(model, df_X, stdevs, n_samples):
    
    gs = GradientShap(model)

    df_X_tensor = helper.df_to_tensor(df_X)                

    # select a set of background examples to take an expectation over
    background = df_X.loc[np.random.choice(df_X.shape[0], df_X.shape[0], replace=False)]
    baseline_dist = helper.df_to_tensor(background)

    gs_attr, delta = gs.attribute(df_X_tensor, stdevs=stdevs, n_samples=n_samples, 
                                                    baselines=baseline_dist, return_convergence_delta=True
                                   )

    gs_attr_df_shapley = pd.DataFrame(gs_attr.numpy(), columns=df_X.columns)
    gs_attr_df_shapley_list = gs_attr_df_shapley.abs().mean().values
    
    return gs_attr_df_shapley_list



@ray.remote(num_returns=1)
def __get_ig_attr__(model, df_X, n_steps):

    # df_X = ray.get(df_X_id)
    # print(df_X)
    df_X_tensor = helper.df_to_tensor(df_X)
    
    ig = IntegratedGradients(model)
    ig_attr = ig.attribute(df_X_tensor, n_steps = n_steps)

    ig_attr_df_shapley = pd.DataFrame(ig_attr.numpy(), columns=df_X.columns)
    ig_attr_df_shapley_list = ig_attr_df_shapley.abs().mean().values #.sort_values(ascending=False).values
    
    return ig_attr_df_shapley_list


@ray.remote(num_returns=1)
def __get_shapley_torch_attr__(model, df_X, n_background):
    
    df_background = df_X.sample(n = n_background)
    df_tensor_background = helper.df_to_tensor(df_background)
    df_X_tensor = helper.df_to_tensor(df_X)                

    explainer_shap = shap.DeepExplainer(model=model, data=df_tensor_background)

    shap_values = explainer_shap.explainer.shap_values(X=df_X_tensor, ranked_outputs=True, check_additivity=False)


    df_shapley_sores = pd.DataFrame(shap_values, columns=df_X.columns)
    df_shapley_sores_list = df_shapley_sores.abs().mean().values #sort_values(ascending=False).values

    return df_shapley_sores_list


@ray.remote(num_returns=1)
def __get_shapley_ensemble_attr__(model, df_X):
    explainer = shap.Explainer(model)
    shap_values = explainer(df_X)
    df_shapley_sores = pd.DataFrame(shap_values.values, columns=df_X.columns)
    df_shapley_sores_list = df_shapley_sores.abs().mean().values #sort_values(ascending=False).values
    return df_shapley_sores_list


@ray.remote(num_returns=1)
def __get_shapley_kernel_attr__(model, df_X,  n_background):
    # df_background = df_X.sample(n = n_background)
    med = df_X.median().values.reshape((1, df_X.shape[1]))
    kernel_explainer = shap.KernelExplainer(model.predict, med)
    kernel_shap_values = kernel_explainer.shap_values(X=df_X)

    # convert 3D into 2D
    if kernel_shap_values.ndim==3:
        kernel_shap_values = [elem for twod in kernel_shap_values for elem in twod]

    df_shapley_sores = pd.DataFrame(kernel_shap_values, columns=df_X.columns)
    df_shapley_sores_list = df_shapley_sores.abs().mean().values #sort_values(ascending=False).values
    return df_shapley_sores_list


@ray.remote(num_returns=1)
def __get_shapley_tree_attr__(model, df_X,  n_background):
    df_background = df_X.sample(n = n_background)
    tree_explainer = shap.TreeExplainer(model.predict, df_background)
    tree_shap_values = tree_explainer.shap_values(X=df_X)# , ranked_outputs=True, check_additivity=False)
    df_shapley_sores = pd.DataFrame(tree_shap_values.values, columns=df_X.columns)
    df_shapley_sores_list = df_shapley_sores.abs().mean().values #sort_values(ascending=False).values
    return df_shapley_sores_list


class Explainable:
    def __init__(self, ensemble_set, df_X) -> None:
        self.df_scores = None
        self.raw = None
        self.ensemble_set = ensemble_set
        self.df_X = df_X
                
        ### IG arguments
        self.ig_n_steps = 50

        ### SHAPLEY arguments
        self.shapley_background_size = 0.3 # 30% of actual dataset
        self.shapley_n_background = int(df_X.shape[0]*self.shapley_background_size)

        ### Gradient Shap
        self.stdevs=0.09
        self.n_samples=1

        # self.load_models()



    def get_attr(self, attr_algos):                
        '''
        Measure variable importance based on SHAP
        attr_algos (list): list of exlainable AI methods, i.e., ['IG', 'SHAP', 'GradientSHAP']. See https://github.com/pytorch/captum and https://github.com/slundberg/shap.
        return pd.Series: list of varaibles sorted as per their importance, zeros equate to no importance at all.
        '''
        
        attr_algos = [x.lower() for x in attr_algos]
        
        col_names = []
        lazy_results = []                    
        
        df_X_id = ray.put(self.df_X)
        
        
        customlogger.info("attribution methods  " + str(attr_algos))
        
        for base_model in self.ensemble_set.base_models:    
            customlogger.info("calculating variable importance on  " + str(base_model.model_file_name))
            

            for attr_algo in attr_algos:                
                if type(base_model.gs.best_estimator) == nn.Sequential:
                    if attr_algo == 'ig':
                        lazy_results.append(__get_ig_attr__.remote(base_model.gs.best_estimator, df_X_id, self.ig_n_steps))
                        col_names.append(base_model.model_file_name + '_' + attr_algo)
                    elif attr_algo == 'shap':          
                        lazy_results.append(__get_shapley_torch_attr__.remote(base_model.gs.best_estimator, df_X_id, self.shapley_n_background))
                        col_names.append(base_model.model_file_name + '_' + attr_algo)
                    elif attr_algo == 'gradientshap':
                        lazy_results.append(__get_gs_attr__.remote(base_model.gs.best_estimator, df_X_id, self.stdevs, self.n_samples))
                        col_names.append(base_model.model_file_name + '_' + attr_algo)
                elif (type(base_model.gs.best_estimator) in [RandomForestRegressor, RandomForestClassifier, LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor ]) & (attr_algo == 'shap'):
                        lazy_results.append(__get_shapley_ensemble_attr__.remote(base_model.gs.best_estimator, df_X_id))
                        col_names.append(base_model.model_file_name + '_' + 'shap')

                elif (type(base_model.gs.best_estimator) in [KNeighborsRegressor, KNeighborsClassifier, BaggingClassifier, BaggingRegressor]) & (attr_algo == 'shap'):
                        lazy_results.append(__get_shapley_kernel_attr__.remote(base_model.gs.best_estimator, df_X_id, self.shapley_n_background))
                        col_names.append(base_model.model_file_name + '_' + 'shap')



        if len(lazy_results)==0:
            return None
        
        results = ray.get(lazy_results)
        
        del df_X_id
                        
        return self.__beautify_scores__(attr_algos, results, col_names)



    def __beautify_scores__(self, attr_algos, results, col_names):                

        df_results = pd.DataFrame(results).T

        if df_results.shape[1] < 1:
            return None
        
        df_results.index = self.df_X.columns
        df_results.columns = col_names 
        df_results['cols'] = df_results.index
                
        self.raw = df_results 
                
        df_scores_ranked = pd.DataFrame()

        for col in df_results:
            if col != 'cols':
                df_scores_ranked[col] = df_results[col].rank(na_option = 'bottom', ascending=True, method='max', pct=False)
                df_scores_ranked.replace(to_replace = df_scores_ranked.min(), value = 0, inplace=True)

        res = df_scores_ranked.mode(axis=1, numeric_only=True).mean(axis=1)
        res.index = df_results['cols']        

        res = res.sort_values(ascending=False)
        
        self.df_scores = res
                
        return res
