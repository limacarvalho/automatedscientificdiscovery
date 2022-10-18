
import numpy as np
import pandas as pd
import os 
import glob
import logging
import pickle

from utils import dasker, helper, config

from torch import nn
import dask

from captum.attr import IntegratedGradients, GradientShap, DeepLift, NoiseTunnel
import shap
import xgboost as xgb



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



    
def __get_ig_attr__(model, df_X, n_steps):

    df_X_tensor = helper.df_to_tensor(df_X)                
    
    ig = IntegratedGradients(model)
    ig_attr = ig.attribute(df_X_tensor, n_steps = n_steps)

    ig_attr_df_shapley = pd.DataFrame(ig_attr.numpy(), columns=df_X.columns)
    ig_attr_df_shapley_list = ig_attr_df_shapley.abs().mean().values #.sort_values(ascending=False).values
    
    return ig_attr_df_shapley_list


def __get_shapley_torch_attr__(model, df_X, n_background):
    
    df_background = df_X.sample(n = n_background)
    df_tensor_background = helper.df_to_tensor(df_background)
    df_X_tensor = helper.df_to_tensor(df_X)                

    explainer_shap = shap.DeepExplainer(model=model, data=df_tensor_background)

    shap_values = explainer_shap.explainer.shap_values(X=df_X_tensor, ranked_outputs=True, check_additivity=False)


    df_shapley_sores = pd.DataFrame(shap_values, columns=df_X.columns)
    df_shapley_sores_list = df_shapley_sores.abs().mean().values #sort_values(ascending=False).values

    return df_shapley_sores_list



def __get_shapley_ensemble_attr__(model, df_X):
    explainer = shap.Explainer(model)
    shap_values = explainer(df_X)
    df_shapley_sores = pd.DataFrame(shap_values.values, columns=df_X.columns)
    df_shapley_sores_list = df_shapley_sores.abs().mean().values #sort_values(ascending=False).values
    return df_shapley_sores_list




class Explainable:
    def __init__(self, df_X, df_y, client) -> None:

        if client is None:
            print("client object is needed")
            logging.exception("client object is needed")
            os._exit()


        self.cluster = client
        self.df_X = df_X
        self.df_y = df_y
        self.models = []
        # self.model_save_path = config.main_dir + "/ml/models/saved/base/xgboost"  #"/ml/models/saved/base/ann/slug/"
        self.model_save_path = "/mnt/c/Users/rwmas/GitHub/auto-learn/ml/models/saved/base/xgboost"  #"/ml/models/saved/base/ann/slug/"
        self.df_scores = None

        ### IG arguments
        self.ig_n_steps = 50

        ### SHAPLEY arguments
        self.shapley_background_size = 0.3 # 30% of actual dataset
        self.shapley_n_background = int(self.df_X.shape[0]*self.shapley_background_size)

        ### Gradient Shap
        self.stdevs=0.09
        self.n_samples=1

        # self.load_models()


    # Load pickled models
    def load_model(self, file_name):
        with open(file_name, "rb") as fin:
            model = pickle.load(fin)

        return model


    def load_models(self):
        model_files = glob.glob(self.model_save_path+'/*')
        for file in model_files:
            model = self.load_model(file)
            self.models.append(model)

        print(f"found {len(self.models)} models")
        logging.exception(f"found {len(self.models)} models")

        

    def inspect(self, model, attr_algo):
        if 'IG' in attr_algo: 
            self.get_ig_attr(model)



    def get_attr(self, attr_algos):        

        # self.models = self.models[0]
        lazy_results = []


        for model in self.models:
            print(f'calculating variable importance on {model}')
            for attr_algo in attr_algos:
                if type(model) == nn.Sequential:
                    if attr_algo == 'IG':
                        res = dask.delayed(__get_ig_attr__)(model, self.df_X, self.ig_n_steps)
                    elif attr_algo == 'SHAP':
                        res = dask.delayed(__get_shapley_torch_attr__)(model, self.df_X, self.shapley_n_background)
                    elif attr_algo == 'GradientSHAP':
                        res = dask.delayed(__get_gs_attr__)(model, self.df_X, self.stdevs, self.n_samples)

                elif type(model) == xgb.core.Booster:
                        res = dask.delayed(__get_shapley_ensemble_attr__)(model, self.df_X)

                lazy_results.append(res)

        results = dask.compute(*lazy_results, scheduler='distributed')

        return self.__beautify_scores__(attr_algos, results)



    def __beautify_scores__(self, attr_algos, results):                
        cols = []
        model_count = 0

        for model_count in range(1, len(self.models)+1):
            for algo in attr_algos:
                col_name = algo + '_model_' +str(model_count)
                cols.append(col_name)


        df_results = pd.DataFrame(results).T

        df_results.index = self.df_X.columns
        df_results.columns = cols 

        self.df_scores = df_results

        return df_results
