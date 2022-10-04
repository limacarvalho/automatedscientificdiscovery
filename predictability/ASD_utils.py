import pandas as pd
#import os
import numpy as np
#import plotly.express as px
import itertools
#import matplotlib.pyplot as plt
#import math
import plotly.graph_objects as go
#import plotly.colors
from plotly.subplots import make_subplots        
#from PIL import ImageColor
#import pickle
import time
#from scipy.spatial import distance
#from scipy.optimize import curve_fit
import dcor

from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
#from sklearn.neighbors import LocalOutlierFactor


def get_column_combinations(all_cols, inputs, outputs):
    
    assert inputs+outputs <= len(all_cols), "More input and output columns specified than there are columns."
    
    # initialise final list of column combinations
    col_combinations = []
    # first, draw possible input tuples
    input_combinations = list(itertools.combinations(all_cols, inputs))
    # now go through all possible input combinations
    for i in input_combinations:
        # get list of possible output columns, i.e. columns that are not yet part of current input columns
        curr_output_cols = [o for o in all_cols if o not in i]
        # now draw from that list all currently possible output combinations
        output_combinations = list(itertools.combinations(curr_output_cols, outputs))
        # add all currently possible output combinations to the current input columns and save in final list
        for oc in output_combinations:
            col_combinations.append(i+(*oc,))
        
    return col_combinations


def get_column_combinations_w_targets(all_cols, inputs, outputs, targets):
    
    assert inputs+outputs <= len(all_cols), "More input and output columns specified than there are columns."
    
    # initialise final list of column combinations
    col_combinations = []

    # if we fix some columns to be considered as targets only, drop these from input list
    if targets:
        all_input_cols = [x for x in all_cols if x not in targets]
    else:
        all_input_cols = all_cols
        
    # first, draw possible input tuples
    input_combinations = list(itertools.combinations(all_input_cols, inputs))
    # now go through all possible input combinations
    for i in input_combinations:
        if targets:
            # if there are pre-defined targets, use these as possible targets
            curr_output_cols = targets
        else:
            # otherwise, use remaining columns
            curr_output_cols = [o for o in all_input_cols if o not in i]
        # now draw from that list all currently possible output combinations
        output_combinations = list(itertools.combinations(curr_output_cols, outputs))
        # append all currently possible output combinations to the current input columns and save in final list
        for oc in output_combinations:
            col_combinations.append(i+(*oc,))
        
    return col_combinations


def data_prep_split(data, inputs, outputs):

    # get x and y value(s)
    curr_x = np.array(data[inputs])
    curr_y = np.array(data[outputs])

    # train test split
    curr_X_train, curr_X_test, curr_y_train, curr_y_test = train_test_split(curr_x, curr_y, random_state=1,
                                                                            test_size=.3, shuffle=True)
    
    return curr_X_train, curr_X_test, curr_y_train, curr_y_test


def rae(true, predicted):

    # define relative absolute error

    numerator = np.sum(np.abs(predicted - true))
    denominator = np.sum(np.abs(np.mean(true) - true))

    return numerator / denominator



def predictability(data, input_cols=1, output_cols=1, col_set=None, primkey_cols=[], targets=[],
                   method="MLP", hidden_layers=None, alphas=None, scoring="r2", scaling=True, 
                   max_iter=10000, n_jobs=-1, verbose=1):
    
    # map scoring to possible options
    scoring_dict = {
        "r2": "r2",
        "MAPE": "neg_mean_absolute_percentage_error",
        "neg_mean_absolute_percentage_error": "neg_mean_absolute_percentage_error",
        "RMSE": "neg_root_mean_squared_error",
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error"
    }
    scoring = scoring_dict[scoring]
    
    # if we want to measure the overall time
    start = time.time()
    
    # initialise the dictionary that is going to save the metrics per tuple
    metric_dict = {}
    
    # dict to save x-/y-train/-test and predicted values for subsequent plotting
    data_dict = {}
    
    # if primary keys are fed in, data columns should not contain these
    data_cols = [col for col in data.columns.to_list() if col not in primkey_cols]
    
    # if set of columns that should be considered is fed in, use this
    if col_set is not None:
        data_cols = list(set(col_set))
    
    # get the list of tuples of input and output columns
    if targets:
        data_tuples = get_column_combinations_w_targets(data_cols, input_cols, output_cols, targets)
    else:
        data_tuples = get_column_combinations(data_cols, input_cols, output_cols)    
    
    # for printing the progress of the analysis
    counter_tuples = 0
    
    # go through all tuples
    for curr_tuple in data_tuples:
        
        # if we want to measure the current tuple's analysis time
        curr_start = time.time()
        
        print("Analysing "+str(curr_tuple)+" now.")
        
        # get current inputs and outputs
        curr_inputs = list(curr_tuple[:input_cols])
        curr_outputs = list(curr_tuple[input_cols:])
        
        # reduce data to current columns and drop NAs
        curr_data = data[curr_inputs+curr_outputs].dropna()
        
        # do data preparations and train-test-split
        curr_X_train, curr_X_test, curr_y_train, curr_y_test = data_prep_split(curr_data, curr_inputs, curr_outputs)
        
        # compute standard deviation of curr_y_test for later scaling of the RMSE
        # TODO: replace with error of y value, if available
        # -> then also use fitting routines that consider these errors
        curr_y_test_std = np.std(curr_y_test)
        
        curr_y_train = curr_y_train.ravel()
        
        #
        # y-mean "prediction"
        #
        curr_y_train_mean = np.mean(curr_y_train)
        curr_y_test_pred_mean = curr_y_train_mean*np.ones(len(curr_X_test))
        # metrics
        curr_mean_r2 = r2_score(curr_y_test, curr_y_test_pred_mean)
        curr_mean_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_mean, squared=False)
        curr_mean_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_mean)
        curr_mean_rae = rae(curr_y_test, curr_y_test_pred_mean)
        curr_mean_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_mean)
        
        #
        # linear regression
        #
        lin_reg = LinearRegression().fit(curr_X_train,curr_y_train)
        curr_y_test_pred_linear = lin_reg.predict(curr_X_test)
        # metrics
        curr_lin_r2 = r2_score(curr_y_test, curr_y_test_pred_linear)
        curr_lin_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_linear, squared=False)
        curr_lin_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_linear)
        curr_lin_rae = rae(curr_y_test, curr_y_test_pred_linear)
        curr_lin_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_linear)
        
        #
        # power law fit
        #
        if ((curr_X_train > 0).all().all()) and ((curr_X_test > 0).all().all()):
            do_pl_fit = True
        else:
            do_pl_fit = False
            
        print("do power law fit: "+str(do_pl_fit))
        
        if do_pl_fit:
            
            curr_X_train_log = np.log(curr_X_train)
            curr_X_test_log = np.log(curr_X_test)
            curr_y_train_log = np.log(curr_y_train)
            
            pl_fit = LinearRegression().fit(curr_X_train_log, curr_y_train_log)
            curr_y_test_pred_pl = pl_fit.predict(curr_X_test_log)
            curr_y_test_pred_pl = np.exp(curr_y_test_pred_pl)
            
            # metrics
            curr_pl_r2 = r2_score(curr_y_test, curr_y_test_pred_pl)
            curr_pl_rmse = mean_squared_error(curr_y_test, curr_y_test_pred_pl, squared=False)
            curr_pl_rmse_std = curr_pl_rmse/curr_y_test_std
            curr_pl_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred_pl)
            curr_pl_rae = rae(curr_y_test, curr_y_test_pred_pl)
            curr_pl_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred_pl)

        # to allow for uniform dicts and later metric dataframes:
        else:
            curr_pl_r2 = None
            curr_pl_rmse = None
            curr_pl_rmse_std = None
            curr_pl_mape = None
            curr_pl_rae = None
            curr_pl_dcor = None
        
        #
        # MLP regression
        print("start MLP routine")
        #
        # list of hidden layer sizes for GridSearch
        if hidden_layers is None:
            hidden_layers = [(12,), 
                              (50,), 
                              (70,5,), 
                              #(40,18,3,)
                            ]
        # list of alpha values for GridSearch
        if alphas is None:
            alphas = [0.001, 
                      0.0001, 
                      0.00001
                     ]

        # via pipeline (with and without scaler)
        if scaling == "yes":
            pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('mlp', MLPRegressor(max_iter=max_iter))
                            ])
            pipe_params = [
                           {'mlp__hidden_layer_sizes': hidden_layers,
                            'mlp__alpha': alphas}
                            ]
            clf = GridSearchCV(pipe,
                               param_grid=pipe_params,
                               cv=3,
                               scoring=scoring,
                               return_train_score=True,
                               verbose=verbose,
                               n_jobs=n_jobs
                               )
        elif scaling == "no":
            pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('mlp', MLPRegressor(max_iter=max_iter))
                            ])
            pipe_params = [{'scaler': ['passthrough'],
                            'mlp__hidden_layer_sizes': hidden_layers,
                            'mlp__alpha': alphas}]
            clf = GridSearchCV(pipe,
                               param_grid=pipe_params,
                               cv=3,
                               scoring=scoring,
                               return_train_score=True,
                               verbose=verbose,
                               n_jobs=n_jobs
                               )
        else:
            pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('mlp', MLPRegressor(max_iter=max_iter))
                            ])
            pipe_params = [{'scaler': ['passthrough'],
                            'mlp__hidden_layer_sizes': hidden_layers,
                            'mlp__alpha': alphas}, 
                           {'mlp__hidden_layer_sizes': hidden_layers,
                            'mlp__alpha': alphas}]
            clf = GridSearchCV(pipe,
                               param_grid=pipe_params,
                               cv=3,
                               scoring=scoring,
                               return_train_score=True,
                               verbose=verbose,
                               n_jobs=n_jobs
                               )
        
        clf.fit(curr_X_train, curr_y_train)
        curr_best_params = clf.best_params_
        curr_y_test_pred = clf.predict(curr_X_test)
        
        # metrics
        curr_mlp_r2 = r2_score(curr_y_test, curr_y_test_pred)
        curr_mlp_rmse = mean_squared_error(curr_y_test, curr_y_test_pred, squared=False)
        curr_mlp_mape = mean_absolute_percentage_error(curr_y_test, curr_y_test_pred)
        curr_mlp_rae = rae(curr_y_test, curr_y_test_pred)
        curr_mlp_dcor = dcor.distance_correlation(curr_y_test, curr_y_test_pred)

        # save metrics into dict
        metric_dict[curr_tuple] = {"MLP r2": curr_mlp_r2, "linear r2": curr_lin_r2, 
                                   "pow. law r2": curr_pl_r2, "mean r2": curr_mean_r2, 
                                    "MLP RMSE": curr_mlp_rmse, "linear RMSE": curr_lin_rmse,
                                    "pow. law RMSE": curr_pl_rmse, "mean RMSE": curr_mean_rmse,
                                    "MLP RMSE/std": curr_mlp_rmse/curr_y_test_std, "linear RMSE/std": curr_lin_rmse/curr_y_test_std,
                                   "pow. law RMSE/std": curr_pl_rmse_std, "mean RMSE/std": curr_mean_rmse/curr_y_test_std,
                                    "MLP MAPE": curr_mlp_mape, "linear MAPE": curr_lin_mape,
                                   "pow. law MAPE": curr_pl_mape, "mean MAPE": curr_mean_mape,
                                    "MLP rae": curr_mlp_rae, "linear rae": curr_lin_rae,
                                    "pow. law rae": curr_pl_rae, "mean rae": curr_mean_rae,
                                    "MLP dcor": curr_mlp_dcor, "linear dcor": curr_lin_dcor,
                                    "pow. law dcor": curr_pl_dcor, "mean dcor": curr_mean_dcor,
                                   }

        # save values into dict
        if do_pl_fit:
            data_dict[curr_tuple] = {"X_train": curr_X_train, "X_test": curr_X_test,
                                     "y_train": curr_y_train, "y_test": curr_y_test,
                                     "y_test_pred": curr_y_test_pred, "y_test_pred_linear": curr_y_test_pred_linear,
                                     "y_test_pred_pl": curr_y_test_pred_pl, "y_test_pred_mean": curr_y_test_pred_mean,
                                     "GridSearchParams": curr_best_params, "scores": clf.cv_results_
                                     }
        else:
            data_dict[curr_tuple] = {"X_train": curr_X_train, "X_test": curr_X_test,
                                     "y_train": curr_y_train, "y_test": curr_y_test, 
                                     "y_test_pred": curr_y_test_pred, "y_test_pred_linear": curr_y_test_pred_linear,
                                     "y_test_pred_mean": curr_y_test_pred_mean,
                                     "GridSearchParams": curr_best_params, "scores": clf.cv_results_
                                     }

        print("The analysis of this tuple took "+str(round(time.time()-curr_start,2))+"s.")
        # for printing the progress of the analysis
        counter_tuples += 1
        print("-----"+str(counter_tuples)+"/"+str(len(data_tuples))+"-----")
    
    print("The whole run took "+str(round(time.time()-start,2))+"s.")
    
    return metric_dict, data_dict

def plot_result(input_datas_dict, plot_comb, plot_along=[]):
    '''
    plot_along = ["linear", "mean", "pl"] or any subsets – always as list
    '''

    # make dict a dataframe, name columns appropriately and compute error of MLP prediction
    results_df = pd.DataFrame([input_datas_dict[plot_comb]["y_test_pred"], input_datas_dict[plot_comb]["y_test"].flatten()]).transpose()
    results_df.columns = ["pred", "true"]
    results_df["error"] = results_df["pred"]-results_df["true"]
    
    fig = make_subplots(
                        rows=1, cols=3,
                        column_widths=[0.7, 0.15, 0.15],
                        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}]],
                        horizontal_spacing=.15
                        )

    #
    # plot results
    #

    # for plotting other than MLP predictions as well, go through additional methods
    if plot_along:
        for comparison in plot_along:
            # even if power law is chosen as additional method, some tuples may not have been fitted via power law
            # due to non-positive values:
            if (comparison == "pl") and ("y_test_pred_pl" not in input_datas_dict[plot_comb].keys()):
                print("no power law fit performed, some columns did not include positive values only")
            else:
                # load data, compute error
                results_df["pred_"+comparison] = input_datas_dict[plot_comb]["y_test_pred_"+comparison]
                results_df["error_"+comparison] = results_df["pred_"+comparison]-results_df["true"]
                # add plot
                fig.add_trace(go.Scatter(
                                          x=results_df["true"],
                                          y=results_df["pred_"+comparison],
                                          mode="markers",
                                          name=comparison+" preds vs trues",
                                          opacity=.8
                                         ),
                              row=1, col=1
                              )
    # add MLP plot
    fig.add_trace(go.Scatter(
                              x=results_df["true"],
                              y=results_df["pred"],
                              mode="markers",
                              name="MLP preds vs trues",
                              opacity=.8
                             ),
                  row=1, col=1
                  )

    #
    # local plot of errors
    #

    # for plotting other than MLP predictions as well, go through additional methods
    if plot_along:
        for comparison in plot_along:
            # even if power law is chosen as additional method, some tuples may not have been fitted via power law
            # due to non-positive values:
            if (comparison == "pl") and ("y_test_pred_pl" not in input_datas_dict[plot_comb].keys()):
                continue
            else:
                fig.add_trace(
                    go.Scatter(
                        x=results_df["error_"+comparison],
                        y=results_df["pred_"+comparison],
                        mode="markers",
                        name="pred. error "+comparison,
                        opacity=.6
                        ),
                    row=1, col=2
                    )
    # add MLP plot
    fig.add_trace(
        go.Scatter(
            x=results_df["error"],
            y=results_df["pred"],
            mode="markers",
            marker_color="Maroon",
            name="pred. error MLP",
            opacity=.6
            ),
        row=1, col=2
        )

    #
    # histogram of errors
    #

    # for plotting other than MLP predictions as well, go through additional methods
    if plot_along:
        for comparison in plot_along:
            # even if power law is chosen as additional method, some tuples may not have been fitted via power law
            # due to non-positive values:
            if (comparison == "pl") and ("y_test_pred_pl" not in input_datas_dict[plot_comb].keys()):
                continue
            else:
                fig.add_trace(
                    go.Histogram(
                        y=results_df["error_"+comparison],
                        nbinsx=int(np.floor(len(results_df["pred"])/10)),
                        name="pred. error "+comparison
                        ),
                    row=1, col=3
                    )
    # add MLP plot
    fig.add_trace(
        go.Histogram(
            y=results_df["error"],
            nbinsx=int(np.floor(len(results_df["pred"])/10)),
            name="pred. error MLP"
            ),
        row=1, col=3
        )

    # set title
    title = str(plot_comb)     

    # set layout, axis labels etc.
    fig.update_layout(
        title=title,
        width=950,
        height=555,
        xaxis=dict(title="true"),
        yaxis=dict(title="pred"),
        xaxis2=dict(title="pred-true"),
        yaxis2=dict(title="pred",
                    matches="y"),
        xaxis3=dict(title="freq"),
        yaxis3=dict(title="pred-true")
    )

    fig.show()
    
#
# OLDER plotting routines
#

def plot_2d_result(data_tuple, metrics, datas, show=False):
    
    # need to order test values for correct connection of data points during line plotting
    pred_MLP_data = list(zip(datas[data_tuple]["X_test"].reshape(len(datas[data_tuple]["X_test"]),), 
             datas[data_tuple]["y_test_pred"]))
    pred_MLP_df = pd.DataFrame(data=pred_MLP_data,
             columns=["X_test", "y_test_pred"],
             ).sort_values(by="X_test")
    
    # get max and min plotting values
    y_min = min(0, min(datas[data_tuple]["y_train"]), min(datas[data_tuple]["y_test"]), min(datas[data_tuple]["y_test_pred"]))
    y_max = max(0, max(datas[data_tuple]["y_train"]), max(datas[data_tuple]["y_test"]), max(datas[data_tuple]["y_test_pred"]))
    
    fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.7, 0.15, 0.15],
    specs=[[{"type": "scatter"}],
           [{"type": "scatter", "t": -.07}],
           [{"type": "histogram", "t": .05}]],
    vertical_spacing=.1,
    #shared_xaxes = True
    )

    fig.add_trace(
        go.Scatter(x=datas[data_tuple]["X_train"].reshape(len(datas[data_tuple]["X_train"]),), 
                   y=datas[data_tuple]["y_train"].flatten(),
                   xaxis="x",
                   yaxis="y",
                   name="training data",
                   mode="markers", marker_color="Maroon", marker_size=3, opacity=.6
                    ),
        row=1, col=1
                 )
    fig.add_trace(
        go.Scatter(x=datas[data_tuple]["X_test"].reshape(len(datas[data_tuple]["X_test"]),), 
                   y=datas[data_tuple]["y_test"].flatten(),  
                   xaxis="x",
                   yaxis="y",
                   name="test data",
                   mode="markers", marker_color="LightSeaGreen", opacity=.8 
                    ),
        row=1, col=1
                 )
    fig.add_trace(
        go.Scatter(x=pred_MLP_df["X_test"],#.reshape(len(datas[data_tuple]["X_test"]),), 
                   y=pred_MLP_df["y_test_pred"], 
                   xaxis="x",
                   yaxis="y",
                   name="predictions MLP",
                   mode='lines+markers',
                   #mode="markers",
                   marker_color="Tomato", 
                    ),
        row=1, col=1
                 )
    # add power law predictions
    fig.add_trace(
        go.Scatter(x=datas[data_tuple]["X_test"].reshape(len(datas[data_tuple]["X_test"]),), 
                   y=datas[data_tuple]["y_test_pred_pl"].flatten(), 
                   xaxis="x",
                   yaxis="y",
                   name="predictions pl",
                   mode="markers", marker_color="SteelBlue", 
                    ),
        row=1, col=1
                 )
        
    # add metrics
    
    fig.add_annotation(text='mean y_train: '+str(round(datas[data_tuple]["y_test_pred_mean"][0])),
                      align="right",
                      showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=1.223,
                        y=.73,
                        bgcolor="white")
    
    fig.add_annotation(text='<b>r2 MLP:   </b>'+str(round(metrics[data_tuple]["MLP r2"],2))+
                       ' <i><br>r2 pow. law:   </i>'+str(round(metrics[data_tuple]["pow. law r2"],2))+
                       ' <i><br>r2 lin. reg.:   </i>'+str(round(metrics[data_tuple]["linear r2"],2))+
                       ' <i><br>r2 mean pred.:   </i>'+str(round(metrics[data_tuple]["mean r2"],2))+
                       ' <b><br>RMSE MLP:   </b>'+str(round(metrics[data_tuple]["MLP RMSE"],2))+
                       ' <i><br>RMSE pow. law:   </i>'+str(round(metrics[data_tuple]["pow. law RMSE"],2))+
                       ' <i><br>RMSE lin. reg.:   </i>'+str(round(metrics[data_tuple]["linear RMSE"],2))+
                       ' <i><br>RMSE nean pred.:   </i>'+str(round(metrics[data_tuple]["mean RMSE"],2))+
                       ' <b><br>RMSE/std MLP:   </b>'+str(round(metrics[data_tuple]["MLP RMSE/std"],2))+
                       ' <i><br>RMSE/std pow. law:   </i>'+str(round(metrics[data_tuple]["pow. law RMSE/std"],2))+
                       ' <i><br>RMSE/std lin. reg.:   </i>'+str(round(metrics[data_tuple]["linear RMSE/std"],2))+
                       ' <i><br>RMSE/std mean pred.:   </i>'+str(round(metrics[data_tuple]["mean RMSE/std"],2))+
                       ' <b><br>MAPE MLP:   </b>'+str(round(metrics[data_tuple]["MLP MAPE"],2))+
                       ' <i><br>MAPE pow. law:   </i>'+str(round(metrics[data_tuple]["pow. law MAPE"],2))+
                       ' <i><br>MAPE lin. reg.:   </i>'+str(round(metrics[data_tuple]["linear MAPE"],2))+
                       ' <i><br>MAPE mean pred.:   </i>'+str(round(metrics[data_tuple]["mean MAPE"],2))+
                       ' <b><br>rae MLP:   </b>'+str(round(metrics[data_tuple]["MLP rae"],2))+
                       ' <i><br>rae pow. law:   </i>'+str(round(metrics[data_tuple]["pow. law rae"],2))+
                       ' <i><br>rae lin. reg.:   </i>'+str(round(metrics[data_tuple]["linear rae"],2))+
                       ' <i><br>rae mean pred.:   </i>'+str(round(metrics[data_tuple]["mean rae"],2))+
                       ' <b><br>dcor MLP:   </b>'+str(round(metrics[data_tuple]["MLP dcor"],2))+
                       ' <i><br>dcor pow. law:   </i>'+str(round(metrics[data_tuple]["pow. law dcor"],2))+
                       ' <i><br>dcor lin. reg.:   </i>'+str(round(metrics[data_tuple]["linear dcor"],2))+
                       ' <i><br>dcor mean pred.:   </i>'+str(round(metrics[data_tuple]["mean dcor"],2)),
                       #' <br>Spearman corr.:   '+str(round(metrics[data_tuple]["Spearman"],2))+
                       #' <br>Pearson corr.:   '+str(round(metrics[data_tuple]["Pearson"],2)), 
                        align='right',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=1.223,
                        y=.68,
                        bgcolor="white",
                        #bordercolor='black',
                        #borderwidth=1
                      )

    # local plot of errors
    fig.add_trace(
        go.Scatter(x=datas[data_tuple]["X_test"].reshape(len(datas[data_tuple]["X_test"]),), 
                   y=datas[data_tuple]["y_test_pred"].flatten()-datas[data_tuple]["y_test"].flatten(), 
                   xaxis="x2",
                   yaxis="y2",
                   mode="markers", marker_color="Tomato",
                   legendgroup="MLP",
                   name="pred. error MLP" 
                    ),
        row=2, col=1
                 )
    fig.add_trace(
        go.Scatter(x=datas[data_tuple]["X_test"].reshape(len(datas[data_tuple]["X_test"]),), 
                   y=datas[data_tuple]["y_test_pred_pl"].flatten()-datas[data_tuple]["y_test"].flatten(), 
                   xaxis="x2",
                   yaxis="y2",
                   mode="markers", marker_color="SteelBlue",
                   legendgroup="pl",
                   name="pred. error pl" 
                    ),
        row=2, col=1
                 )
    
    # add line as separator
    fig.add_shape(type='line',
                x0=-.05,
                y0=.1,
                x1=1.05,
                y1=.1,
                line=dict(color='white',),
                xref='paper',
                yref='paper'
    )
    
    # histogram of errors
    fig.add_trace(
        go.Histogram(x=datas[data_tuple]["y_test_pred"]-datas[data_tuple]["y_test"].flatten(),
                   xaxis="x3",
                   yaxis="y3",
                   nbinsx=100,
                   marker_color='Tomato',
                   legendgroup="MLP",
                   name="pred. error MLP",
                    showlegend=False),
        row=3, col=1
                 )
    fig.add_trace(
        go.Histogram(x=datas[data_tuple]["y_test_pred_pl"].flatten()-datas[data_tuple]["y_test"].flatten(),
                   xaxis="x3",
                   yaxis="y3",
                   nbinsx=100,
                   marker_color='SteelBlue',
                   legendgroup="pl",
                   name="pred. error pl",
                    showlegend=False),
        row=3, col=1
                 )
    
    fig.update_layout(
        title=data_tuple[1]+'  vs.  '+data_tuple[0],
        xaxis=dict(
            gridcolor='white',
            gridwidth=2,
            #type='log',
        ),
        yaxis=dict(
            title=data_tuple[1],
            gridcolor='white',
            gridwidth=2,
            #type='log',
        ),
        yaxis_range=[y_min*1.01,y_max*1.01],
        xaxis2=dict(title=data_tuple[0],
                    matches="x"),
        yaxis2=dict(title="pred. error"),
        xaxis3=dict(title=r"$\text{error } y_{pred}-y$"),
        yaxis3=dict(title="frequency"),
        legend=dict(bgcolor="white"),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        width=920,
        height=720
    )
    fig.update_layout(legend_tracegroupgap=0)
    
    if show==True:
        fig.show()
    else:
        return fig


def plot_3d_result(data_tuple, metrics, datas, show=False):
    
    
    '''fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.7, 0.15, 0.15],
    specs=[[{"type": "scatter3d"}],
           [{"type": "scatter", "t": -.07}],
           [{"type": "histogram", "t": .05}]],
    vertical_spacing=.1,
    #shared_xaxes = True
    )'''
    
    fig = go.Figure(data=[
        go.Scatter3d(x=[i[0] for i in datas[data_tuple]["X_train"]],
                     y=[i[1] for i in datas[data_tuple]["X_train"]],
                     z=datas[data_tuple]["y_train"].flatten(),
                     #xaxis="x",
                     #yaxis="y",
                     name="training data",
                     mode="markers",
                     marker_color=datas[data_tuple]["y_train"].flatten(),
                     marker_size=3,
                     opacity=.6
                    ),
        #row=1, col=1
                         ]
                   )
    
     
    fig.update_layout(
        title=data_tuple[2]+'  vs.  '+data_tuple[0]+' & '+data_tuple[1],
        scene=dict(xaxis=dict(title=data_tuple[0],
            gridcolor='white',
            gridwidth=2,
            #type='log',
        ),
        yaxis=dict(
            title=data_tuple[1],
            gridcolor='white',
            gridwidth=2,
            #type='log',
        ),
        zaxis=dict(
            title=data_tuple[2],
            gridcolor='white',
            gridwidth=2,
            #type='log',
        )
        ),
        #yaxis_range=[y_min*1.01,y_max*1.01],
        #xaxis2=dict(title=data_tuple[0], matches="x"),
        #yaxis2=dict(title="pred. error"),
        #xaxis3=dict(title=r"$\text{error } y_{pred}-y$"),
        #yaxis3=dict(title="frequency"),
        legend=dict(bgcolor="white"),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        width=920,
        height=720
    )
    
    if show==True:
        fig.show()
    else:
        return fig



