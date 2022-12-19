# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


import predictability.utils as asdpu
import predictability.core as asdpc
import relevance.relevance as relevance
import complexity.dim_reduce.dimreduce_main as complexity

####from predictability.utils import get_column_combinations
####from predictability.src.ASD_predictability_utils.utils import get_column_combinations
####from predictability.bin.main import predictability
####from predictability.src.ASD_predictability_utils.utils import plot_result
####from complexity.dim_reduce import dimreduce_main
####from xai import relevance


# Set streamlit layout
st.set_page_config(layout="wide")

# Set streamlit session state
if "discovery_type" not in st.session_state:
    st.session_state["discovery_type"] = "Predictability"

df_input = st.session_state["df_input"]

# Implement mainframe output
st.title("Data discovery")
st.markdown("")
st.header("Data overview")
####df_input = st.session_state["df_input"]


# Print normalized dataframe based on user selection
st.dataframe(df_input.head())
st.markdown("***")

# Create function for sidebar selection: discovery task
def handle_ml_select():
    if st.session_state.ml_type:
        st.session_state.discovery_type=st.session_state.ml_type

# Set selection of discovery task on sidebar
ml_select = st.sidebar.radio("Discovery task:", ["Predictability", "Relevance", "Grouping", "Complexity"], on_change=handle_ml_select, key="ml_type")

# Implement if statements based on discovery task selection
if st.session_state["discovery_type"] == "Predictability":
    st.header("Discover the predictability of your data")
    st.markdown("""
    To begin the predictability task, you have to choose between some options.   
    """)
 
    predict_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    predict_num_columns_without = df_input.select_dtypes(include = np.number).columns.to_list()
    predict_num_columns.insert(0, "All")
    pred_target_column = st.multiselect("Target columns (numeric):", predict_num_columns, help="The subset of columns that should be treated exclusively as targets.")
    pred_target_column_count = len(pred_target_column)
    if len(pred_target_column) == 0:
        pred_target_column = "None"
    if "All" in pred_target_column:
        pred_target_column.clear()
        pred_target_column = predict_num_columns_without    
    pred_primekey_cols = st.multiselect("Primekey columns (numeric):", predict_num_columns, help="The subset of columns corresponding to primary keys. These will neither be used as inputs nor outputs.")
    if len(pred_primekey_cols) == 0:
        pred_primekey_cols = "None"
    if "All" in pred_primekey_cols:
        pred_primekey_cols.clear()
        pred_primekey_cols = predict_num_columns_without             
    pred_col_set = st.multiselect("Column set (numeric):", predict_num_columns, help="The (sub-)set of columns that should be considered.")
    if len(pred_col_set) == 0:
        pred_col_set = "None"
    if "All" in pred_col_set:
        pred_col_set.clear()
        pred_col_set = predict_num_columns_without 
    pred_input_column = st.slider('Input fit:', min_value=0, max_value=pred_target_column_count, value=1, step=1, help="The number of input columns for the fit. For a 4-1 fit, input_cols = 4.")
    pred_output_column = st.slider('Output fit:', min_value=0, max_value=pred_target_column_count, value=1, step=1, help="The number of target columns for the fit. For a 4-1 fit, output_cols = 1.")
    pred_refined_n_best = st.slider('Refined-n-best:', min_value=0, max_value=100, value=1, step=1, help="Sets the number of how many of the best results will go into the refined_predictability routine.") 
    pred_ml_method = st.selectbox("ML-method:", ("kNN", "MLP"), help="Choose between kNN (k-nearest-neighbours) or MLP (Multi-Layer Perceptron).")
    pred_greedy = st.checkbox('Use greedy algorithm')

    if pred_ml_method and pred_input_column and pred_output_column:
        ###### Part 1, Predictability: get_column_combinations of ml-algorithm ###### 
        st.markdown("***")
        st.subheader("Get combinations of your data")
        st.markdown("""
        This function can be used to determine the overall number of combinations the predictability routine analyses
        given the number of data columns, fitting type etc. This allows to estimate the overall runtime of the predictability
        routine.
        """)
        if "pred_get_combinations" not in st.session_state:
            st.session_state["pred_get_combinations"] = False      
       
        def pred_combinations_click():
            st.session_state["pred_get_combinations"] = True       
       
        if st.button("Get combinations", on_click=pred_combinations_click) or st.session_state["pred_get_combinations"]:
            pred_target_column_ext = pred_target_column
            if "None" in pred_target_column:
                pred_target_column_ext = []
            get_column_combinations = asdpu.get_column_combinations(all_cols=df_input.columns, inputs=pred_input_column, outputs=pred_output_column, targets=pred_target_column_ext, amount_only=False, return_targets=False)
            
            # printed dataframe based on selected targets
            st.write(get_column_combinations)
        else:
            st.markdown("""
            You have not chosen this task.
            """)
    
        ###### Part 2, Predictability: predictability function of ml-algorithm ######  
        if "button_pred_start_discovery" not in st.session_state:
            st.session_state["button_pred_start_discovery"] = False
        if "button_pred_refined_predictability" not in st.session_state:
            st.session_state["button_pred_refined_predictability"] = False 
        
        def pred_discovery_click():
            st.session_state["button_pred_start_discovery"] = True

        def pred_refined_click():
            st.session_state["button_pred_refined_predictability"] = True

        st.markdown("***")
        st.subheader("Predictability")
        st.markdown("""
        Discover the predictability of your dataset.
        """)

        if st.button("Start discovery", on_click=pred_discovery_click) or st.session_state["button_pred_start_discovery"]:
            ###### Implement tbe code of Predictability ######
            metrics_dict, datas_dict = asdpc.run_predictability(data=df_input, input_cols=pred_input_column, output_cols=pred_output_column, col_set=pred_col_set, primkey_cols=pred_primekey_cols, targets=pred_target_column, method=pred_ml_method, greedy=pred_greedy, refined_n_best=pred_refined_n_best)
            st.spinner(text="Calculation in progress...")
            st.markdown("""
            Discovery finished.
            """)                  
            # Visualize the output of the predictability part           
            st.markdown("""
            Output of the predictability part:
            """)
            pred_plot_along = st.multiselect("Plot-along:", ("linear", "mean", "pl"), help="Allows for specifying further prediction methods to be plotted along the kNN/MLP ones.")
            struc_dict = datas_dict[list(datas_dict.keys())[0]]
                
            if pred_refined_n_best == 0:
                st.write(asdpu.plot_result(input_datas_dict=datas_dict, plot_comb=struc_dict, refined_dict=False, refined_input_datas_dict=None, plot_along=pred_plot_along))
            else:
                st.write(asdpu.plot_result(input_datas_dict=datas_dict, plot_comb=struc_dict, refined_dict=True, refined_input_datas_dict=None, plot_along=pred_plot_along))
                
            ###### Extended part with refine_predictability ###### 
  
            if st.button('Use refined predictability routine.', on_click=pred_refined_click) or st.session_state["button_pred_refined_predictability"]:
                st.markdown("***")
                if "button_pred_refined_discovery" not in st.session_state:
                    st.session_state["button_pred_refined_discovery"] = False 
                
                def pred_refined_discovery_click():
                    st.session_state["button_pred_refined_discovery"] = True

                pred_time_left_for_this_task = st.slider('Time left for this task (seconds):', min_value=0.0, max_value=60.0, value=0.1, help="Time in seconds that specifies for how long the routine should run.")
                pred_plot_along_refined = st.multiselect("Plot-along:", ("linear", "mean", "pl", "init"), help="Allows for specifying further prediction methods to be plotted along the kNN/MLP ones, based on the refined predictability routine.")
                
                if st.button("Start refined discovery", on_click=pred_refined_discovery_click) or st.session_state["button_pred_refined_discovery"]:                
                    list_n_best = asdpu.tuple_selection(all_metrics=metrics_dict, n_best=pred_refined_n_best)
                    evaluation_metrics, all_data = asdpc.refine_predictability(best_tuples=list_n_best, data_dict=datas_dict, time_left_for_this_task=pred_time_left_for_this_task, use_ray=True, generations=100, population_size=100, n_jobs=-1)
                    st.markdown("""
                    Refined discovery finished.
                    """)                    
                    struc_dict_refined = all_data[list(all_data.keys())[0]]
                    st.write(asdpu.plot_result(input_datas_dict=all_data, plot_comb=struc_dict_refined, refined_dict=True, refined_input_datas_dict=datas_dict, plot_along=pred_plot_along))

        else:
            st.markdown("""
            You have not chosen this task.
            """)
    else:
        st.markdown("""
        You have to set all options and columns.
        """)

elif st.session_state["discovery_type"] == "Relevance":
    st.header("Discover the relevance of your data")
    st.markdown("""
    To begin the relevance task, you have to choose between some options.   
    """)
    relevance_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    relevance_num_columns_without = df_input.select_dtypes(include = np.number).columns.to_list()
    relevance_num_columns.insert(0, "All")
    relevance_column = st.multiselect("Input columns (numeric):", relevance_num_columns)#[st.session_state["predict_target_column"]])
    relevance_target = st.selectbox("Target column (numeric):", relevance_num_columns_without)#[st.session_state["predict_target_column"]])
    
    # If it is not a list
    #df_input_ext = df_input[relevance_column]
 
    # Additional part
    ##relevance_xgb_objective = st.checkbox('Use XGBoost model')
    ##relevance_lgbm_objective = st.checkbox('Use lightgbm model')
    relevance_pred_class = st.selectbox("Modeling task:", ("regression", "classification"), help="Specify problem type, i.e., 'regression' or 'classification'.")
    relevance_list_base_models = st.multiselect("Base models:", ('All', 'briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf'), help="List of base models to be used to fit on the data.")
    relevance_n_trials = st.slider("Sampled parameter settings:", min_value=0, max_value=300, value=100, help="Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.")
    relevance_boosted_round = st.slider("N-estimators:", min_value=0, max_value=300, value=100, help="N_estimators parameter for XGBoost and LightGBM.")    
    relevance_max_depth = st.slider("Max tree depth:", min_value=0, max_value=100, value=30, help="Max tree depth parameter for XGBoost, LightGBM and RandomForest.") 
    relevance_rf_n_estimators = st.slider("N-estimators, RandomForest:", min_value=0, max_value=3000, value=1500, help="N_estimators parameter of RandomForest.") 
    relevance_bagging_estimators = st.slider("Bagging estimators:", min_value=0, max_value=300, value=100, help="Bagging estimators.")
    relevance_n_neighbors = st.slider("NNeighbors of KNN:", min_value=0, max_value=100, value=30, help="N-Neighbors of KNN.")
    relevance_cv_splits = st.slider("Cross-validation splits:", min_value=0, max_value=20, value=3, help="Determines the cross-validation splitting strategy.")
    relevance_ensemble_bagging_estimators = st.slider("Ensemble bagging estimators:", min_value=0, max_value=100, value=50, help="N_estimators parameter of Bagging. This is the second baggin method which is used an an ensemble on top of base estimators.")
    relevance_ensemble_n_trials = st.slider("Ensemble-n-trials:", min_value=0, max_value=100, value=50, help="Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.")
    relevance_attr_algos = st.multiselect("Xai:", ('All', 'IG', 'SHAP', 'GradientSHAP', 'knockoffs'), help="Xai methods.")
    relevance_fdr = st.slider("Fdr:", min_value=0.0, max_value=1.0, value=0.1, help="Target false discovery rate.")
    relevance_fstats = st.multiselect("Fstats methods:", ('All', 'lasso', 'ridge', 'randomforest'), help="Methods to calculate fstats.")
    relevance_knockoff_runs = st.slider("Knockoff runs:", min_value=0, max_value=100000, value=20000, help="Number of reruns for each knockoff setting.")    
    relevance_df = df_input
    if "All" in relevance_column:
        relevance_column.clear()
        relevance_column = relevance_num_columns_without    
    if "All" in relevance_list_base_models:
        relevance_list_base_models.clear()
        relevance_list_base_models = ['briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf']
    if "All" in relevance_attr_algos:
        relevance_attr_algos.clear()
        relevance_attr_algos = ['IG', 'SHAP', 'GradientSHAP', 'knockoffs']   
    if "All" in relevance_fstats:
        relevance_fstats.clear()
        relevance_fstats = ['lasso', 'ridge', 'randomforest']
    if  "regression" in relevance_pred_class:
        metric_func = "r2_score"
    if  "classification" in relevance_pred_class:
        metric_func = "f1_score"    

    relevance_options = {'xgb_objective': 'binary:logistic', 'lgbm_objective': 'binary', 'pred_class': str(relevance_pred_class), 'score_func': None, 'metric_func': str(metric_func), 'list_base_models': relevance_list_base_models, 'n_trials': int(relevance_n_trials), 'boosted_round': int(relevance_boosted_round), 'max_depth': int(relevance_max_depth), 'rf_n_estimators': int(relevance_rf_n_estimators), 'bagging_estimators': int(relevance_bagging_estimators), 'n_neighbors': int(relevance_n_neighbors), 'cv_splits': int(relevance_cv_splits), 'ensemble_bagging_estimators': int(relevance_ensemble_bagging_estimators), 'ensemble_n_trials': int(relevance_ensemble_n_trials), 'attr_algos': relevance_attr_algos, 'fdr': float(relevance_fdr), 'fstats': relevance_fstats, 'knockoff_runs': int(relevance_knockoff_runs)}

    if relevance_column and relevance_target:
    
        ###### Part 1, Relevance function of ml-algorithm ######  
        st.markdown("***")
        st.subheader("Relevance")
        st.markdown("""
        Discover the relevance of your dataset.
        """)
        relevance_algorithm_start = st.button("Start discovery")

        if relevance_algorithm_start == True:
            try:
                ###### Implement tbe code of Relevance ######
                return_relevance = relevance.relevance(relevance_df, relevance_column, relevance_target, relevance_options)
                
                # Visualize the output (the return values) of the relevance function
                st.markdown("""
                Output of the relevance part:
                """)
                st.write(return_relevance)
            except:
                st.markdown("""
                Algorithm Error! You should restart the app.
                """)
        else:
            st.markdown("""
            You have not chosen this task.
            """)
elif st.session_state["discovery_type"] == "Grouping":
    st.markdown("""
    You selected the task: Grouping
    """)
elif st.session_state["discovery_type"] == "Complexity":
    st.header("Discover the complexity of your data")
    st.markdown("""
    To begin the complexity task, you have to choose between some options.   
    """)
    complex_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    complex_num_columns_without = df_input.select_dtypes(include = np.number).columns.to_list()
    complex_num_columns.insert(0, "All")
    complex_column = st.multiselect("Columns (numeric):", complex_num_columns, help="Select columns of your dataframe.")
    complex_ident = st.text_input('Identifier:', placeholder="Enter an identifier")
    complex_cutoff_loss = st.slider("Cutoff loss", min_value=0.0, max_value=1.0, help="Cutoff for loss of dim reduction quality control.")
    complex_ml_method = st.multiselect("ML-method:", ("All","py_pca", "py_pca_sparse", "py_pca_incremental", "py_truncated_svd", "py_crca", "py_sammon", "r_adr", "r_mds", "r_ppca", "r_rpcag", "r_ispe"), help="Dim reduction functions to use.")
    complex_data_high = df_input
    if "All" in complex_column:
        complex_column.clear()
        complex_column = complex_num_columns_without     
    if "All" in complex_ml_method:
        complex_ml_method = ["py_pca", "py_pca_sparse", "py_pca_incremental", "py_truncated_svd", "py_crca", "py_sammon", "r_adr", "r_mds", "r_ppca", "r_rpcag", "r_ispe"]
    if complex_column and complex_ident and complex_cutoff_loss:
    
        ###### Part 1, Complexity function of ml-algorithm ######  
        st.markdown("***")
        st.subheader("Complexity")
        st.markdown("""
        Discover the complexity of your dataset.
        """)
        complex_algorithm_start = st.button("Start discovery")

        if complex_algorithm_start == True:
            try:
                ###### Implement tbe code of Complexity ######
                intrinsic_dimension, best_results, df_summary = complexity.intrinsic_dimension(complex_data_high, str(complex_ident), complex_column, float(complex_cutoff_loss), complex_ml_method)

                # Visualize the output (the return values) of the complexity function
                st.markdown("""
                Output of the complexity part:
                """)
                st.write(intrinsic_dimension)
                st.write(best_results)
                st.write(df_summary)
            except:
                st.markdown("""
                Algorithm Error! You should restart the app.
                """)
        else:
            st.markdown("""
            You have not chosen this task.
            """)
    else:
        st.markdown("""
        You have to set all options and columns.
        """)