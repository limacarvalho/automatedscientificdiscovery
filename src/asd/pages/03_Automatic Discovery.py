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
####from predictability.utils import get_column_combinations
####from predictability.src.ASD_predictability_utils.utils import get_column_combinations
####from predictability.bin.main import predictability
####from predictability.src.ASD_predictability_utils.utils import plot_result
#from complexity.dim_reduce import dimreduce_main
#from xai import relevance


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
#df_input = st.session_state["df_input"]


# Print normalized dataframe based on user selection
st.dataframe(df_input.head())
st.markdown("***")

# Create function for sidebar selection: discovery task
def handle_ml_select():
    if st.session_state.ml_type:
        st.session_state.discovery_type=st.session_state.ml_type
        #st.empty()

# Set selection of discovery task on sidebar
ml_select = st.sidebar.radio("Discovery task:", ["Predictability", "Relevance", "Grouping", "Complexity"], on_change=handle_ml_select, key="ml_type")

# Implement if statements based on discovery task selection
if st.session_state["discovery_type"] == "Predictability":
    st.header("Discover the predictability of your data")
    st.markdown("""
    To begin the predictability task, you have to choose between some options.   
    """)
    predict_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    pred_target_column = st.multiselect("Target columns:", predict_num_columns, help="The subset of columns that should be treated exclusively as targets.")
    pred_target_column_count = len(pred_target_column)
    if len(pred_target_column) == 0:
        pred_target_column = "None"    
    pred_primekey_cols = st.multiselect("Primekey columns:", predict_num_columns, help="The subset of columns corresponding to primary keys. These will neither be used as inputs nor outputs.")
    if len(pred_primekey_cols) == 0:
        pred_primekey_cols = "None"    
    pred_col_set = st.multiselect("Column set:", predict_num_columns, help="The (sub-)set of columns that should be considered.")
    if len(pred_col_set) == 0:
        pred_col_set = "None"
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
        pred_comb_start = st.button("Get combinations")

        if pred_comb_start == True:
            ###### Implement tbe code of Predictability ######
            pred_target_column_ext = pred_target_column
            if pred_target_column == "None":
                pred_target_column_ext = []
            get_column_combinations = asdpu.get_column_combinations(all_cols=df_input.columns, inputs=pred_input_column, outputs=pred_output_column, targets=pred_target_column_ext, amount_only=False, return_targets=False)
            
            # printed dataframe based on selected targets
            st.write(get_column_combinations)
            # Integrate variable with connection to ml-task: Predictability
        else:
            st.markdown("""
            You have not chosen this task.
            """)
    
        ###### Part 2, Predictability: predictability function of ml-algorithm ######  
        st.markdown("***")
        st.subheader("Predictability")
        st.markdown("""
        Discover the predictability of your dataset.
        """)
        pred_algorithm_start = st.button("Start discovery")

        if pred_algorithm_start == True:
            try:
                ###### Implement tbe code of Predictability ######
                metrics_dict, datas_dict = asdpc.run_predictability(data=df_input, input_cols=pred_input_column, output_cols=pred_output_column, col_set=pred_col_set, primkey_cols=pred_primekey_cols, targets=pred_target_column, method=pred_ml_method, greedy=pred_greedy, refined_n_best=pred_refined_n_best)
                st.spinner(text="Calculation in progress...")

                #if metrics_dict and datas_dict:
                # or button !!!
                #metrics_dict, datas_dict = asdpu.predictability(data=df_input_changed, input_cols=pred_input_column, output_cols=pred_output_column, col_set=None, targets=pred_target_column, method=pred_ml_method, random_state_split=None, #refined=True, greedy=pred_greedy)
                # rause pred_metrics = pd.DataFrame.from_dict(metrics_dict).transpose()
                #run_predictability
                #plot_along
                #pred_output = asdpu.plot_result(datas_dict, list(datas_dict.keys())[0], plot_along=["linear", "mean"])
                #st.session_state["pred_output"] = pred_output
                
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
                #plot_result = asdpu.plot_result(input_datas_dict=datas_dict, plot_comb=struc_dict, plot_along=pred_plot_along)            
                st.write(asdpu.plot_result(input_datas_dict=datas_dict, plot_comb=struc_dict, plot_along=pred_plot_along))
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

elif st.session_state["discovery_type"] == "Relevance":
    st.header("Discover the relevance of your data")
    st.markdown("""
    To begin the relevance task, you have to choose between some options.   
    """)
    relevance_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    relevance_num_columns_without = df_input.select_dtypes(include = np.number).columns.to_list()
    relevance_num_columns.insert(0, "All")
    relevance_column = st.multiselect("Numeric input columns:", relevance_num_columns)#[st.session_state["predict_target_column"]])
    relevance_target = st.selectbox("Numeric target column:", relevance_num_columns_without)#[st.session_state["predict_target_column"]])
    
    # If it is not a list
    #df_input_ext = df_input[relevance_column]
 
    # Additional part
    #relevance_xgb_objective = st.checkbox('Use XGBoost model')
    #relevance_lgbm_objective = st.checkbox('Use lightgbm model')
    relevance_pred_class = st.selectbox("Modeling task:", ("regression", "classification"), help="Specify problem type, i.e., 'regression' or 'classification'.")
    relevance_list_base_models = st.multiselect("Base models:", ('All', 'briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf'), help="List of base models to be used to fit on the data.")
    relevance_n_trials = st.slider("Sampled parameter settings:", min_value=0, max_value=300, help="Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.")
    relevance_boosted_round = st.slider("N-estimators:", min_value=0, max_value=300, help="N_estimators parameter for XGBoost and LightGBM.")    
    relevance_max_depth = st.slider("Max tree depth:", min_value=0, max_value=100, help="Max tree depth parameter for XGBoost, LightGBM and RandomForest.") 
    relevance_rf_n_estimators = st.slider("N-estimators, RandomForest:", min_value=0, max_value=3000, help="N_estimators parameter of RandomForest.") 
    relevance_bagging_estimators = st.slider("Bagging estimators:", min_value=0, max_value=300, help="Bagging estimators.")
    relevance_n_neighbors = st.slider("NNeighbors of KNN:", min_value=0, max_value=100, help="N-Neighbors of KNN.")
    relevance_cv_splits = st.slider("Cross-validation splits:", min_value=0, max_value=20, help="Determines the cross-validation splitting strategy.")
    relevance_ensemble_bagging_estimators = st.slider("Ensemble bagging estimators:", min_value=0, max_value=100, help="N_estimators parameter of Bagging. This is the second baggin method which is used an an ensemble on top of base estimators.")
    relevance_ensemble_n_trials = st.slider("Ensemble-n-trials:", min_value=0, max_value=100, help="Number of parameter settings that are sampled. n_trials trades off runtime vs quality of the solution.")
    relevance_attr_algos = st.multiselect("Xai:", ('All', 'IG', 'SHAP', 'GradientSHAP', 'knockoffs'), help="Xai methods.")
    relevance_fdr = st.slider("Fdr:", min_value=0.0, max_value=1.0, help="Target false discovery rate.")
    relevance_fstats = st.multiselect("Fstats methods:", ('All', 'lasso', 'ridge', 'randomforest'), help="Methods to calculate fstats.")
    relevance_knockoff_runs = st.slider("Knockoff runs:", min_value=0, max_value=100000, help="Number of reruns for each knockoff setting.")    
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

    relevance_options = {'xgb_objective': 'binary:logistic', 'lgbm_objective': 'binary', 'pred_class': str(relevance_pred_class), 'score_func': None, 'metric_func': None, 'list_base_models': relevance_list_base_models, 'n_trials': int(relevance_n_trials), 'boosted_round': int(relevance_boosted_round), 'max_depth': int(relevance_max_depth), 'rf_n_estimators': int(relevance_rf_n_estimators), 'bagging_estimators': int(relevance_bagging_estimators), 'n_neighbors': int(relevance_n_neighbors), 'cv_splits': int(relevance_cv_splits), 'ensemble_bagging_estimators': int(relevance_ensemble_bagging_estimators), 'ensemble_n_trials': int(relevance_ensemble_n_trials), 'attr_algos': relevance_attr_algos, 'fdr': float(relevance_fdr), 'fstats': relevance_fstats, 'knockoff_runs': int(relevance_knockoff_runs)}

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
    # Integrate variable with connection to ml-task: Relevance
elif st.session_state["discovery_type"] == "Grouping":
    st.markdown("""
    You selected the task: Grouping
    """)
    # Integrate variable with connection to ml-task: Grouping
elif st.session_state["discovery_type"] == "Complexity":
    st.header("Discover the complexity of your data")
    st.markdown("""
    To begin the complexity task, you have to choose between some options.   
    """)
    complex_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    complex_column = st.multiselect("Numeric columns:", complex_num_columns)#[st.session_state["predict_target_column"]])
    complex_ident = st.text_input('Identifier:', placeholder="Enter an identifier")
    complex_cutoff_loss = st.slider("Cutoff loss", min_value=0.0, max_value=1.0)
    complex_ml_method = st.selectbox("ML-method:", ("All","py_pca", "py_pca_sparse", "py_pca_incremental", "py_truncated_svd", "py_crca", "py_sammon", "r_adr", "r_mds", "r_ppca", "r_rpcag", "r_ispe"))
    complex_data_high = df_input
    if complex_ml_method == "All":
        complex_ml_method = []

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
                #dimreduce_main.data_id = complex_ident
                #dimreduce_main.columns = complex_column
                #dimreduce_main.data_high = complex_data_high
                #dimreduce_main.functions = complex_ml_method 
                #dimreduce_main.cutoff_loss = complex_cutoff_loss
                #intrinsic_dimension, best_results, df_summary = dimreduce_main.complexity(data_high, data_id, columns, cutoff_loss, functions)

                # Visualize the output (the return values) of the complexity function
                st.markdown("""
                Output of the complexity part:
                """)
                st.write(dimreduce_main.intrinsic_dimension)
                st.write(dimreduce_main.best_results)
                st.write(dimreduce_main.df_summary)
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