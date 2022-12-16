# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


import predictability.utils as asdpu
import predictability.core as asdpc

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
                # oder button !!!
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
                plot_result = asdpu.plot_result(input_datas_dict=datas_dict, plot_comb=struc_dict, plot_along=pred_plot_along)            
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
    relevance_column = st.multiselect("Numeric input columns:", relevance_num_columns)#[st.session_state["predict_target_column"]])
    relevance_target = st.selectbox("Numeric target column:", relevance_num_columns)#[st.session_state["predict_target_column"]])
    
    # Additional part
    #relevance_xgb_objective = st.checkbox('Use XGBoost model')
    #relevance_lgbm_objective = st.checkbox('Use lightgbm model')
    relevance_pred_class = st.selectbox("Predicitve Modeling task:", ("regression", "classification"))
    relevance_list_base_models = st.selectbox("Base models:", ('briskbagging', 'briskknn', 'briskxgboost', 'slugxgboost', 'sluglgbm','slugrf'))
    relevance_n_trials = st.slider("Sampled parameter settings:", min_value=0, max_value=300)
    relevance_boosted_round = st.slider("N-estimators parameter for XGBoost and LightGBM:", min_value=0, max_value=300)    
    relevance_max_depth = st.slider("Max tree depth parameter for XGBoost, LightGBM and RandomForest:", min_value=0, max_value=100) 
    relevance_rf_n_estimators = st.slider("N-estimators parameter of RandomForest:", min_value=0, max_value=3000) 
    relevance_bagging_estimators = st.slider("Bagging estimators:", min_value=0, max_value=300)
    relevance_n_neighbors = st.slider("NNeighbors of KNN:", min_value=0, max_value=100)
    relevance_cv_splits = st.slider("Determine the cross-validation splitting strategy:", min_value=0, max_value=20)
    relevance_ensemble_bagging_estimators = st.slider("N-estimators parameter of Bagging:", min_value=0, max_value=100)
    relevance_ensemble_n_trials = st.slider("Number of parameter settings that are sampled:", min_value=0, max_value=100)
    relevance_attr_algos = st.selectbox("Xai methods:", ('IG', 'SHAP', 'GradientSHAP', 'knockoffs'))
    relevance_fdr = st.slider("Number of parameter settings that are sampled:", min_value=0.0, max_value=1.0)
    relevance_fstats = st.selectbox("Fstats methods:", ('lasso', 'ridge', 'randomforest'))
    relevance_knockoff_runs = st.slider("Number of reruns for each knockoff setting:", min_value=0, max_value=100000)    

    #relevance_ident = st.text_input('Identifier:', placeholder="Enter an identifier")
    #relevance_cutoff_loss = st.slider("Cutoff loss", min_value=0.0, max_value=1.0)
    #relevance_ml_method = st.selectbox("ML-method:", ("All","py_pca", "py_pca_sparse", "py_pca_incremental", "py_truncated_svd", "py_crca", "py_sammon", "r_adr", "r_mds", "r_ppca", "r_rpcag", "r_ispe"))
    relevance_df = df_input
    #if relevance_ml_method == "All":
    #    relevance_ml_method = []

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
                #relevance.relevance(relevance_df, relevance_column, relevance_target)
                

                # Visualize the output (the return values) of the relevance function
                st.markdown("""
                Output of the relevance part:
                """)
                #st.write(dimreduce_main.intrinsic_dimension)
                #st.write(dimreduce_main.best_results)
                #st.write(dimreduce_main.df_summary)
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