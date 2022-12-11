# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
#from predictability.src.ASD_predictability_utils.utils import get_column_combinations
#from predictability.bin.main import predictability
#from predictability.src.ASD_predictability_utils.utils import plot_result

# Set streamlit layout
st.set_page_config(layout="wide")

# Set streamlit session state
if "discovery_type" not in st.session_state:
    st.session_state["discovery_type"] = "Predictability"
df_input = st.session_state["df_input"]

# Implement mainframe output
st.title("Data normalization and discovery")
st.markdown("")
st.header("Data normalization")
st.markdown("""
Your normalized data:
""")

# Implement sidebar output
st.sidebar.markdown("""
Which kind of useless data do you want to delete?
""")

# Implement sidebar user input: checkbox
delete_dupl = st.sidebar.checkbox('Delete duplicates')
delete_null = st.sidebar.checkbox('Delete null values')
df_input_changed = pd.DataFrame(df_input)

# Drop duplicated values
if delete_dupl == True:
    df_input_changed = df_input_changed.drop_duplicates(keep=False)
    st.session_state["df_input_changed"] = df_input_changed

# Replace empty column with numpy NAN
if delete_null == True:
    df_input_changed = df_input_changed.replace(r'^\s*$', np.nan, regex=True)
    st.session_state["df_input_changed"] = df_input_changed

# Print normalized dataframe based on user selection
st.dataframe(df_input_changed.head())
st.markdown("***")

# Create function for sidebar selection: discovery task
def handle_ml_select():
    if st.session_state.ml_type:
        st.session_state.discovery_type=st.session_state.ml_type
        #st.empty()

# Set selection of discovery task on sidebar
st.sidebar.markdown("***")
ml_select = st.sidebar.radio("Discovery task:", ["Predictability", "Relevance", "Grouping", "Complexity"], on_change=handle_ml_select, key="ml_type")

# Implement if statements based on discovery task selection
if st.session_state["discovery_type"] == "Predictability":
    st.header("Discover the predictions")
    st.markdown("""
    To begin the predictability task, you have to choose between some options.   
    """)
    predict_num_columns = df_input.select_dtypes(include = np.number).columns.to_list()
    pred_target_column = st.multiselect("Numeric target columns:", predict_num_columns)#[st.session_state["predict_target_column"]])
    pred_input_column = st.multiselect("Numeric input columns:", predict_num_columns)#[st.session_state["predict_target_column"]])
    pred_output_column = st.multiselect("Numeric output columns:", predict_num_columns)#[st.session_state["predict_target_column"]])
    pred_ml_method = st.selectbox("ML-method:", ("kNN", "linear", "mean", "pow. law"))
    pred_greedy = st.checkbox('Use greedy algorithm')

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
        # Implement tbe code 
        #get_column_combinations(all_cols=df_input_changed.columns, inputs=pred_input_column, outputs=pred_output_column, targets=pred_target_column)
        
        # printed dataframe based on selected targets
        st.dataframe(df_input[pred_target_column])
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
        # Implement tbe code of Predictability
        #metrics_dict, datas_dict = predictability(data=df_input_changed, input_cols=pred_input_column, output_cols=pred_output_column, col_set=None, targets=pred_target_column, method=pred_ml_method, random_state_split=None, #refined=True, greedy=pred_greedy)
        #pred_metrics = pd.DataFrame.from_dict(metrics_dict).transpose()
        #pred_output = plot_result(datas_dict, list(datas_dict.keys())[0], plot_along=["linear", "mean"])
        #st.session_state["pred_output"] = pred_output
        st.markdown("""
        Your outputs of the predictability part are saved.
        """)
    else:
        st.markdown("""
        You have not chosen this task.
        """)
elif st.session_state["discovery_type"] == "Relevance":
    st.markdown("""
    You selected the task: Relevance
    """)
    # Integrate variable with connection to ml-task: Relevance
elif st.session_state["discovery_type"] == "Grouping":
    st.markdown("""
    You selected the task: Grouping
    """)
    # Integrate variable with connection to ml-task: Grouping
elif st.session_state["discovery_type"] == "Complexity":
    st.markdown("""
    You selected the task: Complexity
    """)
    # Integrate variable with connection to ml-task: Complexity