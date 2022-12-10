# Import libraries
import streamlit as st
import pandas as pd
import numpy as np

# Set streamlit layout
st.set_page_config(layout="wide")

# Set streamlit session state
if "discovery_type" not in st.session_state:
    st.session_state["discovery_type"] = "Predictability"
df_input = st.session_state["df_input"]

# Implement mainframe output
st.title("Data normalization and machine learning model selection")
st.markdown("")
st.subheader("Your dataset based on your selected normalization:")
st.markdown("")

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
st.dataframe(df_input_changed)

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
    st.markdown("""
    You selected the task Predictability
    """)
    # Integrate variable with connection to ml-task: Predictability
elif st.session_state["discovery_type"] == "Relevance":
    st.markdown("""
    You selected the task Relevance
    """)
    # Integrate variable with connection to ml-task: Relevance
elif st.session_state["discovery_type"] == "Grouping":
    st.markdown("""
    You selected the task Grouping
    """)
    # Integrate variable with connection to ml-task: Grouping
elif st.session_state["discovery_type"] == "Complexity":
    st.markdown("""
    You selected the task Complexity
    """)
    # Integrate variable with connection to ml-task: Complexity