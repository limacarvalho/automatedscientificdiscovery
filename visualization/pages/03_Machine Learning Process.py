import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(layout="wide")

df_input = st.session_state["df_input"]
st.title("Data normalization and machine learning model selection")
st.markdown("")
st.subheader("Your dataset based on your selected normalization:")
#st.dataframe(df_input)
st.markdown("")
st.sidebar.markdown("""
Which kind of useless data do you want to delete?
""")
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

st.dataframe(df_input_changed)

def handle_click_without():
    #if st.session_state.column_kind:
    #    st.session_state.type=st.session_state.column_kind
        st.empty()

st.sidebar.markdown("***")
column_type = st.sidebar.radio("Which kind of machine learning task do you prefer?", ["Complexity", "Predictability", "Auto-ML"], on_change=handle_click_without, key="column_kind")