import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
df_input = st.session_state["df_input"]

if "type" not in st.session_state:
    st.session_state["type"] = "Numerical"

# Pandas implementation for selection of categorical or continous data
num_cols = df_input.select_dtypes(include = np.number).columns.to_list()
num_cols_cat = df_input.select_dtypes(include = np.number).copy(deep=True)
cat_cols = df_input.select_dtypes(exclude = np.number).columns.to_list()
    
def handle_click_without():
    if st.session_state.column_kind:
        st.session_state.type=st.session_state.column_kind
        st.empty()

thre = st.sidebar.slider("Choose a threshold to classify categories:", 0, 500)
for col in num_cols_cat.columns:
    if num_cols_cat[col].nunique() < thre:
        num_cols_cat[col] = num_cols_cat[col].astype('category')
num_cols_cat1 = num_cols_cat.select_dtypes(include = "category").columns.to_list()
types = {"Numerical": num_cols, "Categorical":cat_cols, "Categorical by selection":num_cols_cat1}
    
st.sidebar.markdown("***")
column_type = st.sidebar.radio("Which kind of analysis do you prefer?", ["Numerical", "Categorical", "Categorical by selection"], on_change=handle_click_without, key="column_kind")
    
    
unique_counts = pd.DataFrame.from_records([(c, num_cols_cat[c].nunique()) for c in num_cols_cat.columns],
                        columns=['Column', 'Qunatity_of_unique_numbers']).sort_values(by=['Qunatity_of_unique_numbers'])
    
st.markdown("""
To classify the data, you have to set the threshold of unique numbers:
""")
st.dataframe(unique_counts)
st.markdown("***")
    
column = st.sidebar.multiselect("Please, select your columns:", types[st.session_state["type"]])