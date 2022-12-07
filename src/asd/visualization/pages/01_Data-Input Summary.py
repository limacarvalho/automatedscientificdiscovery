import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import io

st.set_page_config(layout="wide")

df_input = st.session_state["df_input"]
st.title("Data summary of the input data")
st.subheader("Your chosen dataset displayed as a table:")
st.dataframe(df_input)
st.markdown("")
st.markdown("")
st.subheader("Basic Analysis:")
     
tab1, tab2, tab3, tab4 = st.tabs(["Data information", "Basic statistics", "NA values", "Duplicated values"])

with tab1:
        buffer = io.StringIO()
        df_input.info(buf=buffer)
        i = buffer.getvalue()
        st.text(i)
with tab2:
        st.dataframe(df_input.describe())
with tab3:
        NA_val = df_input.isnull().sum().to_frame('NA values')
        NA_val.index.names = ["Column name"]
        st.write(NA_val)
with tab4:
        duplicated_val = df_input.apply(lambda x: x.duplicated()).sum().to_frame('Duplicated values')
        st.write(duplicated_val)
st.markdown("")
st.markdown("")
st.markdown("***")
st.subheader("Exploratory Data Analysis:")
if st.button('Generate EDA-Report'):
        profile=ProfileReport(df_input, minimal=True, explorative=True, title="Uploaded dataset", progress_bar=True) 
        st_profile_report(profile)