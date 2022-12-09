# Import libraries
import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import io

# Set streamlit layout
st.set_page_config(layout="wide")

# Set streamlit session state
if "df_input" not in st.session_state:
    df_input = pd.read_csv(r"datasets/20220727_covid_159rows_52cols_2020.csv")    
    st.session_state["df_input"] = df_input

# Implement mainframe output
st.title("Data summary of the input data")
st.markdown("")
st.markdown("")

# Implement sidebar user input
st.sidebar.header("Data selection:")
st.sidebar.markdown("""
[The default dataset contains Covid 2020 data](https://gitlab.com/automatedscientificdiscovery/datasets/-/blob/Gtomi-main-patch-17962/covid/20220727_covid_159rows_52cols_2020.csv)
""")

# Implement sidebar upload button
file_upload = st.sidebar.file_uploader("Please, open your .csv file", type=["csv"])

# Implement if statement based on file uploader
if file_upload is not None:
    st.write("Your chosen dataset is used.")
    st.write("")
    st.write("")
    df_input = pd.read_csv(file_upload)
    st.session_state["df_input"] = df_input        
    st.write("As a brief overview, your dataset si displayed:")
    st.dataframe(df_input)
    st.markdown("")
    st.markdown("")
else:
    st.write("The default dataset (Covid) is used. If you want a different dataset, please upload a .csv file.")
    st.write("")
    st.write("")
    df_input = pd.read_csv(r"datasets/20220727_covid_159rows_52cols_2020.csv")
    st.session_state["df_input"] = df_input
    st.write("As a brief overview, the default dataset (Covid) is displayed:")  
    st.dataframe(df_input)
    st.markdown("")
    st.markdown("")

# Implement mainframe output
st.subheader("Basic Analysis:")

# Implement tabs with different calculations     
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data information", "Basic statistics", "NA values", "Duplicated values", "Report"])

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
with tab5:
        profile=ProfileReport(df_input, minimal=True, explorative=True, title="Uploaded dataset", progress_bar=True) 
        st_profile_report(profile)