# Standard library imports
import io

# Third party imports
import pandas as pd
import streamlit as st
from pathlib import Path
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Set streamlit layout
st.set_page_config(layout="wide")

# Current OS Path/Directory of the script
main_asd_path = Path(__file__).parents[1]

# Implement mainframe output
st.title("Data summary of the input data")
st.markdown("")
st.markdown("")

# Implement sidebar output
st.sidebar.header("Data selection:")

# Implement sidebar upload button
file_upload = st.sidebar.file_uploader("Please, open your .csv file", type=["csv"])

# Implement if statements based on file uploader
if file_upload is not None:
    st.write("Your chosen dataset is used.")
    st.write("")
    st.write("")
    df_input = pd.read_csv(file_upload)
    st.session_state["df_input"] = df_input
    st.write("As a brief overview, your dataset is displayed:")
    st.dataframe(df_input)
    st.markdown("")
    st.markdown("")
elif "df_input" not in st.session_state:
    st.write("The default dataset contains Covid 2020 data. If you want a different dataset, please upload a .csv file.")
    st.write("")
    st.write("")
    df_input = pd.read_csv(f"{main_asd_path}/datasets/20220727_covid_159rows_52cols_2020.csv")
    st.session_state["df_input"] = df_input
    st.write("As a brief overview, the default dataset (Covid) is displayed:")
    st.dataframe(df_input)
    st.markdown("")
    st.markdown("")
elif "df_input" in st.session_state:
    df_input = st.session_state["df_input"]
    st.write("Your chosen dataset is used.")
    st.write("")
    st.write("")
    st.write("As a brief overview, your dataset is displayed:")
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
    NA_val = df_input.isnull().sum().to_frame("NA values")
    NA_val.index.names = ["Column name"]
    st.write(NA_val)
with tab4:
    duplicated_val = df_input.apply(lambda x: x.duplicated()).sum().to_frame("Duplicated values")
    st.write(duplicated_val)
with tab5:
    profile = ProfileReport(df_input, minimal=True, explorative=True, title="Uploaded dataset", progress_bar=True)
    st.button("Generate Report", on_click=st_profile_report, args=[profile])
