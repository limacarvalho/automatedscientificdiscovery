# import libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")

#@st.cache(allow_output_mutation=True)

st.title("Automated-Scientific-Discovery Project App")
st.markdown("""
This app helps you with the automatic evaluation of your data. To do this, you must upload your data and select a preferred machine-learning method under the "Machine Learning Process" menu item.
""")
st.markdown("***")

st.sidebar.header("Data selection:")
st.sidebar.markdown("""
[The default dataset contains Covid 2020 data](https://gitlab.com/automatedscientificdiscovery/datasets/-/blob/Gtomi-main-patch-17962/covid/20220727_covid_159rows_52cols_2020.csv)
""")

file_upload = st.sidebar.file_uploader("Please, open your .csv file", type=["csv"])

#if "df_input" not in st.session_state:
if file_upload is not None:
    st.write("Your chosen dataset is used.")
    st.write("")
    st.write("")
    df_input = pd.read_csv(file_upload)
    st.session_state["df_input"] = df_input        
    st.write("As a brief overview, here are the first three lines of your dataset:")
    st.dataframe(df_input.head(3))
else:
    st.write("The default dataset (Covid) is used. If you want a different dataset, please upload a .csv file.")
    st.write("")
    st.write("")
    df_input = pd.read_csv(r"datasets/20220727_covid_159rows_52cols_2020.csv")
    st.session_state["df_input"] = df_input
    st.write("As a brief overview, here are the first three lines of default dataset (Covid):")  
    st.dataframe(df_input.head(3))
