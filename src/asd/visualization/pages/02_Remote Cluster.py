import streamlit as st
import pandas as pd
import io

st.set_page_config(layout="wide")

df_input = st.session_state["df_input"]
st.title("Create remote cluster")

cluster_name = st.text_input('Cluster name:', placeholder="Enter a name")
st.write('Your cluster name is:', cluster_name)
st.markdown("")
access_key = st.text_input('Access key:', placeholder="Enter an access key")
st.write('Your access key is:', access_key)
st.markdown("")
secret_access_key = st.text_input('Secrect access key:', placeholder="Enter a scecret access key")
st.write('Your secret access key is:', secret_access_key)