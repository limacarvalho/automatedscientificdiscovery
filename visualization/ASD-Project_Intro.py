# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import io

# Set streamlit layout
st.set_page_config(layout="wide")

#@st.cache(allow_output_mutation=True)

# Implement mainframe output
st.title("Automated-Scientific-Discovery Project App")

# Please, insert the project summary
st.markdown("""
This app helps you with the automatic evaluation of your data. To do this, you must upload your data and select a preferred machine-learning method under the "Machine Learning Process" menu item.
""")
st.markdown("***")