# Import libraries
import streamlit as st
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()

# Set streamlit layout
st.set_page_config(
    page_title="ASD - JupyterLab",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/883px-Jupyter_logo.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

# Define the external URL
if st.button("Start Jupyter", type="primary"):
    st.markdown(
        '<meta http-equiv="refresh" target="_blank" content="0;url=http://localhost:8888/" />', unsafe_allow_html=True
    )
