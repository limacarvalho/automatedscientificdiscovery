# Import libraries
import streamlit as st
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()

# Set streamlit layout
st.set_page_config(
    page_title="ASD - Dashboards",
    page_icon="https://www.ipp.mpg.de/assets/touch-icon-32x32-a66937bcebc4e8894ebff1f41a366c7c7220fd97a38869ee0f2db65a9f59b6c1.png",
    layout="wide",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)
