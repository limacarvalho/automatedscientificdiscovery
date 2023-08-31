# Import libraries
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Set streamlit layout
st.set_page_config(
    page_title="ASD - Home",
    page_icon="https://www.ipp.mpg.de/assets/touch-icon-32x32-a66937bcebc4e8894ebff1f41a366c7c7220fd97a38869ee0f2db65a9f59b6c1.png",
    layout="wide",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

# @st.cache(allow_output_mutation=True)

# Implement mainframe output
st.title("Automated Scientific Discovery (ASD)")

# Please, insert the project summary
st.markdown(
    """
---

![MPI IPP Logo](https://www.ipp.mpg.de/assets/institutes/headers/ipp-desktop-en-8eddb93380e5adb577545e12fd93ab4e2892c2e489e109e2607a478628aca8f2.svg)

Automated Scientific Discovery is an open-source Machine Learning/Bayesian Statistics tool aimed at discovering predictability and data relationships in scientific databases/datasets. This tool is designed for scientists without AI domain knowledge and
focuses on a limited signal domain (scientific signals/parameters) and a limited purpose (finding predictability).

## Features

-   Utilizes MLP neural networks, Kernel density estimation, and Gaussian processes
-   Employs Bayesian methods for optimizing hyperparameters and calculating uncertainties
-   Novelty detection for analyzing points and detecting novel features
-   Sensitivity analysis for identifying relevant predictive parameters
-   Built upon widely used tools such as Ray.io, Torch, XGBoost, Streamlit, Sklearn, Scipy, Optuna, etc
-   Tested on datasets from nuclear fusion and climate science

## Howto
-   TBD

## Contributing

We welcome contributions to improve and extend the project. To see the list of core members and start contributing, please visit https://github.com/limacarvalho/automatedscientificdiscovery

## License

This project is licensed under the MIT License.
"""
)
st.markdown("***")