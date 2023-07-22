# Import libraries
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Set streamlit layout
st.set_page_config(layout="wide")

# @st.cache(allow_output_mutation=True)

# Implement mainframe output
st.title("Automated Scientific Discovery (ASD)")

# Please, insert the project summary
st.markdown(
    """
---

![MPI IPP Logo](https://www.ipp.mpg.de/assets/institutes/headers/ipp-desktop-de-5b77946a9fe513bfee29e3020802db2cab74d92e920731557a284e1ef9261788.svg)

Automated Scientific Discovery is an open-source Machine Learning/Bayesian Statistics tool aimed at discovering predictability and data relationships in scientific databases/datasets. This tool is designed for scientists without AI domain knowledge and
focuses on a limited signal domain (scientific signals/parameters) and a limited purpose (finding predictability).

## Features

-   Utilizes MLP neural networks, Kernel density estimation, and Gaussian processes
-   Employs Bayesian methods for optimizing hyperparameters and calculating uncertainties
-   Novelty detection for analyzing points and detecting novel features
-   Sensitivity analysis for identifying relevant predictive parameters
-   Built upon widely used tools such as Ray.io, Torch, XGBoost, Streamlit, Sklearn, Scipy, Optuna, etc
-   Tested on datasets from nuclear fusion and climate science

## Contributing

We welcome contributions to improve and extend the project. To see the list of core members and start contributing, please visit https://github.com/limacarvalho/automatedscientificdiscovery

## License

This project is licensed under the MIT License.
"""
)
st.markdown("***")