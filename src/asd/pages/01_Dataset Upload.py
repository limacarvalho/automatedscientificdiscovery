# Standard library imports
import csv
import io
import random
from pathlib import Path
from typing import NoReturn

# Third party imports
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()


def generate_csv(size_mb: int) -> NoReturn:
    """
    Generate a CSV file of a specified size in megabytes with randomly generated data,
    including some duplicated values for realism.

    The CSV file has rows of 10 cells, each cell with a floating point number of 17 digits.

    Parameters:
    size_mb (int): The size of the CSV file to be generated, in megabytes.

    Returns:
    NoReturn: This function doesn't return anything; it writes to a CSV file.

    """
    # The row_size is calculated by taking into account that each row has 10 cells with 17 digits and a comma separator
    row_size = 10 * 17 + 10
    rows = int(size_mb * 1024 * 1024 / row_size)
    file_name = f"{size_mb}mb_data.csv"
    last_row = []

    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # For each row, generate a list of 10 random floating point numbers each having 17 digits
        random_int_for_duplicates = random.randint(2, 5)
        for i in range(rows):
            if (
                i % random_int_for_duplicates == 0
            ):  # A random integer between 2 and 5 is used to calculate which row is duplicated for some realism
                data = last_row
            else:
                data = [str(random.uniform(-999999999999999.0, 999999999999999.0)) for _ in range(10)]
                last_row = data

            # Write the generated data to the CSV file
            writer.writerow(data)
        logging.debug(f"### Randomly generated numerical data written in {file_name} ###")


# Set streamlit layout
st.set_page_config(
    page_title="ASD - Dataset Upload",
    page_icon="https://www.ipp.mpg.de/assets/touch-icon-32x32-a66937bcebc4e8894ebff1f41a366c7c7220fd97a38869ee0f2db65a9f59b6c1.png",
    layout="wide",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

# Current OS Path/Directory of the script
main_asd_path = Path(__file__).parents[1]

# Implement mainframe output
st.title("Summary of the input data")
st.markdown("")
st.markdown("")

# Implement sidebar output
st.sidebar.header("Upload CSV dataset:")

# Implement sidebar upload button
file_upload = st.sidebar.file_uploader("Please, load your .csv file", type=["csv"])

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
    st.write(
        "The automatically loaded dataset contains randomly generated numerical data for demo purposes. Please upload your .csv file, using the right-menu section 'Upload CSV dataset'."
    )
    st.write("")
    st.write("")
    # Generates a random numerical dataset (~ approximately 10 MB in size)
    generate_csv(10)
    df_input = pd.read_csv("10mb_data.csv")
    st.session_state["df_input"] = df_input
    st.write("As a brief overview, the default dataset (randomly generated) is displayed:")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data information", "Basic statistics", "NA values", "Duplicated values", "Report"]
)

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
