from pathlib import Path
import datatable as dt
import pandas as pd
from utilities_asd_team  import convert_datatypes

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 150) # print: change width of display
pd.options.display.max_colwidth = 100 # print: max width column
import warnings
warnings.filterwarnings('ignore')


## read data table
base_path = Path(__file__).parent
print(base_path)
# HDB5.2.3_nans.csv,
path = (base_path /'test/data/HDB5.2.3_no_nans.csv').resolve()
data = dt.fread(path).to_pandas()

## important: convert datatypes first!
data = convert_datatypes(data)
print('success:', data.shape, '\n', data.dtypes)


# ## IMPORTANT PARAMETERS!! that are the parameters numbers that matter
# TAUTH * TAUC93, Estimated thermal energy confinement time in seconds * Correction factor for thermal confinement time
# WTOT Estimated total plasma energy content in Joules.
# WTH Estimated thermal plasma energy content in Joules.
# PL Estimated Loss Power not corrected for charge exchange and unconfined orbit losses in watts.
# PLTH Estimated Loss Power corrected for charge exchange and unconfined orbit losses in Watts


## your code