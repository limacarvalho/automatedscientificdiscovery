import numpy as np
import pandas as pd


def nan_content_column(df_col: pd.DataFrame) -> float:
    '''
    calculates the NaN content (in percent) per polumn.
    :param df_col: column of dataframe
    :return: float, percent NaN values in column
    '''
    le = df_col.shape[0]
    df_col = pd.DataFrame(df_col)
    colname = df_col.columns.tolist()[0]

    # this is insane! isinstance(df_col, object) and type(df_col) are not working
    if df_col[colname].dtypes == object:
        rows_nans = df_col[df_col[colname] == 'NaN']
        try:
            nans = round(rows_nans.shape[0] / le * 100, 1)
        except:
            nans = -1
    else:
        n_nans = df_col[colname].isnull().sum()
        try:
            nans = round(n_nans / le * 100, 1)
        except:
            nans = -1
    return nans


# float columns
floats = ['TIME', 'T1', 'T2', 'ELMFREQ', 'ELMMAX', 'ELMDUR', 'ELMINT', 'OLTIME', 'LHTIME',
     'MEFF', 'XGASA', 'PGASA', 'FUELRATE', 'XGASA', 'RGEO', 'RMAG', 'AMIN', 'KAPPA', 'KAPPAA',
     'KAREA', 'VOL', 'DELTA', 'DELTAU', 'DELTAL', 'INDENT', 'AREA', 'SURFFORM', 'SEPLIM', 'XPLIM',
     'DALFMP', 'DALFDV', 'BT', 'VSURF', 'Q95', 'SH95', 'BEILI2', 'BEIMHD', 'BEPMHD', 'BETMHD', 'BEPDIA',
     'TAUCR', 'RHOINV', 'ZEFF', 'ZEFFNEO', 'ENBI', 'COCTR', 'SPIN', 'TORQ', 'TORQBM', 'TORQIN', 'WFICRHP', 'WFICFORM',
     'WFANIIC', 'TAUDIA', 'TAUMHD', 'TAUTH1', 'TAUTH2', 'TAUTOT', 'TAUTH', 'TAUC92', 'TAUC93', 'H89', 'HITER96L',
     'H93', 'HITER92Y', 'HEPS97', 'HIPB98Y', 'HIPB98Y1', 'HIPB98Y2', 'HIPB98Y3', 'HIPB98Y4'
    ]
# ineger columns
ints = ['INDEX', 'LCUPDATE', 'DATE', 'SHOT', 'TIME_ID', 'BGASA', 'BGASZ', 'BGASA2', 'BGASZ2', 'XGASZ', 'IP',
     'BMHDMDIA', 'NEL', 'NEV', 'NE0', 'NE0TSC', 'NESEP', 'NESOL', 'PMAIN', 'PRAD', 'POHM', 'PINJ', 'PINJ2', 'BSOURCE2',
     'PNBI', 'PFLOSS', 'ECHFREQ', 'PECRHC', 'ICFREQ', 'PECRH', 'PICRHC', 'PICRH', 'PALPHA', 'DWDIA', 'DWDIAPAR',
     'DWMHD', 'TEV', 'TE0', 'TE0TSC', 'TIV', 'TI0', 'TICX0', 'VTOR0', 'WDIA', 'WMHD', 'WKIN', 'WEKIN', 'WIKIN',
     'WFPER', 'WFPAR', 'WFFORM', 'WFICRH', 'WTOT', 'WTH', 'PL', 'PLTH', 'SELDB1', 'SELDB2', 'SELDB2X'
    ]
# object columns
objects = ['TOK', 'TOK_ID', 'DIVNAME', 'AUXHEAT', 'PHASE', 'HYBRID', 'ITB', 'ITBTYPE', 'ELMTYPE', 'TPI', 'ISEQ',
     'IEML', 'PGASZ',  'PELLET', 'CONFIG', 'WALMAT', 'DIVMAT', 'LIMMAT', 'EVAP', 'IGRADB', 'PREMAG', 'FBS', 'RHOQ2',
     'NELFORM', 'DNELDT', 'PDIV', 'GP_MAIN', 'GP_DIV', 'BSOURCE', 'ECHMODE', 'ECHLOC', 'ICSCHEME', 'ICANTEN',
     'DWHC', 'OMGAIMP0', 'OMGAIMPH', 'OMGAM0', 'OMGAMH', 'VTORV', 'VTORIMP', 'WROT', 'WFANI', 'ICFORM', 'STANDARD',
     'IAEA92', 'DB2P5', 'DB2P8', 'DB3IS', 'DB3V5', 'IAE2000N', 'IAE2000X', 'HMWS2003', 'IAE2004S', 'IAE2004I',
     'DB3DONLY', 'HMWS2005', 'OJK2006', 'SELDB3', 'SELDB3X', 'STD3', 'SELDB4', 'STDDB4V5', 'SELDB5'
    ]



def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    python gets the datatypes wrong especially when empty strings are present.
    A float value column becomes a object column (string) and so on.
    This function converts the datatypes and substitutes all kinds of stupid values
    into NaNs.
    Please feel free to have a look at the original dataset!
    :param df: pd.DataFrame, dataset
    :return df: pd.DataFrame, dataset with correct datatypes
    '''
    for col in df.columns:
        # remove special character such as  b'', there are entire columns filled with it,
        if isinstance(df[col], str):
            df[col] = df[col].str.decode('utf-8')

        # substitute booleans
        if df[col].dtype == bool or df[col].astype(str).str.contains('False|True').any():
            df[col] = df[col].astype(str).replace('True', 1, regex=True)
            df[col] = df[col].astype(str).replace('False', 0, regex=True)
            df[col] = df[col].astype(np.int8, errors='ignore')

        # replace columns with only one value with NaNs
        if len(df[col].unique()) == 1:
            df[col] = 'NaN'

        # NONE means: something was not applied or was not there, so it has a meaning
        df[col] = df[col].replace('NONE      ', 'NONE')

        # substitute empty values with custom value (NaN), please feel free to laugh!
        df[col] = df[col].replace(' ', '')
        df[col] = df[col].replace('???????',   'NaN')
        df[col] = df[col].replace('UNKNOWN   ','NaN')
        df[col] = df[col].replace('UNKNOWN',   'NaN')
        df[col] = df[col].replace('nan',       'NaN')
        df[col] = df[col].replace('', 'NaN')
        df[col] = df[col].fillna('NaN')

        if col in floats:
            df[col] = df[col].astype(np.float16, errors='ignore')
        elif col in objects:
            df[col] = df[col].astype(np.str, errors='ignore')
        # Python refuses to handle columns with ints and NaNs properly.
        # most int parameters will be converted to floats, its at least a number!
        elif col in ints:
            try:
                df[col] = df[col].astype(int, errors='ignore')
            except:
                df[col] = df[col].astype(np.float, errors='ignore')
        else:
            print(col, 'not found in floats, ints and objects')

    return df



def parameter_categories(parameters) -> pd.DataFrame:
    '''
    categories for each parameter. Gives a hint what each parameter is measuring.
    Makes it a bit easier to select the important parameters.
    :param parameters: list, list of strings with column names (df.columns)
    :return: pd.DataFrame, dataframe with one column with category for each parameter
    '''
    col_categories = pd.DataFrame(index=range(0, len(parameters)-1), columns=['CATEGORY'])
    # skip firts column: 'TOK' Tokamaks are the colnames
    categories = {'general': ['TOK', 'TOK_ID', 'DIVNAME', 'LCUPDATE', 'DATE', 'SHOT', 'TIME', 'TIME_ID', 'T1', 'T2',
                              'AUXHEAT', 'PHASE', 'HYBRID', 'ITB', 'ITBTYPE', 'OLTIME', 'LHTIME', 'TPI', 'ISEQ'],

                  'general_elms': ['ELMTYPE', 'ELMFREQ', 'ELMMAX', 'ELMDUR', 'ELMINT'],

                  'plasma_composition': ['MEFF', 'PGASA', 'PGASZ', 'BGASA', 'BGASZ', 'BGASA2', 'BGASZ2', 'PELLET',
                                         'FUELRATE', 'XGASA', 'XGASZ'],

                  'geometry': ['CONFIG', 'RGEO', 'RMAG', 'AMIN', 'KAPPA', 'KAPPAA', 'KAREA', 'DELTA', 'DELTAU',
                               'DELTAL', 'INDENT', 'AREA', 'VOL', 'SURFFORM', 'SEPLIM', 'XPLIM'],

                  'machine_condition': ['WALMAT', 'DIVMAT', 'LIMMAT', 'EVAP', 'DALFMP', 'DALFDV'],

                  'magnetics': ['IGRADB', 'BT', 'IEML', 'PREMAG', 'IP', 'VSURF', 'Q95', 'SH95', 'BEILI2', 'BEIMHD',
                                'BEPMHD', 'BETMHD', 'BEPDIA', 'BMHDMDIA', 'TAUCR', 'FBS', 'RHOQ2', 'RHOINV'],

                  'densities': ['NEL', 'NELFORM', 'DNELDT', 'NEV', 'NE0', 'NE0TSC', 'NESEP', 'NESOL', 'PMAIN',
                                'PDIV', 'GP_MAIN', 'GP_DIV'],

                  'impurities': ['ZEFF', 'ZEFFNEO', 'PRAD', 'POHM', 'ENBI', 'PINJ', 'BSOURCE', 'PINJ2', 'BSOURCE2',
                                 'COCTR', 'PNBI', 'PFLOSS', 'ECHFREQ', 'ECHMODE', 'ECHLOC', 'PECRHC', 'PECRH',
                                 'ICFREQ', 'ICSCHEME', 'ICANTEN', 'PICRHC', 'PICRH', 'PALPHA', 'DWDIA', 'DWDIAPAR',
                                 'DWMHD', 'DWHC'],

                  'input_powers': ['POHM', 'ENBI', 'PINJ', 'BSOURCE', 'PINJ2', 'BSOURCE2', 'COCTR', 'PNBI', 'PFLOSS',
                                   'ECHFREQ', 'ECHMODE', 'ECHLOC', 'PECRHC', 'PECRH', 'ICFREQ', 'ICSCHEME', 'ICANTEN',
                                   'PICRHC', 'PICRH', 'PALPHA', 'DWDIA', 'DWDIAPAR', 'DWMHD', 'DWHC'],

                  'temperatures': ['TEV', 'TE0', 'TE0TSC', 'TIV', 'TI0', 'TICX0'],

                  'plasma_rotation': ['OMGAIMP0', 'OMGAIMPH', 'OMGAM0', 'OMGAMH', 'SPIN', 'TORQ', 'TORQBM',
                                      'TORQIN', 'VTOR0', 'VTORV', 'VTORIMP'],

                  'energies': ['WDIA', 'WMHD', 'WKIN', 'WEKIN', 'WIKIN', 'WROT', 'WFPER', 'WFPAR',
                              'WFFORM', 'WFANI', 'WFICRH', 'WFICRHP', 'WFICFORM', 'ICFORM', 'WFANIIC'],

                  'energy_confinement_times': ['TAUDIA', 'TAUMHD', 'TAUTH1', 'TAUTH2'],

                  'recommended_variables': ['WTOT', 'WTH', 'PL', 'PLTH', 'TAUTOT', 'TAUTH', 'TAUC92', 'TAUC93',
                                            'H89', 'HITER96L', 'H93', 'HITER92Y', 'HEPS97', 'HIPB98Y', 'HIPB98Y1',
                                            'HIPB98Y2', 'HIPB98Y3', 'HIPB98Y4'],

                  'standart_dataset_flags': ['STANDARD', 'SELDB1', 'SELDB2', 'SELDB2X', 'IAEA92', 'DB2P5', 'DB2P8',
                                             'DB3IS', 'DB3V5', 'IAE2000N', 'IAE2000X', 'HMWS2003', 'IAE2004S',
                                             'IAE2004I', 'DB3DONLY', 'HMWS2005', 'OJK2006', 'SELDB3', 'SELDB3X',
                                             'STD3', 'SELDB4', 'STDDB4V5', 'SELDB5']
                  }

    for count, par in enumerate(parameters):
        for key, value in categories.items():
            if par in value:
                col_categories.loc[count,:'CATEGORY'] = key
    return col_categories





'''
############# remove 133 columns (parameters) with NaNs
I plotted the NaN contents and selected the columns manually.
2 criteria:
 1) keep as many entries as possible from the tokamaks with most and most important entries:
    Jet, Asdex upgrade, Asdex, Textor and JFT2M.
 2) keep as many useful parameters as possible

 nii, no important information
 !!, we would loose a lot of rows from important experiments
 **, dataset flags, references to publications, no cientific information
'''
removed_columns = [
    'T1', # nii, start time of window, 100 percent missing for: TDEV TEXTOR TFTR
    'T2', # nii, start time of window, 100 percent missing for: TDEV TEXTOR TFTR
    'HYBRID', # !!
    'ITB', # !!
    'ITBTYPE', # !!
    'ELMTYPE', # !!
    'ELMFREQ', # !!
    'ELMMAX', # !!
    'ELMDUR', # !!
    'ELMINT', # !!
    'OLTIME', # !!
    'LHTIME', # !!
    'TPI', # nii, Time point indicator (ASDEX only).
    'FUELRATE', # !!
    'DELTAU', # !!
    'DELTAL', # !!
    'XPLIM', # !!
    'DALFMP', # !!
    'DALFDV', # !!
    'IGRADB', # !!
    'PREMAG', # !!
    # 'VSURF', # no efect when removed
    'SH95', # !!
    'BEILI2', # !!
    'BEIMHD', # !!
    'BEPMHD', # !!
    'BETMHD', # !!
    'BEPDIA', # !!
    'BMHDMDIA', # !!
    'RMAG', # !!
    'DELTAU', # !!
    'DELTAL', # !!
    # 'SEPLIM', # no efect when removed
    'XPLIM', # !!
    'DALFMP', # !!
    'DALFDV', # !!
    'IEML', # !!
    'PREMAG', # !!
    'SH95', # !!
    'BEILI2', # !!
    'BEIMHD', # !!
    'BEPMHD', # !!
    'BETMHD', # !!
    'BEPDIA', # !!
    'BMHDMDIA', # !!
    'TAUCR', # !!
    'FBS', # !!
    'RHOQ2', # !!
    'RHOINV', # !!
    'NEV', # !!
    'NE0', # !!
    'NE0TSC', # !!
    'NESEP', # !!
    'NESOL', # !!
    'PMAIN', # !!
    'PDIV', # !!
    'GP_MAIN', # !!
    'GP_DIV', # !!
    'ZEFF', # !!
    'ZEFFNEO', # !!
    'PRAD', # !!
    'COCTR', # !!
    'ECHFREQ', # !!
    'ECHMODE', # !!
    'ECHLOC', # !!
    'DWDIA', # !!
    'DWDIAPAR', # !!
    'DWMHD', # !!
    'TEV', # !!
    'TE0', # !!
    'TE0TSC', # !!
    'TIV', # !!
    'TI0', # !!
    'TICX0', # !!
    'OMGAIMP0', # !!
    'OMGAIMPH', # !!
    'OMGAM0', # !!
    'OMGAMH', # !!
    'SPIN', # !!
    'TORQ', # !!
    'TORQBM', # !!
    'TORQIN', # !!
    'VTOR0', # !!
    'VTORV', # !!
    'VTORIMP', # !!
    'WDIA', # !!
    'WMHD', # !!
    'WKIN', # !!
    'WEKIN', # !!
    'WIKIN', # !!
    'WROT', # !!
    'WFPER', # !!
    'WFPAR', # !!
    'WFANI', # !!
    'WFICRH', # !!
    'WFICRHP', # !!
    'WFICFORM', # !!
    'WFANIIC', # !!
    'TAUDIA', # !!
    'TAUMHD', # !!
    'TAUTH1', # !!
    'TAUTH2', # !!
    'H89', # !!
    'HITER96L', # !!
    'H93', # !!
    'HITER92Y', # !!
    'HEPS97', # !!
    'HIPB98Y', # !!
    'HIPB98Y1', # !!
    'HIPB98Y2', # !!
    'HIPB98Y3', # !!
    'HIPB98Y4', # !!
    'STANDARD', # ** dataset flags
    'SELDB1', # ** dataset flags
    'SELDB2', # ** dataset flags
    'SELDB2X', # ** dataset flags
    'IAEA92', # ** dataset flags
    'DB2P5', # ** dataset flags
    'DB2P8', # ** dataset flags
    'DB3IS', # ** dataset flags
    'DB3V5', # ** dataset flags
    'IAE2000N', # ** dataset flags
    'IAE2000X', # ** dataset flags
    'HMWS2003', # ** dataset flags
    'IAE2004S', # ** dataset flags
    'IAE2004I', # ** dataset flags
    'DB3DONLY', # ** dataset flags
    'HMWS2005', # ** dataset flags
    'OJK2006', # ** dataset flags
    'SELDB3', # ** dataset flags
    'SELDB3X', # ** dataset flags
    'STD3', # ** dataset flags
    'SELDB4', # ** dataset flags
    'STDDB4V5', # ** dataset flags
    'SELDB5', # ** dataset flags
    ]
