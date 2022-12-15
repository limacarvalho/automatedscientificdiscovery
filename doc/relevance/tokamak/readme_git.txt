WELCOME TO THE TOKAMAK DATASET

here are useful information from the official dataset documentation and data features.
Its a summary of tokamak fusion experiments worldwide without taking too much care about data standards.
I cleaned that mess up for you.
HDB5.2.3 is the latest version of this database and has been published by G Verdoolaege et al. in 2021.


main_asd_team.py 

-code for loading the dataset and converting all columns into correct datatypes.
              


HDB5.2.3_no_nans.csv  

- the cleaned up Tokamak dataset (no NaNs) with 71 columns, 10.241 row	
- shape: 71 columns, 10.241 rows
- this is the one you want to work with



HDB5.2.3.csv  

- the original Tokamak dataset from: https://osf.io/drwcq/
- shape: 192 columns, 14.153 rows	



DB5.2.3_documentation.pdf
- official documentation of the database. source: pdf: https://osf.io/593q6




info.txt  text version of the pdf merged with useful information about data.

	Tokamak entries:
    		evaluation entries for each tokamak

	Compact Summary:
    		table with all important information and parameter info as one liner

	Complete Summary:
    		contains data structure and parameter information.
    		- header: parameter, parameter category, datatype, percent NaNs
    		- top 10 value_counts (most present values)
    		- bins, bins numeric values (data distribution, min, max values)
    		- parameter information from DB5.2.3_variables.pdf, DB5.2.3_variables_txt.txt



publication_Tokamak_database_hdb5.2.3.pdf

- publication on the tokamak database DB5.2.3





## STEPS for cleaning up HDB5.2.3.csv
1) substitute nonesense values with NaNs -> HDB5.2.3_nans.csv
2) remove columns (utilities_asd_team.py -> removed_columns)
3) remove all rows with NaNs 

