# Geochemical-levelling

Levelling of geochemical data between surveys is a vital step in using datasets together. This code can apply a number of approaches to eliminate inter-laboratory differences from multi-generational and spatially isolated geochemical surveys. This codes allow the use of a variety of levelling methods: re-analysis, single standards, and multiple standards. The methodology and effectiveness of each of these methods are outlined in Main, P.T. and Champion, D.C., 2022. Levelling of multi-generational and spatially isolated geochemical surveys. Journal of Geochemical Exploration.

## Dependencies
* numpy - 1.13.3
* scipy - 0.19.1
* pandas - 0.25.3
* math - 
* matplotlib - 2.0.2
* seaborn - 0.9.0
* sklearn - 0.20.1

## Running
Run Parameters
*LEVEL – This variable is use to determine if the data is getting levelled or the correction factors are getting generated. Use 1 for levelling or 0 for generating correction factors.
*LIN – This is used to specify the method for levelling/correction factor generation. 0 is used for population statistics of a single standard. 1 is used for linear regression of either reanalysis or multiple standards.
*ROBUST – Used to determine if a robust Huber Regressor (ref) is used for linear regression or not. 0 is to used ordinary least squares, 1 is to use a Huber Regressor.
*STANDARDS = True - 
*ID_COLUMN – Only used if STANDARDS is set to True. This specifies the column name where the standards are named.
*STANDARD_CUTOFF  – Only used if STANDARDS is set to True. The number of times a value in the ID_COLUMN is repeated before being flagged as a standard.
*DATAONE_FILENAME = r"C:\Users\u29043\Desktop\Ebagoola_Batch_1.xlsx"
*DATATWO_filename = r"C:\Users\u29043\Desktop\Ebagoola_Batch_2.xlsx"
*DATASET_ONE_NAME – The name of the first data set, this is the axis legend label.
*DATASET_TWO_NAME – The name of the second data set, this is the axis legend label.
*SaveLocation – The folder location to save the files.
*FileName – The name for the file that will be created.


## Output data


