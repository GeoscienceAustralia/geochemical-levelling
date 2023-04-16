# Geochemical-levelling

Levelling of geochemical data between surveys is a vital step in using datasets together. This code can apply a number of approaches to eliminate inter-laboratory differences from multi-generational and spatially isolated geochemical surveys. This codes allow the use of a variety of levelling methods: re-analysis, single standards, and multiple standards. The methodology and effectiveness of each of these methods are outlined in Main, P.T. and Champion, D.C., 2022. Levelling of multi-generational and spatially isolated geochemical surveys. Journal of Geochemical Exploration.

This script was developed as part of Geoscience Australia’s Exploring for the Future program. Geoscience Australia’s Exploring for the Future program provides precompetitive information to inform decision-making by government, community and industry on the sustainable development of Australia's mineral, energy and groundwater resources. By gathering, analysing and interpreting new and existing precompetitive geoscience data and knowledge, we are building a national picture of Australia’s geology and resource potential. This leads to a strong economy, resilient society and sustainable environment for the benefit of all Australians. This includes supporting Australia’s transition to net zero emissions, strong, sustainable resources and agriculture sectors, and economic opportunities and social benefits for Australia’s regional and remote communities. The Exploring for the Future program, which commenced in 2016, is an eight year, $225m investment by the Australian Government.

## Dependencies
The code was developed with the following dependencies and their versions:
* numpy - 1.13.3
* scipy - 0.19.1
* pandas - 0.25.3
* math - 
* matplotlib - 2.0.2
* seaborn - 0.9.0
* sklearn - 0.20.1

## Running
File Requirements: 
* The files should be excel .xlsx files with limited extraneous where possible. Whilst the script has an in built parser to find the elements unnecessary column may produce errors due to incorrect assignment. Example files can be found in the examples folder.
* No modifications should be made to the generated correction factor files prior to running any levelling.
* If using linear regression of reanalysed samples the first dataset should be the original dataset and the second the reanalysis.
* If levelling the data, the first dataset should be the data to be levelled and the second the generated correction factors file.

Run Parameters:
* LEVEL – This variable is use to determine if the data is getting levelled or the correction factors are getting generated. Use 1 for levelling or 0 for generating correction factors.
* LIN – This is used to specify the method for levelling/correction factor generation. 0 is used for population statistics of a single standard. 1 is used for linear regression of either reanalysis or multiple standards.
* ROBUST – Used to determine if a robust Huber Regressor (ref) is used for linear regression or not. 0 is to used ordinary least squares, 1 is to use a Huber Regressor.
* STANDARDS – If using linear regression of multiple standards set as True, for all other cases set as False. 
* ID_COLUMN – Only used if STANDARDS is set to True. This specifies the column name where the standards are named.
* STANDARD_CUTOFF  – Only used if STANDARDS is set to True. The number of times a value in the ID_COLUMN is repeated before being flagged as a standard.
* DATAONE_FILENAME – the path to the first data set. The path should be surrounded my quotation marks and preceded by an r e.g.  r"C:\Users\ \Desktop\First_Data_Set.xlsx".
* DATATWO_filename – the path to the second data set. The path should be surrounded my quotation marks and preceded by an r e.g.  r"C:\Users\ \Desktop\First_Data_Set.xlsx".
* DATASET_ONE_NAME – The name of the first data set, this is the axis legend label.
* DATASET_TWO_NAME – The name of the second data set, this is the axis legend label.
* SaveLocation – The folder location to save the files.
* FileName – The name for the file that will be created.

## Output data

* When running in correction factor mode, the program will output a series of plots (.png), one for each element, and an excel file containing the correction factors. 
* When running in levelling mode, the script will output a single excel file containing the levelled data.

## Dataset References
### Example 1
*	Group 1 – Northern Australia Geochemical Survey total digestion <75 µm – Bastrakov, E. N., Main, P. T., Wygralak, A. S., Wilford, J., Czarnota, K. & Khan, M. 2018. Northern Australia Geochemical Survey data release 1: total (fine fraction) and MMI™ element contents. record 2018/06. Geoscience Australia.
*	Group 2 – Levelled Geochemical Baseline of Australia – Main, P.T. and Champion, D.C., 2022. Levelling of multi-generational and spatially isolated geochemical surveys. Journal of Geochemical Exploration, 240, p.107028.
### Example 2
*	Group 1 – Levelled Geochemical Baseline of Australia – Main, P.T. and Champion, D.C., 2022. Levelling of multi-generational and spatially isolated geochemical surveys. Journal of Geochemical Exploration, 240, p.107028.
*	Group 2 – Levelled Geochemical Baseline of Australia – Main, P.T. and Champion, D.C., 2022. Levelling of multi-generational and spatially isolated geochemical surveys. Journal of Geochemical Exploration, 240, p.107028.
### Example 3
*	Group 1 – Northern Australia Geochemical Survey total digestion <75 µm ICP-MS data –Bastrakov, E. N., Main, P. T., Wygralak, A. S., Wilford, J., Czarnota, K. & Khan, M. 2018. Northern Australia Geochemical Survey data release 1: total (fine fraction) and MMI™ element contents. record 2018/06. Geoscience Australia.
*	Group 2 – Northern Australia Geochemical Survey total digestion <2 mm ICP-MS data – Main, P. T., Bastrakov, E. N., Wygralak, A. S., Czarnota, K. & Khan, M. 2019. Northern Australia Geochemical Survey Data Release 2: Total (Coarse Fraction), Aqua Regia (Coarse and Fine Fraction), and Fire Assay (Coarse and Fine Fraction) Element Contents. Geoscience Australia Record, 2019/02.

## eCat Id: 147662


