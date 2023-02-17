'''
LEVEL - can be set to 0 to calcualte the correction factors, and set to 1 in
    order to level the data.
LIN - use 0 to use population statistics (single standard multiplicaiton) or 1
    to use linear regression.
Robust - use 0 for ordinary least squares, use 1 for robust Huber regression
    (only used for linear regression)
DATAONE_FILENAME - should be the original dataset for linear regression or
     the data to level.
DATATWO_filename - should be the re-analysed dataset for linear regression
    or the correction factors for levelling.
DATAONE - Legend information - X for LR (should be the original dataset).
DATATWO - Legend information - Y for LR (should be the re-analysed dataset).
SaveLocation - file save location.
FileName - name of the file to be saved.
'''

import math as mt
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import sys

# Setting for running
LEVEL = 0
LIN = 0
ROBUST = 0
# multi-standard linear correction
STANDARDS = False
ID_COLUMN = ' ' # only needed if STANDARDS is set to True
STANDARD_CUTOFF = 3
# Files to run on
DATAONE_FILENAME = r" "
DATATWO_filename = r" "
# Figure legends
DATASET_ONE_NAME =  ' '
DATASET_TWO_NAME =  ' '
# Save location (set a folder)
SaveLocation = r' '
FileName = ' '

def data_load():
    '''
    This function is used to load in two datasets and then parse them if
    correction factors are being generated. This function will return different
    variables depending on whether levelling is being run or not.

    if LEVEL == 0 Returns
    -------
    data1_element_list : array
        numpy array containing the list of elements in the dataset, calculated
        through the parse function.
    data1 : dataframe
        pandas dataframe containing the first dataset, the element column
        names have been changed to a standard format using the parse function.
    data2_element_list : array
        numpy array containing the list of elements in the dataset, calculated
        through the parse function.
    data2 : dataframe
        pandas dataframe containing the second dataset, the element column
        names have been changed to a standard format using the parse function.

    if LEVEL != 0 Returns
    -------
    data1 : dataframe
        pandas dataframe containing the first dataset, the element column
        names have been changed to a standard format using the parse function.
    data2 : dataframe
        pandas dataframe conting the second dataset, no parsing has been
        performed as this is typically the correction factor dataset.
    '''
    try:
        data1 = pd.read_excel(DATAONE_FILENAME, header = 0)
    except FileNotFoundError:
        print ("Unable to open first datset")
        sys.exit()
    try:
        data2 = pd.read_excel(DATATWO_filename, header = 0)
    except FileNotFoundError:
        print ("Unable to open second datset")
        sys.exit()
    try:
        # parse the data in order to split out the element only data
        data1_element_list, data1 = parse(data1)
        if LEVEL == 0:
            data2_element_list, data2 = parse(data2)
            return data1_element_list, data1, data2_element_list, data2

        else:
            return data1, data2
    except Exception as e:
        print ("Unable to parse data")
        print (e)
        sys.exit()

def stats(x,y):
    '''
    The stats function takes two data columns referred to X and Y and runs
    population statistics to determine if the two datasets are from the same
    population. The function first uses a Shapiro test on both datasets to
    determine normality. If both datasets are normally distributed then a
    Welche's t-test is used, else a Wilcoxon signed-rank test is used. The
    function then returns the test that was used, the p value and the median
    value for each dataset.

    Parameters
    ----------
    x : array
        The data to be used as for the X-axis.
    y : array
        The data to be used as for the Y-axis.

    Returns
    -------
    method : string
        The statistical  method used to compare the two populations.
    p-value : float
        The p-value.
    xmedian : flaot
        Median value for the x array.
    ymedian : float
        Median value for the y array.
    '''
    #print (x)
    cleaned_x = [i for i in x if mt.isnan(i) == False]
    cleaned_y = [i for i in y if mt.isnan(i) == False]
    # WX, PX = st.shapiro(cleaned_x)
    # WY, PY = st.shapiro(cleaned_y)
    # STAT, WELCH = st.ttest_ind(cleaned_x, cleaned_y, equal_var = False)
    # if PX > 0.05 and PY >0.05:
    #     method = "Welche's T-Test"
    #     STAT, POP = st.ttest_ind(cleaned_x, cleaned_y, equal_var = False)
    # else:
    #     method = "Wilcoxon signed-rank test"
    #     STAT, POP = st.ranksums(cleaned_x,cleaned_y)
    # print (st.ks_2samp(cleaned_x, cleaned_y))
    stat = st.ks_2samp(cleaned_x, cleaned_y)
    method = 'kolmogorov-smirnov'
    xmedian = np.median(cleaned_x)
    ymedian = np.median(cleaned_y)
    return method,stat[1], xmedian, ymedian

def linreg(x,y):
    '''
    Calculates the slope, intercept, and r2 of two datasets using ordinary
    least squares linear regression.

    Parameters
    ----------
    x : array
        The data to be used as for the X-axis.
    y : array
        The data to be used as for the Y-axis.

    Returns
    -------
    m : float
        Slope of the linear regression.
    c : float
        Intercept of the linear regression.
    r2 : float
        R2 of the linear regression.

    '''
    n = len(x)
    i = 0
    xy = np.zeros([n])
    x2 = np.zeros([n])
    for i in range(0,n):
        xy[i] = x[i]*y[i]
        x2[i] = x[i]**2
    m = (n*(sum(xy))-(sum(x)*sum(y)))/(n*(sum(x2))-(sum(x))**2)
    c = (sum(y)-m*sum(x))/n
    # R2 of the regression
    yscore = (1/len(y))*sum(y)
    sstot = np.zeros([len(y)])
    f = np.zeros([len(y)])
    ssres = np.zeros([len(y)])
    for i in range(0,len(y)):
        sstot[i] = (y[i] - yscore)**2
        f[i] = (m*x[i]) + c
        ssres[i] = (y[i] - f[i])**2
    sstot = sum(sstot)
    ssres = sum(ssres)
    r2 = 1-(ssres/sstot)
    return m, c, r2
#

def parse(geochem_data):
    '''
    Parses the header information in order to find geochemical elements and
    oxides. The function will return a cut down version of the header and the
    data to just include the geochemical data.

    Parameters
    ----------
    geochem_data : dataframe
        A pandas dataframe containing geochemistry data with headers to be
        parsed.

    Returns
    -------
    element_list : array
        An array containing the elements and oxides within the dataset.
    geochem_data : dataframe
        The input dataframe returned with the header information updated to
        searchable .

    '''
    common_headers = ("Lat", 'Long', 'Latitude', 'Longitude', 'Lab','Sample',
              'Time','Date', 'Group','Elev', 'Type', 'Site', 'Comment',
              'Depth','Size', 'LAT', 'LONG', 'Lab No', 'STATE', 'majors',
              'Recode','Name', 'East','North', 'LOI', 'SAMPLE', "GRAIN",
              "BATCH","Survey", "ID", "Standard", "Sample", "Colour", "batch",
              "sampleno", "SampleID", "Sampleno", "Jobno", "Pair", "Order",
              "Internal", "External", "METHOD", "SampleNo", 'Sample No',
              'Sample ID', 'External Lab No.', 'Internal Lab No.', 'Batch',
              'METHOD MILL', 'GA Sample No.', 'ENO', 'SITEID', 'LATITUDE',
              'LONGITUDE', 'BV_ID', 'Pair', 'Batch', 'Order', 'Chem',
              'Sampleid')
    elements = ('SiO2', 'TiO2','Al2O3', 'Fe2O3', 'FeO','MnO', 'MgO', 'CaO',
                'Na2O','K2O', 'P2O5', 'SO3', "H", "He", "Li", "Be", "B", "C",
                "N","O", "F", "Ne", 'Na', 'Mg', 'Al', 'Si','P', 'S', 'Cl',
                'Ar','K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                'Y', 'Zr','Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                'In', 'Sn', 'Sb', 'Te','Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
                'Nd','Pm','Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl','Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu','Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'LOI',
                'I', 'ORGC')
    header = list(geochem_data)
    index = []
    element_list = []
    for i, value in enumerate(header):
        if any(x in value for x in elements):
            if value not in common_headers:
                index.append(i)
            elif value in common_headers[2]:
                header[i] = 'Latitude'
        elif value in common_headers[3]:
            header[i] = 'Longitude'
    for i in index:
        temp_store = []
        for count, value in enumerate(elements):
            if value in header[i]:
                if len(value) > len(temp_store):
                    temp_store = value
        header[i] = temp_store
        element_list.append(temp_store)
    geochem_data.columns = header
    return element_list, geochem_data

#***Correction Factors***
def standard_correction_factors(element_list, dataset_one, dataset_two):
    '''
    Used for producing the correction factor using a single standard that is
    used in both datasets. If multiple shared standards exist then it is
    best to use the multi standard methodology.

    Parameters
    ----------
    element_list : TYPE
        DESCRIPTION.
    dataset_one : TYPE
        Pandas dataframe containing  the first dataset.
    dataset_two : TYPE
        Pandas dataframe containing  the first dataset.

    Returns
    -------
    None.

    '''
    # set up temporary arrays to store the statistics
    store = np.zeros([len(element_list),4])
    method = np.zeros([len(element_list)], dtype ="S64")
    elements = np.zeros([len(element_list)], dtype ="S64")
    #
    for i, element in enumerate(element_list):
        try:
            #X = dataset_one[element]
           # Y = dataset_two[element]
            X = dataset_one[element].apply(str).str.replace('<','-')
            Y = dataset_two[element].apply(str).str.replace('<','-')
            print(X,Y)
            X = X.astype('float64')
            Y = Y.astype('float64')
            print (element)
            #remove empty values to prevent them interfereing with the stats
            cleaned_x = [x for x in X if mt.isnan(x) == False]
            cleaned_y = [x for x in Y if mt.isnan(x) == False]
            print (X, cleaned_x)
            method[i],store[i,0], store[i,1], store[i,2] = stats(cleaned_x, cleaned_y)
            store[i,3] = store[i,1]/store[i,2]
            sns.distplot(cleaned_x, hist = True, kde = True,
                      kde_kws = {'shade': True, 'linewidth': 3})
            sns.distplot(cleaned_y, hist = True, kde = True,
                      kde_kws = {'shade': True, 'linewidth': 3})
            plt.axvline(store[i,1],linestyle=':')
            plt.axvline(store[i,2],  color='darkorange', linestyle=':')
            legend_one = DATASET_ONE_NAME + '\nmedian = ' +\
                str(round(store[i,1],2))
            legend_two = DATASET_TWO_NAME + '\nmedian = ' +\
                str(round(store[i,2],2))
            plot_title = method[i].decode("utf-8") + '\nP = ' +\
                str(round(store[i,0], 3))
            plt.legend([legend_one, legend_two], title=plot_title)
            plt.title(element)
            save_name = r"{}\{}.png".format(SaveLocation,element)
            plt.savefig(save_name, format = 'png', dpi = 900)
            plt.close()
            plt.clf()
        except KeyError:
            '''
            if the element from dataset one is not found in dataset two then
            this element will get skipped and and a note of not applicable
            applied.
            '''
            print (('{} not in second dataset').format(element))
            method[i],store[i,0], store[i,1], store[i,2] = 'Not Applicable',\
                np.nan, np.nan, np.nan

        elements[i] = element
    header = ["Element", "Test", "P-Value", DATASET_ONE_NAME + " Mean",
                DATASET_TWO_NAME + " Mean", "Correction Factor"]
    results = np.vstack((elements, method, store[:,0],store[:,1],
                         store[:,2], store[:,3])).T
    results = np.vstack((header, results))
    dataset = pd.DataFrame(results)
    save = r"{}\{}.xlsx".format(SaveLocation,FileName)
    dataset.to_excel(save, index=False, header = False)

def multi_standard(dataset_one, dataset_two, element):
    '''
    Identifies the standards present within the first dataset (dataset_one)
    based upon repetition of a value in a column. Two global variables set
    the parameter for finding standards, ID_COLUMN and STANDARD_CUTOFF. The
    ID_COLUMN value represents the column name in the pandas dataframe. The
    STANDARD_CUTOFF is a set value where a name must be repeated that many times
    before being added as a standard. The function then returns two numpy
    arrays (one for each dataset) containing the median for each standard.

    Parameters
    ----------
    dataset_one : dataframe
        A pandas dataframe containing the first dataset.
    dataset_two : dataframe
        A pandas dataframe containing the second dataset.
    element : string
        The element to use, this must be the same as the column  name in
        the data frame.

    Returns
    -------
    x : array
        a numpy array containing the median for each standard for the first
        dataset.
    y : array
        a numpy array containing the median for each standard for the second
        dataset.

    '''
    stds_count = dataset_one[ID_COLUMN].value_counts()
    stds = stds_count[stds_count > STANDARD_CUTOFF]
    stds_list = stds.index.tolist()
    x = np.zeros([len(stds_list)])
    y = np.zeros([len(stds_list)])
    print (stds_list)
    i = 0
    for standard in stds_list:
        dataone_slice = dataset_one.loc[dataset_one[ID_COLUMN] ==
                                        standard,element]
        datatwo_slice = dataset_two.loc[dataset_two[ID_COLUMN] ==
                                        standard,element]
        dataone_slice = dataone_slice.apply(str).str.replace('<','-')
        datatwo_slice = datatwo_slice.apply(str).str.replace('<','-')
        dataone_slice = dataone_slice.astype('float64')
        datatwo_slice = datatwo_slice.astype('float64')
        x[i] = np.median(dataone_slice)
        y[i] = np.median(datatwo_slice)
        i +=1
    print (x,y)
    return x,y

def linear_correction_factor(element_list, dataset_one, dataset_two):
    '''
    Calculates a linear correction factor for each element. The function saves
    a plot for each element with the linear regression, and a excel file
    containing the results for each element. This excel file is used for
    correcting the data if levelling is enabled. The first dataset should be
    the original data, dataset 2 the re-analysis.

    Parameters
    ----------
    element_list : array
        an array containing the elements in the dataset, can be generated by
        parsing the data using the parse function.
    dataset_one : dataframe
        A pandas dataframe containing the first dataset.
    dataset_two : dataframe
        A pandas dataframe containing the second dataset.

    Returns
    -------
    None.

    '''

    # print ("Creating correction factors from linear regression")
    store =  np.zeros([3,len(element_list)])
    #elements_used = ([])
    for i, element in enumerate(element_list):
        print (element)
        #try:
        if STANDARDS == True:
            cleanx, cleany = multi_standard(dataset_one, dataset_two, element)
        else:
            try:
                X = dataset_one[element].apply(str).str.replace('<','-')
                Y = dataset_two[element].apply(str).str.replace('<','-')
                print(X,Y)
                X = X.astype('float64')
                Y = Y.astype('float64')
                # idex_x = X[X.gt(0)].index
                # idex_y = Y[Y.gt(0)].index
                x_test = X.gt(0)
                y_test = Y.gt(0)
                if len(X[X.gt(0)].index) > len(Y[Y.gt(0)].index):
                    index = np.zeros([len(Y[Y.gt(0)].index)])
                elif len(X[X.gt(0)].index) < len(Y[Y.gt(0)].index):
                    index = np.zeros([len(X[X.gt(0)].index)])
                else:
                    index = np.zeros([len(X[X.gt(0)].index)])
                counter = 0
                for count, logic in enumerate(x_test):
                    if logic == True and y_test[count] == True:
                        index[counter] = count
                        counter += 1
                cleanx = X.loc[index].to_numpy()
                cleany = Y.loc[index].to_numpy()
            except KeyError:
                cleanx = [0,0]
                cleany = [0,0]
        if sum(cleanx) >0 and sum(cleany) >0:
            slope, intercept, fit = linreg(cleanx, cleany)
            if ROBUST == 1:
                cleanx = cleanx.reshape(-1,1)
                cleany = cleany.reshape(-1,1)
                huber = linear_model.HuberRegressor().fit(cleanx, cleany)
                huber_score = round(huber.score(cleanx, cleany),3)
                slope = huber.coef_[0]
                intercept = huber.intercept_
                fit = huber_score
            store[0,i], store[1,i], store[2,i] = slope, intercept, fit
            #elements_used.append(element)
            fitx = np.linspace(-5,200000,100)
            fity = slope * fitx + intercept
            plt.plot(fitx, fity, color = 'lightcoral')
            plt.plot([0,9999999999],[0,9999999999],color = 'k',
                      linestyle='dashed')
            min_x_value = min(cleanx)-(min(cleanx)*0.1)
            if min_x_value < 0:
                min_x_value = 0
            min_y_value = min(cleany)-(min(cleany)*0.1)
            if min_y_value < 0:
                min_y_value = 0
            plt.xlim(min_x_value, max(cleanx)+(max(cleanx)*0.1))
            plt.ylim(min_y_value, max(cleany)+(max(cleany)*0.1))
            plt.scatter(cleanx, cleany, s = 1)
            plt.title(element)
            plt.xlabel(DATASET_ONE_NAME)
            plt.ylabel(DATASET_TWO_NAME)
            TITLE = ('N = ' + str(len(cleanx)) + '\ny = ' +
                      str(round(slope, 2)) + 'x + ' + str(round(intercept, 2))+
                      '\nR$\mathregular{^2}$ = ' + str(round(fit,3)))
            plt.legend(title=TITLE)
            SAVENAME = r"{}\{}-LR.png".format(SaveLocation,element)
            plt.savefig(SAVENAME, format = 'png', dpi = 900)
            plt.close()
            plt.clf()
    store = np.vstack((element_list,store))
    dataset = pd.DataFrame(store)
    print (dataset)
    dataset.index = ("Element","Slope", "Interpcept", "R2")
    save = r"{}\{}.xlsx".format(SaveLocation,FileName)
    dataset.to_excel(save, index=True, header = False)
    # print ("Finished")

def levelling(dataset, corrections, use_p = True, r2 = False):
    '''
    Levels geochemical data using correction factors generated by
    standard_correction_factors and linear_correction_factor. The levelling
    mode is set through the use of a global variable LIN (1 for linear
    corrections).

    Parameters
    ----------
    dataset : dataframe
        The geochemical dataset.
    corrections : dataframe
        The corrections dataset.
    use_p : Boolean, optional
        The calculated  p value from population statistics. The default is True.
    r2 : boolean, optional
        The R2 value of the linear regression. The default is False.

    Returns
    -------
    None.

    '''
    print ("Starting Corrections")
    print (corrections)
    corrections.index = corrections['Element']
    if LIN == 0:
        for element in corrections['Element']:
            if use_p == True and corrections['P-Value'].loc[element] < 0.05:
                print (element, 'levelled')
                dataset[element] = dataset[element] *\
                    corrections['Correction Factor'].loc[element]
            elif use_p == False:
                try:
                    print (element, 'levelled')
                    dataset[element] = dataset[element] *\
                        corrections['Correction Factor'].loc[element]
                except KeyError:
                    print (element, 'Skipped')
            else:
                print (element, 'Not levelled')
    if LIN == 1:
        element_list = corrections.columns.values
        for element in element_list[1:]:
            slope = corrections[element].loc['Slope']
            intercept = corrections[element].loc['Interpcept'] # fix spelling
            print (element)
            print (dataset[element])
            try:
                for index, value in enumerate (dataset[element]):
                    if value >0:
                        dataset[element].loc[index] = slope*value+intercept
                     # dataset[element].loc[index] = (value/slope) - intercept
            except KeyError:
                print ("Skipped")
            print (dataset[element])
    save = r"{}\{}.xlsx".format(SaveLocation,FileName)
    dataset.to_excel(save, index=False, header = True)
    print ("Finished Corrections")


def main():
    '''
    Only used when the script is running stand alone. This will not run if
    calling this code from another script.

    Returns
    -------
    None.

    '''
    if LEVEL == 0:
        data1_element_list, data1, data2_element_list, data2 = data_load()
        print (data1_element_list, data2_element_list)
        if LIN == 0:
            standard_correction_factors(data1_element_list, data1, data2)
    #Linear regression calculations
        elif LIN == 1:
            #multi_standard(data1, data2, 'SiO2')
            linear_correction_factor(data1_element_list, data1, data2)
    # #***Levelling***
    if LEVEL == 1:
        data1, data2 = data_load()
        levelling(data1, data2,use_p = False)

if __name__ == "__main__":
    main()
