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
LIN = 1
ROBUST = 0
STANDARDS = True
ID_COLUMN = 'SampleID'
STANDARD_CUTOFF = 3
# Files to run on
DATAONE_FILENAME = r"C:\Users\u29043\Desktop\Ebagoola_Batch_1.xlsx"
DATATWO_filename = r"C:\Users\u29043\Desktop\Ebagoola_Batch_2.xlsx"
#DATAONE_FILENAME = r"C:\Users\u29043\Desktop\GeorgeTown_Orig.xlsx"
#DATAONE_FILENAME = r"C:\Users\u29043\Desktop\Levelling_Testing\GT_LR_Corrcted.xlsx"
#DATATWO_filename = r"C:\Users\u29043\Desktop\GeorgeTown_ReAnalysis.xlsx"
#DATATWO_filename = r"C:\Users\u29043\Desktop\Levelling_Testing\GT_LR_Corrctions.xlsx"
# Figure legends
DATASET_ONE_NAME =  'GT_Orig'
DATASET_TWO_NAME =  'GT_rerun'
# Save location (set a folder)
SaveLocation = r'C:\Users\u29043\Desktop\Levelling_Testing'
#FileName = 'GT_LR_Corrctions'
FileName = 'GT_LR_Corrctions'

def data_load():
    '''



    Returns
    -------
    TYPE
        DESCRIPTION.

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
    """
    the stats function takes two data coloumns reffered to X and Y and runs
    population statistcis to determine if the two datsest are from the same
    population. The function first uses a Shapiro test on both datasets to
    determine normality. If both datsests are normally distributed then a
    Welche's t-test is used, else a Wilcoxon signed-rank test is used. The
    function then returns the test that was used, the p value and the median
    value for each dataset.

    :param filename: the full unc path to the xlsx document
    :return: returns the method used, the P value, the X dataset median, and
    the y dataset median.
    """
    cleaned_x = [i for i in x if mt.isnan(x) == False]
    cleaned_y = [i for i in y if mt.isnan(x) == False]
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
    """
    linreg is an implemntation of least squares linear regression.

    :param X: The data to be used as for the X-axis.
    :param Y: The data to be used as for the Y-axis.
    :return: returns the slope, intercept, and the R2 for the input data.
    """
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
    Parses the header informaiton in order to find geochemical elements and
    oxides. The function will return a cut down version of the header and the
    data to just include the geochemical data.

    Parameters
    ----------
    header : array of strings
        The array containing headers for each of the coloumns.

    Returns
    -------
    sndhead : array of strings
        A cut down header containing only chemical elements.
    fullhead : array of strings
        The full header, with Longitude and Latitude changed to a standardised
        name and the elements changed to standard notation.

    '''
    common_headers = ("Lat", 'Long', 'Latitude', 'Longitude', 'Lab','Sample',
              'Time','Date', 'Group','Elev', 'Type', 'Site', 'Comment',
              'Depth','Size', 'LAT', 'LONG', 'Lab No', 'STATE', 'majors',
              'Recode','Name', 'East','North', 'LOI', 'SAMPLE', "GRAIN",
              "BATCH","Survey", "ID", "Standard", "Sample", "Colour", "batch",
              "sampleno", "SampleID", "Sampleno", "Jobno", "Pair", "Order",
              "Internal", "External", "METHOD", "SampleNo", 'Sample No',
              'Sample ID', 'External Lab No.', 'Internal Lab No.', 'Batch',
              'METHOD MILL', 'GA Sample No.')
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
                'I')
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
    used in both datasets. If multpile shared standards exist then it is
    best to use the multi standard methodology.


    Parameters
    ----------
    element_list : TYPE
        DESCRIPTION.
    dataset_one : TYPE
        Pandas dataframe containg the first dataset.
    dataset_two : TYPE
        Pandas dataframe containg the first dataset.

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
            X = dataset_one[element]
            Y = dataset_two[element]
            print (element)
            #remove empty values to prevent them interfereing with the stats
            cleaned_x = [x for x in X if mt.isnan(x) == False]
            cleaned_y = [x for x in Y if mt.isnan(x) == False]
            method[i],store[i,0], store[i,1], store[i,2] = stats(X, Y)
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


    Parameters
    ----------
    dataset_one : TYPE
        DESCRIPTION.
    dataset_two : TYPE
        DESCRIPTION.
    element : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    stds_count = dataset_one[ID_COLUMN].value_counts()
    stds = stds_count[stds_count > STANDARD_CUTOFF]
    stds_list = stds.index.tolist()
    x = np.zeros([len(stds_list)])
    y = np.zeros([len(stds_list)])
    print (stds_list)
    i = 0
    for standard in stds_list:
        dataone_slice = dataset_one.loc[dataset_one[ID_COLUMN] == standard,element]
        datatwo_slice = dataset_two.loc[dataset_two[ID_COLUMN] == standard,element]
        x[i] = np.median(dataone_slice)
        y[i] = np.median(datatwo_slice)
        i +=1
    return x,y

def linear_correction_factor(element_list, dataset_one, dataset_two):
    '''


    Parameters
    ----------
    element_list : TYPE
        DESCRIPTION.
    dataset_one : TYPE
        DESCRIPTION.
    dataset_two : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #dataset 1 should be the original data, dataset 2 the re-analysis

    # print ("Creating correction factors from linear regression")
    store =  np.zeros([3,len(element_list)])
    print (store)
    #elements_used = ([])
    for i, element in enumerate(element_list):
        print (element)
        #try:
        if STANDARDS == True:
            cleanx, cleany = multi_standard(dataset_one, dataset_two, element)
        else:
            X = dataset_one[element].apply(str).str.replace('<','-')
            Y = dataset_two[element].apply(str).str.replace('<','-')
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
        plt.xlim(min(cleanx)-(min(cleanx)*0.1),
                  max(cleanx)+(max(cleanx)*0.1))
        plt.ylim(min(cleany)-(min(cleany)*0.1),
                  max(cleany)+(max(cleany)*0.1))
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


    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    corrections : TYPE
        DESCRIPTION.
    use_p : TYPE, optional
        DESCRIPTION. The default is True.
    r2 : TYPE, optional
        DESCRIPTION. The default is False.

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
                     #   dataset[element].loc[index] = (value/slope) - intercept
            except KeyError:
                print ("Skipped")
            print (dataset[element])
    save = r"{}\{}.xlsx".format(SaveLocation,FileName)
    dataset.to_excel(save, index=False, header = True)
    print ("Finished Corrections")


def main():
    '''


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