'''
LEVEL - can be set to 0 to calcualte the correction factors, and set to 1 in
    order to level the data.
LIN - use 0 to use population statistics (single standard multiplicaiton) or 1
    to use linear regression.
Robust - use 0 for ordinary least squares, use 1 for robust Huber regression
    (only used for linear regression)
DATAONE_filename - should be the original dataset for linear regression or
     the data to level.
DATATWO_filename - should be the re-analysed dataset for linear regression
    or the correction factors for levelling.
DATAONE - Legend information - X for LR (should be the original dataset).
DATATWO - Legend information - Y for LR (should be the re-analysed dataset).
SaveLocation - file save location.
FileName - name of the file to be saved.
'''

import numpy as np
import scipy.stats as st
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import sys

# Setting for running
LEVEL = 1
LIN = 0
Robust = 1
# Files to run on
DATAONE_filename = r""
DATATWO_filename = r""
# Figure legends
DATASET_ONE_NAME =  ''
DATASET_TWO_NAME =  ''
# Save location (set a folder)
SaveLocation = r''
FileName = ''

def data_load():
    '''



    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    try:
        data1 = pd.read_excel(DATAONE_filename, header = 0)
    except FileNotFoundError:
        print ("Unable to open first datset")
        sys.exit()
    try:
        data2 = pd.read_excel(DATATWO_filename, header = 0)
    except FileNotFoundError:
        print ("Unable to open second datset")
        sys.exit()
    try:
        # parse the dat in order to split out the element only data
        data1_element_list, data1 = parse(data1)
        if LEVEL == 0:
            data2_element_list, data2 = parse(data2)
            return data1_element_list, data1, data2_element_list, data2

        else:
            return data1, data2
            # else:
        #     CF_HEADER = DATATWO2[1]
        #     CF = DATATWO2[0]
    except Exception as e:
        print ("Unable to parse data")
        print (e)
        sys.exit()

def stats(X,Y):
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
    cleaned_x = [x for x in X if mt.isnan(x) == False]
    cleaned_y = [x for x in Y if mt.isnan(x) == False]
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

def LINREG(x,y):
    """
    LINREG is an implemntation of least squares linear regression.

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
    #f = np.zeros([len(y)])
    ssres = np.zeros([len(y)])
    for i in range(0,len(y)):
        sstot[i] = (y[i] - yscore)**2
        m[i] = (m*x[i]) + c
        ssres[i] = (y[i] - m[i])**2
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

# def plotting(cleaned_x, cleaned_y, store, element, method, i):
#     sns.distplot(cleaned_x, hist = True, kde = True,
#              kde_kws = {'shade': True, 'linewidth': 3})
#     sns.distplot(cleaned_y, hist = True, kde = True,
#              kde_kws = {'shade': True, 'linewidth': 3})
#     plt.axvline(store[i,1],linestyle=':')
#     plt.axvline(store[i,2],  color='darkorange', linestyle=':')
#     legend_one = DATASET_ONE_NAME + '\nmedian = ' +\
#         str(round(store[i,1],2))
#     legend_two = DATASET_TWO_NAME + '\nmedian = ' +\
#         str(round(store[i,2],2))
#     plot_title = method[i].decode("utf-8") + '\nP = ' +\
#         str(round(store[i,0], 3))
#     plt.legend([legend_one, legend_two], title=plot_title)
#     plt.title(element)
#     save_name = r"{}\{}.png".format(SaveLocation,element)
#     plt.savefig(save_name, format = 'png', dpi = 900)
#     plt.close()
#     plt.clf()

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
            #plotting(cleaned_x, cleaned_y, store, element, method i)
            #plotting of the standard
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
    for i, element in enumerate(element_list):
#         try:
        X = dataset_one[element]
        Y = dataset_two[element]
        cleanx = [x for x in X if (mt.isnan(x) == False)]
        cleany = [x for x in Y if (mt.isnan(x) == False)]
        slope, intercept, fit = LINREG(cleanx, cleany)
        if Robust == 1:
            huber = linear_model.HuberRegressor().fit(cleanx, cleany)
            huber_score = round(huber.score(cleanx, cleany),3)
            slope = huber.coef_[0]
            intercept = huber.intercept
            fit = huber_score
#         ARX = np.asarray(CLEANX)
#         ARX = ARX.reshape(-1,1)
#         ARY = np.asarray(CLEANY)
#         ARY = ARY.reshape(-1,1)
#         line_X = np.arange(ARX.min(), ARX.max())[:, np.newaxis]
#         reg = linear_model.TheilSenRegressor(random_state=0).fit(ARX, ARY)
#         huber = linear_model.HuberRegressor().fit(ARX, ARY)
#         REG_Score = round(reg.score(ARX, ARY),3)
#         Huber_Score = round(huber.score(ARX, ARY),3)
#         print (DATA1_element_list[J], huber.coef_, huber.intercept_)
#         line_y_thiel = reg.predict(line_X)
#         line_y_huber = huber.predict(line_X)
#
#             plt.plot(line_X, line_y_huber, color='lightcoral',
#                      label='Huber regressor R2 = {}'.format(Huber_Score))
        store[0,i], store[1,i], store[2,i] = slope, intercept, fit
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
    # except Exception as e:
    #     print ("Unable to run Linear regression correction calculation")
    #     print (e)
    #     sys.exit()
    # HEAD = (["Slope"], ["Intercept"], ["R2"])
    # STORE = np.hstack((HEAD, STORE))
    # TMP = ([" "])
    # HEADER = np.concatenate((TMP, DATA1_element_list))
    # RESULTS = np.vstack((HEADER, STORE))
    # HEAD = ("","Slope", "Interpcept", "R2")
    dataset = pd.DataFrame(store)
    dataset.index = ("Elements","Slope", "Interpcept", "R2")
    save = r"{}\{}.xlsx".format(SaveLocation,FileName)
    dataset.to_excel(save, index=True, header = True)
    # print ("Finished")

def levelling(dataset, corrections, use_p = True):
    print ("Starting Corrections")
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
    # if LIN == 1:
    #     CF = np.delete(CF.T, 0, axis = 0)
    #     CF = CF.T
    #     CF_HEADER = np.delete(CF_HEADER,0)
    #     SLOPE = CF[0,:]
    #     INTERCEPT = CF[1,:]
    #     I = 0
    #     J = 0
    #     for I in range(0,len(CF_HEADER)):
    #         for J in range(0, len(DATA1[:,0])):
    #             if mt.isnan(SLOPE[I] ) == False:
    #                 if DATA1[J,I] > 0:
    #                     CORRECTED[J, I] = SLOPE[I]*DATA1[J,I]+INTERCEPT[I]
    #                     if CORRECTED[J, I] < 0:
    #                         CORRECTED[J, I] = np.nan
    #                 else:
    #                     CORRECTED[J, I] = np.nan
    #             else:
    #                 if DATA1[J,I] > 0:
    #                     CORRECTED[J, I] = DATA1[J, I]
    #                 else:
    #                     CORRECTED[J, I] = np.nan
    # RESULTS = np.vstack((DATA_HEADER, CORRECTED))
    # DATASET = pd.DataFrame(RESULTS)
    save = r"{}\{}.xlsx".format(SaveLocation,FileName)
    dataset.to_excel(save, index=False, header = True)
    print ("Finished Corrections")


def main():
    if LEVEL == 0 and LIN == 0:
        data1_element_list, data1, data2_element_list, data2 = data_load()
        standard_correction_factors(data1_element_list, data1, data2)
    #Linear regression calculations
    if LEVEL == 0 and LIN == 1:
        linear_correction_factor(data1_element_list, data1, data2)
    # #***Levelling***
    if LEVEL == 1:
        data1, data2 = data_load()
        levelling(data1, data2,use_p = False)

if __name__ == "__main__":
    main()