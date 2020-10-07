import numpy as np
import scipy.stats as st
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import sys

# Setting for running
LEVEL = 0 # use 0 to calculate correction factors, use 1 to level the data
LIN = 0 # use 0 for population statistcis, use 1 for linear regression
robust = 1 # use 0 for ordinary least squares, use 1 for robust Huber regression (only used for linear regression)
# Files to run on
#should be the original dataset for linear regression or the data to level
DATAONE_filename = r'C:\Users\u29043\Desktop\Levelling Paper\NGSA_Thomson\Eulo_Toompine_TILL-1.xlsx'
#should be the re-analysed dataset for linear regression or the correction factors for levelling
DATATWO_filename = r'C:\Users\u29043\Desktop\Levelling Paper\NGSA_Thomson\NGSA_Standards-cut_TILL-1.xlsx' 
# Figure legends
DATAONE =  'Eulo-Toompine'# Legend information - X for LR (should be the original dataset)
DATATWO =  'NGSA' # Legend information - Y for LR (should be the re-analysed dataset)
# Save location (set a folder)
SaveLocation = r'C:\Users\u29043\Desktop\New Code Test\Standards'
FileName = 'StandardsTest' #Correction factor file name

def IMPORT(filename): #imports data from excel sheets, must be xlsx no csv
    xl = pd.ExcelFile(filename)
    sheets = xl.sheet_names
    Sheet = sheets[0]
    info= pd.io.excel.ExcelFile.parse(xl, Sheet)
    DATA = pd.DataFrame(info, index = info.index)
    GENDATA = DATA.values
    HEADER = info.columns
    return GENDATA, HEADER

def STATS(X,Y):
    CLEANX = [x for x in X if (mt.isnan(x) == False)]
    CLEANY = [x for x in Y if (mt.isnan(x) == False)]
    WX, PX = st.shapiro(CLEANX)
    WY, PY = st.shapiro(CLEANY)
    STAT, WELCH = st.ttest_ind(CLEANX, CLEANY, equal_var = False) 
    if PX > 0.05 and PY >0.05:
        METHOD = r"Welche's T-Test"
        STAT, POP = st.ttest_ind(CLEANX, CLEANY, equal_var = False) 
    else:
        METHOD = r"Wilcoxon signed-rank test"
        STAT, POP = st.ranksums(CLEANX,CLEANY)
#    MEANX = sum(CLEANX)/len(CLEANX)
#    MEANY = sum(CLEANY)/len(CLEANY)
    print (st.ks_2samp(CLEANX, CLEANY))
    xmedian = np.median(CLEANX)
    ymedian = np.median(CLEANY)
    return METHOD,POP, xmedian, ymedian

def BOXPLOT(VALUES):#calculates the breaks for box plots
    CLEAN = [x for x in VALUES if (mt.isnan(x) == False)]
    IQR = np.percentile(CLEAN, 75) - np.percentile(CLEAN, 25)
    WT = np.percentile(CLEAN, 75) + 1.5*IQR
    WL = np.percentile(CLEAN, 25) - 1.5*IQR
    CLEANX = [x for x in CLEAN if (x < WT)]
    CLEANX = [x for x in CLEAN if (x > WL)]
    return CLEANX   
 
def LINREG(X,Y):
    # linear regression
    N = len(X)
    I = 0
    XY = np.zeros([N])
    X2 = np.zeros([N])
    for I in range(0,N):
        XY[I] = X[I]*Y[I]
        X2[I] = X[I]**2
    M = (N*(sum(XY))-(sum(X)*sum(Y)))/(N*(sum(X2))-(sum(X))**2)
    C = (sum(Y)-M*sum(X))/N
    # R2 of the regression
    YSCORE = (1/len(Y))*sum(Y)
    SStot = np.zeros([len(Y)])
    F = np.zeros([len(Y)])
    SSres = np.zeros([len(Y)])
    for I in range(0,len(Y)):
        SStot[I] = (Y[I] - YSCORE)**2
        F[I] = (M*X[I]) + C
        SSres[I] = (Y[I] - F[I])**2   
    SStot = sum(SStot)
    SSres = sum(SSres)
    R2 = 1-(SSres/SStot)
    return M, C, R2
#
def PARSE(HEADER, DATA): #used to parse the header to find the elements and Lat and Longs
    LENGH = len(HEADER)
    NEWHEAD = np.zeros([LENGH], dtype = "U16")
    # list of common headers, add things here if you need them
    COMMON = ("Lat", 'Long', 'Latitude', 'Longitude', 'Lab','Sample', 'Time', 'Date', 'Group',
              'Elev', 'Type', 'Site', 'Comment', 'Depth', 'Size', 'LAT', 
              'LONG', 'Lab No', 'STATE', 'majors', 'Recode', 'Name', 'East',
              'North', 'LOI', 'SAMPLE', "GRAIN", "BATCH", "Survey", "ID", "Standard")
    ELEMENTS = ('SiO2', 'TiO2','Al2O3', 'Fe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'SO3', "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 'Na', 'Mg', 'Al', 'Si',
     'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
     'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
     'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
     'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd','Pm','Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
     'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
     'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
     'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 
     'Hs', 'Mt', 'LOI', 'I')
    I = 0
    K = 0
    for I in range(0, LENGH):
        check = 0
        for K in range(0, len(COMMON)):
            if COMMON[K] in HEADER[I]:
                NEWHEAD[I] = 'NaN'
                check = 1
            elif K == (len(COMMON)-1) and check == 0:
                NEWHEAD[I] = HEADER[I]
    NaNCOUNT = sum(1 for item in NEWHEAD if item==('NaN'))
    SNDHEAD = np.zeros([len(NEWHEAD)-NaNCOUNT], dtype = "U16")
    FULLHEAD = np.zeros(len(HEADER), dtype = 'U64')
    for I in range(0,len(FULLHEAD)):
        FULLHEAD[I] = HEADER[I]
    for I in range(0, len(FULLHEAD)):
        if COMMON[0] in FULLHEAD[I] or COMMON[2] in FULLHEAD[I]:
            FULLHEAD[I] = "Latitude"
        elif COMMON[1] in FULLHEAD[I] or COMMON[3] in FULLHEAD[I]:
            FULLHEAD[I] = "Longitude"
    K = 0
    I = 0
    COUNT = 0
    TSTORE = 'a'
    for I in range(0, len(NEWHEAD)):
        CHECK = 0
        if NEWHEAD[I] != 'NaN':
            for K in range(0,len(ELEMENTS)):
                if ELEMENTS[K] in FULLHEAD[I]:
                    if CHECK == 0:
                        TSTORE = ELEMENTS[K]
    
                        CHECK = 1
                    elif CHECK == 1 and len(TSTORE) == 1 and ELEMENTS[K] != 'I':
                        TSTORE = ELEMENTS[K]
            FULLHEAD[I] = TSTORE
    K = 0
    I = 0  
    COUNT = 0
    NaNCOUNT = sum(1 for item in NEWHEAD if item==('NaN'))
    DELROWS = np.zeros([NaNCOUNT])
    for I in range(0, len(NEWHEAD)):
        if NEWHEAD[I] =='NaN':
            DELROWS[COUNT] = I
            COUNT = COUNT + 1
            K = K
        else:
            SNDHEAD[K] = NEWHEAD[I]
            K = K + 1
    NEWHE = np.zeros([len(SNDHEAD)], dtype = "U16")
    CUTDATA = DATA
    for I in range(len(DELROWS)-1, -1, -1):
        CUTDATA = np.delete(CUTDATA, I,1)
    TSTORE = "a"
    for I in range(0, len(SNDHEAD)):
        check = 0
        for K in range(0, len(ELEMENTS)):
            if ELEMENTS[K] in SNDHEAD[I]:
                #print ELEMENTS[K], SNDHEAD[I] 
                if check == 1:
                    if len(TSTORE) == 1 and len(ELEMENTS[K]) >1:
                        TSTORE = ELEMENTS[K]
                elif check == 0:          
                    TSTORE = ELEMENTS[K]
                check = 1
                NEWHE[I] = TSTORE
            if check == 0 and K == 107:
                NEWHE[I] = 'NaN'
                print ("Element not found")
    NaNCOUNT = sum(1 for item in NEWHE if item==('NaN'))
    if NaNCOUNT == 0:
        SNDHEAD = NEWHE
    else:
        SNDHEAD = np.zeros([len(NEWHE)-NaNCOUNT], dtype = "U16")
        K = 0
        for I in range(0, len(NEWHE)):
            if NEWHE[I] =='NaN':
                K = K
            else:
                SNDHEAD[K] = NEWHE[I]
                K = K + 1 
        #print SNDHEAD
    return FULLHEAD, SNDHEAD, CUTDATA
#***Correction Factors***
#Open the data
if LEVEL == 0: # function to load and parse the data to determine levelling factors
    try:
        DATA = IMPORT(DATAONE_filename)
        DATA1_HEADER = DATA[1]
        DATA1 = DATA[0]
    except:
         sys.exit("Unable to open first datset")
    try:
        DATATWO2 = IMPORT(DATATWO_filename)
        DATA2_HEADER = DATATWO2[1]
        DATA2 = DATATWO2[0]
    except:
        sys.exit("Unable to open second datset")
    try:
        # parse the dat in order to split out the element only data
        full_data_header, DATA1_HEADER, DATA1 = PARSE(DATA1_HEADER,DATA1)
        full_data2_header, DATA2_HEADER, DATA2 = PARSE(DATA2_HEADER,DATA2)
    except:
        sys.exit("Unable to parse data")
    #check the number of elements is the same
    if len(DATA1_HEADER) != len (DATA2_HEADER):
        print("Different number of elements")
        print(DATA1_HEADER)
        print(DATA2_HEADER)
else:
    try:
        DATA = IMPORT(DATAONE_filename)
        DATA_HEADER = DATA[1]
        DATA = DATA[0]
        full_data_header, DATA1_HEADER, DATA1 = PARSE(DATA1_HEADER,DATA1)
    except:
         sys.exit("Unable to open first datset")
    try:
        DATATWO2 = IMPORT(DATATWO_filename)
        CF_HEADER = DATATWO2[1]
        CF = DATATWO2[0]
    except:
        sys.exit("Unable to open second datset")
# population statistics
if LEVEL == 0 and LIN == 0:
    STORE = np.zeros([len(DATA2_HEADER),3])
    TYPE = np.zeros([len(DATA2_HEADER)], dtype ="S64")
    HEAD = np.zeros([len(DATA2_HEADER)], dtype ="S64")
    CORRECTION = np.zeros([len(DATA2_HEADER)])
    slope = np.zeros([len(DATA2_HEADER)])
    intercept = np.zeros([len(DATA2_HEADER)])
    I = 0
    for I in range(0, len(DATA1_HEADER)):
        X = DATA1[:,I]
        Y = DATA2[:,I]
        print (DATA1_HEADER[I])
        CLEANX = [x for x in X if (mt.isnan(x) == False)]
        CLEANY = [x for x in Y if (mt.isnan(x) == False)]
        print ("Kurtosis: ",st.kurtosis(CLEANX), st.kurtosis(CLEANY))
        print ("Skew: ",st.skew(CLEANX), st.skew(CLEANY))
        sns.distplot(CLEANX, hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
        sns.distplot(CLEANY, hist = True, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
        #TYPE[I],STORE[I,0], STORE[I,1], STORE[I,2] = STATS(DATA1[:,I], DATA2[:,I])
        TYPE[I],STORE[I,0], STORE[I,1], STORE[I,2] = STATS(X, Y)
        plt.axvline(STORE[I,1],linestyle=':')
        plt.axvline(STORE[I,2],  color='darkorange', linestyle=':')
        LEGEONE = DATAONE + '\nmedian = ' + str(round(STORE[I,1],2))
        LEGETWO = DATATWO + '\nmedian = ' + str(round(STORE[I,2],2))
        TITLE = TYPE[I].decode("utf-8") + '\nP = ' + str(round(STORE[I,0], 3))
        plt.legend([LEGEONE, LEGETWO], title=TITLE)
        plt.title(DATA1_HEADER[I])
        SAVENAME = r"{}\{}.png".format(SaveLocation,DATA1_HEADER[I])
        plt.savefig(SAVENAME, format = 'png', dpi = 900)
        plt.close()
        plt.clf()
        #
        HEAD[I] = DATA2_HEADER[I]
    HEADING = ["Element", "Test", "P-Value", "Dataset 1 Mean", "Dataset 2 Mean", "Slope", "Intercept"]
    RESULTS = np.vstack((HEAD, TYPE, STORE[:,0],STORE[:,1],STORE[:,2],slope,intercept)).T
    RESULTS = np.vstack((HEADING, RESULTS))
    DATASET = pd.DataFrame(RESULTS)
    SAVE = r"{}\{}.xlsx".format(SaveLocation,FileName)
    DATASET.to_excel(SAVE, index=False, header = False)
#
#Linear regression calculations    
if LEVEL == 0 and LIN == 1:
    #dataset 1 should be the original data, dataset 2 the re-analysis
    print ("Creating correction factors from linear regression")
    N =  len(DATA1[:,0])
    I = 0
    J = 0
    STORE =  np.zeros([3,len(DATA1[0])])
    for J in range (0, len(DATA1[0])):
        try:
            X = np.zeros([N])
            Y = np.zeros([N])
            for I in range(0,N):
                if DATA1[I,J] >0 and DATA2[I,J] >0:
                    X[I] = DATA1[I,J]
                    Y[I] = DATA2[I,J]
                else:
                    X[I] = np.nan
                    Y[I] = np.nan
                CLEANX = [x for x in X if (mt.isnan(x) == False)]
                CLEANY = [x for x in Y if (mt.isnan(x) == False)]
            SLOPE, INTERCEPT, FIT = LINREG(CLEANX, CLEANY)
            #Robust linear regression RANSAC algorithm
            ARX = np.asarray(CLEANX)
            ARX = ARX.reshape(-1,1)
            ARY = np.asarray(CLEANY)
            ARY = ARY.reshape(-1,1)
            line_X = np.arange(ARX.min(), ARX.max())[:, np.newaxis]
            reg = linear_model.TheilSenRegressor(random_state=0).fit(ARX, ARY)
            huber = linear_model.HuberRegressor().fit(ARX, ARY)
            REG_Score = round(reg.score(ARX, ARY),3)
            Huber_Score = round(huber.score(ARX, ARY),3)
            print (DATA1_HEADER[J], huber.coef_, huber.intercept_)
            line_y_thiel = reg.predict(line_X)
            line_y_huber = huber.predict(line_X)
            if robust == 1:
                SLOPE = huber.coef_[0]
                INTERCEPT = huber.intercept_
                FIT = Huber_Score
                plt.plot(line_X, line_y_huber, color='lightcoral',label='Huber regressor R2 = {}'.format(Huber_Score))
            STORE[0,J], STORE[1,J], STORE[2,J] = SLOPE, INTERCEPT, FIT 
            FITX = x = np.linspace(-5,200000,100)
            FITY = SLOPE * FITX + INTERCEPT
            plt.plot(FITX, FITY, color = 'lightcoral')
            plt.plot([0,9999999999],[0,9999999999],color = 'k',linestyle='dashed')
            plt.xlim(min(CLEANX)-(min(CLEANX)*0.1),max(CLEANX)+(max(CLEANX)*0.1))
            plt.ylim(min(CLEANY)-(min(CLEANY)*0.1), max(CLEANY)+(max(CLEANY)*0.1))
            #sns.regplot(x=CLEANX,y= CLEANY, robust=False, color='red', scatter_kws={'s':1})
            plt.scatter(CLEANX, CLEANY, s = 1)
            plt.title(DATA1_HEADER[J])
            plt.xlabel(DATAONE)
            plt.ylabel(DATATWO)
            TITLE = ('N = ' + str(len(CLEANX)) + '\ny = ' + str(round(SLOPE, 2)) +
                     'x + ' + str(round(INTERCEPT, 2)) + '\nR$\mathregular{^2}$ = ' + 
                     str(round(FIT,3)))
            plt.legend(title=TITLE)
            SAVENAME = r"{}\{}-LR.png".format(SaveLocation,DATA1_HEADER[J])
            plt.savefig(SAVENAME, format = 'png', dpi = 900)
            plt.close()
            plt.clf()
        except:
            pass
    HEAD = (["Slope"], ["Intercept"], ["R2"])
    STORE = np.hstack((HEAD, STORE))
    TMP = ([" "])
    HEADER = np.concatenate((TMP, DATA1_HEADER))
    RESULTS = np.vstack((HEADER, STORE))
    HEAD = ("","Slope", "Interpcept", "R2")
    DATASET = pd.DataFrame(RESULTS)
    SAVE = r"{}\{}.xlsx".format(SaveLocation,FileName)
    DATASET.to_excel(SAVE, index=False, header = False)
    print ("Finished")
#***Levelling***
#
if LEVEL == 1:
    print ("Starting Corrections")
    LEN, WID = np.shape(DATA)
    CORRECTED = np.zeros([LEN, WID])
    if LIN == 0:
        I = 0
        J = 0
        for I in range(0, len(CF[:,0])):
            for J in range(0, len(DATA[:,0])):
                if mt.isnan(CF[I,1] ) == False:
                    if DATA[J,I] > 0:
                        CORRECTED[J, I] = DATA[J,I] * CF[I,1]
                    else:
                        CORRECTED[J, I] = np.nan
                else:
                    if DATA[J,I] > 0:
                        CORRECTED[J, I] = DATA[J, I]
                    else:
                        CORRECTED[J, I] = np.nan
    if LIN == 1:
        CF = np.delete(CF.T, 0, axis = 0)
        CF = CF.T
        CF_HEADER = np.delete(CF_HEADER,0)
        SLOPE = CF[0,:]
        INTERCEPT = CF[1,:]
        I = 0
        J = 0
        for I in range(0,len(CF_HEADER)):
            for J in range(0, len(DATA[:,0])):
                if mt.isnan(SLOPE[I] ) == False:
                    if DATA[J,I] > 0:
                        #CORRECTED[J, I] = (DATA[J,I] - INTERCEPT[I])/SLOPE[I]
                        CORRECTED[J, I] = SLOPE[I]*DATA[J,I]+INTERCEPT[I]
                        if CORRECTED[J, I] < 0:
                            CORRECTED[J, I] = np.nan
                    else:
                        CORRECTED[J, I] = np.nan
                else:
                    if DATA[J,I] > 0:
                        CORRECTED[J, I] = DATA[J, I]
                    else:
                        CORRECTED[J, I] = np.nan 
    RESULTS = np.vstack((DATA_HEADER, CORRECTED))
    DATASET = pd.DataFrame(RESULTS)
    SAVE = r"{}\{}.xlsx".format(SaveLocation,FileName)
    DATASET.to_excel(SAVE, index=False, header = False)
    print ("Finished Corrections")
