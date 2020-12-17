
import time_to_orbit as tto
import function_folder as ff
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import os
import shutil
import io
import re
import requests
import csv
import pymms
# from tqdm import tqdm
from cdflib import epochs
from urllib.parse import parse_qs
import urllib3
import warnings
from scipy.io import readsav
from getpass import getpass
import cdflib
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyaC as PyAs
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
import seaborn as sn

#functions that need to be brought in: mms.download_files, tto.MrMMS_SDC_API, tto.sort_files, ff.cdf_todf

# os.rename("/Users/Riley/PycharmProjects/mms_data/ /Users/Riley/PycharmProjects/mms_data/heliopy/heliopy/data/mms /mms1","/Users/Riley/PycharmProjects/mms_data/loaded data")
# shutil.move("/Users/Riley/PycharmProjects/mms_data/ /Users/Riley/PycharmProjects/mms_data/heliopy/heliopy/data/mms /mms1","/Users/Riley/PycharmProjects/mms_data/loaded data")
# os.replace("/Users/Riley/PycharmProjects/mms_data/ /Users/Riley/PycharmProjects/mms_data/heliopy/heliopy/data/mms /mms1","/Users/Riley/PycharmProjects/mms_data/loaded data")
/Users/Riley/PycharmProjects/mms_data/Evaluating_Parameters.py

def fgm_data(start_date, end_date):
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'

    # FGM
    b_vname = '_'.join((sc, 'fgm', 'b', 'gse', mode, level))
    mms = tto.MrMMS_SDC_API(sc, 'fgm', mode, level,
                            start_date=start_date, end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]

    fgm_data = ff.cdf_to_df(files, b_vname)  # tto.from_cdflib(files, b_vname,
    # start_date, end_date)

    # fgm_data['data'] = fgm_data['data'][:, [3, 0, 1, 2]]
    # fgm_data['color'] = ['Black', 'Blue', 'Green', 'Red']
    # fgm_data[fgm_data['LABL_PTR_1']]['data'] = ['|B|', 'Bx', 'By', 'Bz']
    fgmabs_data = features(fgm_data['mms1_fgm_b_gse_srvy_l2_0'])
    fgmx_data = features(fgm_data['mms1_fgm_b_gse_srvy_l2_1'])
    fgmy_data = features(fgm_data['mms1_fgm_b_gse_srvy_l2_2'])
    fgmz_data = features(fgm_data['mms1_fgm_b_gse_srvy_l2_3'])
    return fgmabs_data, fgmx_data, fgmy_data, fgmz_data

def velocity_data(start_date, end_date):
    # FPI DIS
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    fpi_mode = 'fast'
    ni_vname = '_'.join((sc, 'dis', 'numberdensity', fpi_mode))
    espec_i_vname = '_'.join((sc, 'dis', 'energyspectr', 'omni', fpi_mode))
    BV_vname = '_'.join((sc, 'dis', 'bulkv_gse', fpi_mode))
    mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
                            optdesc='dis-moms',
                            start_date=start_date, end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]

    ni_data = ff.cdf_to_df(files,ni_vname)
    # start_date, end_date)
    # especi_data = tto.from_cdflib(files, espec_i_vname,
    #                               start_date, end_date)
    BV_data = ff.cdf_to_df(files, BV_vname)  # tto.from_cdflib(files, BV_vname,
    # start_date, end_date)
    BVX_data = features(BV_data['mms1_dis_bulkv_gse_fast_0'])
    BVY_data = features(BV_data['mms1_dis_bulkv_gse_fast_1'])
    BVZ_data = features(BV_data['mms1_dis_bulkv_gse_fast_2'])
    density = features(ni_data['mms1_dis_numberdensity_fast'])
    return BVX_data, BVY_data, BVZ_data, density

def temp_para_data(start_date, end_date):
    # get the temperature parameters
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    fpi_mode = 'fast'
    Tpara_vname = '_'.join((sc, 'des', 'temppara', 'fast'))
    mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level, optdesc='des-moms', start_date=start_date,
                            end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]
    Temppara_data = ff.cdf_to_df(files, Tpara_vname)  # tto.from_cdflib(files, Tpara_vname, start_date, end_date)
    tpara = features(Temppara_data['mms1_des_temppara_fast'])
    return tpara

def temp_perp_data(start_date, end_date):
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    fpi_mode = 'fast'
    Tperp_vname = '_'.join((sc, 'des', 'tempperp', 'fast'))
    mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level, optdesc='des-moms', start_date=start_date,
                            end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]
    Tempperp_data = ff.cdf_to_df(files, Tperp_vname)  # tto.from_cdflib(files, Tperp_vname, start_date, end_date)
    tperp = features(Tempperp_data['mms1_des_tempperp_fast'])
    return tperp

def Pressure_data(start_date, end_date):
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    fpi_mode = 'fast'
    Pressure_vname = '_'.join((sc, 'des', 'prestensor_gse', 'fast'))
    mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level, optdesc='des-moms', start_date=start_date,
                            end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]
    Pressure_data = ff.cdf_to_df(files,
                                 Pressure_vname)  # tto.from_cdflib(files, Pressure_vname, start_date, end_date)
    pscalar = features((Pressure_data['mms1_des_prestensor_gse_fast_1'] + Pressure_data[
        'mms1_des_prestensor_gse_fast_2'] + Pressure_data['mms1_des_prestensor_gse_fast_3']) / 3)
    return pscalar

def anis_temp(start_date, end_date):
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    fpi_mode = 'fast'
    Tpara_vname = '_'.join((sc, 'des', 'temppara', 'fast'))
    mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level, optdesc='des-moms', start_date=start_date,
                            end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]
    tpara = ff.cdf_to_df(files, Tpara_vname)

    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    fpi_mode = 'fast'
    Tperp_vname = '_'.join((sc, 'des', 'tempperp', 'fast'))
    mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level, optdesc='des-moms', start_date=start_date,
                            end_date=end_date)
    files = mms.download_files()
    files = tto.sort_files(files)[0]
    tperp = ff.cdf_to_df(files, Tperp_vname)
    tanis = []
    tscalar = []
    tanis = features((1 - (tperp.div(tpara))))  # features
    tscalar = features((tpara.add(tperp).div(3)))  # features
    return tanis, tscalar

def data(start_date, end_date):
    f1, f2, f3, f4 = fgm_data(start_date, end_date)
    v1, v2, v3 = velocity_data(start_date, end_date)
    tpara = temp_para_data(start_date, end_date)
    tperp = temp_perp_data(start_date, end_date)
    pscalar = Pressure_data(start_date, end_date)
    tanis, tscalar = anis_temp(start_date, end_date)
    return [f1, f2, f3, f4, pscalar, tperp, v1, v2, v3, tpara, tanis, tscalar]

# now identify the 500 eV and 1000 eV channels for the energy spectogram.

# Mean and STD Function
def mean(data):
    # su = 0
    # for i in range(len(data)):
    #     su += data[i]
    # su = su/len(data)
    mean = np.mean(data)
    return mean

def STD(data):
    standard_devi = np.std(data)
    return standard_devi

# zero crossings function
# Come back to this because the zero crossings variable will only account
# For the last parameter but we need it to account for all the parameters
# in data types

def zero_counter(data):
    zero_crossings = 0
    for i in range(len(data), 1):
        if (data[i - 1] > 0 and data[i] < 0):
            zero_crossings += 1
        if (data[i - 1] < 0 and data[i] > 0):
            zero_crossing += 1
    return zero_crossings

# Plataeus Function
# we need to rewrite this function.
def Plateaus_slopes(data):
    plateaus = 0
    increasing = 0
    decreasing = 0
    # delta = np.diff(data, axis=1)
    # gradient = np.sign(delta)

    for i in range(0, len(data)):
        if data[i+1]>data[i]+1 or data[i]-1:
            increasing += 1
        elif gradient[0, i] < 0:
            decreasing += 1
        elif gradient[0, i] == 0:
            plateaus += 1
    return plateaus, increasing, decreasing

# Global Extreme Function
def Extreme(data):
    maxs = np.amax(data)
    mins = np.amin(data)
    return maxs, mins

# Q-Parameter Function
def quality_factor(data, M=2):

    smoothed_data = [data[0]]
    for i, value in enumerate(data[0, 1:]):
        smoothed_data.append((smoothed_data[i - 1] * (2 ** M - 1) + value) / 2 ** M)
    return np.subtract(data, np.transpose(smoothed_data))

def features(data):
    # qu = quality_factor(data)
    E1, E2 = Extreme(data)
    zc = zero_counter(data)
    S = STD(data)
    m = mean(data)
    row = [m, S, zc, E1, E2]
    return row

# positive cases
start_date1 = dt.datetime(2015, 10, 25, 13, 56, 20)
end_date1 = dt.datetime(2015, 10, 25, 13, 58, 20)

start_date2 = dt.datetime(2015, 10, 30, 5, 14, 30)
end_date2 = dt.datetime(2015, 10, 30, 5, 17, 50)

start_date3 = dt.datetime(2015, 12, 6, 0, 22, 50)
end_date3 = dt.datetime(2015, 12, 6, 0, 25, 40)

start_date4 = dt.datetime(2015, 12, 8, 0, 3, 0)
end_date4 = dt.datetime(2015, 12, 8, 0, 5, 0)

start_date5 = dt.datetime(2015, 12, 8, 0, 6, 20)
end_date5 = dt.datetime(2015, 12, 8, 0, 11, 40)

start_date6 = dt.datetime(2015, 12, 30, 1, 5, 10)
end_date6 = dt.datetime(2015, 12, 30, 1, 7, 30)  # new

start_date7 = dt.datetime(2016, 10, 24, 10, 56, 10)
end_date7 = dt.datetime(2016, 10, 24, 10, 58, 50)

start_date8 = dt.datetime(2016, 10, 24, 16, 54, 10)
end_date8 = dt.datetime(2016, 10, 24, 17, 5, 50)

start_date9 = dt.datetime(2016, 10, 27, 14, 44, 30)
end_date9 = dt.datetime(2016, 10, 27, 14, 46, 30)

start_date10 = dt.datetime(2016, 10, 28, 11, 26, 50)
end_date10 = dt.datetime(2016, 10, 28, 11, 28, 40)

start_date11 = dt.datetime(2016, 12, 4, 5, 35, 50)
end_date11 = dt.datetime(2016, 12, 4, 5, 39, 0)

start_date12 = dt.datetime(2016, 12, 4, 6, 16, 0)
end_date12 = dt.datetime(2016, 12, 4, 6, 23, 40)

start_date13 = dt.datetime(2017, 1, 8, 13, 4, 50)
end_date13 = dt.datetime(2017, 1, 8, 13, 4, 30)
# negative cases

start_date14 = dt.datetime(2015, 12, 1, 12, 27, 50)
end_date14 = dt.datetime(2015, 12, 1, 12, 30, 20)  # it says MP, i think that that's what we want

start_date15 = dt.datetime(2015, 12, 2, 1, 14, 0)
end_date15 = dt.datetime(2015, 12, 2, 1, 15, 20)  # it says MP again

start_date16 = dt.datetime(2015, 12, 2, 5, 13, 40)
end_date16 = dt.datetime(2015, 12, 2, 5, 24, 30)

start_date17 = dt.datetime(2015, 12, 2, 5, 41, 10)
end_date17 = dt.datetime(2015, 12, 2, 5, 44, 50)

start_date18 = dt.datetime(2015, 12, 2, 5, 51, 50)
end_date18 = dt.datetime(2015, 12, 2, 5, 55, 30)

start_date19 = dt.datetime(2015, 12, 7, 1, 15, 20)
end_date19 = dt.datetime(2015, 12, 7, 1, 16, 10)

start_date20 = dt.datetime(2015, 12, 27, 21, 29, 50)
end_date20 = dt.datetime(2015, 12, 27, 21, 34, 40)

start_date21 = dt.datetime(2015, 12, 29, 7, 29, 20)
end_date21 = dt.datetime(2015, 12, 29, 7, 33, 40)

start_date22 = dt.datetime(2015, 12, 29, 9, 24, 40)
end_date22 = dt.datetime(2015, 12, 29, 9, 30, 0)

start_date23 = dt.datetime(2015, 12, 30, 1, 3, 10)
end_date23 = dt.datetime(2015, 12, 30, 1, 4, 20)

start_date24 = dt.datetime(2015, 11, 15, 2, 33, 40)
end_date24 = dt.datetime(2015, 11, 15, 2, 34, 30)

start_date25 = dt.datetime(2015, 11, 15, 14, 7, 15)
end_date25 = dt.datetime(2015, 11, 15, 14, 12, 5)

start_date26 = dt.datetime(2015, 11, 20, 1, 5, 10)
end_date26 = dt.datetime(2015, 11, 20, 1, 8, 0)

start_date27 = dt.datetime(2015, 11, 20, 1, 21, 50)
end_date27 = dt.datetime(2015, 11, 20, 1, 25, 10)

start_date28 = dt.datetime(2015, 11, 20, 14, 8, 0)
end_date28 = dt.datetime(2015, 11, 20, 14, 10, 20)

start_date29 = dt.datetime(2015, 11, 21, 1, 28, 40)
end_date29 = dt.datetime(2015, 11, 21, 1, 32, 50)

start_date30 = dt.datetime(2015, 11, 25, 2, 54, 20)
end_date30 = dt.datetime(2015, 11, 25, 2, 55, 20)

start_date31 = dt.datetime(2015, 12, 8, 0, 30, 40)
end_date31 = dt.datetime(2015, 12, 8, 0, 32, 40)

start_date32 = dt.datetime(2015, 12, 8, 0, 30, 40)
end_date32 = dt.datetime(2015, 12, 8, 0, 31, 30)
time_int = [start_date1, end_date1, start_date2, end_date2, start_date3, end_date3, start_date4, end_date4,
            start_date5, end_date5, start_date6, end_date6, start_date7, end_date7, start_date8, end_date8,
            start_date9, end_date9, start_date10, end_date10, start_date11, end_date11, start_date12, end_date12,
            start_date13, end_date13, start_date14, end_date14, start_date15, end_date15, start_date16, end_date16,
            start_date17, end_date17, start_date18, end_date18, start_date19, end_date19, start_date20, end_date20,
            start_date21, end_date21, start_date22, end_date22, start_date23, end_date23, start_date24, end_date24,
            start_date25, end_date25, start_date26, end_date26, start_date27, end_date27, start_date28, end_date28,
            start_date29, end_date29, start_date30, end_date30, start_date31, end_date31, start_date32, end_date32]
Y = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# print(fgm_data(time_int[1],time_int[2]))
# print(velocity_data(time_int[1],time_int[2]))
# print(temp_perp_data(time_int[1],time_int[2]))
# print(temp_para_data(time_int[1],time_int[2]))
# print(Pressure_data(time_int[1],time_int[2]))
X = []
A = []
B = []
C = []
D = []
E = []
F = []
G = []
H = []
I = []
J = []
K = []
for i in range(0, len(time_int), 2):
    hold1, hold2, hold3, hold4,  = (fgm_data(time_int[i], time_int[i + 1]))
    A.append(hold1)
    B.append(hold2)
    C.append(hold3)
    D.append(hold4)
    hold5, hold6, hold7,hold8 = velocity_data(time_int[i], time_int[i + 1])
    K.append(hold8)
    E.append(hold5)
    F.append(hold6)
    G.append(hold7)
    H.append(temp_para_data(time_int[i], time_int[i + 1]))
    I.append(temp_perp_data(time_int[i], time_int[i + 1]))
    J.append(Pressure_data(time_int[i], time_int[i + 1]))

A = pd.DataFrame(A)
B = pd.DataFrame(B)
C = pd.DataFrame(C)
D = pd.DataFrame(D)
E = pd.DataFrame(E)
F = pd.DataFrame(F)
G = pd.DataFrame(G)
H = pd.DataFrame(H)
I = pd.DataFrame(I)
J = pd.DataFrame(J)
K = pd.DataFrame(K)
X = pd.concat([A, B, C, D, E, F, G, H, I, J],axis=1)#, K], axis=1)
X = pd.DataFrame(normalize(X))

# for some reason incorporating the number density decreases the accuracy of the model, this is possibly due to overfitting so we will exclude it.




def model1( X, Y,xtest,ytest):
    logistic_regression = LogisticRegression(class_weight='balanced')
    logistic_regression.fit(X, Y)
    ypred = logistic_regression.predict(xtest)
    ytest = np.transpose(pd.DataFrame.to_numpy(ytest))
    ytest = ytest.flatten()
    confusion_matrix = pd.crosstab(ytest, ypred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy of the Logistic Regression model: ', metrics.accuracy_score(ytest, ypred))
    plt.show()
    return ypred

def model2(X, Y,xtest,ytest):
    model = svm.SVC()
    model.fit(X, Y)
    ypred = model.predict(xtest)
    ytest = np.transpose(pd.DataFrame.to_numpy(ytest))
    ytest = ytest.flatten()
    confusion_matrix = pd.crosstab(ytest, ypred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy of the Support Vector Machine model: ', metrics.accuracy_score(ytest, ypred))
    plt.show()
    return ypred

def model3(X, Y,xtest,ytest):
    model = tree.DecisionTreeClassifier()
    model.fit(X, Y)
    ypred = model.predict(xtest)
    ytest = np.transpose(pd.DataFrame.to_numpy(ytest))
    ytest = ytest.flatten()
    confusion_matrix = pd.crosstab(ytest, ypred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy of the Decsion Tree classifier: ', metrics.accuracy_score(ytest, ypred))
    plt.show()
    return ypred



LR = LogisticRegression(class_weight='balanced')
SV = svm.SVC()
DTC = tree.DecisionTreeClassifier()
Ensemble_model = VotingClassifier(estimators=[('LR', LR), ('SVM', SV), ('DTC', DTC)], voting='hard')
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
Ensemble_model.fit(xtrain, ytrain)
ypred = Ensemble_model.predict(xtest)
print('model accuracy: ', metrics.accuracy_score(ytest, ypred))


# this function will time frame contains a reconnection jet.
def Ensemble_model(X,Y,xtest):
    LR = LogisticRegression(class_weight='balanced')
    SV = svm.SVC()
    DTC = tree.DecisionTreeClassifier()
    Ensemble_model = VotingClassifier(estimators=[('LR', LR), ('SVM', SV), ('DTC', DTC)], voting='hard')
    Ensemble_model.fit(X, Y)
    ypred = Ensemble_model.predict(xtest)
    if ypred == 1:
        print('This time frame does contain a magnetopause reconnection jet!')
    if ypred == 0:
        print('This time frame contains no magnetopause reconnection jet.')
