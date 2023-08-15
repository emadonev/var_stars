import numpy as np
import pandas as pd
import scipy as sc
from astroML.utils.decorators import pickle_results
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.datasets import fetch_LINEAR_sample
from ztfquery import lightcurve
import dask.dataframe as dd 
from astropy.timeseries import LombScargle

#--------
import sys
sys.path.insert(0,'../src/')
sys.path
from ZTF_data import*
from config import*

'''
This Python file is used to calculate the periods of LINEAR and ZTF data with one term or multi-term. 
'''

# data
ZTF_data = data_ztf()
data = fetch_LINEAR_sample(data_home='../inputs') # fetching the data from astroML data library

def calculating_period(data_type, n_terms, name, nyquist=350, testing=True):
    '''
    This function calculates the period of light curves from either LINEAR or ZTF data. 

    Arguments:
        data_type(str): either LINEAR or ZTF
        nterms(int): number of terms for the Lomb-Scargle algorithm
        nyquist(int): the highest frequency the algorithm should search. The higher the value, the more it takes to run. Default is 350
        testing(bool): True: we are testing the function with a Default of 30 values, False: running the function for all the data. Default is True.
        name(str): name of the file we want to save/load
    '''
    if testing:
        if data_type == 'LINEAR':
            file_path = '../outputs/'+name+'.csv'
            if os.path.isfile(file_path):
                print('Loading LINEAR data!')
                LC = pd.read_csv("../outputs/"+name+".csv")
            else:
                LC = pd.DataFrame() # creating the empty DataFrame
                ids = [x for x in data.ids]

                for i in ids:
                    t, mag, mager = data.get_light_curve(i).T # get the data for every light curve
                    ls = LombScargle(t, mag, mager, nterms=n_terms) # set up a LombScargle object to model the frequency and power
                    frequency, power = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power

                    period = 1. / frequency # calculating the periods
                    best_period = period[np.argmax(power)] # choosing the period with the highest power
                    best_frequency = frequency[np.argmax(power)] # choosing the frequency with the highest power
                    N = len(t) # number of points used in calculation

                    lc_periods = pd.DataFrame([i, best_frequency, best_period, N]) # create Dataframe which you will append to main DataFrame
                    lc_periods = lc_periods.transpose() # transpose in order to create a row of data
                    columns = ['ID', 'Frequency','Period','N'] # determine columns
                    lc_periods.columns = columns # assign columns
                    LC = pd.concat([LC, lc_periods], axis=0) # concatenate tables 
                    print(f'Current ID:{i}') # print current ID

                LC.reset_index(drop=True, inplace=True)
                LC.to_csv('../outputs/'+name+'.csv', index=False)
        elif data_type == 'ZTF':
            file_path = '../outputs/'+name+'.csv'

            if os.path.isfile(file_path):
                print('Loading ZTF data!')
                LC = pd.read_csv("../outputs/"+name+".csv")
            else:
                LC = pd.DataFrame() # creating the empty DataFrame
                ids = [x for x in range(30)]
                for i in ids:
                    if ZTF_data[i][1].shape[0] > 0:
                        t, mag, mager = ZTF_data[i][1]['mjd'], ZTF_data[i][1]['mag'],ZTF_data[i][1]['magerr'] # get the data for every light curve
                        ls = LombScargle(t, mag, mager, nterms=nterms) # set up a LombScargle object to model the frequency and power
                        f, p = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power

                        period = 1. / f # calculating the periods
                        best_p = p[np.argmax(p)] # choosing the period with the highest power
                        best_f = f[np.argmax(p)] # choosing the frequency with the highest power
                        N = len(t) # number of points used in calculation

                        lc_p_ztf = pd.DataFrame([i, best_f, best_p, N]) # create Dataframe which you will append to main DataFrame
                        lc_p_ztf = lc_p_ztf.transpose() # transpose in order to create a row of data
                        columns = ['ID', 'Frequency','Period','N'] # determine columns
                        lc_p_ztf.columns = columns # assign columns
                        LC = pd.concat([LC, lc_p_ztf], axis=0) # concatenate tables 
                        print(f'Current ID:{i}') # print current ID
                    else:
                        lc_p_ztf = pd.DataFrame([i, 0, 0, 0])
                        lc_p_ztf = lc_p_ztf.transpose()
                        columns = ['ID', 'Frequency','Period','N']
                        lc_p_ztf.columns = columns 
                        LC = pd.concat([LC, lc_p_ztf], axis=0)
                        print(f'Current ID:{i}') # print current ID

                LC.reset_index(drop=True, inplace=True)
                LC.to_csv('../outputs/'+name+'.csv', index=False)
        else:
            print("Wrong input for data_type! Can only be LINEAR or ZTF.")
    else:
        if data_type == 'LINEAR':
            file_path = '../outputs/'+name+'.csv'
            if os.path.isfile(file_path):
                print('Loading LINEAR data!')
                LC = pd.read_csv("../outputs/"+name+".csv")
            else:
                LC = pd.DataFrame() # creating the empty DataFrame
                for i in data.ids:
                    t, mag, mager = data.get_light_curve(i).T # get the data for every light curve
                    ls = LombScargle(t, mag, mager, nterms=nterms) # set up a LombScargle object to model the frequency and power
                    frequency, power = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power

                    period = 1. / frequency # calculating the periods
                    best_period = period[np.argmax(power)] # choosing the period with the highest power
                    best_frequency = frequency[np.argmax(power)] # choosing the frequency with the highest power
                    N = len(t) # number of points used in calculation

                    lc_periods = pd.DataFrame([i, best_frequency, best_period, N]) # create Dataframe which you will append to main DataFrame
                    lc_periods = lc_periods.transpose() # transpose in order to create a row of data
                    columns = ['ID', 'Frequency','Period','N'] # determine columns
                    lc_periods.columns = columns # assign columns
                    LC = pd.concat([LC, lc_periods], axis=0) # concatenate tables 
                    print(f'Current ID:{i}') # print current ID

                LC.reset_index(drop=True, inplace=True)
                LC.to_csv('../outputs/'+name+'.csv', index=False)
        elif data_type == 'ZTF':
            file_path = '../outputs/'+name+'.csv'

            if os.path.isfile(file_path):
                print('Loading ZTF data!')
                LC = pd.read_csv("../outputs/"+name+".csv")
            else:
                LC = pd.DataFrame() # creating the empty DataFrame
                for i in num:
                    if ZTF_data[i][1].shape[0] > 0:
                        t, mag, mager = ZTF_data[i][1]['mjd'], ZTF_data[i][1]['mag'],ZTF_data[i][1]['magerr'] # get the data for every light curve
                        ls = LombScargle(t, mag, mager, nterms=nterms) # set up a LombScargle object to model the frequency and power
                        f, p = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power

                        period = 1. / f # calculating the periods
                        best_p = p[np.argmax(p)] # choosing the period with the highest power
                        best_f = f[np.argmax(p)] # choosing the frequency with the highest power
                        N = len(t) # number of points used in calculation

                        lc_p_ztf = pd.DataFrame([i, best_f, best_p, N]) # create Dataframe which you will append to main DataFrame
                        lc_p_ztf = lc_p_ztf.transpose() # transpose in order to create a row of data
                        columns = ['ID', 'Frequency','Period','N'] # determine columns
                        lc_p_ztf.columns = columns # assign columns
                        LC = pd.concat([LC, lc_p_ztf], axis=0) # concatenate tables 
                        print(f'Current ID:{i}') # print current ID
                    else:
                        lc_p_ztf = pd.DataFrame()
                        LC = pd.concat([LC, lc_p_ztf], axis=0)
                        print(f'Current ID:{i}') # print current ID

                LC.reset_index(drop=True, inplace=True)
                LC.to_csv('../outputs/'+name+'.csv', index=False)
        else:
            print("Wrong input for data_type! Can only be LINEAR or ZTF.")
    return LC