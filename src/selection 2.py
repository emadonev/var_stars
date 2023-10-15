# IMPORTING LIBRARIES
# --------------------

# AstroML & Astropy
from astroML.datasets import fetch_LINEAR_sample
from astropy.timeseries import LombScargle
from astroML.datasets import fetch_LINEAR_sample
from astroML.datasets import fetch_LINEAR_geneva
from astropy.timeseries import TimeSeries
from astropy.table import Table


# ZTF
from ztfquery import lightcurve

# Basic libraries
import random
import pickle
import os
import sys
from tqdm import tqdm

# Plotting
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

# DataFrame analysis
import pandas as pd
import dask.dataframe as dd 

# Math libraries
import numpy as np
import scipy as sc
from scipy.stats import norm

# Multithreading/multiprocessing libraries
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import threading

# Importing custom libraries
# ----------------------------
sys.path.insert(0,'../src/')
from ZTF_data import *
from config import*
from descriptive_stats import *
from plots import *

'''
This Python file contains the code necessary to calculate preliminary LINEAR IDs and select already calculated ones if file not present which are good. 
'''

def select_LINEAR(NAME):
    '''
    This function selects correct LINEAR Ids and calculates the preliminary periods of light curves.

    Arguments:
        NAME(string): name to save with
    '''

    if os.path.isfile('../outputs/'+NAME+'.txt'):
        colnames = ['ID','Porig','Pnew','ra','dec', 'ug','gi','iK','JK','logP','Ampl','skew','kurt','magMed','nObs','LCtype']
        L = Table.read('../outputs/'+NAME+'.txt', format='ascii', names=colnames)
        Properties = [] # creating a list
        for i in list(L.columns): # for every column in the .txt file
            a = list(L[i]) # append all the values from that column
            Properties.append(a) # create a nested list with all the column values

        j = 0 # counter
        LINEAR_periods = pd.DataFrame() # we create a new dataframe
        for i in list(L.columns): # for every column
            LINEAR_periods.insert(j, i, Properties[j]) # we make a new column with all the data inserted
            j += 1 # update the counter

    else:
        # CALCULATING PERIOD OF LINEAR LIGHT CURVES with nyquist=350, nterms=3
        #----------------
        name = 'LINEAR_periods'
        file_path = '../outputs/'+name+'.csv' # define file path
        if os.path.isfile(file_path): # if this file exists
            print('Loading LINEAR data!')
            LINEAR_data = pd.read_csv("../outputs/"+name+".csv") # load the already calculated data
        else:
            LC =  {}
            for n, i in enumerate(tqdm(data.ids)):
                t, mag, mager = data.get_light_curve(i).T # get the data for every light curve
                ls = LombScargle(t, mag, mager, nterms=3) # set up a LombScargle object to model the frequency and power
                frequency, power = ls.autopower(nyquist_factor=350) # calculate the frequency and power

                period = 1. / frequency # calculating the periods
                best_period = period[np.argmax(power)] # choosing the period with the highest power
                best_frequency = frequency[np.argmax(power)] # choosing the frequency with the highest power
                N = len(t) # number of points used in calculation

                LC[i] = [best_period, N]     # save values in dictionary with the id as the key

                if (n % 100) == 0:
                    # save dictionary to pkl file
                    with open('../outputs/LINEAR_periods.pkl', 'wb') as fp:
                        pickle.dump(LC, fp)           

            # SAVE AS A DATAFRAME
            # ----------------------
            cols = ['Period', 'N'] # define the columns
            LINEAR_data = pd.DataFrame.from_dict(LC, orient='index', columns=cols) # transfer into a DataFrame based on the index
            LINEAR_data.index.name='index' # name the index
            LINEAR_data.to_csv("../outputs/"+name+".csv") # save the .csv file


        PeriodsLinear = np.array(LINEAR_data['Period']) # selecting all the periods we just calculated
        IDs = [x for x in data.ids] # making a list of ids

        # ACCESSING THE OFFICIAL PERIODS
        #------------
        dataPeriods = fetch_LINEAR_geneva() # accessing the good data
        PLorig = 10**(dataPeriods['logP'].T) # accessing the periods
        ID_orig = list(dataPeriods['LINEARobjectID'].T) # accessing the good ID's

        ibad = igood = 0 # current number of good and bad star periods, counters
        goodPeriodOrig = [] # list for good original periods
        goodPeriodNew = [] # list of good newly calculated periods
        goodIDs = [] # list of good ID's from our whole dataset
        goodOrigIndex = [] # list of the good  ID


        # FINDING 'GOOD' LINEAR PERIODS
        #---------------------------------
        for id in ID_orig: # for every star in the good dataset 
            if id in IDs:# if the original ID is in our list of ID's
                # this star exists in the "geneva" subsample
                index = IDs.index(id) # the index is where the id from the good list is in our list
                goodPeriodOrig.append(PLorig[ID_orig.index(id)]) # append the good period with the correct index
                goodPeriodNew.append(PeriodsLinear[index]) # append the newly calculated period with the correct index
                goodIDs.append(id) # append the correct id
                goodOrigIndex.append(ID_orig.index(id)) # append the index of our id
                igood += 1 # count the number of good stars
            else: 
                # not in the "geneva" list 
                ibad += 1 # count the number of bad stars
        PeriodOrig = np.array(goodPeriodOrig)
        PeriodNew = np.array(goodPeriodNew)

        fout = open(NAME, "w") # create a new file in which we can write
        fout.write("    LINEAR ID     Porig         Pnew") # write up 3 columns
        fout.write("        ra             dec          ug      gi      iK      JK       logP       Ampl    skew    kurt  magMed nObs LCtype \n")    # add the rest of the columns (metadata)
        for i in range(0,len(goodIDs)): # for every ID in the new good IDs list
            ID = goodIDs[i] # current ID is the i element in the good list
            Porig = goodPeriodOrig[i] # the original period
            Pnew = goodPeriodNew[i] # the new period
            s = str("%12s " % ID) + str("%12.8f  " % Porig) + str("%12.8f  " % Pnew) # formats values into a string 's', aligning them in columns
            OrigIndex = goodOrigIndex[i] # getting the original index
            LID = dataPeriods[OrigIndex ]['LINEARobjectID'] # accessing the LINEAR ID of that object
            assert ID == LID # asserting that they are the same, so we can continue with the code
            ra = dataPeriods[OrigIndex]['ra'] #accessing the rectascension
            dec = dataPeriods[OrigIndex]['dec'] # accessing the declination
            s = s + str("%12.7f " % ra) + str("%12.7f  " % dec) # formatting previous values into columns
            for q in ['ug','gi','iK','JK']: 
                s = s + str("%7.2f " % dataPeriods[OrigIndex][q]) # loops through the filters and assigns their columns
            s = s + str("%12.8f " % dataPeriods[OrigIndex]['logP']) # adds the logP
            for q in ['Ampl','skew','kurt','magMed']: # loops through the rest of the statistics
                s = s + str("%7.2f " % dataPeriods[OrigIndex][q]) # formats the statistics
            s = s + str("%4.0f " % dataPeriods[OrigIndex]['nObs'])  # adds number of observations
            s = s + str("%2.0f " % dataPeriods[OrigIndex]['LCtype']) # adds light curve type
            s = s + "\n" # next record is in a new line or row
            fout.write(s) # add this string of values as a row
        fout.close() # when finished close

        colnames = ['ID','Porig','Pnew','ra','dec', 'ug','gi','iK','JK','logP','Ampl','skew','kurt','magMed','nObs','LCtype']
        L = Table.read('../outputs/'+NAME+'.txt', format='ascii', names=colnames)

        Properties = [] # creating a list
        for i in list(L.columns): # for every column in the .txt file
            a = list(L[i]) # append all the values from that column
            Properties.append(a) # create a nested list with all the column values

        j = 0 # counter
        LINEAR_periods = pd.DataFrame() # we create a new dataframe
        for i in list(L.columns): # for every column
            LINEAR_periods.insert(j, i, Properties[j]) # we make a new column with all the data inserted
            j += 1 # update the counter
            
    return LINEAR_periods
