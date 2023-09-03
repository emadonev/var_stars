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
from selection import *

'''
This Python file contains all the functions necessary to calculate the LINEAR and ZTF periods.
'''

# ORIGINAL AUTOPOWER CALCULATION
def doPeriodsOrig(time, mag, magErr, nterms, lsPS=False, nyquist=300):
    """
    This function calculates the period of a LINEAR light curve using .autopower()

    Arguments:
        time(array): array of time values for the light curve
        mag(array): array of magnitude values for the light curve
        magErr(array): array of magnitude errors for the light curve
        nterms(int): number of terms for the Fourier fitting
        lsPS(Bool): flag, Default False
        nyquist(int): highest frequency, Default 300
    """
    try:
        ls = LombScargle(time, mag, magErr, nterms=nterms) # set up a LombScargle object to model the frequency and power
        if (1):
            frequency, power = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power
        else:
            frequency = np.arange(0.4, 40, 0.2e-04)
            power = ls.power(frequency)  
        period = 1. / frequency # calculating the periods
        best_period = period[np.argmax(power)] # choosing the period with the highest power
        if lsPS: 
            return best_period, frequency, power
        else:
            return best_period
    except:
        print('failed for ID=', Lid)
        return 'Error in doPeriods'

# BETTER FREQUENCY GRID
# -------------------------
# + "zoom-in" around the highest LS power peak 

# note: freqFac=1.02 allows search for Blazhko periods longer than 50*basic period, so ~25 days and longer

def doPeriods(time, mag, magErr, nterms, lsPS=False, nyquist=100, freqFac=1.02):
    """
    This function calculates the period of a LINEAR light curve using .autopower()

    Arguments:
        time(array): array of time values for the light curve
        mag(array): array of magnitude values for the light curve
        magErr(array): array of magnitude errors for the light curve
        nterms(int): number of terms for the Fourier fitting
        lsPS(Bool): flag, Default False
        nyquist(int): highest frequency, Default 300
        freqFac(int): max frequency for later Blazhko analysis, Default 1.02
    """
    try:
        ls = LombScargle(time, mag, magErr, nterms=nterms) # set up a LombScargle object to model the frequency and power
        frequencyAuto, powerAuto = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power
        best_freq = frequencyAuto[np.argmax(powerAuto)]
        frequency = np.arange(best_freq/freqFac, best_freq*freqFac, 5e-6)
        power = ls.power(frequency)  # compute LS periodogram again
        period = 1. / frequency
        best_period = period[np.argmax(power)] # choosing the period with the highest power
        if lsPS: 
            return best_period, frequency, power
        else:
            return best_period
    except:
        print('failed for ID=', Lid)
        return 'Error in doPeriods'

# CALCULATE ZTF PERIOD
# ----------------------

def getZTFperiod(ZTFdata, nterms, ZTFbands=['zg', 'zr', 'zi'], lsPS=False, nyquist=300, orig=False):
    """
    This function calculates the period of a ZTF light curve by taking the median of the periods of the 3 filters.

    Arguments:
        ZTFdata(array): an array of ZTF data for a light curve
        nterms(int): number of terms for the Fourier fitting
        ZTFbands(list): list of filters, Default ["zg", "zr", "zi"]
        lsPS(Bool): flag, Default False
        nyquist(int): highest frequency, Default 300
        orig(bool): using the original doPeriodsOrig function or the new doPeriods function, default False
    """
    failed = []
    ZTFperiods = []
    if ZTFdata.empty == True:
        #print("Empty")
        ZTFbestPeriod = 0
        Zfreq = np.array(())
        Zpow = np.array(())
        ZTFperiods.append(ZTFbestPeriod)
    else:
        #print("Not empty")
        for b in ZTFbands:
            BandData = ZTFdata.loc[ZTFdata['filtercode'] == b]
            timeZ = BandData['mjd']
            magZ = BandData['mag']
            magErrZ = BandData['magerr']
            if lsPS:
                try:
                    if (orig):
                        ZTFperiod, Zfreq, Zpow = doPeriodsOrig(timeZ, magZ, magErrZ, nterms, lsPS=lsPS)
                    else:
                        ZTFperiod, Zfreq, Zpow = doPeriods(timeZ, magZ, magErrZ, nterms, lsPS=lsPS)
                    ZTFperiods.append(ZTFperiod)
                except:
                    ZTFperiod = -9.99
                    failed.append(b)

            else:
                ZTFperiods.append(doPeriods(timeZ, magZ, magErrZ, nterms))
        ZTFbestPeriod = np.median(ZTFperiods)
        if lsPS:
            return ZTFbestPeriod, Zfreq, Zpow
        else:
            return ZTFbestPeriod
    return ZTFbestPeriod, Zfreq, Zpow