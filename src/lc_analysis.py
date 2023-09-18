# IMPORTING LIBRARIES
# --------------------

# AstroML & Astropy
from astroML.datasets import fetch_LINEAR_sample
from astropy.timeseries import LombScargle
from astroML.datasets import fetch_LINEAR_sample
from astroML.datasets import fetch_LINEAR_geneva
from astropy.timeseries import TimeSeries
from astropy.table import Table


# Basic libraries
import random
import pickle
import os
import sys

# Plotting
import seaborn as sns
from matplotlib import pyplot as plt

# DataFrame analysis
import pandas as pd

# Math libraries
import numpy as np
import scipy as sc
from scipy.stats import norm
from scipy.signal import find_peaks

# Importing custom libraries
# ----------------------------
sys.path.insert(0,'../src/')
from ZTF_data import *
from config import*
from descriptive_stats import *
from plots import *
from selection import *

'''
This Python file contains all the functions necessary to analyze LINEAR and ZTF light curves using period calculations, periodogram analysis, light curve fitting and goodness of fit measurements. 
'''

# ORIGINAL AUTOPOWER CALCULATION
# ---------------------------------
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

# HELPER FUNCTIONS
# ------------------
def sigG(x):
    '''
    Calculates the interquartile range and then adjusts it with the 0.741 parameter. This creates a more robust version of the standard deviation without bias.

    Arguments:
        x(array) = data we want to calculate the interquartile range with
    '''
    return 0.741*(np.percentile(x,75)-np.percentile(x,25))

def sort3arr(a, b, c):
    ''' 
    Sorts the light curve fit data.
    '''
    ind = np.argsort(a)
    return a[ind], b[ind], c[ind]

def getCoordinatesFromLINEARid(tbl, id0):
    return tbl[tbl['ID'].astype(str)==id0]['ra'][0], tbl[tbl['ID'].astype(str)==id0]['dec'][0]


# retrieve ZTF data for a single object specified by (RA, Dec)
def getZTFlightcurve(ra, dec, radius=3.0):
    # matching radius is given in arcsec
    try:
       lcq = lightcurve.LCQuery()
       res = lcq.from_position(ra, dec, radius)
       ZTFdata = res.data[['mjd', 'mag', 'magerr', 'catflags', 'filtercode']]
       # M. Graham recommends to get rid of obvious spurious points
       ZTFdata = ZTFdata.loc[ZTFdata['catflags'] < 32768]
    except Exception as e:
       print(e)
    return ZTFdata

# LC ANALYSIS
# -------------
def LCanalysisFromP(time, mag, magErr, P, ntermsModels):
    '''
    Analyzes light curves with phasing, fitting and calculating metrics such as chi-2 and rms scatter.

    Arguments:
        time(array) = array of time data for a light curve
        mag(array) = array of magnitude data for a light curve
        magErr(array) = array for the magnitude error data for a light curve
        P(float) = period of the light curve
        ntermsModels(int) = number of modes with which to fit
    '''
    # defining dictionary to store the results
    LCanalysisResults = {}
    
    # FITTING
    # ----------
    mtf = MultiTermFit(2*np.pi/P, ntermsModels) # defining a MultiTermFit object by converting the period into angular frequency and determining the number of modes for the fit
    mtf.fit(time, mag, magErr) # fit to the desired data

    # a = the phase grid = folding the light curve with the period (phased light curve) (y axis)
    # b = model fit = the actual fit of the light curve (fit)
    # c = phased time data = the 'adjusted' or phased time data after the folding process (x axis)
    a, b, c = mtf.predict(1000, return_phased_times=True, adjust_offset=False)

    # storing values
    LCanalysisResults['modelPhaseGrid'] = a
    LCanalysisResults['modelFit'] = b
    LCanalysisResults['dataPhasedTime'] = c
    
    # INTERPOLATION
    # ---------------
    # we interpolate the light curve to remove blank spots without data, correct points so they align (fit and actual value)
    A = np.max(b) - np.min(b) # calculate the amplitude of model Fit
    mmax = np.min(b) # maximum magnitude

    # SAVING
    LCanalysisResults['A'] = A # amplitude
    LCanalysisResults['mmax'] = mmax # saving maximum magnitude
    LCanalysisResults['modTemplate'] = (b - mmax)/A # normalizing the model template using amplitude and max magnitude
    LCanalysisResults['dataTemplate'] = (mag - mmax)/A # normalizing the data template using amplitude and max magnitude
    LCanalysisResults['dataTemplateErr'] = magErr/A # normalizing the error template using amplitude
    
    # FITTING PART 2
    # -----------------
    modelFit2data = np.interp(c, a, LCanalysisResults['modTemplate']) # the model is fitted to the interpoated version of the data using our model template
    LCanalysisResults['modelFit2data'] = modelFit2data # saving the fitted data
    delmag = LCanalysisResults['dataTemplate'] - modelFit2data # the difference between the data template and the fitted data
    LCanalysisResults['rms'] = sigG(delmag) # RMS is calculated
    LCanalysisResults['chi'] = delmag/LCanalysisResults['dataTemplateErr'] # chi-2 is calculated (dividing the difference by the template error)
    LCanalysisResults['chi2dof'] = np.sum(LCanalysisResults['chi']**2)/np.size(LCanalysisResults['chi']) # calculating the degree of freedom
    LCanalysisResults['chi2dofR'] = sigG(LCanalysisResults['chi']) # normalized version of chi-2
    
    return LCanalysisResults 

# PLOTTING LIGHT CURVES
# -----------------------
def makeLCplot4(L1, L2, Z1, Z2, plotrootname='LCplot4', plotSave=False):
    '''
    This function plots 4 light curves: LINEAR light curves with the best period and the mean period, as well as the ZTF light curves with the best period and the mean period.
    '''
    fig, axs = plt.subplots(2,2, figsize=(14,10))  

    ### LINEAR plots
    ## TOP LEFT: with best-fit LINEAR period
    axs[0,0].set(xlabel='Data phased with BEST-FIT LINEAR period', ylabel='LINEAR normalized light curve')
    axs[0,0].set_xlim(-0.1, 1.1)
    axs[0,0].set_ylim(1.3, -0.3)
    # data
    xx, yy, zz = sort3arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'])
    axs[0,0].errorbar(xx, yy, zz, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for Plinear
    axs[0,0].plot(L1['modelPhaseGrid'], L1['modTemplate'], 'black', markeredgecolor='black', lw=2, fillstyle='top', linestyle='solid')
    # fit for mean period
    axs[0,0].plot(L2['modelPhaseGrid'], L2['modTemplate'], 'red', markeredgecolor='red', lw=2, fillstyle='top', linestyle='solid')

    ## TOP RIGHT: with the mean period
    axs[0,1].set(xlabel='Data phased with the MEAN period', ylabel='LINEAR normalized light curve')
    axs[0,1].set_xlim(-0.1, 1.1)
    axs[0,1].set_ylim(1.3, -0.3)
    # data
    xx, yy, zz = sort3arr(L2['dataPhasedTime'], L2['dataTemplate'], L2['dataTemplateErr'])
    axs[0,1].errorbar(xx, yy, zz, fmt='.r', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for mean period
    axs[0,1].plot(L2['modelPhaseGrid'], L2['modTemplate'], 'red', markeredgecolor='red', lw=2, fillstyle='top', linestyle='solid')
    
    
    ### ZTF plots
    ## BOTTOM LEFT: with best-fit ZTF period
    axs[1,0].set(xlabel='data phased with BEST-FIT ZTF period', ylabel='ZTF normalized light curve')
    axs[1,0].set_xlim(-0.1, 1.1)
    axs[1,0].set_ylim(1.3, -0.3)
    # data
    xx, yy, zz = sort3arr(Z1['dataPhasedTime'], np.array(Z1['dataTemplate']), np.array(Z1['dataTemplateErr']))
    axs[1,0].errorbar(xx, yy, zz, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for Pztf
    axs[1,0].plot(Z1['modelPhaseGrid'], Z1['modTemplate'], 'black', markeredgecolor='black', lw=2, fillstyle='top', linestyle='solid')
    # fit for mean period
    axs[1,0].plot(Z2['modelPhaseGrid'], Z2['modTemplate'], 'red', markeredgecolor='red', lw=2, fillstyle='top', linestyle='solid')

    ## BOTTOM RIGHT: with the mean period
    axs[1,1].set(xlabel='data phased with the MEAN period', ylabel='ZTF normalized light curve')
    axs[1,1].set_xlim(-0.1, 1.1)
    axs[1,1].set_ylim(1.3, -0.3)
    # data
    xx, yy, zz = sort3arr(Z2['dataPhasedTime'], np.array(Z2['dataTemplate']), np.array(Z2['dataTemplateErr']))
    axs[1,1].errorbar(xx, yy, zz, fmt='.r', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for mean period
    axs[1,1].plot(Z2['modelPhaseGrid'], Z2['modTemplate'], 'red', markeredgecolor='red', lw=2, fillstyle='top', linestyle='solid')
    
    if plotSave:
        plotName = plotrootname + '.png'
        plt.savefig(plotName, dpi=600)
        print('saved plot as:', plotName) 
    plt.show()     
    return

# PERIODOGRAM ANALYSIS
# ----------------------
def periodogram_blazhko(power, limit, perc_xshift):
    '''
    This function analyzes the peaks of periodograms from all RR Lyrae stars and determines if they are possible Blazhko stars.

    Arguments:
        power(array) arraw of power
        limit(float): a bar above which peaks are found
        shift(integer): how big of a range you want to zoom into; # In % how much to cut the x-axis range
    '''
    matches = []
    c = 0
    peaks_inside = []
    distance = []
    cless = 0
    cmore = 0
    indicator = 2
    x = 0
    a = 0
    

    while x != 1:
        peaks, _ = find_peaks(power, height=limit)
        if any(peaks) != True:
            limit -= 0.05
        else:
            x = 1

    if len(peaks)>50:
        indicator = 0
    else:

        while a != 1:
            max_peak_x = peaks[np.argmax(power[peaks])]
            min_lim = max_peak_x - int((len(power) * perc_xshift))
            max_lim = max_peak_x + int((len(power) * perc_xshift))
            p = power[min_lim:max_lim]
            
            if any(p) == False:
                perc_xshift -= 0.05
            elif len(p) == 1:
                perc_xshift += 0.05
            else:
                a = 1

        for x in range(len(peaks)):
            matches.append((power[peaks][x], peaks[x]))

        
        for x in power[peaks]:
            if x in p:
                c += 1
                peaks_inside.append(x)

        x_peaks = []
        for x in matches:
            if x[0] in peaks_inside:
                x_peaks.append(x[1])
        x_peaks.remove(max_peak_x)    

        
        for i in x_peaks:
            if max_peak_x<i:
                distance.append(i-max_peak_x)
                cmore += 1
            else:
                distance.append(max_peak_x-i)
                cless += 1

        if cmore>=1 and cless>=1:
            if ((c-1)==2 or (c-1)==3):
                indicator = 1
            else:
                within_range = [a for a in distance if a < 700]
                if len(within_range) in [2, 3]:
                    outof_range = [b for b in distance if b > 1000]
                    if len(outof_range) == len(distance) - len(within_range):
                        indicator = 1
                    else:
                        indicator = 0
                else:
                    indicator = 0
        else:
            indicator = 0

    return indicator, limit