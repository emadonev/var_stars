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
    # fit for powertf
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
def periodogram_blazhko(power, limit, perc_xshift, limit_percentage, verbose:bool=False):
    '''
    This function analyzes the peaks of periodograms from all RR Lyrae stars and determines if they are possible Blazhko stars.

    Arguments:
        power(array) arraw of power
        limit(float): a bar above which peaks are found
        perc_xshift(float): in % how much to cut the x-axis range
        limit_percentage(float): percentage of inclusion of subsequent peaks around the maximum
        verbose(bool): testing mode, includes printing
    '''
    
    # DEFINING LISTS AND COUNTERS
    # -----------------------------
    matches = [] # list for the x-axis and power value matches
    peaks_inside = [] # list of all the peaks which are inside our cut range
    distance = [] # a list of all the distances between peaks
    peaks = [] # the starting list where the peaks will go
    
    a = 0 # counter for cutting the periodogram    
    c = 0 # counter for counting peaks inside specified range
    cless = 0 # counter for peaks before the max peak
    cmore = 0 # counter for peaks after the max peak
    indicator = np.nan # indicator of the blazhko effect
    
    # STEP 1: check if we have enough datapoints to even use
    if len(power)<5: # if there are less than 5 datapoints
        indicator = np.nan # assign the indicator as none
        if verbose: print("Blazhko effect cannot be calculated.")
    else:
        # if enough datapoints, continue with peak finding!
        
        # STEP 2: finding peaks
        while (len(peaks) == 0): # since we defined peaks as an empty list, it will stay that way until we find some peaks
            peaks, _ = find_peaks(power, height=limit) # find peaks above a certain limit
            limit -= 0.05 # lower the limit in case we find no peaks, if peaks found limit does not affect the code late
        # if we are testing, print the number of peaks, the limiting factor and plot the periodogram
        if verbose: 
            print(f"The number of peaks:{len(peaks)}")
            print(f"The limit:{limit}")
            plt.plot(power)
            plt.title("Full power spectrum")
            plt.show()
            
        # CHECK 1: what if there are too many peaks --> indicator of very noisy and not good data
        if len(peaks)>50: # if the number of peaks is too large
            indicator = 0 # set the indicator to 0
            if verbose: print(f"There are too many peaks >>> indicator is 0.")
        else: # if not, continue with the function
        
            # STEP 3: cutting the power spectrum to a specified range
            while a != 1: # while our counter is not 1, meaning we set it to 1 when the range is defined and cut
                
                max_peak_x = peaks[np.argmax(power[peaks])] # finding the highest peak and its x coordinate
                min_lim = max_peak_x - int((len(power) * perc_xshift)) # the lower limit is the x coordinate of the maximum value minus a certain percentage of the total data
                max_lim = max_peak_x + int((len(power) * perc_xshift))# the higher limit is the x coordinate of the maximum value plus a certain percentage of the total data
                p = power[min_lim:max_lim] # cutting the power spectrum
                
                # ADJUST 1: what if when lowering the percentage to cut we reach 0 because the maximum value is so close to the edges of the power spectrum?
                if perc_xshift == 0.0: # if percentage is 0
                    end_distance = len(power)-max_peak_x # calculating the end distance (from max_peak_x to the end of the list)
                    if max_peak_x > end_distance: # if the max_peak is closer to the end of the list
                        p = power[(max_peak_x-end_distance):] # cut from the distance to the end before the max peak all the way to the end
                        if verbose: 
                            print(f"The maximum value is too close to the end:{len(p)}")
                            plt.plot(p)
                            plt.title("The cut power spectrum")
                            plt.show()
                    else: # if the max peak is closer to the first element of the list
                        p = power[0:(max_peak_x*2)] # cut from 0 to twice the max peak
                        if verbose: 
                            print(f"The maximum value is too close to the beginning:{len(p)}")
                            plt.plot(p)
                            plt.title("The cut power spectrum")
                            plt.show()
                    a = 1 # when finished with adjusting, break the loop
                
                # ADJUST 2: what if there were no values in peaks? --> the range is to large and then it cannot cut because the cut goes beyond the list
                elif any(p) == False:
                    perc_xshift -= 0.05 # lessen the percentage so the range falls inside the list
                    perc_xshift = round(perc_xshift, 2) # make sure the percentage is rounded
                    if verbose: print(f"Range is too large:{perc_xshift}")
                # ADJUST 3: what if there is only 1 value for the peaks: the maximum peak --> the range is too small and cannot find other peaks
                elif len(p) == 1:
                    perc_xshift += 0.05 # enlarge the percentage so more peaks fall inside the range
                    perc_xshift = round(perc_xshift, 2)# make sure the percentage is rounded
                    if verbose: print(f"Range is too small:{perc_xshift}")
                else:
                    a = 1 # the process of adjusting is finished
                    if verbose: 
                        print(f"Range of cut:{len(p)}")
                        plt.plot(p)
                        plt.title("The cut power spectrum")
                        plt.show()
                    
            # STEP 4: find the x values of the peaks inside the specified range
            #--------------
            # STEP 4a: create matches of the power of every peak and its x value
            for x in range(len(peaks)):
                matches.append((power[peaks][x], peaks[x]))

            # STEP 4b: count the number of peaks inside the specified range (p)
            for y in power[peaks]:
                if y in p:
                    c += 1
                    peaks_inside.append(y)
            if verbose: print(f"Number of peaks inside the cut range:{c}")
            # STEP 4c: accessing the x values of the peaks inside the specified range
            x_peaks = []
            for j in matches: # for every match
                if j[0] in peaks_inside: # if the y value is inside the range
                    x_peaks.append(j[1]) # append the x value of the peak
            x_peaks.remove(max_peak_x)    
            
            # STEP 5: counting the number of peaks before and after the maximum value: crucial for Blazhko Effect recognition
            for i in x_peaks: # for every value inside the range
                if max_peak_x<i: # after the max peak
                    distance.append(i-max_peak_x) # calculate the distance from the peak to the max peak
                    cmore += 1 # count number of peaks after the max peak
                else:
                    distance.append(max_peak_x-i) # calculate the distance from the peak to the max peak
                    cless += 1 # count number of peaks before the max peak
            if verbose: print(f"Number of peaks before:{cless}, and after:{cmore}")
            # STEP 6: Determining if blazhko effect potentially present
            #--------
            # CHECK 2: there has to be at least one peak before and at least one after the main peak
            if cmore>=1 and cless>=1:
                # STEP 6a: if there are 2 or 3 peaks only in the range
                if ((c-1)==2 or (c-1)==3):
                    indicator = 1 # make the indicator as a positive mark
                    if verbose: print(f"Blazhko effect indicated")
                else:
                    # STEP 6b: check if even though there are more peaks that the other peaks are far away from the 2 or 3 close to the max peak
                    within_range = [a for a in distance if a < round((limit_percentage*len(p)))] # if the peaks within range have distances below 700 
                    if verbose: print(f"Limiting number for peak counting:{round((limit_percentage*len(p)))}")
                    if len(within_range) in [2, 3]: # if there are 2 or 3 peaks within the close range
                        outof_range = [b for b in distance if b > round(((limit_percentage+0.2)*len(p)))] # calculate the amount of peaks outside of the range
                        if verbose: 
                            print(f"Limiting number for peak counting over the range:{round(((limit_percentage+0.2)*len(p)))}")
                        if len(outof_range) == len(distance) - len(within_range): # assertion of number of the peaks
                            indicator = 1 # set the indicator as positive
                            if verbose: print(f"Blazhko effect indicated!")
                        else:
                            # if the assertion is incorrect, indicator is negative
                            indicator = 0
                            if verbose: print(f"Blazhko effect probably not present.")
                    else:
                        # if there are too many peaks withing range, set indicator to negative (0)
                        indicator = 0
                        if verbose: print(f"Blazhko effect probably not present.")
            else:
                # if there are not at least one peaks on each side, set indicator to 0
                indicator = 0
                if verbose: print(f"Blazhko effect probably not present.")

    return indicator, limit, distance # save the indicator value,the limit and the distances