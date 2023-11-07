# IMPORTING LIBRARIES
# --------------------

# AstroML & Astropy
from astroML.datasets import fetch_LINEAR_sample
from astropy.timeseries import LombScargle
from astroML.datasets import fetch_LINEAR_sample
from astroML.datasets import fetch_LINEAR_geneva
from astropy.timeseries import TimeSeries
from astropy.table import Table
from astroML.time_series import MultiTermFit

# ZTF
from ztfquery import lightcurve

# Basic libraries
import random
import pickle
import os
import sys
from tqdm import tqdm

# Plotting
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual

# DataFrame analysis
import pandas as pd
import dask.dataframe as dd 

# Math libraries
import numpy as np
import scipy as sc
from scipy.stats import norm
from scipy.signal import find_peaks
from array import array
# Importing custom libraries
# ----------------------------
sys.path.insert(0,'../src/')
from ZTF_data import *
from config import*
from descriptive_stats import *
from plots import *
from selection import *
from lc_analysis import *

'''
This Python file contains all the functions and classes necessary to determine Blazhko effect candidates from 
light curve data.
'''

# CALCUlATING LIGHT CURVE PROPERTIES
# ===================================

# LINEAR PERIOD CALCULATION
# ---------------------------

# First try with auto frequency grid
# and then "zoom-in" around the highest LS power peak 
# note: freqFac=1.02 allows search for Blazhko periods longer than 50*basic period, so ~25 days and longer
# note: freqFac=1.05 allows search for Blazhko periods longer than 20*basic period, so ~10 days and longer
def doPeriods(time, mag, magErr, nterms, lsPS=True, nyquist=100, freqFac=1.05):
    '''
    This function calculates the best period for RR Lyrae stars using the Lomb-Scargle periodogram. It first tries with the auto
    frequency grid, then it zooms in around the highest Lomb-Scargle power peak and searches for the best period. 

    Notes:
        - the freqFactor = 1.02 alows search for Blazhko periods longer than 50*basic period, so ~25 days and longer
        - the freqFactor = 1.05 allows search for Blazhko periods longer than 20*basic period, so ~10 days and longer

    Arguments:
        time(array): time data array for light curve
        mag(array): magnitude data array for light curve
        magErr(array): magnitude error data array for light curve
        nterms(int): number of Fourier terms with which to fit for best period
        lsPs(bool): decide if you want to save periodogram or not, default is True so yes
        nyquist(int): highest frequency of search
        freqFac(float): frequency for searching (defining the grid)
    '''
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

# calculating LINEAR periods
def LINEARLS(LINEARids, LINEARlightcurves, order, verbose=False):
    '''
    This function accesses the LINEAR data and calculates the period.

    Arguments:
        LINEARids(list): list of all LINEAR ids
        LINEARlightcurves(array): light curve data
        order(int): the order of light curve in the list
        verbose(bool): printing statements  
    '''
    
    # array index for this LINEARid
    LINEARid = LINEARids[order]
    
    if verbose:
        print('------------------------------------------------------------')
        print('Period and light curve analysis for LINEAR ID =', LINEARid)
    ### first prepare light curve data
    # LINEAR light curve for this star (specified by provided LINEARid)
    tL, mL, mLerr = LINEARlightcurves[LINEARid].T

    ### now compute periods (using LombScargle from astropy.timeseries)
    nterms = 3
    # LINEAR-only period
    if verbose:
        print('  computing LINEAR period...')
    Plinear, fL, pL = doPeriods(tL, mL, mLerr, nterms, lsPS=True)    
    if verbose:
        print('            LINEAR period = ', Plinear)
    return Plinear, fL, pL, tL, mL, mLerr

def getCoordinatesFromLINEARid(tbl, id0):
    '''
    Accesses the coordinates for ZTF data via LINEAR id values inside the table.

    Arguments:
        tbl(array): table of all the necessary light curve information
        id0(int): id of specific star you are trying to acccess data for
    '''
    return tbl[tbl['id'].astype(str)==id0]['ra'][0], tbl[tbl['id'].astype(str)==id0]['dec'][0]

# retrieve ZTF data for a single object specified by (RA, Dec)
def getZTFlightcurve(ra, dec, radius=3.0):
    '''
    This function uses the right ascension and declination coordinates to find LINEAR counterparts in ZTF data.

    Arguments:
        ra(float): right ascension values
        da(float): declination values
        radius(float): radius to search the sky with
    '''
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

def ZTFs(ZTFdata, lsPS=False, verbose=False):
    """
    This function calculates the period of a ZTF light curve by taking the median of the periods of the 3 filters.

    Arguments:
        ZTFdata(array): an array of ZTF data for a light curve
        nterms(int): number of terms for the Fourier fitting
        ZTFbands(list): list of filters, Default ["zg", "zr", "zi"]
        lsPS(Bool): flag, Default False
        nyquist(int): highest frequency, Default 300
        orig(bool): using the original doPeriodsOrig function or the new doPeriods function, default False
        verbose(bool): printing statements
    """
    ZTFperiod_ograms = []
    ZTFbands=['zg', 'zr', 'zi']
    nterms = 3

    if verbose:
        print('And now for the ZTF counterpart -------------')

    if ZTFdata.empty == True:
        #print("Empty")
        ZTFbestPeriod = 0
        Zfreq = np.array(())
        Zpow = np.array(())
        ZTFperiod_ograms.append((ZTFbestPeriod, Zfreq, Zpow))
    else:
        if verbose:
            print('  computing ZTF period...')
        for b in ZTFbands:
            BandData = ZTFdata.loc[ZTFdata['filtercode'] == b]
            timeZ = BandData['mjd']
            magZ = BandData['mag']
            magErrZ = BandData['magerr']
            ZTFperiod, Zfreq, Zpow = doPeriods(timeZ, magZ, magErrZ, nterms, lsPS=lsPS)
            ZTFperiod_ograms.append((ZTFperiod, Zfreq, Zpow))

            ZTFperiod_ograms = np.sort(ZTFperiod_ograms, axis=0)
            ZTFbestPeriod = ZTFperiod_ograms[1][0]
            ZTFbestfreq = ZTFperiod_ograms[1][1]
            Zbestpow = ZTFperiod_ograms[1][2]
    
    if verbose:
        print('            ZTF period = ', ZTFbestPeriod)

    return ZTFbestPeriod, ZTFbestfreq, Zbestpow, timeZ, magZ, magErrZ

# FITTING LIGHT CURVES
# ----------------------

# helper functions.
# -----------------
def sort3arr(a, b, c):
    '''
    This function sorts 3 arrays by their indexes.
    '''
    ind = np.argsort(a)
    return a[ind], b[ind], c[ind]

def sort4arr(a, b, c, d):
    '''
    This function sorts 4 arrays by their indexes.
    '''
    ind = np.argsort(a)
    return a[ind], b[ind], c[ind], d[ind]

def sigG(x):
    '''
    This function normalizes statistical values by using the interquartile range.
    '''
    return 0.741*(np.percentile(x,75)-np.percentile(x,25))

# light curve fitting from period.
# ---------------------------------
def LCanalysisFromP(time, mag, magErr, P, ntermsModels):
    '''
    This function fits light curve data with a sinusoidal wave using a certain number of terms and the period of
    the periodic light curve.

    Arguments:
        time(array): time data array of light curve
        mag(array): magnitude data array of light curve
        magErr(array): magnitude error data array of light curve
        P(float): the best fit period
        ntermsModels(int): the number of terms with which to fit the light curve
    '''
    LCanalysisResults = {}
    # first compute best-fit models for given period
    mtf = MultiTermFit(2*np.pi/P, ntermsModels)
    mtf.fit(time, mag, magErr)
    a, b, c = mtf.predict(1000, return_phased_times=True, adjust_offset=False)
    LCanalysisResults['modelPhaseGrid'] = a
    LCanalysisResults['modelFit'] = b
    LCanalysisResults['dataPhasedTime']= c
    # light curve template normalization: mag = A * t(phi) + mmax, where
    # phi is phase, t is template, A is amplitude and mmax is the magnitude at 
    #       maximum light (note: numerically it is the minimum value of mag)
    # also: we are using models for computing amplitude and mmax to avoid noise in data
    A = np.max(b) - np.min(b) 
    mmax = np.min(b) 
    LCanalysisResults['A'] = A
    LCanalysisResults['mmax'] = mmax
    LCanalysisResults['modTemplate'] = (b - mmax)/A 
    LCanalysisResults['dataTemplate'] = (mag - mmax)/A 
    LCanalysisResults['dataTemplateErr'] = magErr/A 
    # for chi2, first interpolate model fit to phases of data values
    modelFit2data = np.interp(c, a, LCanalysisResults['modTemplate'])
    LCanalysisResults['modelFit2data'] = modelFit2data 
    delmag = LCanalysisResults['dataTemplate'] - modelFit2data
    LCanalysisResults['rms'] = sigG(delmag)
    LCanalysisResults['chi'] = delmag/LCanalysisResults['dataTemplateErr']
    LCanalysisResults['chi2dof'] = np.sum(LCanalysisResults['chi']**2)/np.size(LCanalysisResults['chi'])
    LCanalysisResults['chi2dofR'] = sigG(LCanalysisResults['chi'])
    return LCanalysisResults 


# LATER ANALYSIS
# ---------------
# L1 = results of period analysis
def makeLCplot(L1, plotrootname='LCplot', plotSave=False):
    fig, ax = plt.subplots(1,1, figsize=(7,5))  

    ax.set(xlabel='data phased with best-fit LINEAR period', ylabel='LINEAR normalized light curve')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.3, -0.3)
    # data
    xx, yy, zz = sort3arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'])
    ax.errorbar(xx, yy, zz, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for Plinear
    ax.plot(L1['modelPhaseGrid'], L1['modTemplate'], 'red', markeredgecolor='red', lw=1, fillstyle='top', linestyle='dashed')
    
    if plotSave:
        plotName = plotrootname + '.png'
        plt.savefig(plotName, dpi=600)
        print('saved plot as:', plotName) 
    plt.show()     
    return
# L1 = results of period analysis

def makeLCplotBySeason(id0, L1, plotrootname='LCplotBySeason', plotSave=False):
    
    fig = plt.figure(figsize=(10, 12))
    fig.subplots_adjust(hspace=0.2, bottom=0.06, top=0.94, left=0.12, right=0.94)
    
    def plotPanel(ax, L1, season):
        ax.set(xlabel='phase', ylabel='normalized phased light curve')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.3, -0.4)
        # fit for Plinear
        ax.plot(L1['modelPhaseGrid'], L1['modTemplate'], 'red', markeredgecolor='red', lw=1, fillstyle='top', linestyle='dashed')
    
        # data
        xx, yy, zz, ww = sort4arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'], L1['obsTimes'])
        tSmin = 52520 + (season-1)*365
        tSmax = 52520 + season*365
        xxS = xx[(ww>tSmin)&(ww<tSmax)]
        yyS = yy[(ww>tSmin)&(ww<tSmax)]
        zzS = zz[(ww>tSmin)&(ww<tSmax)]
        wwS = ww[(ww>tSmin)&(ww<tSmax)]
        ax.errorbar(xxS, yyS, zzS, fmt='.b', ecolor='blue', lw=1, ms=4, capsize=1.5, alpha=0.3)
        textString = "LINEAR season " + str(season)
        ax.text(0.03, 0.96, textString, ha='left', va='top', transform=ax.transAxes)
        textString = "MJD=" + str(tSmin) + ' to ' + str(tSmax)
        ax.text(0.53, 0.96, textString, ha='left', va='top', transform=ax.transAxes)

        
    # plot each season separately 
    for season in range(1,7):
        # plot the power spectrum
        ax = fig.add_subplot(321 + season-1)
        plotPanel(ax, L1, season)
        if (season==1):
            ax.set(title='LINEAR object {0}'.format(id0))

    if plotSave:
        plotName = plotrootname + '.png'
        plt.savefig(plotName, dpi=600)
        print('saved plot as:', plotName) 
    plt.show()     
    return


def plotBlazhkoPeaksLINEAR(id0, fL, pL, fac=1.008, plotSave=False, verbose=True):
    
    print('LINEAR ID=', id0)
    flin = fL[np.argmax(pL)]

    fFolded, pFolded, fMainPeak, fBlazhkoPeak, BlazhkoPeriod, BpowerRatio, Bsignificance = \
    getBlazhkoPeak(fL, pL, verbose=verbose)

    ## at some point, we will read periodograms back from files...
    fig = plt.figure(figsize=(10, 12))
    fig.subplots_adjust(hspace=0.1, bottom=0.06, top=0.94, left=0.12, right=0.94)

    # plot the power spectrum
    ax = fig.add_subplot(321)

    ax.plot(fL, pL, c='b')
    ax.plot([flin, flin], [0,1], lw = 1, c='r', ls='--')
    ax.plot([fBlazhkoPeak, fBlazhkoPeak], [0, 0.7*np.max(pFolded)], lw = 1, c='r', ls='--')
    ax.plot([2*flin-fBlazhkoPeak, 2*flin-fBlazhkoPeak], [0, 0.7*np.max(pFolded)], lw = 1, c='r', ls='--')
    # show 1 year alias
    f1yr = flin+1/365.0
    ax.plot([f1yr, f1yr], [0,0.7*np.max(pFolded)], lw = 1, ls='-.', c='green')
    f1yr = flin-1/365.0
    ax.plot([f1yr, f1yr], [0,0.7*np.max(pFolded)], lw = 1, ls='-.', c='green')

    ax.text(0.03, 0.96, "LINEAR", ha='left', va='top', transform=ax.transAxes)
    if (fBlazhkoPeak > flin*fac):
        ax.set_xlim(0.99*(2*flin-fBlazhkoPeak), 1.01*fBlazhkoPeak)
    else:
        ax.set_xlim(flin/fac, flin*fac)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ylim = ax.get_ylim()
    ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
    if ymax>1.0: ymax=1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Lomb-Scargle power')
    ax.set_xlabel('frequency (d$^{-1}$)')

    # plot folder power spectrum
    ax = fig.add_subplot(322)

    ax.plot(fFolded, pFolded, c='b')
    ax.plot([fBlazhkoPeak, fBlazhkoPeak], [0,0.4*np.max(pFolded)], lw = 1, ls='--', c='r')
    # show 1 year alias
    f1yr = flin+1/365.0
    ax.plot([f1yr, f1yr], [0,0.4*np.max(pFolded)], lw = 1, ls='-.', c='green')
    
    powerFar = pFolded[fFolded>fBlazhkoPeak]  # frequencies beyond the second peak
    powerFarMedian = np.median(powerFar)      # the median power
    powerFarRMS = np.std(powerFar)            # standard deviation, i.e. "sigma"
    noise5sig = powerFarMedian+5*powerFarRMS
    
    if (fBlazhkoPeak > flin*fac):
        ax.plot([flin+0.5*(fBlazhkoPeak-flin), 1.01*fBlazhkoPeak], [noise5sig, noise5sig], lw = 1, ls='--', c='cyan')
        ax.set_xlim(flin, 1.01*fBlazhkoPeak)
    else:
        ax.plot([flin+0.5*(fBlazhkoPeak-flin), flin*fac], [noise5sig, noise5sig], lw = 1, ls='--', c='cyan')
        ax.set_xlim(flin, flin*fac)

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ylim = ax.get_ylim()
    ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
    if ymax>1.0: ymax=1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel('folded power')
    ax.set_xlabel('frequency (d$^{-1}$)')

    if plotSave:
        plotName = '../plots/Blazhko.png'
        plt.savefig(plotName, dpi=600)
        print('saved plot as:', plotName) 
    plt.show()     
    return fBlazhkoPeak   

## plot LINEAR light curve and mark seasons
def plotLINEARmarkSeasons(id0, LINEARlightcurves):
    tL, mL, mLerr = LINEARlightcurves[id0].T
    fig, ax = plt.subplots()
    ax.errorbar(tL, mL, mLerr, fmt='.b', ecolor='blue')
    ax.set(xlabel='Time (days)', ylabel='LINEAR magnitude', title='LINEAR object {0}'.format(id0))
    ax.invert_yaxis()
    plt.xlim(np.min(tL)-200, np.max(tL)+200)

    for s in range(1, 8):
        tS = 52550 + (s-1)*365
        ax.plot([tS, tS], [np.min(mL)-0.1, np.max(mL)+0.1], c='r')
    plt.show()     
    return

# plot standard plots to support visual analysis
def plotAll(idList, LINEARmetadata, LINEARlightcurves, verbose=True):
    for id0 in idList:
        Pcomparison, fL, pL, LINEAR_Plinear = LINEARLS(LINEARmetadata, LINEARlightcurves, id0) 
        fBlazhkoPeak = plotBlazhkoPeaksLINEAR(id0, fL, pL, fac=1.008, plotSave=False, verbose=verbose)
        plotLINEARmarkSeasons(id0, LINEARlightcurves)
        makeLCplotBySeason(id0, LINEAR_Plinear)
    return fL[np.argmax(pL)], fBlazhkoPeak


# BLAZHKO PEAK ANALYSIS
# -------------------------
# given frequency and Lomb-Scargle power, return parameters for a candidate Blazhko peak
def getBlazhkoPeak(freq, LSpow, verbose=False):
    '''
    This function searches for the Blazhko effect in periodograms of light curves. It searches for 2 subsequent peaks by
    folding the light curve and searching for local peaks. It also accounts for year aliases. 

    Arguments:
        freq(array): frequency array
        LSpow(array): lomb-scargle power array
        verbose(bool): print statements
    '''
    # no. of points
    Npts = np.size(LSpow)
    # index for the main peak
    imax = np.argmax(LSpow)
    # 1 year alias frequency (factor 1.02 to push it a bit over the maximum)
    f1yr = freq[imax] + 1.02/365
    # iDelta is the max. width for folding around the main peak
    if (imax < Npts/2):
        iDelta = imax
    else:
        iDelta = Npts - imax
    # folded versions 
    fFolded = freq[imax:imax+1+iDelta]  
    pLeft = LSpow[imax-iDelta:imax+1]  
    pRight = LSpow[imax:imax+1+iDelta]
    pFolded = 0*fFolded
    for i in range(0, iDelta-1):
        # multiply the two branches to increase SNR 
        pFolded[i] = pLeft[-i-1] * pRight[i] 
    # now search for the strongest secondary minimum (after the main one at index=0)
    foundMin = 0
    foldedMax = 0 
    ifoldedMax = 0
    # NB: the first point is the highest by construction (the main peak)
    for i in range(1, iDelta):
        if ((foundMin==0)&(pFolded[i] > pFolded[i-1])):
            # the first time we passed through a local minimum
            if (fFolded[i]>f1yr): foundMin = 1
        if foundMin:
            # after the first local minimum, remember the maximum power and its location
            if (pFolded[i] > foldedMax):
                foldedMax = pFolded[i]
                ifoldedMax = i
    # done, return useful quantities       
    fMainPeak = freq[imax] # location of the main peak
    fBlazhkoPeak = fFolded[ifoldedMax] # location of the second strongest peak
    BlazhkoPeriod = 1/(fBlazhkoPeak - fMainPeak) # expression for Blazhko period
    BpowerRatio = pFolded[ifoldedMax]/fFolded[0] # the ratio of power for the 2nd and 1st peaks
    # now compare the second peak's strength to the power at larger frequencies (presumably noise)
    powerFar = pFolded[fFolded>fBlazhkoPeak]  # frequencies beyond the second peak
    powerFarMedian = np.median(powerFar)      # the median power
    powerFarRMS = np.std(powerFar)            # standard deviation, i.e. "sigma"
    Bsignificance = (pFolded[ifoldedMax]-powerFarMedian)/powerFarRMS  # how many sigma above median?
    if (verbose):
        print('main frequency (1/day):', fMainPeak)
        print('detected second peak at index:', ifoldedMax)
        print('Blazhko peak frequency (1/day):', fBlazhkoPeak)
        print('Blazhko peak relative strength:', BpowerRatio)
        print('median power beyond Blazhko peak:', powerFarMedian)
        print('power rms beyond Blazhko peak:', powerFarRMS)
        print('Blazhko peak significance:', Bsignificance)
        print('Blazhko period (day):', BlazhkoPeriod)
    return fFolded, pFolded, fMainPeak, fBlazhkoPeak, BlazhkoPeriod, BpowerRatio, Bsignificance

# BUILDING THE VISUAL INTERFACE
# ================================
# Building a class for the visual interface
class BlazhkoAnalyzer:
    '''
    This class builds a customizable interface for visual inspection of BE candidates.

    INPUTS:
        length(int) = how large is your dataset in length
        data_lc(dataframe) = the dataset we are inspecting
        save_data(dataframe) = where to send BE visual candidates
        ids(list) = list of LINEAR ids
        period(array) = array of periodograms
        fit(array) = array of fits for light curves
    '''

    # initialization of the class
    def __init__(self, length, data_lc, save_data, ids, period, fit):
        # initialize every variable in use for this class
        self.length = length
        self.data_lc = data_lc
        self.save_data = save_data
        self.ids = ids
        self.period = period
        self.fit = fit
        # used for the for loop
        self.current_i = None

        # initialize plotting 
        self.gen = self.plot_light_curves()
        # initalize window for where to show plot
        self.output_plot = widgets.Output()
        
        # Buttons initialization
        self.button_keeping = widgets.Button(description='Keep')
        self.button_continue = widgets.Button(description='Continue')
        
        # Assigning functions to the buttons
        self.button_keeping.on_click(self.on_keep_click)
        self.button_continue.on_click(self.on_continue_click)
        
        # display buttons and plot
        display(self.output_plot, self.button_keeping, self.button_continue)
        
        # starting the for loop
        self.on_continue_click(None)
    
    # DEFINING NECESSARY FUNCTIONS
    # -------
    def plot_light_curves(self):
        '''
        This function plots the light curve data, periodograms and displays important information.
        '''
        for i in range(self.length):
            self.current_i = i # counter for the for loop
            LID = self.ids[i]
            blazhko_analysis(self.data_lc, Lid=LID, order=i, PD=self.period, fits=self.fit, name=str(LID)) # plot
            yield # don't continue until button is pressed

    def on_continue_click(self, b):
        '''
        This button defines what happens when the CONTINUE button is clicked: the program moves
        on to the next light curve.
        '''
        with self.output_plot:
            clear_output(wait=True) # clear the previous output
            try:
                next(self.gen) # generate the next plot and update current_i
            except StopIteration: # when the for loop is finished, disable the button
                print("No more plots.")
                self.button_continue.disabled = True

    def on_keep_click(self, b):
        '''
        This function defines what happens when the KEEP button is clicked: the program
        saves the specific row or light curve information into the save_data dataframe, for later use.
        '''
        row = pd.DataFrame(self.data_lc.iloc[[int(self.current_i)]]) # assign the current row we are analyzing
        # concatenate that row with the save_data dataframe
        self.save_data = pd.concat([self.save_data, row.reset_index(drop=True)], ignore_index=True, axis=0)

    # Saving the save_data dataframe for outside the class
    def get_save_data(self):
        return self.save_data