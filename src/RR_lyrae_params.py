# IMPORTING LIBRARIES
# --------------
# AstroML & Astropy
from astroML.datasets import fetch_LINEAR_sample
from astropy.timeseries import LombScargle
from astroML.time_series import MultiTermFit

import sys
from tqdm import tqdm

import pandas as pd
import numpy as np
sys.path.insert(0,'../src/')
from helper import*
from blazhko_analysis import*

'''
This Python file contains functions used to calculate the parameters of RR Lyrae stars.
'''

def doPeriods(time, mag, magErr, nterms, nyquist=100, freqFac=1.05, verbose=False):
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
        if verbose: print('Engaging in period calculation...')
        ls = LombScargle(time, mag, magErr, nterms=nterms) # set up a LombScargle object to model the frequency and power
        frequencyAuto, powerAuto = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power
        best_freq = frequencyAuto[np.argmax(powerAuto)]
        frequency = np.arange(best_freq/freqFac, best_freq*freqFac, 5e-6)
        power = ls.power(frequency)  # compute LS periodogram again
        period = 1. / frequency
        best_period = period[np.argmax(power)] # choosing the period with the highest power
        return best_period, frequency, power
    
    except:
        if verbose: print('There is no data!')
        # if there is no data, assign everything to 0 or empty
        best_period = 0.0
        frequency = np.array(())
        power = np.array(())
        return best_period, frequency, power

def LINEARLS(dataL, Lid, nterms, verbose=False):
    '''
    This function accesses the LINEAR data and calculates the period.

    Arguments:
        LINEARids(list): list of all LINEAR ids
        LINEARlightcurves(array): light curve data
        order(int): the order of light curve in the list
        verbose(bool): printing statements  
    '''
    
    if verbose:
        print('------------------------------------------------------------')
        print('Period and light curve analysis for LINEAR ID =', Lid)
    ### first prepare light curve data
    # LINEAR light curve for this star (specified by provided LINEARid)
    tL, mL, mLerr = dataL[Lid].T

    ### now compute periods (using LombScargle from astropy.timeseries)
    # LINEAR-only period
    if verbose:
        print('computing LINEAR period...')
    Plinear, fL, pL = doPeriods(tL, mL, mLerr, nterms)    
    
    if verbose:
        print('LINEAR period = ', Plinear)

    return Plinear, fL, pL, tL, mL, mLerr

def ZTFs(ZTFdata, Lid, nterms, verbose=False):
    """
    This function calculates the period of a ZTF light curve by taking the median of the periods of the 3 filters.

    Arguments:
        ZTFdata(array): dataframe of ZTF data for a light curve
        nterms(int): number of terms for the Fourier fitting
        ZTFbands(list): list of filters, Default ["zg", "zr", "zi"]
        lsPS(Bool): flag, Default False
        verbose(bool): printing statements
    """

    # variables
    ZTFperiod_ograms = []
    ZTFbands=['zg', 'zr', 'zi']

    if verbose:
        print('And now for the ZTF counterpart -------------')
        print('  computing ZTF period...')

    BandData, timeZ, magZ, magErrZ = None, None, None, None
    for b in ZTFbands:
        if verbose: print('Period calculation for ',b, 'filter.')

        BandData = ZTFdata.loc[ZTFdata['filtercode'] == b]
        timeZ = BandData['mjd'].to_numpy()
        magZ = BandData['mag'].to_numpy()
        magErrZ = BandData['magerr'].to_numpy()     

        ZTFperiod, Zfreq, Zpow = doPeriods(timeZ, magZ, magErrZ, nterms)
        if ZTFperiod > 0:
            ZTFperiod_ograms.append((ZTFperiod, Zfreq, Zpow, timeZ, magZ, magErrZ))
        
    ZTFperiod_ograms.sort(key=lambda x: x[0], reverse=True)
    if len(ZTFperiod_ograms) < 3:
        ZTFbestPeriod, ZTFbestfreq, Zbestpow,timeZ, magZ, magErrZ = ZTFperiod_ograms[0]
    else:
        ZTFbestPeriod, ZTFbestfreq, Zbestpow,timeZ, magZ, magErrZ = ZTFperiod_ograms[1]


    if verbose:
        print('            ZTF period = ', ZTFbestPeriod)

    return ZTFbestPeriod, ZTFbestfreq, Zbestpow, timeZ, magZ, magErrZ

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
    LCanalysisResults['chi'] = delmag/LCanalysisResults['dataTemplateErr']
    LCanalysisResults['chi2dof'] = np.sum(LCanalysisResults['chi']**2)/np.size(LCanalysisResults['chi'])
    LCanalysisResults['chi2dofR'] = sigG(LCanalysisResults['chi'])
    return LCanalysisResults 