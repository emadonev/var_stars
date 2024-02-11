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

def doPeriods(time, mag, magErr, nterms, lsPS=True, nyquist=100, freqFac=1.05, verbose=False):
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
        if verbose:print('Engaging in calculation, please wait...')
        ls = LombScargle(time, mag, magErr, nterms=nterms) # set up a LombScargle object to model the frequency and power
        frequencyAuto, powerAuto = ls.autopower(nyquist_factor=nyquist) # calculate the frequency and power
        if verbose:print('Frequency and power have been calculated.')
        best_freq = frequencyAuto[np.argmax(powerAuto)]
        frequency = np.arange(best_freq/freqFac, best_freq*freqFac, 5e-6)
        power = ls.power(frequency)  # compute LS periodogram again
        period = 1. / frequency
        best_period = period[np.argmax(power)] # choosing the period with the highest power
        if verbose:print('The best period is, ',best_period)
        if lsPS: 
            return best_period, frequency, power
        else:
            return best_period
    except:
        # if there is no data, assign everything to 0 or empty
        if verbose: print('Period calculation unsuccesful.')
        best_period = 0.0
        frequency = np.array(())
        power = np.array(())
        return best_period, frequency, power

def LINEARLS(LINEARlightcurves, Lid, verbose=False):
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
    tL, mL, mLerr = LINEARlightcurves.T

    ### now compute periods (using LombScargle from astropy.timeseries)
    nterms = 3
    # LINEAR-only period
    if verbose:
        print('  computing LINEAR period...')
    Plinear, fL, pL = doPeriods(tL, mL, mLerr, nterms, Lid, lsPS=True)    
    if verbose:
        print('            LINEAR period = ', Plinear)
    return Plinear, fL, pL, tL, mL, mLerr

def ZTFs(ZTFdata, Lid, lsPS=True, verbose=False):
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
    nterms = 3

    if verbose:
        print('And now for the ZTF counterpart -------------')

    if ZTFdata.empty == True:
        ZTFbestPeriod, ZTFbestfreq, Zbestpow = 0, 0, 0
        Zfreq, Zpow = np.array(()), np.array(())
        ZTFperiod_ograms.append((ZTFbestPeriod, Zfreq, Zpow))
        timeZ,magZ,magErrZ = np.array(()), np.array(()), np.array(())
    else:
        if verbose:
            print('  computing ZTF period...')
        for b in ZTFbands:
            BandData = ZTFdata.loc[ZTFdata['filtercode'] == b]
            timeZ = BandData['mjd']
            magZ = BandData['mag']
            magErrZ = BandData['magerr']                
            ZTFperiod, Zfreq, Zpow = doPeriods(timeZ, magZ, magErrZ, nterms, Lid, lsPS=lsPS)
            ZTFperiod_ograms.append((ZTFperiod, Zfreq, Zpow))
            
        ZTFperiod_ograms.sort(key=lambda x: x[0], reverse=True)
        if len(ZTFperiod_ograms) < 3:
            ZTFbestPeriod, ZTFbestfreq, Zbestpow = ZTFperiod_ograms[0]
        else:
            ZTFbestPeriod, ZTFbestfreq, Zbestpow = ZTFperiod_ograms[1]
    if verbose:
        print('            ZTF period = ', ZTFbestPeriod)

    return ZTFbestPeriod, ZTFbestfreq, Zbestpow, Zfreq, Zpow, timeZ, magZ, magErrZ

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

def RR_lyrae_analysis(end, i, Lid, dataL, dataZ, lc_analysis, ZTF_data_best, fits, periodograms, verbose=False):
    '''
    This function analyzes RR Lyrae light curve data by calculating periods, fitting light curves and conducting BE
    candidate analysis of local peaks. 

    Arguments:
        end(str) = how to save this iteration of the dataset
        i(int) = iterable
        Lids(list) = list of LINEAR ids
        ztfdata(dict) = dictionary of ZTF data
        lc_analysis(dict) = dictionary to save light curve analysis
        ZTF_data_lc(list) = place to save best ztf data
        fits(list) = list to save light curve fits
        periodograms(list) = list to save periodograms
    '''
    
            # accessing data
    
    if verbose:
        print('Current i:',i)
        print('Current LINEAR ID:', Lid)
        print(f'Shape of linear:{dataL.shape}, shape of ztf:{dataZ.shape}')

    # calculating the periods
    Plinear, fL, pL, tL, mL, meL = LINEARLS(dataL, Lid)
    Pztf, Zbestf, Zbestp, fZ, pZ, tZ, mZ, meZ = ZTFs(dataZ, Lid)

    if verbose:
        print(f'Plinear: {Plinear}, ZTFperiod: {Pztf}')

    # saving the ZTF data
    ZTF_data_best.append((Lid, (tZ, mZ, meZ)))

    # blazhko periodogram analysis
    fFoldedL, pFoldedL, fMainPeakL, fBlazhkoPeakL, BlazhkoPeriodL, BpowerRatioL, BsignificanceL = getBlazhkoPeak(fL, pL)
    if fZ.size==0 or pZ.size==0 or Plinear==0 or Pztf==0 or dataZ.shape[0]==0:
        fFoldedZ, pFoldedZ, fMainPeakZ, fBlazhkoPeakZ, BlazhkoPeriodZ, BpowerRatioZ, BsignificanceZ = np.array(()), np.array(()), 0, 0, 0, 0, 0
    else:
        fFoldedZ, pFoldedZ, fMainPeakZ, fBlazhkoPeakZ, BlazhkoPeriodZ, BpowerRatioZ, BsignificanceZ = getBlazhkoPeak(fZ, pZ)

    # period analysis
    Plinear = round(Plinear, 6)
    Pztf = round(Pztf, 6)
    Pmean = round((Plinear+Pztf)/2, 6)
    Pratio = round((Pztf/Plinear), 6)

    # saving the periodograms
    periodograms.append((Lid, (fL, pL, fFoldedL, pFoldedL), (fZ, pZ, fFoldedZ, pFoldedZ)))

    # fitting the light curves
    ntermsModels = 6

    if verbose: print('Starting to fit light curves!')

    if tZ.size==0 or mZ.size == 0 or meZ.size == 0 or Plinear == 0.0 or Pztf == 0.0:
        if verbose: print(f'We engaged with these parameters: tZ={tZ.size}, mZ={mZ.size}, meZ={meZ.size}, plinear={Plinear}, pztf={Pztf}')
        LINEAR_Plinear = {
            'modelPhaseGrid': np.array(()), 
            'modelFit': np.array(()), 
            'dataPhasedTime': np.array(()), 
            'A': 0.0, 
            'mmax': 0.0, 
            'modTemplate': np.array(()), 
            'dataTemplate': np.array(()), 
            'dataTemplateErr': np.array(()), 
            'modelFit2data': np.array(()), 
            'rms': 0.0, 
            'chi': 0.0, 
            'chi2dof': 0.0, 
            'chi2dofR': 0.0
        }
        LINEAR_Pmean = {
            'modelPhaseGrid': np.array(()), 
            'modelFit': np.array(()), 
            'dataPhasedTime': np.array(()), 
            'A': 0.0, 
            'mmax': 0.0, 
            'modTemplate': np.array(()), 
            'dataTemplate': np.array(()), 
            'dataTemplateErr': np.array(()), 
            'modelFit2data': np.array(()), 
            'rms': 0.0, 
            'chi': 0.0, 
            'chi2dof': 0.0, 
            'chi2dofR': 0.0
        }
        ZTF_Pztf = {
            'modelPhaseGrid': np.array(()), 
            'modelFit': np.array(()), 
            'dataPhasedTime': np.array(()), 
            'A': 0.0, 
            'mmax': 0.0, 
            'modTemplate': np.array(()), 
            'dataTemplate': np.array(()), 
            'dataTemplateErr': np.array(()), 
            'modelFit2data': np.array(()), 
            'rms': 0.0, 
            'chi': 0.0, 
            'chi2dof': 0.0, 
            'chi2dofR': 0.0
        }
        ZTF_Pmean = {
            'modelPhaseGrid': np.array(()), 
            'modelFit': np.array(()), 
            'dataPhasedTime': np.array(()), 
            'A': 0.0, 
            'mmax': 0.0, 
            'modTemplate': np.array(()), 
            'dataTemplate': np.array(()), 
            'dataTemplateErr': np.array(()), 
            'modelFit2data': np.array(()), 
            'rms': 0.0, 
            'chi': 0.0, 
            'chi2dof': 0.0, 
            'chi2dofR': 0.0
        }
    else:
        LINEAR_Plinear = LCanalysisFromP(tL, mL, meL, Plinear, ntermsModels)
        LINEAR_Pmean = LCanalysisFromP(tL, mL, meL, Pmean, ntermsModels)


        ZTF_Pztf = LCanalysisFromP(tZ, mZ, meZ, Pztf, ntermsModels)
        ZTF_Pmean = LCanalysisFromP(tZ, mZ, meZ, Pmean, ntermsModels)

    STAR = [Plinear, Pztf, Pmean, Pratio, np.size(tL), LINEAR_Plinear['rms'], round(LINEAR_Plinear['chi2dof'], 1), round(LINEAR_Plinear['chi2dofR'], 1),LINEAR_Pmean['rms'],
        round(LINEAR_Pmean['chi2dof'],1), round(LINEAR_Pmean['chi2dofR'],1), round(LINEAR_Plinear['mmax'],2), round(LINEAR_Plinear['A'],2),
        np.size(tZ), ZTF_Pztf['rms'], round(ZTF_Pztf['chi2dof'],1), round(ZTF_Pztf['chi2dofR'],1), ZTF_Pmean['rms'], round(ZTF_Pmean['chi2dof'],1), round(ZTF_Pmean['chi2dofR'],1), round(ZTF_Pztf['mmax'],2), round(ZTF_Pztf['A'],2),
        fMainPeakL, fBlazhkoPeakL, BlazhkoPeriodL, BpowerRatioL, BsignificanceL, fMainPeakZ, 
        fBlazhkoPeakZ, BlazhkoPeriodZ, BpowerRatioZ, BsignificanceZ]
        
    lc_analysis[Lid] = STAR
    fits.append((Lid, (LINEAR_Plinear, LINEAR_Pmean, ZTF_Pztf, ZTF_Pmean)))

    return lc_analysis, periodograms, fits, ZTF_data_best

