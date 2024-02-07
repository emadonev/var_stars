from astropy.timeseries import LombScargle
from astroML.datasets import fetch_LINEAR_geneva
from astropy.table import Table
from ztfquery import lightcurve

import os
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np

'''
This Python script contains all of the functions necessary for selecting RR Lyrae stars and
then subsequently finding the corresponding ZTF stars.
'''

# SELECTION OF RR LYRAE STARS
# ==============
def select_good_LINEAR(LDATA):
    '''
    This function selects correct LINEAR Ids and calculates the preliminary periods of light curves.

    Arguments:
        NAME(string): name to save with
        LDATA = LINEAR data
    '''
    # ----------------------
    # convert to dataframe for easier manipulation
    IDs = [x for x in LDATA.ids] # making a list of ids

    # ACCESSING THE OFFICIAL PERIODS
    #------------
    dataPeriods = fetch_LINEAR_geneva() # accessing the good data
    dataPeriods = pd.DataFrame(dataPeriods)
    ID_orig = list(dataPeriods['LINEARobjectID'].to_numpy()) # accessing the good ID's

    LINEAR_data = pd.DataFrame(())

    for id in ID_orig: # for every star in the good dataset 
        if id in IDs:# if the original ID is in our list of ID's
            index = ID_orig.index(id)
            row = pd.DataFrame(dataPeriods.iloc[[int(index)]]) # assign the current row we are analyzing
            # concatenate that row with the save_data dataframe
            LINEAR_data = pd.concat([LINEAR_data, row.reset_index(drop=True)], ignore_index=True, axis=0)

    return LINEAR_data

# ZTF DATA ACCESS
# ==============
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
    except:
        ZTFdata = pd.DataFrame(())
    return ZTFdata