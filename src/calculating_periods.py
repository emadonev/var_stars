# IMPORTS
#----------

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
#--------

#DESCRIPTION
#------------
'''
This Python file is used to calculate the periods of LINEAR and ZTF data with one term or multi-term. 
'''

# DATA
#--------
#ZTF_data = data_ztf()
data = fetch_LINEAR_sample(data_home='../inputs')


#-----------
# FUNCTIONS
#-----------

def period_comp_LINEAR(ID):
    t, mag, mager = data.get_light_curve(ID).T # get the data for every light curve
    ls = LombScargle(t, mag, mager, nterms=4) # set up a LombScargle object to model the frequency and power
    frequency, power = ls.autopower(nyquist_factor=350) # calculate the frequency and power

    period = 1. / frequency # calculating the periods
    best_period = period[np.argmax(power)] # choosing the period with the highest power
    best_frequency = frequency[np.argmax(power)] # choosing the frequency with the highest power
    N = len(t) # number of points used in calculation

    parameters = [ID, best_period, best_frequency, N]
    return parameters