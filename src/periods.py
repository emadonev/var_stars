import numpy as np
import pandas as pd
import scipy as sc
from astroML.utils.decorators import pickle_results
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.datasets import fetch_LINEAR_sample
from ztfquery import lightcurve
import dask.dataframe as dd 
#--------
import sys
sys.path.insert(0,'../src/')
sys.path
from ZTF_data import*

'''
This Python file is used to calculate the periods of LINEAR and ZTF data with one term or multi-term. 
'''

