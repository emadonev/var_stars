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
This Python file is used to create descriptive statistics tables for a variety of data sources. 
'''

data = fetch_LINEAR_sample(data_home='../inputs') # fetching the data from astroML data library
ZTF_data = data_ztf('ZTF_light_curves.npy')


def lc_descriptive_stats_table_LINEAR(data_type):
    '''
    This function creates a table of descriptive statistics for LINEAR data.
    The arguments are mean, median, standard deviation, minimum, maximum, skewness, kurtosis and amplitude.

    Arguments:
        data_type(string): the data column we want (1 or 2, where 1 is magnitude and 2 is magnitude error).
    '''
    descriptive_stats = pd.DataFrame()
    print(f"Argument received: {data_type}")

    if data_type == 'mag':
        for i in data.ids:
            light_curve = data.get_light_curve(i) # getting the light curve information
            df = pd.DataFrame(light_curve.T[1], columns=[i]) # creating a table for every light curve
            dt = df.agg([np.mean, np.median, np.std, np.min, np.max]) # calculating the statistics for every value in the magnitudes column
            dt = dt.transpose() # making the rows (mean, median,...) into columns
            dt['skew'], dt['kurtosis'] = sc.stats.skew(df[i]), sc.stats.kurtosis(df[i]) # calculating skewness and kurtosis
            descriptive_stats = pd.concat([descriptive_stats, dt], axis=0) # add the data to the existing table
        descriptive_stats.index.name='index'
        descriptive_stats['Amplitude'] = descriptive_stats['amax'] - descriptive_stats['amin']
    elif data_type == 'magerr':
        for j in data.ids:
            light_curve = data.get_light_curve(j) # getting the light curve information
            df = pd.DataFrame(light_curve.T[2], columns=[j]) # creating a table for every light curve
            dt = df.agg([np.mean, np.median, np.std, np.min, np.max]) # calculating the statistics for every value in the magnitudes column
            dt = dt.transpose() # making the rows (mean, median,...) into columns
            dt['skew'], dt['kurtosis'] = sc.stats.skew(df[j]), sc.stats.kurtosis(df[j]) # calculating skewness and kurtosis
            descriptive_stats = pd.concat([descriptive_stats, dt], axis=0) # add the data to the existing table
        descriptive_stats.index.name='index'
        descriptive_stats['Amplitude'] = descriptive_stats['amax'] - descriptive_stats['amin']
    else:
        print('Wrong input for data_type! Must be either mag or magerr.')
    return descriptive_stats

def lc_descriptive_stats_table_ZTF(data_type, DATA):
    '''
    This function creates a table of descriptive statistics for ZTF data.
    The arguents are mean, median, standard deviation, minimum, maximum, skewness, kurtosis and amplitude.

    Arguments:
        data_type(str): can be 'mag' or 'magerr', denoting which type to analyze.
        DATA(list): input data based on filter
    '''
    num = [x for x in range(7010)]
    descriptive_stats = pd.DataFrame()
    if data_type == 'mag':
        for i in num: # looping over the id's of the dataset
            if DATA[i][1].shape[0] > 0:
                df = pd.DataFrame(DATA[i][1]['mag']) 
                dt = df.agg([np.mean, np.median, np.std, np.min, np.max]) 
                dt = dt.transpose()
                dt['skew'], dt['kurtosis'] = sc.stats.skew(df['mag']), sc.stats.kurtosis(df['mag'])
                descriptive_stats = pd.concat([descriptive_stats, dt], axis=0) 
            else:
                dt = pd.DataFrame()
                descriptive_stats = pd.concat([descriptive_stats, dt], axis=0)
        descriptive_stats.index.name='index'
        descriptive_stats['Amplitude'] = descriptive_stats['amax'] - descriptive_stats['amin']
    elif data_type == 'magerr':
        for i in num: # looping over the id's of the dataset
            if DATA[i][1].shape[0] > 0:
                df = pd.DataFrame(DATA[i][1]['magerr']) 
                dt = df.agg([np.mean, np.median, np.std, np.min, np.max]) 
                dt = dt.transpose()
                dt['skew'], dt['kurtosis'] = sc.stats.skew(df['magerr']), sc.stats.kurtosis(df['magerr'])
                descriptive_stats = pd.concat([descriptive_stats, dt], axis=0) 
            else:
                dt = pd.DataFrame()
                descriptive_stats = pd.concat([descriptive_stats, dt], axis=0)
        descriptive_stats.index.name='index'
        descriptive_stats['Amplitude'] = descriptive_stats['amax'] - descriptive_stats['amin']
    return descriptive_stats

def lc_descriptive_dataframe(data, column):
    '''
    This function creates a table for descriptive statistics for DataFrame objects.

    Arguments:
        data(DataFrame): the dataframe we are analyzing
        column(str): the column which we want to analyze
    '''
    descriptive_stats = pd.DataFrame()

    df = pd.DataFrame(data[column]) # creating a table for every light curve
    dt = df.agg([np.mean, np.median, np.std, np.min, np.max]) # calculating the statistics for every value in the magnitudes column
    dt = dt.transpose() # making the rows (mean, median,...) into columns
    dt['skew'], dt['kurtosis'] = sc.stats.skew(df), sc.stats.kurtosis(df) # calculating skewness and kurtosis
    descriptive_stats = pd.concat([descriptive_stats, dt], axis=0) # add the data to the existing table
    descriptive_stats.index.name='index'
    descriptive_stats['Amplitude'] = descriptive_stats['amax'] - descriptive_stats['amin']
    return descriptive_stats
