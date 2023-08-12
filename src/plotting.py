import numpy as np
import seaborn as sns
import pandas as pd
import scipy as sc
from matplotlib import pyplot as plt
from scipy.stats import norm
from astroML.utils.decorators import pickle_results
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.datasets import fetch_LINEAR_sample
from ztfquery import lightcurve
from matplotlib import ticker
import random
import pickle
#---------
import sys
sys.path.insert(0,'../src/')
sys.path
from ZTF_data import*

'''
This Python file is used to plot correlations, descriptive statistics and various forms of light curves. All of the plotting functions are found here.
'''

#data
data = fetch_LINEAR_sample(data_home='../inputs') # fetching the data from astroML data library
ZTF_data = data_ztf()

def plotting_descriptive_stats(data, bins):
    '''
    A function which only plots the descriptive statistics, or the histogram, of any data metric. The plot is structured as a 4x2 grid.
    The metrics are: mean, median, standard deviation, minimum, maximum, skewness, kurtosis and amplitude.

    Arguments:
    data(pandas DataFrame object): the data we want to analyze,
    bins(int): the number of bins in the histogram,
    '''

    # the list of columns
    columns = ['mean', 'median', 'std', 'amin', 'amax', 'skew','kurtosis','Amplitude']

    fig, ax = plt.subplots(4,2, figsize=(20,15)) # creating subplots with 2 columms and 3 rows
    ax = ax.flatten() # flatten the axes
    for i in range(len(columns)): #plot the same type of graph for every property
        ax[i].set_title(columns[i].upper()) # the title of the graph is the column name
        sns.histplot(data=data,x=columns[i],bins=100,ax=ax[i])
    plt.tight_layout()
    plt.show()

def plotting_correlations(data, x, y, hue, palette):
    '''
    Function for plotting statistical correlations. The graph is a scatterplot.

    Arguments:
    data(DataFrame): data we want to analyze,
    x(str): which column is the x axis,
    y(str): which column is the y axis,
    hue(str): by which column we are applying the hue,
    palette(str): color palette
    '''
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        palette=palette,
        hue=hue)
    plt.tight_layout()
    plt.show()

def plotting_lc(data_type, num, rows, columns):
    '''
    Function for plotting light curves from both LINEAR and ZTF datasets. 

    Arguments:
    data_type(str): either 'LINEAR' or 'ZTF' to denote which datatype to use. 
    num(int): number of light curves to plot
    rows(int): number of rows for the subplots
    columns(int): number of columns for the subplots
    '''
    fig, axs = plt.subplots(rows,columns, figsize=(20,18)) # creating subplots with 2 columms and 3 rows
    axs = axs.flatten() # flatten the axes
    if data_type=='LINEAR':
        indexes = [ random.choice(data.ids) for i in range(num)]
        for i in range(num):
            light_curve = data.get_light_curve(indexes[i]) # accessing light curve data
            time, mag, mag_error = light_curve.T # assigning time, magnitude and errors
            ax = axs[i]    
            ax.errorbar(time, mag, mag_error, fmt='.b', ecolor='blue')
            ax.set(xlabel='Time (days)', ylabel='magitude',title='LINEAR object {0}'.format(data.ids[10]))
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(plt.NullFormatter()) # no numbers on the x axis
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) # 4 ticks on the x axis
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4)) # 4 ticks on the y axis
        plt.tight_layout()
        plt.show()
    elif data_type=='ZTF':
        fig, axs = plt.subplots(rows,columns, figsize=(20,18)) 
        axs = axs.flatten() 
        indexes = [ random.randint(0,7009) for i in range(num)]
        for i in range(num):
            lc = ZTF_data[indexes[i]][1]
            time, mag, mag_error = lc['mjd'], lc['mag'], lc['magerr']
            ax = axs[i]    
            ax.errorbar(time, mag, yerr=mag_error, fmt='.b', ecolor='b')
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) 
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.set_title(indexes[i])
        plt.tight_layout()
        plt.show()
    else:
        print('Incorrect data_type. It must be either LINEAR or ZTF.')

def plotting_lc_phased(data, data_type, num, rows, columns):
    '''
    Function for plotting phased light curves from both LINEAR and ZTF datasets.

    Arguments:
    data(DataFrame): data from which we plot
    data_type(str): either 'LINEAR' or 'ZTF' to denote which datatype to use
    num(int): number of light curves to plot
    rows(int): number of rows for the subplots
    columns(int): number of columns for the subplots
    '''
    fig, axs = plt.subplots(rows,columns, figsize=(20,18)) # creating subplots with 2 columms and 3 rows
    axs = axs.flatten() # flatten the axes
    indexes = [ random.randint(0,7010) for i in range(num)]
    if data_type=='LINEAR':
        for i in range(num):
            period = data.iloc[indexes[i]]['Period']
            light_curve = data.get_light_curve(indexes[i]) # accessing light curve data
            time, mag, mag_error = light_curve.T
            phase = (time / period) % 1
            ax = axs[i]    
            ax.errorbar(phase, mag, yerr=mag_error, fmt='.b', ecolor='b')
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) 
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.set_title(indexes[i])
        plt.tight_layout()
        plt.show()
    elif data_type=='ZTF':
        for i in range(num):
            period = data.iloc[indexes[i]]['Period']
            lc = data[indexes[i]][1]
            time, mag, mag_error = lc['mjd'], lc['mag'], lc['magerr']
            phase = (time / period) % 1
            ax = axs[i]    
            ax.errorbar(phase, mag, yerr=mag_error, fmt='.b', ecolor='b')
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) 
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.set_title(indexes[i])
        plt.tight_layout()
        plt.show()