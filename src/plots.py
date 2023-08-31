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
from matplotlib.font_manager import FontProperties
import random
import pickle
#---------
import sys
sys.path.insert(0,'../src/')
sys.path
from ZTF_data import*

font = FontProperties()
font.set_family('avenir')
font.set_name('Avenir')
font.set_style('normal')
font.set_size('xx-large')


'''
This Python file is used to plot correlations, descriptive statistics and various forms of light curves. All of the plotting functions are found here.
'''

#data
dataL = fetch_LINEAR_sample(data_home='../inputs') # fetching the data from astroML data library

def lc_LINEAR():
    fig, axs = plt.subplots(3,2, figsize=(30,18)) # creating subplots with 2 columms and 3 rows
    axs = axs.flatten() # flatten the axes

    num = 6
    indexes = [ random.choice(dataL.ids) for i in range(num)]

    for i in range(num):
        light_curve = dataL.get_light_curve(indexes[i]) # accessing light curve data
        time, mag, mag_error = light_curve.T # assigning time, magnitude and errors
        ax = axs[i]    
        ax.errorbar(time, mag, mag_error, fmt='.k', ecolor='gray',lw=1, ms=4, capsize=1.5)
        ax.set_xlabel('Time (days)',fontproperties=font)
        ax.set_ylabel('Magnitude', fontproperties=font)
        ax.set_title('LINEAR object {0}'.format(indexes[i]), fontproperties=font)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(plt.NullFormatter()) # no numbers on the x axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) # 4 ticks on the x axis
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4)) # 4 ticks on the y axis
    fig.tight_layout()
    plt.show()

def lc_ZTF(DATA, c, color):
    '''
    This function draws 6 randomly selected ZTF light curves based on their filter.
    Arguments: 
        DATA(DataFrame): data from which to draw from
        c(string): color of cap
        color(string): color of line
    '''
    fig, axs = plt.subplots(3,2, figsize=(30,18)) # creating subplots with 2 columms and 3 rows
    axs = axs.flatten() # flatten the axes

    num = 6
    indexes = [ random.randint(0,7009) for i in range(num)]

    for i in range(num):
        lc = DATA[indexes[i]][1]
        time, mag, mag_error = lc['mjd'], lc['mag'], lc['magerr']
        ax = axs[i]    
        ax.errorbar(time, mag, mag_error, fmt=c, ecolor=color,lw=1, ms=4, capsize=1.5)
        ax.set_xlabel('Time (days)',fontproperties=font)
        ax.set_ylabel('Magnitude', fontproperties=font)
        ax.set_title('ZTF object {0}'.format(indexes[i]), fontproperties=font)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(plt.NullFormatter()) # no numbers on the x axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) # 4 ticks on the x axis
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4)) # 4 ticks on the y axis
    fig.tight_layout()
    plt.show()