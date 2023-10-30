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