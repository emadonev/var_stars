# IMPORTING LIBRARIES
# --------------------
import numpy as np
import pandas as pd

'''
This Python file contains helper functions necessary for this project.
'''


def sort3arr(a, b, c):
    '''
    This function sorts 3 arrays by their indexes.
    '''
    if len(a) != len(b) or len(a) != len(c):
        raise ValueError("Arrays must have the same length")
    
    a = pd.Series(a)
    b = pd.Series(b)
    c = pd.Series(c)

    ind = np.argsort(a)
    return a.iloc[ind], b.iloc[ind], c.iloc[ind]

def sort4arr(a, b, c, d):
    '''
    This function sorts 4 arrays by their indexes.
    '''
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    c = pd.Series(c).reset_index(drop=True)
    d = pd.Series(d).reset_index(drop=True)
    ind = np.argsort(a)
    return a.iloc[ind], b.iloc[ind], c.iloc[ind], d.iloc[ind]

def sigG(x):
    '''
    This function normalizes statistical values by using the interquartile range.
    '''
    return 0.741*(np.percentile(x,75)-np.percentile(x,25))
