# IMPORTING LIBRARIES
# --------------------
import numpy as np

'''
This Python file contains helper functions necessary for this project.
'''


def sort3arr(a, b, c):
    '''
    This function sorts 3 arrays by their indexes.
    '''
    if len(a) != len(b) or len(a) != len(c):
        raise ValueError("Arrays must have the same length")

    ind = np.argsort(a)
    return a[ind], b[ind], c[ind]

def sort4arr(a, b, c, d):
    '''
    This function sorts 4 arrays by their indexes.
    '''
    ind = np.argsort(a)
    return a[ind], b[ind], c[ind], d[ind]

def sigG(x):
    '''
    This function normalizes statistical values by using the interquartile range.
    '''
    return 0.741*(np.percentile(x,75)-np.percentile(x,25))
