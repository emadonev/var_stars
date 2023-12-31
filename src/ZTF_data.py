import dask.dataframe as dd 
from ztfquery import lightcurve
from astroML.datasets import fetch_LINEAR_sample
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

data = fetch_LINEAR_sample(data_home='../inputs') # fetching the data from astroML data library

'''
This Python file contains 3 functions which are used for querying ZTF light curves using equatorial
coordinates of LINEAR light curves.
'''

def getZTFlightcurve(ra:float, 
                     dec:float, 
                     iD,
                     radius:float=3.0):
    """Defines a function for acessing the light curve data based on which area of the sky it should search

    Arguments:
        ra (float): rectascension coordinate of star
        dec (float): declination coordinate of star
        radius(float, default=3.0): how wide should the search radius be
    
    """
    try: 
        lcq = lightcurve.LCQuery() #this object is used to query for the data
        res = lcq.from_position(ra, dec, radius) # search for data which satisfies the beforementioned parameters
        try:
            # from the found dataset, access these select columns
            ZTFdata = res.data[['mjd', 'mag', 'magerr', 'catflags', 'filtercode']] 
        except:
            # if this doesn't work, print error message
            print(f"Something went wrong when extracting DataFrame: {iD}")
            try:
                # eliminate any points which are above a select number: M. Graham recommends to get rid of obvious spurious points
                ZTFdata = dd.ZTFdata.loc[ZTFdata['catflags']< 32768].compute()
            except:
                # if this doesn't work, print error message
                print(f"Something went wrong when filerting catflags: {iD}")
        finally:   
            # when done with everything, always remove select columns
            ZTFdata = ZTFdata.drop(['catflags'],axis=1)
    except:
        # if the light curve could not be found, assign the data to None
        print(f"Something went wrong with finding the light curve: {iD}")
        ZTFdata = None
    return ZTFdata # save the select light curve

def lc_access(iD):
    '''
    Defines a function for acessing the light curve data based on equatorial coordinates of the LINEAR dataset.

    Arguments:
        iD (integer): ID of every light curve in the LINEAR dataset
    '''
    StarAttributes = data.targets[iD] # access attributes of a light curve from the LINEAR dataset
    ra, dec = StarAttributes[3], StarAttributes[4] # access the equatorial coordinates
    light = (iD,getZTFlightcurve(ra, dec, iD)) # search for light curve
    return light # save the data

def data_ztf(name):
    '''
    Defines a function for creating a ZTF dataset using the previous functions. 

    Arguments:
        name(str): provide name to save file or load with
    '''
    file_path = name

    if os.path.isfile(name):
        print("Loading the data!")
        ZTF_data = np.load(name, allow_pickle=True) # loading the data
    else:
        print("Accessing the data!")
        num_cores = os.cpu_count()
        num = [x for x in range(7010)]
        ZTF_data = [] # empty list for the data

        # the asynchronous querying for data
        with ProcessPoolExecutor(max_workers=num_cores) as exe:
            ZTF_data = list(exe.map(lc_access, num))
        np.save(name, np.array(ZTF_data, dtype=object), allow_pickle=True) # saving the data as an .npy file which can be used across notebooks
    return ZTF_data