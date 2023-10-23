i = 294
verbose = True
# STEP 1: Gather the data
if verbose: print('Gathering data')
# --------------------------
# LINEAR
Lid = Lids[i]
if verbose: print(f'Lid: {Lid}   Index: {i}')

tL, mL, mLerr = data.get_light_curve(Lid).T
ra = rectascension[i]
dec = declination[i]

# ZTF
ZTFdata = getZTFlightcurve(ra, dec)
if verbose: 
    print(f'ZTF data >>> shape:{ZTFdata.shape}')
    print(f'ZTF data >>> columns:{ZTFdata.columns}')

# STEP 2: Calculating the periods + accessing the periodograms
if verbose: print('\nCalculating the periods + accessing the periodograms')
# ---------------------------------------------------------------
nterms = 3
Plinear, fL, pL = doPeriods(tL, mL, mLerr, nterms, lsPS=True)
if ZTFdata.empty:
    if verbose: print('ZTFdata is empty!')
    Pztf, fZ, pZ = 0,np.array(()),np.array(())
else:
    Pztf, fZ, pZ = getZTFperiod(ZTFdata, nterms, ZTFbands=['zg', 'zr', 'zi'], lsPS=True)

Pmean = (Plinear+Pztf)/2
Pratio = Pztf/Plinear
if verbose: 
    print(f'Pmean: {Pmean}')
    print(f'Pratio: {Pratio}')

Lindicator, Llimit = periodogram_blazhko(pL, 0.3, 0.2, 0.1,verbose=True)
if ZTFdata.empty==True:
    Zindicator, Zlimit = np.nan, np.nan
else:
    Zindicator, Zlimit = periodogram_blazhko(pZ, 0.3, 0.2, 0.1,verbose=True)