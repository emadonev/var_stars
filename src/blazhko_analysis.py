# IMPORTING LIBRARIES
# ======================
from astroML.datasets import fetch_LINEAR_sample
from astropy.timeseries import LombScargle
from astroML.time_series import MultiTermFit
import ipywidgets as widgets
from IPython.display import display, clear_output

import pickle
import os
import sys
from tqdm import tqdm

import pandas as pd
import numpy as np
sys.path.insert(0,'../src/')
from helper import*
import BE_plotting

'''
This Python file contains all the functions and classes necessary to determine Blazhko effect candidates from 
light curve data.
'''

# BLAZHKO PEAK ANALYSIS
# -------------------------
# given frequency and Lomb-Scargle power, return parameters for a candidate Blazhko peak
def getBlazhkoPeak(freq, LSpow, verbose=False):
    '''
    This function searches for the Blazhko effect in periodograms of light curves. It searches for 2 subsequent peaks by
    folding the light curve and searching for local peaks. It also accounts for year aliases. 

    Arguments:
        freq(array): frequency array
        LSpow(array): lomb-scargle power array
        verbose(bool): print statements
    '''
    # no. of points
    Npts = np.size(LSpow)
    # index for the main peak
    imax = np.argmax(LSpow)
    # 1 year alias frequency (factor 1.02 to push it a bit over the maximum)
    f1yr = freq[imax] + 1.02/365
    # iDelta is the max. width for folding around the main peak
    if (imax < Npts/2):
        iDelta = imax
    else:
        iDelta = Npts - imax
    # folded versions 
    fFolded = freq[imax:imax+1+iDelta]  
    pLeft = LSpow[imax-iDelta:imax+1]  
    pRight = LSpow[imax:imax+1+iDelta]
    pFolded = 0*fFolded
    for i in range(0, iDelta-1):
        # multiply the two branches to increase SNR 
        pFolded[i] = pLeft[-i-1] * pRight[i] 
    # now search for the strongest secondary minimum (after the main one at index=0)
    foundMin = 0
    foldedMax = 0 
    ifoldedMax = 0
    # the first point is the highest by construction (the main peak)
    for i in range(1, iDelta):
        if ((foundMin==0)&(pFolded[i] > pFolded[i-1])):
            # the first time we passed through a local minimum
            if (fFolded[i]>f1yr): foundMin = 1
        if foundMin:
            # after the first local minimum, remember the maximum power and its location
            if (pFolded[i] > foldedMax):
                foldedMax = pFolded[i]
                ifoldedMax = i
    # done, return useful quantities       
    fMainPeak = freq[imax] # location of the main peak
    fBlazhkoPeak = fFolded[ifoldedMax] # location of the second strongest peak
    if (fBlazhkoPeak - fMainPeak)==0:
        BlazhkoPeriod = 0
    else:
        BlazhkoPeriod = 1/(fBlazhkoPeak - fMainPeak) # expression for Blazhko period
        if BlazhkoPeriod == np.inf:
            BlazhkoPeriod = 0
    BpowerRatio = pFolded[ifoldedMax]/fFolded[0] # the ratio of power for the 2nd and 1st peaks
    if BpowerRatio==np.inf:
        BpowerRatio = 0
    # now compare the second peak's strength to the power at larger frequencies (presumably noise)
    powerFar = pFolded[fFolded>fBlazhkoPeak]  # frequencies beyond the second peak
    powerFarMedian = np.median(powerFar)      # the median power
    powerFarRMS = np.std(powerFar)            # standard deviation, i.e. "sigma"
    if powerFarRMS==0:
        Bsignificance = 0
    else:
        Bsignificance = (pFolded[ifoldedMax]-powerFarMedian)/powerFarRMS  # how many sigma above median?
        if Bsignificance==np.inf:
            Bsignificance = 0

    if (verbose):
        print('main frequency (1/day):', fMainPeak)
        print('detected second peak at index:', ifoldedMax)
        print('Blazhko peak frequency (1/day):', fBlazhkoPeak)
        print('Blazhko peak relative strength:', BpowerRatio)
        print('median power beyond Blazhko peak:', powerFarMedian)
        print('power rms beyond Blazhko peak:', powerFarRMS)
        print('Blazhko peak significance:', Bsignificance)
        print('Blazhko period (day):', BlazhkoPeriod)
    return fFolded, pFolded, round(fMainPeak, 4), round(fBlazhkoPeak, 4), round(BlazhkoPeriod, 4), round(BpowerRatio, 4), round(Bsignificance, 4)

# BLAZHKO EFFECT CANDIDATES
# =============================
def blazhko_determine(df, dfnew, indic, bscore):
    '''
    This algorithm sorts through a DataFrame of light curve parameters and decides which stars are
    BE candidates. The parameters we use for determining BE candidates are amplitude, chi2 of 2 both LINEAR and ZTF, 
    period and the periodogram analysis (sign of local peaks for BE).

    Arguments:
        df(DataFrame) = input dataframe
        dfnew(DataFrame) = new dataframe for inputing good candidates
    '''
    df_stat = pd.DataFrame([[0 for i in range(9)] for j in range(df.shape[0])], columns=['L2', 'Z2', 'LZ3','LZ4','LZ5','LINEAR periodogram', 'ZTF periodogram', 'Amplitude', 'Period difference'])
    for i in range(df.shape[0]):
        
        # STEP 1: getting rid of bad data
        # ===============
        # Amplitude for each star needs to be less than 2 mags
        if (df['Lampl'][i]<2 or df['Zampl'][i]<2) and (df['Plinear'][i]<4 or df['Pztf'][i]<4) and (df['NdataLINEAR'][i]>200 and df['NdataZTF'][i]>150) and (df['Pratio'][i]>0.8 and df['Pratio'][i]<1.2):
            # STEP 2: determine periodogram likelihood of BE
            # ================
            dPmin = 0.01
            #--- determining if LINEAR part has periodogram indication of BE ---
            # no daily alias of main period
            LINEAR_pd_period = (np.abs(df['Plinear'][i]-0.5)>dPmin)&(np.abs(df['Plinear'][i]-1.0)>dPmin)&(np.abs(df['Plinear'][i]-2.0)>dPmin)
            # blazhko period must be within RR Lyrae range
            LINEAR_pd_pB = (df['BlazhkoPeriodL'][i]>35)&(df['BlazhkoPeriodL'][i]<325) 
            # relative strength and significance must be above 0.05 and 5 respectively
            LINEAR_pd_sig = (df['BpowerRatioL'][i]>0.05)&(df['BsignificanceL'][i]>5)

            #--- determining if ZTF part has periodogram indication of BE ---
            ZTF_pd_period = (np.abs(df['Pztf'][i]-0.5)>dPmin)&(np.abs(df['Pztf'][i]-1.0)>dPmin)&(np.abs(df['Pztf'][i]-2.0)>dPmin)
            ZTF_pd_pB = (df['BlazhkoPeriodZ'][i]>35)&(df['BlazhkoPeriodZ'][i]<325) 
            ZTF_pd_sig = (df['BpowerRatioZ'][i]>0.05)&(df['BsignificanceZ'][i]>5)
            #---
            BE = 0
            # if a star has indication of BE via both its periodograms
            if ((LINEAR_pd_period&LINEAR_pd_pB&LINEAR_pd_sig)&(ZTF_pd_period&ZTF_pd_pB&ZTF_pd_sig)):
                BE += 1
                df.loc[i, indic] = 'LZ'
            # indication of BE via LINEAR periodogram
            if (LINEAR_pd_period&LINEAR_pd_pB&LINEAR_pd_sig):
                BE += 1
                df.loc[i, indic] = 'L'
                df_stat.loc[i, 'LINEAR periodogram'] = 1
            # indication of BE via ZTF periodogram
            if (ZTF_pd_period&ZTF_pd_pB&ZTF_pd_sig):
                BE += 1
                df.loc[i, indic] = 'Z'
                df_stat.loc[i, 'ZTF periodogram'] = 1
            # ---
            # STEP 03: if a star has BE indication via periodogram, it is immediately selected
            # otherwise it goes through the scoring mechanism
            if BE>0:
                row = pd.DataFrame(df.iloc[[int(i)]])
                dfnew = pd.concat([dfnew, row.reset_index(drop=True)], ignore_index=True, axis=0)
            else:
                # select period, chi2 and amplitude values
                period = df['dP'][i]
                chiL = df['L_chi2dofR'][i]
                chiZ = df['Zchi2dofR'][i]
                ampl = df['Ampl_diff'][i]

                # ---
                # assign starting scores
                SCORE = 0

                # ---
                
                # CHI^2 scores
                # sector L2
                if (chiL > 1.5 and chiL < 3.0) and (chiZ < 1.8):
                    SCORE += 2
                    df_stat.loc[i, 'L2'] = 1
                # sector Z2
                elif (chiZ > 1.8 and chiZ < 3.5) and (chiL < 1.5):
                    SCORE += 2
                    df_stat.loc[i, 'Z2'] = 1
                # sector LZ3
                elif (chiL > 1.5 and chiL < 3.0) and (chiZ > 1.8 and chiZ < 3.5):
                    SCORE += 3
                    df_stat.loc[i, 'LZ3'] = 1
                # sectors LZ4
                elif ((chiL > 3.0) and (chiZ < 3.5)):
                    SCORE += 4
                    df_stat.loc[i, 'LZ4'] = 1
                elif ((chiL < 3.0) and (chiZ > 3.5)):
                    SCORE += 4
                    df_stat.loc[i, 'LZ4'] = 1
                # sector LZ5
                elif (chiL > 3.0 and chiZ > 3.5):
                    SCORE += 6
                    df_stat.loc[i, 'LZ5'] = 1
            
                # AMPL score
                # ----------------
                if ampl>0.05 and ampl<0.15: 
                    SCORE += 1
                    df_stat.loc[i, 'Amplitude'] = 1
                if ampl>0.15 and ampl<2: 
                    SCORE += 2
                    df_stat.loc[i, 'Amplitude'] = 1

                # PERIOD scores
                # -----------------
                if period > 0.00002 and period < 0.00005: 
                    SCORE += 2
                    df_stat.loc[i, 'Period difference'] = 1
                if period >= 0.00005: 
                    SCORE += 4
                    df_stat.loc[i, 'Period difference'] = 1

                # TOTAL SCORE calculation
                df.loc[i, bscore] = SCORE

                # if a star has a score of 5 or more, it is selected as a Blazhko candidate
                # 
                if (SCORE>4):
                    row = pd.DataFrame(df.iloc[[int(i)]])
                    dfnew = pd.concat([dfnew, row.reset_index(drop=True)], ignore_index=True, axis=0)
        else:
            pass
    return dfnew, df_stat

# BUILDING THE VISUAL INTERFACE
# ================================
# Building a class for the visual interface
class BE_analyzer:
    '''
    This class is used as an interface for user visual analysis: it contains 4 plots for the 4 phases of visual analysis.
    At the bottom there is a KEEP and CONTINUE button, if KEEP is clicked, the star is saved in a database and then the 
    interface moves onto the next star.

    Arguments:
        linear_ids(list): list of LINEAR IDs
        tot(int): total number of stars
        database_lightc(array): database of light curve data
        be_cand(DataFrame): data of blazhko candidates
        lightc_fits(array): data for light curve fits
        lightc_per(array): data for periodograms
        Zdata(array): collection of ZTF light curves
        Ldata(array): collection of LINEAR light curves
        plotSave(bool): default False, saving the total plot as an image
    '''
    def __init__(self, linear_ids, tot, database_lightc, be_cand, lightc_fits, lightc_per, Zdata, Ldata,plotSave=True):
        self.linear_ids = linear_ids
        self.database_lightc = database_lightc
        self.be_cand = be_cand
        self.lightc_fits = lightc_fits
        self.lightc_per = lightc_per
        self.Zdata = Zdata
        self.Ldata = Ldata
        self.total_num = tot
        self.plotSave = plotSave

        self.current_i = None
        self.generate = self.plot_BE_data()
        
        self.keep_button = widgets.Button(description='KEEP')
        self.con_button = widgets.Button(description='CONTINUE')
        self.keep_button.on_click(self.click_keep)
        self.con_button.on_click(self.click_con)
        
        self.output = widgets.Output()
        #display(self.output, self.keep_button, self.con_button)


    def plot_BE_data(self):
        #print('Engaging in plotting!')
        for i in range(len(self.linear_ids)):
            self.current_i = i
            # access the LINEAR id
            LID = self.linear_ids[self.current_i]
            for n, j in enumerate(self.lightc_fits):
                    if j[0]==LID:
                        break

            # select light curve fits
            L1 = self.lightc_fits[n][1][0]
            L2 = self.lightc_fits[n][1][2]

            # select the periodogram data
            for o, k in enumerate(self.lightc_per):
                    if k[0]==LID:
                        break

            fL = self.lightc_per[o][1][0]
            pL = self.lightc_per[o][1][1]
            fZ = self.lightc_per[o][2][0]
            pZ = self.lightc_per[o][2][1]

            

            fFoldedL = self.lightc_per[o][1][2]
            pFoldedL = self.lightc_per[o][1][3]
            fFoldedZ = self.lightc_per[o][2][2]
            pFoldedZ = self.lightc_per[o][2][3]

            # selecting ZTF data
            lc = self.Ldata.get_light_curve(LID)
            tL = lc.T[0]
            for f, g in enumerate(self.Zdata):
                if g[0] == LID:
                    break
            tZ = self.Zdata[f][1]

            # plotting all the graphs for visual analysis
            if self.plotSave:
                BE_plotting.plotAll(LID, n, i, self.total_num, L1, L2, self.database_lightc, fL, pL, fZ, pZ, fFoldedL, fFoldedZ, pFoldedL, pFoldedZ, self.Ldata, tL, tZ, self.Zdata, plotSave=True)
            else:
                BE_plotting.plotAll(LID, n, i, self.total_num, L1, L2, self.database_lightc, fL, pL, fZ, pZ, fFoldedL, fFoldedZ, pFoldedL, pFoldedZ, self.Ldata, tL, tZ, self.Zdata, plotSave=False)
            
            yield

    def click_keep(self, b):
        row = pd.DataFrame(self.database_lightc.iloc[[int(self.current_i)]]) # assign the current row we are analyzing
        # concatenate that row with the save_data dataframe
        self.be_cand = pd.concat([self.be_cand, row.reset_index(drop=True)], ignore_index=True, axis=0)

        with self.output:
            clear_output(wait=True)  # clear the previous output
            #print('Clearing output!')
            try:
                #print('Next image generated!')
                next(self.generate)  # generate the next plot and update current_i
            except StopIteration:  # when the for loop is finished, disable the button
                print("No more plots.")
                self.con_button.disabled = True

    def click_con(self, b):
       #print('Button clicked!')
       with self.output:
        clear_output(wait=True)  # clear the previous output
        #print('Clearing output!')
        try:
            #print('Next image generated!')
            next(self.generate)  # generate the next plot and update current_i
        except StopIteration:  # when the for loop is finished, disable the button
            print("No more plots.")
            self.con_button.disabled = True
    
    def get_save_data(self):
        return self.be_cand
    
    def display_interface(self):
        # Create a layout for the widgets
        self.layout = widgets.VBox([self.output, self.keep_button, self.con_button])
        # Display the layout
        display(self.layout)


def category_analysis(begin_data, fits, periodogr, ztf_data, dataLINEAR, id_list=None,parameter=None, value=None, plotSave=True):
    '''
    This function takes in a certain parameter and then generates a seperate dataset and interface
    in order to analyze it for Blazhko stars.

    Arguments:
        begin_data(DataFrame) = starting dataframe from which we take data
        parameter(string) = the parameter of interest
        value(string) = the value equal to which we select candidates
        fits = the fits dataset
        periodogr = the periodogram dataset
        ztf_data = the ZTF dataset
        dataLINEAR = the LINEAR dataset
        end(string) = with which we save the data
    '''

    # if there is a parameter specified, select all values with te desired parameter
    if parameter:
        new_dataset = begin_data.loc[(begin_data[parameter] == value)]
        new_dataset = new_dataset.reset_index(drop=True)
        print(f'This dataset has {new_dataset.shape[0]} stars.')

        # ----

        # access LINEAR ids and go through the entire process of blazhko star analysis 
        length = new_dataset.shape[0]
        Lids = new_dataset['LINEAR id'].to_numpy()

        blazhko_analyzer = pd.DataFrame(())
        if plotSave:
            analysis = BE_analyzer(Lids, length, new_dataset, blazhko_analyzer, fits, periodogr, ztf_data, dataLINEAR, plotSave=True)
            analysis.display_interface()
        else:
            analysis = BE_analyzer(Lids, length, new_dataset, blazhko_analyzer, fits, periodogr, ztf_data, dataLINEAR)
            analysis.display_interface()
    else:
        # if instead a list of IDS is provided instead of a specific parameter, repeat all previous steps 
        # but with a small select of IDs for plotting and analyzing
        if id_list:
            new_dataset = begin_data[begin_data['LINEAR id'].isin(id_list)]
            new_dataset = new_dataset.reset_index(drop=True)
            print(f'This dataset has {new_dataset.shape[0]} stars.')

            # ----

            length = new_dataset.shape[0]
            Lids = new_dataset['LINEAR id'].to_numpy()

            blazhko_analyzer = pd.DataFrame(())
            if plotSave:
                analysis = BE_analyzer(Lids, length, new_dataset, blazhko_analyzer, fits, periodogr, ztf_data, dataLINEAR, plotSave=True)
                analysis.display_interface()
            else:
                analysis = BE_analyzer(Lids, length, new_dataset, blazhko_analyzer, fits, periodogr, ztf_data, dataLINEAR)
                analysis.display_interface()
        else:
            # if nothing else is specified, start the visual analysis process for all stars
            print(f'This dataset has {begin_data.shape[0]} stars.')
            length = begin_data.shape[0]
            Lids = begin_data['LINEAR id'].to_numpy()

            blazhko_analyzer = pd.DataFrame(())
            if plotSave:
                analysis = BE_analyzer(Lids, length, begin_data, blazhko_analyzer, fits, periodogr, ztf_data, dataLINEAR,plotSave=True)
                analysis.display_interface()
            else:
                analysis = BE_analyzer(Lids, length, begin_data, blazhko_analyzer, fits, periodogr, ztf_data, dataLINEAR)
                analysis.display_interface()
    return analysis
