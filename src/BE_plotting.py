# IMPORTING LIBRARIES
# -----------------
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import ticker
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import ipywidgets as widgets
from IPython.display import display, clear_output

import numpy as np
import pandas as pd
import sys
import os

# CONFIGURATION
# -------------
sns.set_theme(style='white') # setting the theme for plotting
sys.path.insert(0,'../src/')
np.random.seed(42)
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'

# configuring plotting colors
colors = ['#465BBB', '#3F8FCE', '#7ABBCE', '#3A3865', '#A82F43', '#612A37', '#DC5433', '#F29457']
b1 = '#465BBB'
b2 = '#3F8FCE'
b3 = '#7ABBCE'
b4 = '#3A3865'
black1 = '#26212A'
black2 = '#2D1E21'
or1 = '#A82F43'
or2 = '#612A37'
or3 = '#DC5433'
or4 = '#F29457'
muted_colors = sns.set_palette(sns.color_palette(colors))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{txfonts}'  # Use TX Times font

# Set up your font properties (optional)
font = FontProperties()
font.set_family('serif')
font.set_name('Times')  # Times font

sys.path.insert(0,'../src/')
from helper import*

# ===============

# PLOTTING LIGHT CURVES
# --------------------------
def makeLCplot_info(L1, L2, dataset, order, Lid, dataL, total_num, plotname='LCplot', plotSave=False):
    '''
    This function plots a single phase of a light curve with fit for both LINEAR and ZTF data, along with 
    a separate box for text data.
    
    Arguments:
        L1(array): phased LINEAR light curve
        L2(array): phased ZTF light curve
        dataset(dataframe): dataset of light curve statistics
        order(int): iteration of light curve
        Lid(int): LINEAR ID
        dataL(array): LINEAR metadata table
        total_num(int): total number of light curves
        plotname(str): name of saved plot, default: 'LCplot'
        plotSave(bool): default: False, if True, plot will be saved
    '''
    # plot setup
    fig, ax = plt.subplots(1,3, figsize=(36,10))   # dimensions of plot
    fig.suptitle('STAR '+str(order+1)+' from '+str(total_num), fontsize=30, fontproperties=font) # title of plot - star number
    fig.set_facecolor('white') # background color

    # plotting LINEAR phased light curve
    # ---------------
    # labels
    ax[0].set_xlabel('data phased with best-fit LINEAR period', fontproperties=font, fontsize=23)
    ax[0].set_ylabel('LINEAR normalized light curve', fontproperties=font, fontsize=23)
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_ylim(1.3, -0.3)
    # data for phased light curve
    xx, yy, zz = sort3arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'])
    ax[0].errorbar(xx, yy, zz, fmt='.k', ecolor=black1, lw=1, ms=4, capsize=1.5, alpha=0.2)
    # plotting the fit based off of LINEAR period
    ax[0].plot(L1['modelPhaseGrid'], L1['modTemplate'], or1, markeredgecolor=or1, lw=2, fillstyle='top', linestyle='dashed')

    # plotting ZTF phased light curve
    # --------------
    ax[1].set_xlabel('data phased with best-fit ZTF period', fontproperties=font, fontsize=23)
    ax[1].set_ylabel('ZTF normalized light curve', fontproperties=font, fontsize=23)
    ax[1].set_xlim(-0.1, 1.1)
    ax[1].set_ylim(1.3, -0.3)
    # data for phased light curve
    xx1, yy1, zz1 = sort3arr(L2['dataPhasedTime'], L2['dataTemplate'], L2['dataTemplateErr'])
    ax[1].errorbar(xx1, yy1, zz1, fmt='.k', ecolor=black1, lw=1, ms=4, capsize=1.5, alpha=0.2)
    # fit for ZTF period
    ax[1].plot(L2['modelPhaseGrid'], L2['modTemplate'], or1, markeredgecolor=or1,  lw=2, fillstyle='top', linestyle='dashed')

    # overview of data calculated for particular light curve pair
    ax[2].axis([0, 8, 0, 10])
    # LINEAR chi2 robust
    ax[2].text(0, 8, 'LINEAR period chi robust: '+str(dataset['L_chi2dofR'][order])+', LINEAR mean period chi robust: '+str(dataset['Lmean_chi2dofR'][order]),fontsize=26,fontproperties=font)
    # ZTF chi2 robust
    ax[2].text(0, 7, 'ZTF period chi robust: '+str(dataset['Zchi2dofR'][order])+', ZTF mean period chi robust: '+str(dataset['Zmean_chi2dofR'][order]),fontsize=26,fontproperties=font)
    # LINEAR chi2 original
    ax[2].text(0, 6, 'LINEAR period chi: '+str(dataset['L_chi2dof'][order])+', LINEAR mean period chi: '+str(dataset['Lmean_chi2dof'][order]),fontsize=26,fontproperties=font)
    # ZTF chi2 original
    ax[2].text(0, 5, 'ZTF period chi: '+str(dataset['Zchi2dof'][order])+', ZTF mean period chi: '+str(dataset['Zmean_chi2dof'][order]),fontsize=26,fontproperties=font)
    # LINEAR and ZTF periods + period difference
    ax[2].text(0, 4, 'LINEAR period: '+str(dataset['Plinear'][order])+', ZTF period: '+str(dataset['Pztf'][order])+', Period difference: '+str(dataset['dP'][order]),fontsize=26,fontproperties=font)
    # average LINEAR magnitude
    ax[2].text(0, 3, 'Average LINEAR magnitude: '+str(round(np.mean(dataL.get_light_curve(Lid).T[1]), 2)),fontsize=26,fontproperties=font)
    # amplitude
    ax[2].text(0, 2, 'LINEAR amplitude:'+str(dataset['Lampl'][order])+', ZTF amplitude:'+str(dataset['Zampl'][order]),fontsize=26,fontproperties=font)
    # if the star has a stronger period or amplitude score, display it
    if dataset['period_vs_amp'][order] != np.nan:
        ax[2].text(0, 1, '- this star has a stronger '+str(dataset['period_vs_amp'][order])+' score',fontsize=26,fontproperties=font)

    ax[2].grid(False)
    ax[2].axis('off')
    
    # if the user specified, please save plots
    if plotSave:
        plotName = plotname + '_' + str(Lid) + '.png'
        plt.savefig('../visual_analysis/'+plotName, dpi=150,bbox_inches = 'tight')
    else:
        plt.show()

    return

# PLOTTING THE PERIODOGRAMS
# =================
def plotBlazhkoPeaks(Lid, order, fL, pL, fZ, pZ, fFoldedL, pFoldedL, fFoldedZ, pFoldedZ, dataset, fac=1.008, plotSave=False):
    '''
    This function plots the periodogram of both LINEAR and ZTF light curves, as well as their folded periodograms
    indicating the presence of the Blazhko effect.

    Arguments:
        order(int): the number of the light curve
        fL(array): array of LINEAR frequency values
        pL(array): array of LINEAR power values
        fZ(array): array of ZTF frequency values
        pZ(array): array of ZTF power values
        fFoldedL(array): frequencies of folded periodogram data LINEAR
        pFoldedL(array): powers of folded periodogram data LINEAR
        fFoldedZ(array): frequencies of folded periodogram data ZTF
        pFoldedZ(array): powers of folded periodogram data ZTF
        dataset(DataFrame): dataset with calculated parameters for every light curve pair
        fac(int): factor for xlim and ylim, default 1.008
        plotSave(bool): default False, does the user want to save the plot
    '''
    flin = fL[np.argmax(pL)]
    fztf = fZ[np.argmax(pZ)]

    # DATA PREP
    # ===========
    # accessing the LINEAR and ZTF Blazhko peaks
    fBlazhkoPeakL = dataset['BlazhkoPeakL'][order]
    # ---
    fBlazhkoPeakZ = dataset['BlazhkoPeakZ'][order]

    # testing if there is periodogram data for a particular star
    # if there isn't, print a statement, and if there is, continue with plotting
    if fL.size==0 or pL.size==0 or fZ.size==0 or pZ.size==0 or fFoldedL.size==0 or pFoldedL.size==0 or fFoldedZ.size==0 or pFoldedZ.size==0:
        print("No available periodogram data.")
    else:
        # plot setup
        fig = plt.figure(figsize=(32, 8))
        fig.subplots_adjust(hspace=0.1, bottom=0.06, top=0.94, left=0.12, right=0.94)
        ax = fig.add_subplot(141)

        # LINEAR COMPLETE PLOT
        # ----------------------
        ax.plot(fL, pL, c=b1, alpha=0.8) # plotting the periodogram
        ax.plot([flin, flin], [0,1], lw = 2, c=or1, ls='--') # plotting in the largest power as the peak
        # plotting the Blazhko peaks
        ax.plot([fBlazhkoPeakL, fBlazhkoPeakL], [0, 0.7*np.max(pFoldedL)], lw = 2, c=or1, ls='--') 
        ax.plot([2*flin-fBlazhkoPeakL, 2*flin-fBlazhkoPeakL], [0, 0.7*np.max(pFoldedL)], lw = 2, c=or1, ls='--') # 
        # show 1 year alias
        f1yr = flin+1/365.0
        ax.plot([f1yr, f1yr], [0,0.7*np.max(pFoldedL)], lw = 2, ls='-.', c=b3)
        f1yr = flin-1/365.0
        ax.plot([f1yr, f1yr], [0,0.7*np.max(pFoldedL)], lw = 2, ls='-.', c=b3)
        
        # setting plot title, limits on the graph
        ax.text(0.03, 0.96, "LINEAR", ha='left', va='top', transform=ax.transAxes,fontproperties=font, fontsize=16)
        if (fBlazhkoPeakL > flin*fac):
            ax.set_xlim(0.99*(2*flin-fBlazhkoPeakL), 1.01*fBlazhkoPeakL)
        else:
            ax.set_xlim(flin/fac, flin*fac)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ylim = ax.get_ylim()
        ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
        if ymax>1.0: ymax=1.0
        ax.set_ylim(0, ymax)
        ax.set_ylabel('Lomb-Scargle power',fontproperties=font, fontsize=23)
        ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font, fontsize=23)

        # LINEAR FOLDED PLOT
        # ---------------------
        ax = fig.add_subplot(142)

        # plotting folded periodogram with the blazhko peak
        ax.plot(fFoldedL, pFoldedL, c=b1, alpha=0.8)
        ax.plot([fBlazhkoPeakL, fBlazhkoPeakL], [0,0.4*np.max(pFoldedL)], lw = 2, ls='--', c=or1)
        # show 1 year alias
        f1yr = flin+1/365.0
        ax.plot([f1yr, f1yr], [0,0.4*np.max(pFoldedL)], lw = 2, ls='-.', c=b3)
        
        powerFar = pFoldedL[fFoldedL>fBlazhkoPeakL]  # frequencies beyond the second peak
        powerFarMedian = np.median(powerFar)      # the median power
        powerFarRMS = np.std(powerFar)            # standard deviation, i.e. "sigma"
        noise5sig = powerFarMedian+5*powerFarRMS
        
        # plot in the noise line above which we found the Blazhko peak
        if (fBlazhkoPeakL > flin*fac):
            ax.plot([flin+0.5*(fBlazhkoPeakL-flin), 1.01*fBlazhkoPeakL], [noise5sig, noise5sig], lw = 2, ls='--', c=b2)
            ax.set_xlim(flin, 1.01*fBlazhkoPeakL)
        else:
            ax.plot([flin+0.5*(fBlazhkoPeakL-flin), flin*fac], [noise5sig, noise5sig], lw = 2, ls='--', c=b2)
            ax.set_xlim(flin, flin*fac)

        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ylim = ax.get_ylim()
        ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
        if ymax>1.0: ymax=1.0
        ax.set_ylim(0, ymax)
        ax.set_ylabel('folded power',fontproperties=font, fontsize=23)
        ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font, fontsize=23)

        # ZTF
        # ========
        
        # PLOTTING THE FULL PERIODOGRAM
        # ---------------
        ax = fig.add_subplot(143)

        # plotting the periodogram
        ax.plot(fL, pL, c=b1, alpha=0.8) # plotting basic periodogram
        # adding the main peak and blazhko peaks
        ax.plot([fztf, fztf], [0,1], lw = 2, c=or1, ls='--')
        ax.plot([fBlazhkoPeakZ, fBlazhkoPeakZ], [0, 0.7*np.max(pFoldedZ)], lw = 2, c=or1, ls='--')
        ax.plot([2*fztf-fBlazhkoPeakZ, 2*fztf-fBlazhkoPeakZ], [0, 0.7*np.max(pFoldedZ)], lw = 2, c=or1, ls='--')

        # show 1 year alias for ztf
        f1yrZ = fztf+1/365.0
        ax.plot([f1yrZ, f1yrZ], [0,0.7*np.max(pFoldedZ)], lw = 2, ls='-.', c=b3)
        f1yrZ = fztf-1/365.0
        ax.plot([f1yrZ, f1yrZ], [0,0.7*np.max(pFoldedZ)], lw = 2, ls='-.', c=b3)

        # adding y-axis text
        ax.text(0.03, 0.96, "ZTF", ha='left', va='top', transform=ax.transAxes,fontproperties=font, fontsize=16)
        if (fBlazhkoPeakZ > fztf*fac):
            ax.set_xlim(0.99*(2*fztf-fBlazhkoPeakZ), 1.01*fBlazhkoPeakZ)
        else:
            ax.set_xlim(fztf/fac, fztf*fac)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ylim = ax.get_ylim()
        ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
        if ymax>1.0: ymax=1.0
        ax.set_ylim(0, ymax)
        ax.set_ylabel('Lomb-Scargle power',fontproperties=font, fontsize=23)
        ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font, fontsize=23)

        # PLOTING FOLDED POWER SEQUENCE
        # ----------------
        ax = fig.add_subplot(144)
        # plotting folded ZTF data along with the Blazhko peaks
        ax.plot(fFoldedZ, pFoldedZ, c=b1, alpha=0.8)
        ax.plot([fBlazhkoPeakZ, fBlazhkoPeakZ], [0,0.4*np.max(pFoldedZ)], lw = 2, ls='--', c=or1)

        # show 1 year alias
        f1yrZ = fztf+1/365.0
        ax.plot([f1yrZ, f1yrZ], [0,0.4*np.max(pFoldedZ)], lw = 2, ls='-.', c=b3)
        
        powerFarZ = pFoldedZ[fFoldedZ>fBlazhkoPeakZ]  # frequencies beyond the second peak
        powerFarMedianZ = np.median(powerFarZ)      # the median power
        powerFarRMSZ = np.std(powerFarZ)            # standard deviation, i.e. "sigma"
        noise5sigZ = powerFarMedianZ+5*powerFarRMSZ
        
        # drawing the noise line for peak recognition
        if (fBlazhkoPeakZ > fztf*fac):
            ax.plot([fztf+0.5*(fBlazhkoPeakZ-fztf), 1.01*fBlazhkoPeakZ], [noise5sigZ, noise5sigZ], lw = 2, ls='--', c=b2)
            ax.set_xlim(flin, 1.01*fBlazhkoPeakZ)
        else:
            ax.plot([flin+0.5*(fBlazhkoPeakZ-fztf), fztf*fac], [noise5sigZ, noise5sigZ], lw = 2, ls='--', c=b2)
            ax.set_xlim(fztf, fztf*fac)

        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ylim = ax.get_ylim()
        ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
        if ymax>1.0: ymax=1.0
        ax.set_ylim(0, ymax)
        ax.set_ylabel('folded power',fontproperties=font, fontsize=23)
        ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font,fontsize=23)

        if plotSave:
            plotName = '../visual_analysis/periodogram'+str(Lid)+'.png'
            plt.savefig(plotName, dpi=150,bbox_inches = 'tight')
            #print('saved plot as:', plotName) 
        plt.show()     
        return   


def plotLINEARmarkSeasons(Lid, ztf_data, order, LINEARlightcurves, plotName='season_plot', plotSave=False):
    '''
    This function plots the total light curves of each light curve pair.

    Arguments:
        Lid(int): LINEAR ID
        ztf_data(array): collection of ZTF light curves
        order(int): number of light curve
        LINEARlightcurves(array): collection of LINEAR light curves
        plotName(string): default 'season_plot', default saving name
        plotSave(bool): default True, saving the plot onto the computer
    '''
    # LINEAR PLOT
    # ================
    tL, mL, mLerr = LINEARlightcurves[Lid].T # accessing LINEAR data
    fig, ax = plt.subplots(1,2, figsize=(32,10))   # setting up the plot
    # drawing LINEAR data
    ax[0].set_ylim(np.min(mL)-0.3, np.max(mL)+0.3)
    #ax[0].set_title('LINEAR object {0}'.format(Lid),fontproperties=font,fontsize=18)
    ax[0].errorbar(tL, mL, mLerr, fmt='.k', ecolor=black1, alpha=0.3)
    ax[0].set_xlabel('Time (days)', fontproperties=font, fontsize=23)
    ax[0].set_ylabel('LINEAR magnitude', fontproperties=font, fontsize=23)
    ax[0].invert_yaxis()
    ax[0].set_xlim(np.min(tL)-230, np.max(tL)+230)

    redL = 0 # counting of seasons, used in another function for determining the number of season plots

    # determine the timespan of measurement and count the number of seasons of observation
    for s in range(1, 8):
        tS = 52550 + (s-1)*365
        per = np.median(mL)
        ax[0].plot([tS, tS], [per-per*0.05, per+per*0.05], c=or3) # plot the season line
        if tS>np.min(tL)-230 and tS<np.max(tL)+230:
            redL += 1 # save the number of seasons

    # ZTF PLOT
    # ==========
    tZ, mZ, meZ = ztf_data[order][1], ztf_data[order][2], ztf_data[order][3]
    ax[1].set_ylim(np.min(mZ)-0.3, np.max(mZ)+0.3)
    ax[1].errorbar(tZ, mZ, meZ, fmt='.k', ecolor=black1, alpha=0.3)
    #ax[0].set_title('ZTF object {0}'.format(order),fontproperties=font,fontsize=18)
    ax[1].set_xlabel('Time (days)', fontproperties=font, fontsize=23)
    ax[1].set_ylabel('ZTF magnitude', fontproperties=font, fontsize=23)
    ax[1].invert_yaxis()
    ax[1].set_xlim(np.min(tZ)-230, np.max(tZ)+230)

    # count number of seasons of obesrvation for ZTF
    redZ = 0 
    for r in range(1, 8):
        tSZ = (np.min(tZ)-50) + (r-1)*365
        ax[1].plot([tSZ, tSZ], [np.min(mZ)-0.1, np.max(mZ)+0.1], c=or3)# plot the season line
        if tSZ>np.min(tZ)-230 and tSZ<np.max(tZ)+230:
            redZ += 1# save the number of seasons

    # plot saving mechanism
    if plotSave:
        plt.savefig('../visual_analysis/'+plotName+str(Lid)+'.png', dpi=150)
    plt.show()   
    
    return redL, redZ

def makeLCplotBySeason(Lid, L1, tL, L2, tZ, redL, redZ, plotrootname='LCplotBySeason', plotSave=False):
    '''
    This function is used for plotting the phased light curve fit for every season of observation compared to the
    light curve data for that season. 

    Arguments:
        Lid(int): LINEAR id
        L1(array): phased light curve fit for LINEAR data
        tL(array): time data for LINEAR
        L2(array): phased light curve fit for ZTF data
        tZ(array): time data for ZTF
        redL(int): number of seasons for LINEAR data
        redZ(int): number of seasons for ZTF data
        plotrootname(str): the default name of the saved plot, 'LCplotBySeason'
        plotSave(bool): default False, saving the plot
    '''
    # setting up the plot
    fig = plt.figure(figsize=(32, 30))
    fig.subplots_adjust(hspace=0.2, bottom=0.06, top=0.94, left=0.12, right=0.94)
    
    fig.suptitle('Seasons for:'+str(Lid), fontsize=30, fontproperties=font)
    
    # new function for plotting LINEAR seasons
    def plotPanelL(ax, L1, season):
        ax.set_xlabel('phase', fontproperties=font, fontsize=14)
        ax.set_ylabel('normalized phased light curve', fontproperties=font, fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.3, -0.4)
        # fit for LINEAR period
        ax.plot(L1['modelPhaseGrid'], L1['modTemplate'], or3, markeredgecolor=or3, lw=2, fillstyle='top', linestyle='dashed')
    
        # plotting the phased light curve data for that particular season
        xx, yy, zz, ww = sort4arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'], tL)
        tSmin = 52523 + (season-1)*365
        tSmax = 52523 + season*365
        condition = (ww > tSmin) & (ww < tSmax)
        # selecting the data based on the season (condition)
        xxS = xx[condition]
        yyS = yy[condition]
        zzS = zz[condition]
        wwS = ww[condition]
        ax.errorbar(xxS, yyS, zzS, fmt='.k', ecolor=black1, lw=1, ms=4, capsize=1.5, alpha=0.3)
        textString = "LINEAR season " + str(season)
        ax.text(0.03, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=23)
        textString = "MJD=" + str(tSmin) + ' to ' + str(tSmax)
        ax.text(0.53, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=23)

        
    # plot each season separately 
    for season in range(1,redL):
        # plot the power spectrum
        ax = fig.add_subplot(5, 3, season)
        plotPanelL(ax, L1, season)
        if (season==1):
            ax.set_title('LINEAR object {0}'.format(Lid), fontproperties=font,fontsize=18)

    # =======
    # ZTF
    # =======

    # creating a new function for plotting the ZTF seasons
    def plotPanelZ(ax, L2, seasonZ):
        # plot setup
        ax.set_xlabel('phase', fontproperties=font, fontsize=14)
        ax.set_ylabel('normalized phased light curve', fontproperties=font, fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.3, -0.4)
        # fit for ZTF period
        ax.plot(L2['modelPhaseGrid'], L2['modTemplate'], or3, markeredgecolor=or3, lw=2, fillstyle='top', linestyle='dashed')
    
        # accessing the phased ZTF data
        xx, yy, zz, ww = sort4arr(L2['dataPhasedTime'], L2['dataTemplate'], L2['dataTemplateErr'], tZ)
        tSmin = (np.min(tZ)-50) + (seasonZ-1-6)*365
        tSmax = (np.min(tZ)-50) + (seasonZ-6)*365
        # selecting ZTF data based on condition (time season)
        xxS = xx[(ww>tSmin)&(ww<tSmax)]
        yyS = yy[(ww>tSmin)&(ww<tSmax)]
        zzS = zz[(ww>tSmin)&(ww<tSmax)]
        wwS = ww[(ww>tSmin)&(ww<tSmax)]
        ax.errorbar(xxS, yyS, zzS, fmt='.k', ecolor=black1, lw=1, ms=4, capsize=1.5, alpha=0.3)
        textString = "ZTF season " + str(seasonZ-6)
        ax.text(0.03, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=23)
        textString = "MJD=" + str(round(tSmin)) + ' to ' + str(round(tSmax))
        ax.text(0.53, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=23)

        
    # plot each season separately 
    for seasonZ in range(redL,redL+redZ-1):
        # plot the power spectrum
        ax = fig.add_subplot(5, 3, seasonZ)
        plotPanelZ(ax, L2, seasonZ)
        if (seasonZ==1):
            ax.set_title('ZTF object {0}'.format(Lid), fontproperties=font,fontsize=18)

    if plotSave:
        plotName = plotrootname +str(Lid)+ '.png'
        plt.savefig('../visual_analysis/'+plotName, dpi=150,bbox_inches = 'tight')
        #print('saved plot as:', plotName) 
    plt.show()     
    return

def plotAll(Lid, orderlc, o, tot, L1, L2, blazhko_can, fL, pL, fZ, pZ, fFoldedL, fFoldedZ, pFoldedL, pFoldedZ, data, tL, tZ,ztf_data,plotSave=False):
    '''
    This function plots all of the graphs necessary for visual analysis of Blazhko stars

    Arguments:
        Lid(int): LINEAR id
        orderlc(int): order of light curve (in the data)
        o(int): order of light curve in overall analysis
        tot(int): total number of light curves
        L1(array): phased light curve data LINEAR
        L2(array): phased light curve data ZTF
        balzhko_can(DataFrame): dataframe with the parameters of Blazhko candidates
        fL(array): frequency values from periodogram for LINEAR
        pL(array): power values from periodogram for LINEAR
        fZ(array): frequency values from periodogram for ZTF
        pZ(array): power values from periodogram for ZTF
        fFoldedL(array): folded frequency values for LINEAR
        pFoldedL(array): folded power values for LINEAR
        fFoldedZ(array): folded frequency values for ZTF
        pFoldedZ(array): folded power values for LINEAR
        data(array): collection of LINEAR light curves
        tL(array): time data for LINEAR
        tZ(array): time data for ZTF
        ztf_data(array): collection of light curves for ZTF
        plotSave(bool): default False, saving the plot on computer
    '''
    if plotSave:
        makeLCplot_info(L1, L2, blazhko_can, o, Lid, data, tot,plotSave=True)
        plotBlazhkoPeaks(Lid, o, fL, pL, fZ, pZ, fFoldedL, pFoldedL, fFoldedZ, pFoldedZ, blazhko_can, fac=1.008, plotSave=True)
        redLin, redZtf = plotLINEARmarkSeasons(Lid, ztf_data, orderlc, data, plotSave=True)
        makeLCplotBySeason(Lid, L1, tL, L2, tZ, redLin, redZtf,plotSave=True)
    else:
        makeLCplot_info(L1, L2, blazhko_can, o, Lid, data, tot)
        plotBlazhkoPeaks(Lid, o, fL, pL, fZ, pZ, fFoldedL, pFoldedL, fFoldedZ, pFoldedZ, blazhko_can, fac=1.008, plotSave=False)
        redLin, redZtf = plotLINEARmarkSeasons(Lid, ztf_data, orderlc, data)
        makeLCplotBySeason(Lid, L1, tL, L2, tZ, redLin, redZtf)
    return