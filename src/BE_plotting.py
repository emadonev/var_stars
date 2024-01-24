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

sns.set_theme() # setting the theme for plotting

# configuring plotting colors
colors = ['#5F6372', '#79A8A4', '#B2AD8F', '#92A186', '#AD8082']
blue = '#5F6372'
turqoise = '#79A8A4'
light_green = '#B2AD8F'
green = '#92A186'
pink = '#AD8082'
muted_colors = sns.set_palette(sns.color_palette(colors))

# configuring fonts for plotting
font = FontProperties()
font.set_family('avenir')
font.set_name('Big Caslon')
font.set_style('normal')
font.set_size('xx-large')

sys.path.insert(0,'../src/')
from helper import*
# ===============

def makeLCplot_info(L1, L2, dataset, order, Lid, dataL, total_num, plotname='LCplot', plotSave=False):
    '''
    This function plots a single phase of a light curve with fit for both LINEAR and ZTF data, along with 
    a separate box for text data.
    
    Arguments:
        L1: fit data for light curve
    '''
    fig, ax = plt.subplots(1,3, figsize=(32,8))   
    fig.suptitle('STAR '+str(order+1)+' from '+str(total_num), fontsize=30, fontproperties=font)
    fig.set_facecolor('white')

    ax[0].set_xlabel('data phased with best-fit LINEAR period', fontproperties=font, fontsize=14)
    ax[0].set_ylabel('LINEAR normalized light curve', fontproperties=font, fontsize=14)
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_ylim(1.3, -0.3)
    # data
    xx, yy, zz = sort3arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'])
    ax[0].errorbar(xx, yy, zz, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for Plinear
    ax[0].plot(L1['modelPhaseGrid'], L1['modTemplate'], pink, markeredgecolor=pink, lw=1, fillstyle='top', linestyle='dashed')

    ax[1].set_xlabel('data phased with best-fit ZTF period', fontproperties=font, fontsize=14)
    ax[1].set_ylabel('ZTF normalized light curve', fontproperties=font, fontsize=14)
    ax[1].set_xlim(-0.1, 1.1)
    ax[1].set_ylim(1.3, -0.3)
    # data
    xx1, yy1, zz1 = sort3arr(L2['dataPhasedTime'], L2['dataTemplate'], L2['dataTemplateErr'])
    ax[1].errorbar(xx1, yy1, zz1, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
    # fit for Plinear
    ax[1].plot(L2['modelPhaseGrid'], L2['modTemplate'], 'red', markeredgecolor='red', lw=1, fillstyle='top', linestyle='dashed')

    ax[2].axis([0, 8, 0, 10])
    ax[2].text(0, 8, 'LINEAR period chi robust: '+str(dataset['L_chi2dofR'][order])+', LINEAR mean period chi robust: '+str(dataset['Lmean_chi2dofR'][order]),fontsize=15,fontproperties=font)
    ax[2].text(0, 7, 'ZTF period chi robust: '+str(dataset['Zchi2dofR'][order])+', ZTF mean period chi robust: '+str(dataset['Zmean_chi2dofR'][order]),fontsize=15,fontproperties=font)
    ax[2].text(0, 6, 'LINEAR period chi: '+str(dataset['L_chi2dof'][order])+', LINEAR mean period chi: '+str(dataset['Lmean_chi2dof'][order]),fontsize=15,fontproperties=font)
    ax[2].text(0, 5, 'ZTF period chi: '+str(dataset['Zchi2dof'][order])+', ZTF mean period chi: '+str(dataset['Zmean_chi2dof'][order]),fontsize=15,fontproperties=font)
    ax[2].text(0, 4, 'LINEAR period: '+str(dataset['Plinear'][order])+', ZTF period: '+str(dataset['Pztf'][order])+', Period difference: '+str(dataset['dP'][order]),fontsize=15,fontproperties=font)
    ax[2].text(0, 3, 'Average LINEAR magnitude: '+str(round(np.mean(dataL.get_light_curve(Lid).T[1]), 2)),fontsize=15,fontproperties=font)
    ax[2].text(0, 2, 'LINEAR amplitude:'+str(dataset['Lampl'][order])+', ZTF amplitude:'+str(dataset['Zampl'][order]),fontsize=15,fontproperties=font)
    if dataset['period_vs_amp'][order] != np.nan:
        ax[2].text(0, 1, '- this star has a stronger '+str(dataset['period_vs_amp'][order])+' score',fontsize=15,fontproperties=font)

    ax[2].grid(False)
    ax[2].axis('off')
    plt.show()
    #print('Finished plotting!')
    
    if plotSave:
        plotName = plotname + '.png'
        plt.savefig('../images_blazhko/'+plotName, dpi=600)
    #return

def plotBlazhkoPeaksLINEAR(Lid, order, fL, pL, fZ, pZ, fFoldedL, pFoldedL, fFoldedZ, pFoldedZ, dataset, fac=1.008, plotSave=False, verbose=False):
    flin = fL[np.argmax(pL)]
    fztf = fZ[np.argmax(pZ)]

    # DATA PREP
    # ===========
    fBlazhkoPeakL = dataset['BlazhkoPeakL'][order]
    # ---
    fBlazhkoPeakZ = dataset['BlazhkoPeakZ'][order]

    ## at some point, we will read periodograms back from files...
    fig = plt.figure(figsize=(32, 8))
    fig.subplots_adjust(hspace=0.1, bottom=0.06, top=0.94, left=0.12, right=0.94)

    # plot the power spectrum
    ax = fig.add_subplot(141)

    ax.plot(fL, pL, c='b')
    ax.plot([flin, flin], [0,1], lw = 1, c=pink, ls='--')
    ax.plot([fBlazhkoPeakL, fBlazhkoPeakL], [0, 0.7*np.max(pFoldedL)], lw = 1, c=pink, ls='--')
    ax.plot([2*flin-fBlazhkoPeakL, 2*flin-fBlazhkoPeakL], [0, 0.7*np.max(pFoldedL)], lw = 1, c=pink, ls='--')
    # show 1 year alias
    f1yr = flin+1/365.0
    ax.plot([f1yr, f1yr], [0,0.7*np.max(pFoldedL)], lw = 1, ls='-.', c=green)
    f1yr = flin-1/365.0
    ax.plot([f1yr, f1yr], [0,0.7*np.max(pFoldedL)], lw = 1, ls='-.', c=green)

    ax.text(0.03, 0.96, "LINEAR", ha='left', va='top', transform=ax.transAxes,fontproperties=font, fontsize=10)
    if (fBlazhkoPeakL > flin*fac):
        ax.set_xlim(0.99*(2*flin-fBlazhkoPeakL), 1.01*fBlazhkoPeakL)
    else:
        ax.set_xlim(flin/fac, flin*fac)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ylim = ax.get_ylim()
    ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
    if ymax>1.0: ymax=1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Lomb-Scargle power',fontproperties=font, fontsize=14)
    ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font, fontsize=14)

    # plot folder power spectrum
    ax = fig.add_subplot(142)

    ax.plot(fFoldedL, pFoldedL, c=blue)
    ax.plot([fBlazhkoPeakL, fBlazhkoPeakL], [0,0.4*np.max(pFoldedL)], lw = 1, ls='--', c=pink)
    # show 1 year alias
    f1yr = flin+1/365.0
    ax.plot([f1yr, f1yr], [0,0.4*np.max(pFoldedL)], lw = 1, ls='-.', c=green)
    
    powerFar = pFoldedL[fFoldedL>fBlazhkoPeakL]  # frequencies beyond the second peak
    powerFarMedian = np.median(powerFar)      # the median power
    powerFarRMS = np.std(powerFar)            # standard deviation, i.e. "sigma"
    noise5sig = powerFarMedian+5*powerFarRMS
    
    if (fBlazhkoPeakL > flin*fac):
        ax.plot([flin+0.5*(fBlazhkoPeakL-flin), 1.01*fBlazhkoPeakL], [noise5sig, noise5sig], lw = 1, ls='--', c=turqoise)
        ax.set_xlim(flin, 1.01*fBlazhkoPeakL)
    else:
        ax.plot([flin+0.5*(fBlazhkoPeakL-flin), flin*fac], [noise5sig, noise5sig], lw = 1, ls='--', c=turqoise)
        ax.set_xlim(flin, flin*fac)

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ylim = ax.get_ylim()
    ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
    if ymax>1.0: ymax=1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel('folded power',fontproperties=font, fontsize=14)
    ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font, fontsize=14)

    # ZTF
    # ========
    
    # PLOTTING THE FULL PERIODOGRAM
    # ---------------
    ax = fig.add_subplot(143)

    # plotting the periodogram
    ax.plot(fL, pL, c=blue) # plotting basic periodogram
    # adding the structure lines
    ax.plot([fztf, fztf], [0,1], lw = 1, c=pink, ls='--')
    ax.plot([fBlazhkoPeakZ, fBlazhkoPeakZ], [0, 0.7*np.max(pFoldedZ)], lw = 1, c=pink, ls='--')
    ax.plot([2*fztf-fBlazhkoPeakZ, 2*fztf-fBlazhkoPeakZ], [0, 0.7*np.max(pFoldedZ)], lw = 1, c=pink, ls='--')

    # show 1 year alias for ztf
    f1yrZ = fztf+1/365.0
    ax.plot([f1yrZ, f1yrZ], [0,0.7*np.max(pFoldedZ)], lw = 1, ls='-.', c=green)
    f1yrZ = fztf-1/365.0
    ax.plot([f1yrZ, f1yrZ], [0,0.7*np.max(pFoldedZ)], lw = 1, ls='-.', c=green)

    # adding y-axis text
    ax.text(0.03, 0.96, "ZTF", ha='left', va='top', transform=ax.transAxes,fontproperties=font, fontsize=10)
    if (fBlazhkoPeakZ > fztf*fac):
        ax.set_xlim(0.99*(2*fztf-fBlazhkoPeakZ), 1.01*fBlazhkoPeakZ)
    else:
        ax.set_xlim(fztf/fac, fztf*fac)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ylim = ax.get_ylim()
    ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
    if ymax>1.0: ymax=1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Lomb-Scargle power',fontproperties=font, fontsize=14)
    ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font, fontsize=14)

    # PLOTING FOLDED POWER SEQUENCE
    # ----------------
    ax = fig.add_subplot(144)

    ax.plot(fFoldedZ, pFoldedZ, c=blue)
    ax.plot([fBlazhkoPeakZ, fBlazhkoPeakZ], [0,0.4*np.max(pFoldedZ)], lw = 1, ls='--', c=pink)

    # show 1 year alias
    f1yrZ = fztf+1/365.0
    ax.plot([f1yrZ, f1yrZ], [0,0.4*np.max(pFoldedZ)], lw = 1, ls='-.', c=green)
    
    powerFarZ = pFoldedZ[fFoldedZ>fBlazhkoPeakZ]  # frequencies beyond the second peak
    powerFarMedianZ = np.median(powerFarZ)      # the median power
    powerFarRMSZ = np.std(powerFarZ)            # standard deviation, i.e. "sigma"
    noise5sigZ = powerFarMedianZ+5*powerFarRMSZ
    
    if (fBlazhkoPeakZ > fztf*fac):
        ax.plot([fztf+0.5*(fBlazhkoPeakZ-fztf), 1.01*fBlazhkoPeakZ], [noise5sigZ, noise5sigZ], lw = 1, ls='--', c=turqoise)
        ax.set_xlim(flin, 1.01*fBlazhkoPeakZ)
    else:
        ax.plot([flin+0.5*(fBlazhkoPeakZ-fztf), fztf*fac], [noise5sigZ, noise5sigZ], lw = 1, ls='--', c=turqoise)
        ax.set_xlim(fztf, fztf*fac)

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ylim = ax.get_ylim()
    ymax = ylim[0] + 1.1 * (ylim[1] - ylim[0])
    if ymax>1.0: ymax=1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel('folded power',fontproperties=font, fontsize=14)
    ax.set_xlabel('frequency (d$^{-1}$)',fontproperties=font,fontsize=14)

    if plotSave:
        plotName = '../plots/Blazhko.png'
        plt.savefig(plotName, dpi=600)
        print('saved plot as:', plotName) 
    plt.show()     
    return   

def plotLINEARmarkSeasons(Lid, ztf_data, order, LINEARlightcurves):
    tL, mL, mLerr = LINEARlightcurves[Lid].T
    fig, ax = plt.subplots(1,2, figsize=(32,8))   
    ax[0].set_ylim(np.min(mL)-0.3, np.max(mL)+0.3)
    ax[0].set_title('LINEAR object {0}'.format(Lid),fontproperties=font,fontsize=14)
    ax[0].errorbar(tL, mL, mLerr, fmt='.b', ecolor=blue)
    ax[0].set_xlabel('Time (days)', fontproperties=font, fontsize=14)
    ax[0].set_ylabel('LINEAR magnitude', fontproperties=font, fontsize=14)
    ax[0].invert_yaxis()
    ax[0].set_xlim(np.min(tL)-200, np.max(tL)+200)

    redL = 0

    for s in range(1, 8):
        tS = 52550 + (s-1)*365
        per = np.median(mL)
        ax[0].plot([tS, tS], [per-per*0.05, per+per*0.05], c=pink)
        if tS>np.min(tL)-200 and tS<np.max(tL)+200:
            redL += 1

    tZ, mZ, meZ = ztf_data[order][1][0], ztf_data[order][1][1], ztf_data[order][1][2]
    ax[1].set_ylim(np.min(mZ)-0.3, np.max(mZ)+0.3)
    ax[1].errorbar(tZ, mZ, meZ, fmt='.b', ecolor=blue)
    ax[0].set_title('ZTF object {0}'.format(order),fontproperties=font,fontsize=14)
    ax[1].set_xlabel('Time (days)', fontproperties=font, fontsize=14)
    ax[1].set_ylabel('ZTF magnitude', fontproperties=font, fontsize=14)
    ax[1].invert_yaxis()
    ax[1].set_xlim(np.min(tZ)-200, np.max(tZ)+200)

    redZ = 0

    for r in range(1, 8):
        tSZ = (np.min(tZ)-50) + (r-1)*365
        ax[1].plot([tSZ, tSZ], [np.min(mZ)-0.1, np.max(mZ)+0.1], c=pink)
        if tSZ>np.min(tZ)-200 and tSZ<np.max(tZ)+200:
            redZ += 1
    plt.show()   
      
    return redL, redZ

def makeLCplotBySeason(Lid, L1, tL, L2, tZ, redL, redZ, plotrootname='LCplotBySeason', plotSave=False):
    
    fig = plt.figure(figsize=(32, 30))
    fig.subplots_adjust(hspace=0.2, bottom=0.06, top=0.94, left=0.12, right=0.94)
    
    fig.suptitle('Seasons for:'+str(Lid), fontsize=30, fontproperties=font)
    
    def plotPanelL(ax, L1, season):
        ax.set_xlabel('phase', fontproperties=font, fontsize=14)
        ax.set_ylabel('normalized phased light curve', fontproperties=font, fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.3, -0.4)
        # fit for Plinear
        ax.plot(L1['modelPhaseGrid'], L1['modTemplate'], pink, markeredgecolor=pink, lw=1, fillstyle='top', linestyle='dashed')
    
        # data
        xx, yy, zz, ww = sort4arr(L1['dataPhasedTime'], L1['dataTemplate'], L1['dataTemplateErr'], tL)
        tSmin = 52520 + (season-1)*365
        tSmax = 52520 + season*365
        condition = (ww > tSmin) & (ww < tSmax)
        xxS = xx[condition]
        yyS = yy[condition]
        zzS = zz[condition]
        wwS = ww[condition]
        ax.errorbar(xxS, yyS, zzS, fmt='.b', ecolor=blue, lw=1, ms=4, capsize=1.5, alpha=0.3)
        textString = "LINEAR season " + str(season)
        ax.text(0.03, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=14)
        textString = "MJD=" + str(tSmin) + ' to ' + str(tSmax)
        ax.text(0.53, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=14)

        
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

    def plotPanelZ(ax, L2, seasonZ):
        ax.set_xlabel('phase', fontproperties=font, fontsize=14)
        ax.set_ylabel('normalized phased light curve', fontproperties=font, fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.3, -0.4)
        # fit for Plinear
        ax.plot(L2['modelPhaseGrid'], L2['modTemplate'], pink, markeredgecolor=pink, lw=1, fillstyle='top', linestyle='dashed')
    
        # data
        xx, yy, zz, ww = sort4arr(L2['dataPhasedTime'], L2['dataTemplate'], L2['dataTemplateErr'], tZ)
        tSmin = (np.min(tZ)-50) + (seasonZ-1-6)*365
        tSmax = (np.min(tZ)-50) + (seasonZ-6)*365
        xxS = xx[(ww>tSmin)&(ww<tSmax)]
        yyS = yy[(ww>tSmin)&(ww<tSmax)]
        zzS = zz[(ww>tSmin)&(ww<tSmax)]
        wwS = ww[(ww>tSmin)&(ww<tSmax)]
        ax.errorbar(xxS, yyS, zzS, fmt='.b', ecolor=blue, lw=1, ms=4, capsize=1.5, alpha=0.3)
        textString = "ZTF season " + str(seasonZ-6)
        ax.text(0.03, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=14)
        textString = "MJD=" + str(tSmin) + ' to ' + str(tSmax)
        ax.text(0.53, 0.96, textString, ha='left', va='top', transform=ax.transAxes, fontproperties=font,fontsize=14)

        
    # plot each season separately 
    for seasonZ in range(redL,redL+redZ-1):
        # plot the power spectrum
        ax = fig.add_subplot(5, 3, seasonZ)
        plotPanelZ(ax, L2, seasonZ)
        if (seasonZ==1):
            ax.set_title('ZTF object {0}'.format(Lid), fontproperties=font,fontsize=18)

    if plotSave:
        plotName = plotrootname + '.png'
        plt.savefig(plotName, dpi=600)
        print('saved plot as:', plotName) 
    plt.show()     
    return

def plotAll(Lid, orderlc, o, tot, L1, L2, blazhko_can, fL, pL, fZ, pZ, fFoldedL, fFoldedZ, pFoldedL, pFoldedZ, data, tL, tZ,ztf_data):
    makeLCplot_info(L1, L2, blazhko_can, o, Lid, data, tot)
    plotBlazhkoPeaksLINEAR(Lid, o, fL, pL, fZ, pZ, fFoldedL, pFoldedL, fFoldedZ, pFoldedZ, blazhko_can, fac=1.008, plotSave=False, verbose=True)
    redLin, redZtf = plotLINEARmarkSeasons(Lid, ztf_data, orderlc, data)
    makeLCplotBySeason(Lid, L1, tL, L2, tZ, redLin, redZtf)
    return