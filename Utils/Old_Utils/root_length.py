#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:07:55 2020

@author: weihuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = '/mnt/WD500/Data'
Run = 'Run6'
saveout = '/mnt/WD500/sesame_meeting/histogram'
df = pd.read_csv(os.path.join(path, Run+'ROOT.csv'))
df = df.dropna(axis=0)
alltube = df['TubeNb'].unique()
nbin=30
stat = df.describe()

for tube in alltube:
    sub_df = df[df['TubeNb']==tube]
    nroot = len(sub_df)
    cultivar = sub_df['Cultivar'].iloc[0]
    moisture = sub_df['Moisture'].iloc[0]
    totL = sub_df['TotLength(cm)'].to_numpy()
    totPA = sub_df['TotProjArea(cm2)'].to_numpy()
    totSA = sub_df['TotSurfArea(cm2)'].to_numpy()
    totAD = sub_df['TotAvgDiam(mm)'].to_numpy()
    totV = sub_df['TotVolume(cm3)'].to_numpy()
    TD = sub_df['TipDiam'].to_numpy()
    

    
    fig, ax = plt.subplots(2,3,figsize=(14,7))
    plt.subplots_adjust(wspace=.4,hspace=.4)
    fig.suptitle('Histograms for features of Tube:%s; num of roots: %s; Cultivar: %s; Moisture: %s' %(tube, nroot, cultivar, moisture))

    ax[0,0].set_title('Total Length for Each Root')
    ax[0,0].hist(totL, bins=np.arange(0, stat.loc['max','TotLength(cm)']*1.1, \
                                  step=stat.loc['max','TotLength(cm)']/nbin))
    
    ax[0,1].set_title('Total ProjArea(cm2)')
    ax[0,1].hist(totPA, bins=np.arange(0, stat.loc['max','TotProjArea(cm2)']*1.1, \
                                   step=stat.loc['max','TotProjArea(cm2)']/nbin))
    

    ax[0,2].set_title('TotSurfArea(cm2)')
    ax[0,2].hist(totSA, bins=np.arange(0, stat.loc['max','TotSurfArea(cm2)']*1.1, \
                                   step=stat.loc['max','TotSurfArea(cm2)']/nbin))
    

    ax[1,0].set_title('TotAvgDiam(mm)')
    ax[1,0].hist(totAD, bins=np.arange(0, stat.loc['max','TotAvgDiam(mm)']*1.1, \
                                    step=stat.loc['max','TotAvgDiam(mm)']/nbin))
    

    ax[1,1].set_title('TotVolume(cm3)')
    ax[1,1].hist(totV, bins=np.arange(0, stat.loc['max','TotVolume(cm3)']*1.1, \
                                    step=stat.loc['max','TotVolume(cm3)']/nbin))
        

    ax[1,2].set_title('TipDiam')
    ax[1,2].hist(TD, bins=np.arange(0, stat.loc['max','TipDiam']*1.1, \
                                    step=stat.loc['max','TipDiam']/nbin))
    
    plt.savefig(os.path.join(saveout, (str(tube)+'_hist.png')))
    
    