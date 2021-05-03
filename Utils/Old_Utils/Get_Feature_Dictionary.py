#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:55:52 2020

@author: weihuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def get_fdic(csv_dir):

    fdic = {} # feature dictionary to save all the features
    df = pd.read_csv(csv_dir) # load data
    df = df.dropna(axis=0) # remove Nan value or empty data
    alltube = df['TubeNb'].unique()
    # stat = df.describe()
    
    for tube in alltube:
        fdic[str(tube)] = {} 
        
        sub_df = df[df['TubeNb']==tube]
        nroot = len(sub_df)
        cultivar = sub_df['Cultivar'].iloc[0]
        moisture = sub_df['Moisture'].iloc[0]
        totL = sub_df['TotLength(cm)'].sum()
        totPA = sub_df['TotProjArea(cm2)'].sum()
        totSA = sub_df['TotSurfArea(cm2)'].sum()
        totAD = sub_df['TotAvgDiam(mm)'].sum()
        totV = sub_df['TotVolume(cm3)'].sum()
        tipD = sub_df['TipDiam'].sum()
        
        # save all features to dictionary
        fdic[str(tube)]['num_of_root'] = nroot
        fdic[str(tube)]['cultivar'] = cultivar
        fdic[str(tube)]['moisture'] = moisture
        fdic[str(tube)]['avg_totL'] = totL/nroot
        fdic[str(tube)]['avg_totPA'] = totPA/nroot
        fdic[str(tube)]['avg_totSA'] = totSA/nroot
        fdic[str(tube)]['avg_totAD'] = totAD/nroot
        fdic[str(tube)]['avg_totV'] = totV/nroot
        fdic[str(tube)]['avg_tipD'] = tipD/nroot
    
    return fdic
    

    

# test code
if __name__ == "__main__":
    # csv_dir = '/mnt/WD500/Data/Run4ROOT.csv'
    # features = get_features(csv_dir)
    path = '/mnt/WD500/Data/Features'
    flist = os.listdir(path)
    runlist = ['Run4', 'Run5', 'Run6', 'Run7']
    saveout = './Results'
    fdic = {}
    
    for run in runlist:
        for file in flist:
            if run in file:
                fdic[run] = get_fdic(os.path.join(path, file))
    
    json = json.dumps(fdic)
    f = open(os.path.join(saveout, 'feature_dict.json'),"w")
    f.write(json)
    f.close()
                