 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:13:56 2022
Script to compute compational time for STARSEED (supplemental figures)
@author: jpeeples
"""

from Utils.Load_Data import load_data
from Utils.Superpixel_Hist_Clustering import SP_clustering
from Utils.Generate_Time_Figures import compute_computational_cost
from Parameters import Parameters
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle

plt.ioff()

Params = Parameters()

num_SP = np.arange(Params['SP_min'],Params['SP_max']+1,Params['SP_step'])

#Main loop to run experiments for STARSEED
ds_count = 0

num_runs = 2
supplement_folder = "Supplementary_Figures"

time_dictionary = {}
for ds_type in Params['ds_methods']:
    
    #Generate dataset for fold
    dataset = load_data(Params['csv_dir'],Params['img_dir'],Params['run_nums'],
                        train_reps=Params['train_reps'],test_reps=Params['test_reps'], 
                        mode=Params['mode'],preprocess=Params['preprocess'],
                        downsample=Params['downsample'],
                        ds_factor=Params['ds_factor'],ds_type=ds_type)
    
    feat_count = 0
    
    combo_count = 0
    for feature in Params['features']:
        
        #Initialize time matrix (# of runs x # of SP)
        time_matrix = np.zeros((num_runs,len(num_SP)))
        
        for run in range(0,num_runs):
        
            #Initialize label count
            label_count = 0
            
            for label in Params['label_types']:
                
                count = 0
                results_folder = (Params['results_location'] + feature + '/' + ds_type + '/')
                
                if Params['embed'] == 'UMAP':
                    SP_folder = (results_folder + Params['preprocess'] + '/' + Params['mode'] +'_Run_' +
                                 str(Params['run_nums']) + '_Random_State_' + str(Params['seed']) + '/' + label + '/')
                else:
                    SP_folder = (results_folder + Params['preprocess'] + '/' + Params['mode'] +'_Run_' +
                                 str(Params['run_nums']) + '_Random_State_' + str(Params['seed']) + '/' )
                    
                cultivar_score = []
                water_levels_score = []
                cross_score = []
              
                
                for SP in num_SP:
    
                    start = time.time()
                    scores, true_labels = SP_clustering(dataset,numSP=SP,
                                                           mode=Params['mode'],
                                                           folder=SP_folder + 'SP_'+ str(SP) + 
                                                           '/', embed = Params['embed'],
                                                           split_data=Params['split_data'],
                                                           features=feature,seed=Params['seed'],
                                                           equal_weight=Params['equal_weight'],
                                                           alpha=Params['alpha'],
                                                           normalize=Params['norm_method'],
                                                           root_only=Params['root_only'],
                                                           num_neighbors=Params['num_neighbors'],
                                                           label_type=label,
                                                           vis_fig_type=Params['vis_fig_type'],
                                                           score_metric=Params['score_metric'])
                    
                    #Record time for all experiments with 1 feature
                    time_elapsed = time.time() - start
                    
                    #Add time to matrix
                    time_matrix[run,count] = time_elapsed
                    
                    #Iterate counter
                    count  += 1
                    
                    print('Finished {} of {} values for Superpixels'.format(count,len(num_SP)))
                      
                combo_count += 1
                label_count +=1
                print('Finished {} of {} label types'.format(label_count,len(Params['label_types']))) 
                print('Finished {} of {} comp combos types'.format(combo_count,
                                                                   len(Params['features'])*num_runs))
        
        feat_count += 1
        
        #Save time matrix to dictionary
        time_dictionary[feature] = time_matrix
        
        print('Finished {} of {} features'.format(feat_count,len(Params['features'])))
    
    ds_count += 1
    print('Finished {} of {} downsampling methods'.format(ds_count,len(Params['ds_methods'])))
    
#Compute and record computational cost
supplement_folder = '{}{}/'.format(Params['results_location'],supplement_folder)

if not os.path.exists(supplement_folder):
    os.makedirs(supplement_folder)

#Save time dictionary incase changes are needed to figure
with open('{}Runs_{}saved_dictionary.pkl'.format(supplement_folder,Params['run_nums']), 'wb') as f:
    pickle.dump(time_dictionary, f)

compute_computational_cost(time_dictionary,Params['features'],
                            supplement_folder,num_SP,Params['run_nums'])




    
    







