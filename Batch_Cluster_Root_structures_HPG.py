# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:13:56 2020
Main script to classify root structure
images using superpixel histogram images
@author: jpeeples
"""
from Utils.Load_Data import load_data
from Utils.Superpixel_Hist_Clustering import SP_clustering
from Utils.Visualization import get_SP_plot_SI, get_Cluster_metrics_plots,plot_norm_table,plot_avg_norm_table
# from prettytable import PrettyTable
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import pdb

plt.ioff()

#Load data
csv_dir = '../../Data/TubeNumberKey.csv'
img_dir = '../../Data/'
# csv_dir = 'T:/Fractal_Images/TubeNumberKey.csv'
# img_dir = 'T:/Fractal_Images/'
# csv_dir = 'C:/Users/jpeeples/Documents/Research/Root_Project/TubeNumberKey.csv'
# img_dir = 'C:/Users/jpeeples/Documents/Research/Root_Project/Data/'

run_nums = [4,5]
mode = 'bw' #rgb or bw
DAP = 'last' #last or all
preprocess = 'gauss' #Either do 'avg' (average filter), 'gauss' (gaussian filter), or None
#Either use 'root_pixels','fractal', 'lacunarity', or 'all' (use all three texture features)
features = ['root_pixels','fractal', 'lacunarity']
# features = ['EHD']
# norm_vals = ['no_norm','texture_feats','all_feats','standardization']
# ds_methods = ['No_Pool','Avg_Pool','Max_Pool']
ds_methods = ['Avg_Pool']
set_preferences = False #Set False to use median as preference or True to set to percentage of root SPs
downsample = True #(set to false for 'avg')
ds_factor = 16
equal_weight = False #Set to False to use feature value in SP as weights for EMD
alpha = 1 #Scale on spatial coordinates
num_imgs = 5 #Number of test images to visualize (26 in total)
#Seed for Random state for TSNE, Clustering and Metrics (should only affect
# root visualization and not clustering performance)
seed = 42
adjusted=True
embed = 'UMAP' #Embedding method to be used, select either 'TSNE','UMAP', or 'MDS'
split_data = True #Cluster training and test images separately (if true, will "predict" test labels)
root_only = True #Only compute EMD where root is present
#num_SP = np.arange(100,500,25)
# num_SP = np.concatenate((np.arange(10,100,10),np.arange(100,200,25)))
num_SP = np.arange(100,300,100)
results_location = 'Results/High_SP_CV/'

reps = [1,2,3,4]
CV_folds = list(itertools.combinations(reps, 3))


#If data split (test/train), peform four fold
for fold in range(3, len(CV_folds)):
    train_reps = list(CV_folds[fold])
    test_reps = list(sorted(set(reps) - set(CV_folds[fold])))

    ds_count = 0
    for ds_type in ds_methods:
        
        #Generate dataset for fold
        dataset = load_data(csv_dir,img_dir,run_nums,train_reps = train_reps,
                         test_reps = test_reps, DAP=DAP,mode=mode,preprocess=preprocess,
                         downsample=downsample,ds_factor=ds_factor,ds_type=ds_type)
        
        feat_count = 0
        for feature in features:
            
            count = 0
            results_folder = (results_location + feature + '/' + ds_type + '/'
                              + 'Fold_' + str(fold+1) + '/')
            
            table_folder = (results_location + 'Fold_' + str(fold+1) +'/' + preprocess + '_' + mode +'_Run_' +
                         str(run_nums) + '_Random_State_' + str(seed) + '/')
            
            SP_folder = (results_folder + preprocess + '/' + mode +'_Run_' +
                         str(run_nums) + '_Random_State_' + str(seed) + '/')
            cultivar_silhoutte = []
            water_levels_silhoutte = []
            cross_silhoutte = []
            cluster_silhoutte = []
            AP_cluster_scores = []
            cultivar_cp_scores = []
            water_cp_scores = []
            cross_cp_scores = []
            cultivar_hg_scores = []
            water_hg_scores = []
            cross_hg_scores = []
            cultivar_v_scores = []
            water_v_scores = []
            cross_v_scores = []
            
            for SP in num_SP:
                
                scores, cluster_scores, _ = SP_clustering(dataset,numSP=SP,mode=mode,
                                                       num_imgs=num_imgs,
                                                       folder=SP_folder + 'SP_'+ str(SP) + 
                                                       '/', embed = embed,
                                                       split_data=split_data,
                                                       features=feature,seed=seed,
                                                       equal_weight=equal_weight,
                                                       alpha=alpha,
                                                       normalize='None',
                                                       root_only=root_only,
                                                       set_preferences = set_preferences)
                
                #Save scores for plotting
                cultivar_silhoutte.append(scores['cultivar'])
                water_levels_silhoutte.append(scores['water_levels'])
                cross_silhoutte.append(scores['cross_treatment'])
                cluster_silhoutte.append(scores['clustering'])
                AP_cluster_scores.append(cluster_scores['Affinity_Propagation'])
                cultivar_cp_scores.append(cluster_scores['Affinity_Propagation']['Cultivar'][1])
                water_cp_scores.append(cluster_scores['Affinity_Propagation']['Water_Levels'][1])
                cross_cp_scores.append(cluster_scores['Affinity_Propagation']['Cross_Treatment'][1])
                cultivar_hg_scores.append(cluster_scores['Affinity_Propagation']['Cultivar'][0])
                water_hg_scores.append(cluster_scores['Affinity_Propagation']['Water_Levels'][0])
                cross_hg_scores.append(cluster_scores['Affinity_Propagation']['Cross_Treatment'][0])
                cultivar_v_scores.append(cluster_scores['Affinity_Propagation']['Cultivar'][2])
                water_v_scores.append(cluster_scores['Affinity_Propagation']['Water_Levels'][2])
                cross_v_scores.append(cluster_scores['Affinity_Propagation']['Cross_Treatment'][2])
                
                #Iterate counter
                count  += 1
                
                print('Finished {} of {} values for Superpixels'.format(count,len(num_SP)))
                  
            # #Generate plots to show accuracy as K is varied
            get_SP_plot_SI(cultivar_silhoutte,num_SP,title_type='Cultivar',folder=SP_folder)
            get_SP_plot_SI(water_levels_silhoutte,num_SP,title_type='Water Levels',folder=SP_folder)
            get_SP_plot_SI(cross_silhoutte,num_SP,title_type='Cross Treatments',folder=SP_folder)
            get_SP_plot_SI(cluster_silhoutte,num_SP,title_type='Affinity Propagation',folder=SP_folder)
            get_Cluster_metrics_plots(AP_cluster_scores,num_SP,title_type='Affinity Propagation',folder=SP_folder,
                                      adjusted=adjusted)
            
            feat_count += 1
            print('Finished {} of {} features'.format(feat_count,len(features)))
            
        ds_count += 1
        print('Finished {} of {} downsampling methods'.format(ds_count,len(ds_methods)))
        

    print('Finished {} of {} Folds'.format(fold+1,len(CV_folds)))

# # #Save dictionary of results
# f = open(overall_folder+"Clustering_Results.pkl","wb")
# pickle.dump(dict_tables,f)
# f.close()  








