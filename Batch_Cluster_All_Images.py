 # -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:13:56 2020
Main script to cluster root structure
images using superpixel histogram images
@author: jpeeples
"""
from Utils.Load_Data import load_data
from Utils.Superpixel_Hist_Clustering import SP_clustering
from Parameters import Parameters
import pandas as pd 
from Utils.Visualization import get_SP_plot_SI, plot_global_feats
import numpy as np
import matplotlib.pyplot as plt
import pdb

plt.ioff()

Params = Parameters()

def generate_excel_sheet(results_location,mean_data,std_data,max_data,title,
                         run_nums,labels,features,SP_data=None):
    
    #Convert to each list to array
    mean_data = np.array(mean_data)
    std_data = np.array(std_data)
    max_data = np.array(max_data)
    
    #Initialize writer object
    writer = pd.ExcelWriter(results_location + 
                            '{}_Class_Scores_{}.xlsx'.format(title,str(run_nums)),
                            engine='xlsxwriter')
    
    #Create column names and dataframes
    temp_cols = list(np.unique(labels))
    
    if SP_data is None:
        temp_cols.append('All Classes')
    DF_avg = pd.DataFrame(mean_data,columns=temp_cols,index=features)
    DF_std = pd.DataFrame(std_data,columns=temp_cols,index=features)
    DF_max = pd.DataFrame(max_data,columns=temp_cols,index=features)
    
    DF_avg.to_excel(writer,sheet_name='Average Scores')
    DF_std.to_excel(writer,sheet_name='Std Scores')
    DF_max.to_excel(writer,sheet_name='Max Scores')
    
    if SP_data is not None:
        DF_SP = pd.DataFrame(SP_data,columns=temp_cols,index=features)
        DF_SP.to_excel(writer,sheet_name='Superpixels')
        
    writer.save()
    writer.close()

#Save average and std for metrics
score_table = np.zeros((len(Params['features']),len(Params['labels']),3)) 
sp_table = np.zeros((len(Params['features']),len(Params['labels'])))

num_SP = np.arange(Params['SP_min'],Params['SP_max']+1,Params['SP_step'])


ds_count = 0
for ds_type in Params['ds_methods']:
    
    #Generate dataset for fold
    dataset = load_data(Params['csv_dir'],Params['img_dir'],Params['run_nums'],
                        train_reps=Params['train_reps'],test_reps=Params['test_reps'], 
                        DAP=Params['DAP'], mode=Params['mode'],
                        preprocess=Params['preprocess'],downsample=Params['downsample'],
                        ds_factor=Params['ds_factor'],ds_type=ds_type)
    
    feat_count = 0
    mean_feats_cultivar = []
    std_feats_cultivar = []
    max_feats_cultivar = []
    mean_feats_water = []
    std_feats_water = []
    max_feats_water = []
    
    for feature in Params['features']:
        
        label_count = 0
        
        for label in Params['label_types']:
            
            count = 0
            results_folder = (Params['results_location'] + feature + '/' + ds_type + '/')
            
            SP_folder = (results_folder + Params['preprocess'] + '/' + Params['mode'] +'_Run_' +
                         str(Params['run_nums']) + '_Random_State_' + str(Params['seed']) + '/' + label + '/')
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

                scores, cluster_scores, true_labels = SP_clustering(dataset,numSP=SP,
                                                       mode=Params['mode'],
                                                       num_imgs=Params['num_imgs'],
                                                       folder=SP_folder + 'SP_'+ str(SP) + 
                                                       '/', embed = Params['embed'],
                                                       split_data=Params['split_data'],
                                                       features=feature,seed=Params['seed'],
                                                       equal_weight=Params['equal_weight'],
                                                       alpha=Params['alpha'],
                                                       normalize=Params['norm_method'],
                                                       root_only=Params['root_only'],
                                                       set_preferences=Params['set_preferences'],
                                                       adjusted=Params['adjusted'],
                                                       num_neighbors=Params['num_neighbors'],
                                                       label_type=label,
                                                       vis_fig_type=Params['vis_fig_type'],
                                                       score_metric=Params['score_metric'])
                
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
                  
            # #Generate plots to show accuracy as number of superpixels is varied
            cultivar_silhoutte = np.array(cultivar_silhoutte)
            water_levels_silhoutte = np.array(water_levels_silhoutte)
            cross_silhoutte = np.array(cross_silhoutte)
            # cluster_silhoutte = np.array(cluster_silhoutte)
            
            get_SP_plot_SI(cultivar_silhoutte[:,-1],num_SP,title_type='Cultivar',
                           folder=SP_folder,metric=Params['score_metric'])
            get_SP_plot_SI(water_levels_silhoutte[:,-1],num_SP,title_type='Water Levels',
                           folder=SP_folder,metric=Params['score_metric'])
            get_SP_plot_SI(cross_silhoutte[:,-1],num_SP,title_type='Cross Treatments',
                           folder=SP_folder,metric=Params['score_metric'])
            # get_SP_plot_SI(cluster_silhoutte[:,-1],num_SP,title_type='Affinity Propagation',folder=SP_folder,metric=score_metric)
            # get_Cluster_metrics_plots(AP_cluster_scores,num_SP,title_type='Affinity Propagation',folder=SP_folder,
            #                           adjusted=adjusted)

            #Save out max scores (maybe average later)
            score_table[feat_count, :, 0] = (np.mean(cultivar_silhoutte[:,-1]), 
                                             np.mean(water_levels_silhoutte[:,-1]), 
                                             np.mean(cross_silhoutte[:,-1]),
                                             np.mean(cluster_silhoutte[0]))
            score_table[feat_count, :, 1] = (np.std(cultivar_silhoutte[:,-1]), 
                                             np.std(water_levels_silhoutte[:,-1]), 
                                             np.std(cross_silhoutte[:,-1]),
                                             np.std(cluster_silhoutte[0]))
            score_table[feat_count, :, 2] = (np.max(cultivar_silhoutte[:,-1]), 
                                             np.max(water_levels_silhoutte[:,-1]), 
                                             np.max(cross_silhoutte[:,-1]),
                                             np.max(cluster_silhoutte[0]))
            sp_table[feat_count, :] = (num_SP[np.argmax(cultivar_silhoutte[:,-1])], 
                                             num_SP[np.argmax(water_levels_silhoutte[:,-1])], 
                                             num_SP[np.argmax(cross_silhoutte[:,-1])],
                                             num_SP[0])
                                             #num_SP[np.argmax(cluster_silhoutte[0])])
            
            label_count +=1
            print('Finished {} of {} label types'.format(label_count,len(Params['label_types'])))           
            
      
        #Visualize global feature results
        plot_global_feats(dataset,feature,SP_folder+'Global_Visuals/',seed=Params['seed'],
                          root_only=Params['root_only'])
        
        #Save out results for each class across each feature
        #Save all values out as np array
        np.save(SP_folder+'{}_Scores'.format('Cultivar'),cultivar_silhoutte)
        np.save(SP_folder+'{}_Scores'.format('Water Levels'),water_levels_silhoutte)
        mean_feats_cultivar.append(np.mean(cultivar_silhoutte,axis=0))
        std_feats_cultivar.append(np.std(cultivar_silhoutte,axis=0))
        max_feats_cultivar.append(np.max(cultivar_silhoutte,axis=0))
        mean_feats_water.append(np.mean(water_levels_silhoutte,axis=0))
        std_feats_water.append(np.std(water_levels_silhoutte,axis=0))
        max_feats_water.append(np.max(water_levels_silhoutte,axis=0))
    
        feat_count += 1
        print('Finished {} of {} features'.format(feat_count,len(Params['features'])))
    
    
    mean_feats_cultivar = np.array(mean_feats_cultivar)
    std_feats_cultivar = np.array(std_feats_cultivar)
    max_feats_cultivar = np.array(max_feats_cultivar)
    mean_feats_water = np.array(mean_feats_water)
    std_feats_water = np.array(std_feats_water)
    max_feats_water = np.array(max_feats_water)
    
    #Create spreadsheets to save out score statistics
    generate_excel_sheet(Params['results_location'],mean_feats_cultivar,std_feats_cultivar,
                         max_feats_cultivar,'Cultivar',Params['run_nums'],true_labels['Cultivar'],
                         Params['features'])
    generate_excel_sheet(Params['results_location'],mean_feats_water,std_feats_water,
                         max_feats_water,'Water Levels',Params['run_nums'],true_labels['Water Level'],
                         Params['features'])
    generate_excel_sheet(Params['results_location'],score_table[:,:,0],score_table[:,:,1],
                         score_table[:,:,2],Params['score_metric'],Params['run_nums'],Params['labels'],
                         Params['features'],SP_data=sp_table)
    
        
    ds_count += 1
    print('Finished {} of {} downsampling methods'.format(ds_count,len(Params['ds_methods'])))
        


    
    







