# -*- coding: utf-8 -*-
"""
Visual code for root images clustering
Code modified from original plotting code 
from Jeff Dale at the University of Missouri

"""

import os
from typing import List

from matplotlib.backend_bases import MouseEvent
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
import itertools
from itertools import compress 

from Utils.Load_Data import load_data
from Utils.Superpixel_Hist_Clustering import SP_clustering
from Utils.Compute_fractal_dim import fractal_dimension
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from itertools import cycle, compress
import json
import pdb

def onclick(event: MouseEvent):

    # We need to modify variables global scope in this function.
    # Global statement is probably not required (and bad practice), but is nice for being explicit

    global ax               # Plotting axes to display image on
    global embedding         # The nx2 array of n points in 2-space that were scatter plotted
    global roots_df         # A pandas dataframe of metadata for each point, such as filename, target / not target, etc
    global data
    global names
    global centers
    global labels
    global feat_dict
    global SI_scores
    global features
    
    # Make a 1x2 numpy array of [x, y] coordinates of click
    x = np.array([event.xdata, event.ydata]).reshape(1, 2)

    # Find nearest neighbor of clicked coords in feats_2d array from global scope
    nearest_ix = pairwise_distances_argmin(x, embedding)[0]

    # Get appropriate row of all_feats_info.
    # Transform all_feats_info to align with feats_2d, then index by nearest_index
    info = roots_df.iloc[nearest_ix, :]
    
    #Grab feature information
    run_num = 'Run' + names[nearest_ix][4]
    tube_num = names[nearest_ix][-3::]
    num_root = feat_dict[run_num][tube_num]["num_of_root"]
    avg_len = feat_dict[run_num][tube_num]["avg_totL"]
    text_feats = data[nearest_ix]['SP_profile'][:,0]
    text_feat_avg = np.round(np.average(text_feats,axis=0),decimals=3)
    text_feat_std = np.round(np.std(text_feats,axis=0),decimals=3)
    # text_feats_names = ['Num Root Pixels', 'Fractal','Lacunarity']
    if features == 'lacunarity':
        global_feat = fractal_dimension(data[nearest_ix]['Img'],compute_lac=True)[-1]
    elif features == 'fractal':
        global_feat = fractal_dimension(data[nearest_ix]['Img'])
    else: #Number of root pixels
        global_feat = np.count_nonzero(data[nearest_ix]['Img'])/(data[nearest_ix]['Img'].size)
    
    columns = ('Cluster', 'Avg Root Length (cm)','# Roots','Avg SP '+features,
               'Global '+ features)
    cell_text = [[labels[nearest_ix]], 
                 np.round([avg_len],decimals=3), [num_root],
                 [r"{} $\pm$ {}".format(text_feat_avg,text_feat_std)],
                 [np.round(global_feat,decimals=3)]]
    
    #Show information under image
    ax[2].table(cellText=cell_text,rowLabels=columns,loc='center')
    ax[2].set_axis_off()

    # Read Root image associated with clicked point
    im = data[nearest_ix]['Img']

    # Clear image preview subplot
    ax[1].cla()

    # Show image preview, cropping according to metadata
    ax[1].imshow(im,cmap='binary')

    # Set title to filename of image and remove x and y axes
    ax[1].set_title('{}: \n Cultivar {} and Water Level {}'.format(names[nearest_ix],
                                                                      info.cultivar,
                                                                      info.water_levels))
    ax[1].set_axis_off()

    # Call plt.pause to force matplotlib to redraw plot
    plt.pause(0.01)
    

def Create_Cluster_Figure(EMD_train,embedding,numSP,EMD_test=None,
                           ax=None,images=None,embed='TSNE',labels=None,
                           class_names=None,seed=42,split_data=True,
                           train_idx=None,test_idx=None,run_nums=[4,5]):  
    
    #Convert distance matrix to similarity matrix for clustering
    if split_data:
        EMD_train = EMD_mat[np.ix_(train_idx,train_idx)]
    else:
        EMD_train = EMD_mat
        
    EMD_train_sim = 1 - EMD_train/np.max(EMD_train)
    
    #Affinity propogation: 
    #preference set based on median of similarity of matrix
    af = AffinityPropagation(affinity='precomputed',random_state=seed).fit(EMD_train_sim)
    cluster_centers_indices = af.cluster_centers_indices_
    labels_af = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    
    if split_data:
        test_labels = []
        for img in test_idx:
            #Grab EMD values from matrix (in future, compute distance and flow matrix)
            #Can add visualization here
            temp_dists = EMD_mat[img,cluster_centers_indices]
            
            # pdb.set_trace()
            #Select minimal distance and assign label
            #Error occurs when convergence does not happen
            try:
                temp_center = cluster_centers_indices[np.argmin(temp_dists)]
                test_labels.append(np.where(cluster_centers_indices==temp_center)[0][0])
            except:
                test_labels.append(labels_af[0])
        
        #Combine test labels with training labels
        test_labels = np.array(test_labels)
        labels_af = np.concatenate((labels_af,test_labels),axis=0)
 
    #Create stack feature maps figure
    fig_stack, ax_stack = plt.subplots(nrows=2,ncols=n_clusters_,figsize=(22,11))
    plt.subplots_adjust(wspace=.7,hspace=.5)

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(14,7))
        plt.subplots_adjust(right=.75)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    center_treatments = []

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels_af == k
        cluster_center = embedding[cluster_centers_indices[k]]
        ax.plot(embedding[class_members, 0], embedding[class_members, 1], col + '.')
        if labels is None:
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=14,label='Cluster '+str(k)+':')
        else:
            labels = labels.astype(int)
            treatment = str(class_names[labels[cluster_centers_indices[k]]])
            center_treatments.append(treatment)
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=14,label=treatment)
            # label='Cluster '+str(k)+': '+ treatment
            ax.annotate(str(k),  xy=(cluster_center[0], cluster_center[1]), color='white',
            fontsize="large", weight='heavy',
            horizontalalignment='center',
            verticalalignment='center')
        for x in embedding[class_members]:
            ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color = col)
                    #Plot testing points    
        if split_data:
           test_pts = np.concatenate((np.zeros(len(train_idx)),np.ones(len(test_idx))),axis=0).astype(bool)
           test_class_members = np.logical_and(class_members,test_pts)
           ax.plot(embedding[test_class_members,0], embedding[test_class_members,1], col + '*')
          
        temp_data = []
        img_idx = np.where(class_members)[0]
        num_imgs = len(img_idx)
        for idx in img_idx:
            temp_data.append(data[idx]['Root_mask'])
        
        temp_data = np.stack(temp_data,axis=0)
        #TBD, fix colorbars
        avg_vals = ax_stack[0,k].imshow(np.average(temp_data,axis=0),aspect='auto',vmin=0,vmax=.3)
        std_vals = ax_stack[1,k].imshow(np.std(temp_data,axis=0),aspect='auto',vmin=0,vmax=.3)
        fig_stack.colorbar(avg_vals,ax=ax_stack[0,k])
        fig_stack.colorbar(std_vals,ax=ax_stack[1,k])
        ax_stack[0,k].set_title('Avg SP {} \nFor Cluster {}\nTotal Images: {}'. format(features,k,num_imgs))
        ax_stack[1,k].set_title('Std SP {} \nFor Cluster {}\nTotal Images: {}'. format(features,k,num_imgs))
        
    #Save stack image
    fig_stack.savefig('Runs_' + str(run_nums) + '_Stacked_Cluster_Feature_Maps.png',
                      dpi=fig_stack.dpi,bbox_inches='tight')
    plt.close(fig=fig_stack)

    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(('Affnity Propogation: Est {:d} clusters with ' +
               '{:d} Superpixels').format(n_clusters_,numSP),y=1.08)
   
    return  cluster_centers_indices, labels_af
        

if __name__ == "__main__":
    
    #Make embedding function
    # csv_dir = 'C:/Users/jpeeples/Documents/Research/Root_Project/TubeNumberKey.csv'
    # img_dir = 'C:/Users/jpeeples/Documents/Research/Root_Project/Data/'
    # csv_dir = 'T:/Fractal_Images/TubeNumberKey.csv'
    # img_dir = 'T:/Fractal_Images/'
    csv_dir = '../../Data/TubeNumberKey.csv'
    img_dir = '../../Data/'
    run_nums = [6,7] #[6,7]
    
    # reading the data from the file 
    with open('Utils/feature_dict.json') as f: 
        data = f.read() 
    # reconstructing the data as a dictionary 
    feat_dict = json.loads(data) 
    
    mode = 'bw' #rgb or bw
    DAP = 'last' #last or all
    preprocess = 'gauss' #Either do 'avg' (average filter), 'gauss' (gaussian filter), or None
    features = 'lacunarity' #Either use 'root_pixels','fractal', or 'lacunarity'
    ds_methods = 'Avg_Pool'
    train_reps = [1,2,4] #[1,2,3]
    downsample = True #(set to false for 'avg')
    ds_factor = 16
    test_reps = [3] #4
    num_imgs = 1
    cluster_vis = True
    equal_weight = False
    
    #Seed for Random state for TSNE, Clustering and Metrics (should only 
    # root visualization and not clustering performance)
    seed = 42
    norm_val = 'None'
    embed = 'UMAP' #Embedding method to be used, select either 'TSNE','UMAP', or 'MDS'
    split_data = True #Cluster training and test images separately (if true, will "predict" test labels)
    SP = 10
    results_folder = 'Results/Clustering/' + features + '/'
    
    SP_folder = (results_folder + features + '_Superpixel_Hist_clustering/SP_imgs/' 
                 + preprocess + '/' + mode +'_Run_' + str(run_nums) + 'Random_State_'
                 + str(seed) + '/')
    #Generate dataset
    dataset = load_data(csv_dir,img_dir,run_nums,train_reps = train_reps,
                    test_reps = test_reps, DAP=DAP,mode=mode,preprocess=preprocess,
                    downsample=downsample,ds_factor=ds_factor,ds_type=ds_methods)
    
    #Get embedding
    (embedding, data, water_levels, cultivar, EMD_mat, names,
     train_idx, test_idx) = SP_clustering(dataset,numSP=SP,mode=mode,
                              num_imgs=num_imgs,
                              folder=SP_folder + 'SP_'+ str(SP) + '/', embed = embed,
                              split_data=split_data,features=features,seed=seed,
                              embed_only=True,normalize=norm_val,equal_weight=equal_weight)
    del dataset
    cross_treatments = tuple(zip(water_levels, cultivar)) 
    class_names = list(itertools.product(list(np.unique(cultivar)),
                                              list(np.unique(water_levels)))) 
    
    roots_df = pd.DataFrame(embedding, columns=('x', 'y'))
    roots_df['water_levels'] = [str(x) for x in water_levels]
    roots_df['cultivar'] = [str(x) for x in cultivar]
    roots_df['run'] = ['Run'+x[4] for x in names]
    roots_df['tube_num'] = [x[-3::] for x in names]
    

    # Create figure with 1x2 subplots, left for scatter plot and right for image preview.
    # Left subplot occupies 2/3 of the width, right subplot occupies 1/3 of the width.
    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    gs = GridSpec(2, 2, width_ratios=(2, 1))
    ax = [
        fig.add_subplot(gs[:, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1,1]),
    ] # type: List[plt.Axes]

    # Create scatter plot of 2d features in left subplot
    if cluster_vis:

        count = 0
        cross_labels = np.zeros(len(cultivar))
        for treatment in class_names:
            
            temp_cultivar = np.where(np.array(cultivar)==treatment[0])
            temp_water_level = np.where(np.array(water_levels)==treatment[1])
            intersection = np.intersect1d(temp_cultivar,temp_water_level)
            cross_labels[intersection] = count
            count += 1
            
        centers,labels = Create_Cluster_Figure(EMD_mat,embedding,SP,embed=embed,ax=ax[0],
                          labels=cross_labels,class_names=class_names,
                          seed=seed,split_data=split_data,train_idx=train_idx,
                          test_idx=test_idx,run_nums=run_nums)
        
        
        # SI_scores = metrics.silhouette_samples(EMD_mat, labels, metric='precomputed',
        #                                       random_state=seed)
        
        
        # #Create dataframe and save
        cluster_df = pd.DataFrame(labels[centers], columns = ['Cluster Center #'])
        cluster_df['Cultivar'] = roots_df['cultivar'].iloc[centers].values
        cluster_df['Water Levels'] = roots_df['water_levels'].iloc[centers].values
        
        #Get average SI scores, biological measures, and texture features
        # corresponding to each cluster center
        # SI_avg = []
        # SI_std = []
        Avg_root_leng_avg = []
        Avg_root_leng_std = []
        num_roots_avg = []
        num_roots_std = []
        text_feat_avg = []
        text_feat_std = []
        global_feat_avg = []
        global_feat_std = []
        
        for temp_center in np.unique(labels):
            # SI_avg.append(np.round(np.average(SI_scores[labels==temp_center]),decimals=4))
            # SI_std.append(np.round(np.std(SI_scores[labels==temp_center]),decimals=4))
            run_num = roots_df['run'].iloc[labels==temp_center].values
            tube_num = roots_df['tube_num'].iloc[labels==temp_center].values
            avg_root_len = []
            num_roots =[]
            global_feat = []
            temp_data = list(compress(data, labels==temp_center))
            #Loop through each image biological measure
            for img in range(0,len(tube_num)):
                try:
                    avg_root_len.append(feat_dict[run_num[img]][tube_num[img]]["avg_totL"])
                    num_roots.append(feat_dict[run_num[img]][tube_num[img]]["num_of_root"])
                    if features == 'lacunarity':
                        global_feat.append(fractal_dimension(temp_data[img]['Img'],compute_lac=True)[-1])
                    elif features == 'fractal':
                        global_feat.append(fractal_dimension(temp_data[img]['Img']))
                    else: #Number of root pixels
                        global_feat.append(np.count_nonzero(temp_data[img]['Img'])/(temp_data[img]['Img'].size))
                except:
                    print('No bio measures for {} Tube {}'.format(run_num[img],tube_num[img]))
            
            Avg_root_leng_avg.append(np.round(np.average(avg_root_len),decimals=3))
            Avg_root_leng_std.append(np.round(np.std(avg_root_len),decimals=3))
            num_roots_avg.append(np.round(np.average(num_roots),decimals=0))
            num_roots_std.append(np.round(np.std(num_roots),decimals=0))
            text_feats = []
            text_feats = [x['SP_profile'][:,0] for x in temp_data]
            text_feat_avg.append(np.round(np.average(text_feats),decimals=4))
            text_feat_std.append(np.round(np.std(text_feats),decimals=4))
            global_feat_avg.append(np.round(np.average(global_feat),decimals=4))
            global_feat_std.append(np.round(np.std(global_feat),decimals=4))
        
     
        # cluster_df['Avg SI Scores'] = ["{:.3f}".format(x) for x in SI_avg]
        # cluster_df['Std SI Scores'] = ["{:.3f}".format(x) for x in SI_std]
        cluster_df['Avg Root Length'] = ["{:.3f}".format(x) for x in Avg_root_leng_avg] 
        cluster_df['Std Root Length'] = ["{:.3f}".format(x) for x in Avg_root_leng_std] 
        cluster_df['Avg # of Roots'] = ["{}".format(x) for x in num_roots_avg]
        cluster_df['Std # of Roots'] = ["{}".format(x) for x in num_roots_std] 
        cluster_df['Avg SP '+ features] = ["{:.3f}".format(x) for x in text_feat_avg]
        cluster_df['Std SP '+ features] = ["{:.3f}".format(x) for x in text_feat_std]
        cluster_df['Avg Global '+ features] = ["{:.3f}".format(x) for x in global_feat_avg]
        cluster_df['Std Global '+ features] = ["{:.3f}".format(x) for x in global_feat_std]
        
        #Save out
        # pdb.set_trace()
        #Possible location: SP_folder + 'SP_'+ str(SP) + '/Cluster_Results.csv' 
        cluster_df.to_csv('Runs_' + str(run_nums) + '_Cluster_Results.csv',index=False)
        
    else:
        markers =['o','*','s','^']
        plt_colors = ['red','green','blue','purple']
        color_markers = list(itertools.product(markers,plt_colors))
        count = 0
        cross_labels = np.zeros(len(cultivar))
        for treatment in class_names:
            
            temp_cultivar = np.where(np.array(cultivar)==treatment[0])
            temp_water_level = np.where(np.array(water_levels)==treatment[1])
            intersection = np.intersect1d(temp_cultivar,temp_water_level)
            cross_labels[intersection] = count
            x = embedding[[intersection],0]
            y = embedding[[intersection],1]
            ax[0].scatter(x, y, color = color_markers[count][1], 
                          marker = color_markers[count][0], label=class_names[count])
            count += 1
      
        ax[0].set_title('Cross Treatments True Labels for ' + str(SP) + ' Superpixels')
        ax[0].legend(class_names,bbox_to_anchor=(1.04, 1), borderaxespad=0.)
        
    # Initialize image preview in right subplot
    ax[1].set_title("Root Image")
    ax[1].set_axis_off()

    # Connect the image preview event handler to this figure's onclick event
    fig.canvas.mpl_connect("button_press_event", onclick)

    # Show the plot
    plt.show()