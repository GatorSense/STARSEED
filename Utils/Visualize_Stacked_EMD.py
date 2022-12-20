# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:49:36 2021
Function to combine images based on labels
and show changes between each class or cluster representative 
@author: jpeeples
"""

import matplotlib.pyplot as plt
import numpy as np
from Utils.Visualize_SP_EMD import Visualize_EMD
from sklearn import preprocessing
import os
import cv2
import pdb
from Utils.Compute_EMD import compute_EMD

def Stack_EMD_Visual(data,labels,folder,feature,class_names=None,root_only=True,
                     lab=False,label_type='Water Level'):
    # pdb.set_trace()
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    if class_names is None:
        class_names = np.unique(labels)
        
    labels_af = labels
    n_clusters_ = len(np.unique(labels))
    
    #Encode labels as numerical values
    le = preprocessing.LabelEncoder()
    labels_af = le.fit_transform(labels_af)
 
    #Create stack feature maps figure
    fig_stack, ax_stack = plt.subplots(nrows=2,ncols=n_clusters_,figsize=(22,11))
    plt.subplots_adjust(wspace=.7,hspace=.5)

    center_treatments = []
    num_class_imgs = []
    stack_data, avg_max, avg_min, std_max, std_min = get_stacked_data(data,n_clusters_,labels_af)
    for i, k in enumerate(zip(range(n_clusters_))):
        class_members = labels_af == k
        if isinstance(class_names[k], float):
            treatment = "%.f%%" %(class_names[k]*100) #convert water level from float to percent
        else:
            treatment = str(class_names[k])
        center_treatments.append(treatment)
      
        img_idx = np.where(class_members)[0]
        num_imgs = len(img_idx)
        num_class_imgs.append(len(img_idx))
        k = k[0]
         
        avg_vals = ax_stack[0,k].imshow(stack_data[k]['Root_mask'],aspect='auto',vmin=avg_min,vmax=avg_max)
        std_vals = ax_stack[1,k].imshow(stack_data[k]['Root_mask_std'],aspect='auto',vmin=std_min,vmax=std_max)
        fig_stack.colorbar(avg_vals,ax=ax_stack[0,k])
        fig_stack.colorbar(std_vals,ax=ax_stack[1,k])
        ax_stack[0,k].set_title('Avg SP {} \nFor {} {}\nTotal Images: {}'. format(feature,label_type,treatment,num_imgs))
        ax_stack[1,k].set_title('Std SP {} \nFor {} {}\nTotal Images: {}'. format(feature,label_type,treatment,num_imgs))
        ax_stack[0,k].tick_params(axis='both', labelsize=0, length = 0)
        ax_stack[1,k].tick_params(axis='both', labelsize=0, length = 0)
        ax_stack[0,k].set_xlabel('(%s)' %(chr(i+97)), fontsize=20)
        ax_stack[1,k].set_xlabel('(%s)' %(chr(i+n_clusters_+97)), fontsize=20)
     
        
    #Save stack image
    fig_stack.savefig(folder + '{}_Stacked_{}_Feature_Maps.png'.format(feature,label_type),
                      dpi=fig_stack.dpi,bbox_inches='tight')
    plt.close(fig=fig_stack)
    
    #Compute EMD visualization for each class
    for class_stack in range(0,len(stack_data)):
        EMD_class_scores = []
        EMD_class_flows = []
        
        #Grab one class and compute EMD between other aggregated feature maps
        temp_indices = np.arange(len(stack_data))
        temp_indices = np.delete(temp_indices,class_stack)
        for temp_class in temp_indices:
            temp_EMD, temp_flow = compute_EMD(stack_data[class_stack]['SP_profile'],stack_data[temp_class]['SP_profile'])
            EMD_class_scores.append(temp_EMD)
            EMD_class_flows.append(temp_flow)
  
        #Visualize images with magnitude and directional arrows
        Visualize_EMD(stack_data[class_stack],np.array(stack_data)[temp_indices],
                      EMD_class_scores,EMD_class_flows,folder,center_treatments[class_stack],
                      label_type, num_class=n_clusters_-1,
                      title='EMD for {}'.format(center_treatments[class_stack]),
                      class_names=np.array(center_treatments)[temp_indices],
                      lab=lab,sp_overlay=False,train_imgs_names=np.repeat(np.array(label_type),len(temp_indices)),
                      arrow_width_scale=80,cmap='binary',stacked=True,
                      max_feat=avg_max,min_feat=avg_min)
        
        Visualize_EMD(stack_data[class_stack],np.array(stack_data)[temp_indices],
                      EMD_class_scores,EMD_class_flows,folder,center_treatments[class_stack],
                      label_type, num_class=n_clusters_-1,
                      title='EMD for {}'.format(center_treatments[class_stack]),
                      class_names=np.array(center_treatments)[temp_indices],
                      lab=lab,sp_overlay=True,train_imgs_names=np.repeat(np.array(label_type),len(temp_indices)),
                      arrow_width_scale=80,cmap='binary', stacked=True,
                      max_feat=avg_max,min_feat=avg_min)
    

def get_stacked_data(data,num_clusters,labels):

    #Return max and std values for 
    stack_data = []
    num_class_imgs = []
    for k in zip(range(num_clusters)):
        class_members = labels == k
          
        temp_data = []
        temp_SP_profile = []
        temp_img = []
        max_avgs = []
        max_stds = []
        min_avgs = []
        min_stds = []
        
        img_idx = np.where(class_members)[0]
        num_imgs = len(img_idx)
        num_class_imgs.append(num_imgs)
        for idx in img_idx:
            temp_data.append(data[idx]['Root_mask'])
            temp_SP_profile.append(data[idx]['SP_profile'])
            temp_img.append(data[idx]['Img'])
        
        temp_data = np.stack(temp_data,axis=0)
        temp_SP_profile = np.stack(temp_SP_profile,axis=0)
        temp_img = np.stack(temp_img,axis=0)
       
        #Save out max and min average and std for colormap
        max_avgs.append(np.max(np.average(temp_data,axis=0)))
        min_avgs.append(np.min(np.average(temp_data,axis=0)))
        max_stds.append(np.max(np.std(temp_data,axis=0)))
        min_stds.append(np.min(np.std(temp_data,axis=0)))

        stack_data.append({'Root_mask': np.average(temp_data,axis=0),
                           'Root_mask_std': np.std(temp_data,axis=0),
                           'SP_profile': np.average(temp_SP_profile,axis=0),
                           'SP_mask': data[idx]['SP_mask'],
                           'Img': np.average(temp_img,axis=0)})
    
    return stack_data, np.max(max_avgs), np.min(min_avgs), np.max(max_stds), np.min(min_stds)
    
    
    
    

