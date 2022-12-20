# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:13:42 2020
Visualize EMD scores and flow from source to destination
Code modified and adapted from: 
https://samvankooten.net/2018/09/25/earth-movers-distance-in-python/
@author: jpeeples
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import math
import os
import pdb


def Visualize_EMD(test_img,train_imgs,EMD_scores,SP_flow, folder,test_label,img_name,
                  num_class=4,arrow_width_scale=2,per_arrows=0.10,title=None,class_names=None,
                  lab=False,sp_overlay=True,train_imgs_names=None, cmap='binary', 
                  stacked=False, max_feat=None,min_feat=None):

    """Plots the flow computed by cv2.EMD
    
    The source images are retrieved from the signatures and
    plotted in a combined image, with the first image in the
    red channel and the second in the green. Arrows are
    overplotted to show moved earth, with arrow thickness
    indicating the amount of moved earth."""
    fig, ax = plt.subplots(2,num_class+1,figsize=(24,12))
    plt.subplots_adjust(wspace=.4,hspace=.4)
    
    #Plot test image and EMD scores
    #Overlay SP images over roots
    #For now, just show root image
    if sp_overlay:
        # ax[0,0].imshow(mark_boundaries(test_img['Img'],test_img['SP_mask'],color=(1,1,0)),
        #                 aspect = 'auto',cmap=cmap)
        ax[0,0].imshow(test_img['Img'],aspect='auto',cmap=cmap)
        ax[0,0].set_title(img_name +': \n' + str(test_label))
    else:
 
        mask_values = ax[0,0].imshow(test_img['Root_mask'], aspect="auto",vmin=min_feat,vmax=max_feat)
        ax[0,0].set_title(img_name +': \n' + str(test_label))
        fig.colorbar(mask_values,ax=ax[0,0])
    
    ax[0,0].tick_params(axis='both', labelsize=0, length = 0)
    ax[0,0].set_xlabel('(a)', fontsize=20)
    y_pos = np.arange(len(class_names))
    rects = ax[1,0].bar(y_pos,EMD_scores)
    ax[1,0].set_xticks(y_pos)
    ax[1,0].set_xticklabels(class_names,rotation=90)
    ax[1,0].patches[np.argmin(EMD_scores)].set_facecolor('#aa3333')
    ax[1,0].set_title('EMD Class Scores: ')
    ax[1,0].set_xlabel('(e)', fontsize=20)
    for current_class in range(0,num_class):
        
        #Plot superpixel images for each class 
        #Superpixel segmentation
        #Just show root image
        if sp_overlay:
            if lab:
                ax[0,current_class+1].imshow(train_imgs[current_class]['Img'],
                                                  aspect="auto",cmap=cmap)
            else:
                ax[0,current_class+1].imshow(train_imgs[current_class]['Img'],
                                              aspect="auto",cmap=cmap)

        else: 
         
            mask_values = ax[0,current_class+1].imshow(train_imgs[current_class]['Root_mask'], 
                                                       aspect="auto",
                                                       vmin=min_feat,vmax=max_feat)
            fig.colorbar(mask_values,ax=ax[0,current_class+1])
        
        if train_imgs is not None:
            ax[0,current_class+1].set_title(train_imgs_names[current_class] +': \n' + str(class_names[current_class]))
        else:
            ax[0,current_class+1].set_title(str(class_names[current_class]))
        ax[0,current_class+1].tick_params(axis='both', labelsize=0, length = 0)
        ax[0,current_class+1].set_xlabel('(%s)'%(chr(current_class+98)), fontsize=20)
        
        img2 = train_imgs[current_class]['Root_mask']
        
        # RGB values should be between 0 and 1
        combined = np.dstack((test_img['Root_mask']/test_img['Root_mask'].max(), 
                              img2/img2.max(), 0*img2))
     
        ax[1,current_class+1].imshow(combined,aspect='auto')
    
        #For stacked images, need to transpose (change for root only, no longer symmetric matrix)
        if stacked:
            flow = SP_flow[current_class].T
        else:
            flow = SP_flow[current_class]
        
        flows = np.transpose(np.nonzero(flow))
      
        # Plot selected top-K flows 
        mags = [] 
        srcs = [] 
        dests = []
        
        for src, dest in flows:
            # Skip the pixel value in the first element, grab the
            # coordinates. It'll be useful later to transpose x/y.
            # start = test_img['SP_profile'][src, 1:][::-1]
            start = test_img['SP_profile'][src,-2::][::-1]
            try: #Indexing issue with last image, need to check later
                end = train_imgs[current_class]['SP_profile'][dest,-2::][::-1]
                if np.all(start == end):
                    # Unmoved earth shows up as a "flow" from a pixel
                    # to that same exact pixel---don't plot mini arrows
                    # for those pixels
                    continue

                mag = np.log(flow[src, dest]+1) * arrow_width_scale
                mags.append(mag)
                srcs.append(src)
                dests.append(dest)
            except:
                pass
        
        # sorted the flows based on magnitude. Plot top-K flows in the figure
        sorted_flows = sorted(zip(mags, srcs, dests), reverse=True)
        if per_arrows is None:
            num_arrows = len(sorted_flows)
        else:
            num_arrows = int(len(sorted_flows)*per_arrows)
        for i in range(num_arrows):
            mag, src, dest = sorted_flows[i]
            start = test_img['SP_profile'][src,-2::][::-1]
            try: #Indexing issue with last image, need to check later
                # end = train_imgs[current_class]['SP_profile'][dest, 1:][::-1]
                end = train_imgs[current_class]['SP_profile'][dest,-2::][::-1]
                
                # Add a random shift to arrow positions to reduce overlap.
                shift = np.random.random(1) * .3 - .15
                start = start + shift
                end = end + shift
                
                ax[1,current_class+1].quiver(*start, *(end - start), angles='xy',
                            scale_units='xy', scale=1, color='white',
                            edgecolor='black', linewidth=mag/3,
                            width=mag, units='dots',
                            headlength=5,
                            headwidth=3,
                            headaxislength=4.5)
            except:
                pass
    
        ax[1,current_class+1].tick_params(axis='both', labelsize= 0, length = 0)
        ax[1,current_class+1].set_xlabel('(%s)'%(chr(current_class+4+98)), fontsize=20)
        if class_names is not None:
            if stacked:
                ax[1,current_class+1].set_title(str(test_label)+ " to " + 
                                                str(class_names[current_class]))
            else:
                ax[1,current_class+1].set_title(img_name+ " to \n" + 
                                                str(train_imgs_names[current_class]))
    
    if sp_overlay:
        folder = folder +'/Root_Images/' 
    else:
        folder = folder + '/SP_Counts/'
        
    #Create folder and save figures
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    if sp_overlay:
        if stacked:
            fig_name = img_name + '_' +str(test_label) + '_Root_Images'
        else:
            fig_name = img_name + '_Root_Images'
    else:
        if stacked:
            fig_name = img_name + '_' + str(test_label) + '_SP_Counts'
        else:
            fig_name = img_name + '_SP_Counts'
      
    #Replace colons in run name with underscore to save figure properly
    fig_name = fig_name.replace(': ','_')
    fig.savefig(folder+fig_name+'.png', dpi=100,bbox_inches='tight')
    plt.close()
            

    





