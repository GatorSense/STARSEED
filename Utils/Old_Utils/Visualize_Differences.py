# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:30:50 2020

@author: jpeeples
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os

def Visualize_differences(img,water_dist, cultivar_dist, img_indices, folder,
                          img_num, water_levels, cultivar):
    
    num_class = len(water_levels)
    
    #Create folder and save figures
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    #Create figures for water levels
    fig_water = plt.figure(figsize=(16,16))
    fig_water.add_subplot(1,num_class+1,1)
    ax = fig_water.gca()
    
    if img.shape[-1]==1:
        img = img[:,:,0]
        
    im = ax[0].imshow(img,cmap='gray')
    plt.title('Test Image: ' + str(img_num))
    plt.axis('off')
    im_ratio = img.shape[0]/img.shape[1]
    plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)

    for current_class in range(1,num_class+1):
        #Generate temp matrix
        temp_dist_img = np.zeros((img.shape[0],img.shape[1]))
        
        #Fill values of image with corresponding distances
        temp_dist_img[img_indices[:,0],img_indices[:,1]] = np.sum(water_dist[current_class-1],axis=0)
        
        #Plot distances
        fig_water.add_subplot(1,num_class+1,current_class+1)
        ax = fig_water.gca()
        im = ax[current_class].imshow(temp_dist_img,'hot',vmin=0,vmax=255)
        plt.title('Water=%.2f' %(water_levels[current_class-1]), fontsize = 12)
        plt.colorbar(im,fraction=0.046*im_ratio,pad=0.04)
        plt.axis('off')
        
    
    #Save figure
    fig_water.savefig(folder+'Img_'+ str(img_num) + '_Water_levels.png',dpi=1000)
    plt.close()

    #Create figures for cultivar levels
    fig_cultivar = plt.figure(figsize=(16,16))
    fig_cultivar.add_subplot(1,num_class+1,1)
    ax = fig_cultivar.gca()    
        
    im = ax.imshow(img,cmap='gray')
    plt.title('Test Image: ' + str(img_num))
    plt.axis('off')
    im_ratio = img.shape[0]/img.shape[1]
    plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)

    for current_class in range(1,num_class+1):
        #Generate temp matrix
        temp_dist_img = np.zeros((img.shape[0],img.shape[1]))
        
        #Fill values of image with corresponding distances
        temp_dist_img[img_indices[:,0],img_indices[:,1]] = np.sum(cultivar_dist[current_class-1],axis=0)
        
        #Plot distances
        fig_cultivar.add_subplot(1,num_class+1,current_class+1)
        ax = fig_cultivar.gca()
        im = ax.imshow(temp_dist_img,cmap='hot',vmin=0,vmax=255)
      
        plt.title('Cultivar=%s' %(cultivar[current_class-1]), fontsize = 12)
        plt.colorbar(im,fraction=0.046*im_ratio,pad=0.04)
        plt.axis('off')
        
    #Save figure
    fig_cultivar.savefig(folder+'Img_'+ str(img_num) + '_Cultivar.png',dpi=1000)
    plt.close()