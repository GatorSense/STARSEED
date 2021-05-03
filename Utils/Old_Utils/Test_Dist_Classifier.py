# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:47:10 2020
Test Classifier that uses correlation between training and test images
@author: jpeeples
"""
import pdb
import numpy as np
from scipy.special import softmax,expit
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
import os
from Utils.Visualize_Differences import Visualize_differences
#Distance classifier (compute distance to nearest pixel roots)

def test_dist_classifier(model,test_dataset,mode='bw',downsample=True,K=1,
                         num_imgs=1,folder='Test_Imgs/'):

    print('Testing Model...')   
    start = time.time()
    
    #Load training data and test data, use half precision
    X_train_water = model['X_train_water']
    X_train_cultivar = model['X_train_cultivar']
    cultivar = model['cultivar']
    water_levels = model['water_levels']
   
    test_imgs = test_dataset['data']
    test_water_levels = test_dataset['water_level_labels']
    test_cultivar = test_dataset['cultivar_labels']
    
    #Delete model and test dataset to clear memory 
    del model,test_dataset
    
    #Downsample training and testing images
    if downsample:
        ds = 16 #2 for AVG, else 16 (comparable sizes)
        X_train_water = X_train_water[:,::ds,::ds]
        X_train_cultivar= X_train_cultivar[:,::ds,::ds]
        test_imgs = test_imgs[:,::ds,::ds]
    #Expand dimension for test dataset if grayscale/bw
    if mode == 'bw':
        test_imgs = np.expand_dims(test_imgs,axis=-1)
        
    #Expand second dimension of test images and repeat tensor 
    # along second dimension for the number of classes
    # assumes number of cultivar and water levels are the same
    X_water_corr = []
    X_cultivar_corr = []
    
    #Get indices of training data
    num_class = X_train_water.shape[0]
    train_water_indices = []
    train_cultivar_indices = []
    
    for current_class in range(0,num_class):
        train_water_indices.append(np.nonzero(X_train_water[current_class]))
        train_cultivar_indices.append(np.nonzero(X_train_cultivar[current_class]))
        
    #Test one image at a time (memory issue with all at once)
    visual_img = 0
    for img in range(0,test_imgs.shape[0]):
        
        temp_test_img = test_imgs[img]
        
        #Get non-zero entries in test img
        temp_test_indices = np.nonzero(temp_test_img)
        
        #For each class compute raw score which is the sum of the distance of the
        #nearest non-zero neighbor over the total number of root pixels in test img
        temp_X_water_corr = np.zeros(num_class)
        temp_X_cultivar_corr = np.zeros(num_class)
        num_root_pixels = np.count_nonzero(temp_test_img)
        
        temp_test_indices = np.vstack((temp_test_indices[0],temp_test_indices[1])).T
        test_water_dist = []
        test_cultivar_dist = []
        for current_class in range(0,num_class):
            
            #Get pairwise distance between spatial dimensions of training and test
            temp_water_train_indices = np.vstack((train_water_indices[current_class][0],
                                                   train_water_indices[current_class][1])).T
            
            temp_cultivar_train_indices = np.vstack((train_cultivar_indices[current_class][0],
                                                   train_cultivar_indices[current_class][1])).T
            
            #Compute distance for water levels
             #Compute euclidean distance between spatial dimension and return distance of K nearest neighbors
            X_water_dist = distance.cdist(temp_water_train_indices,temp_test_indices,'euclidean')
            X_water_dist = np.sort(X_water_dist,axis=0)[:K,:]
            
            #Compute for cultivar
            X_cultivar_dist = distance.cdist(temp_cultivar_train_indices,temp_test_indices,'euclidean')
            X_cultivar_dist = np.sort(X_cultivar_dist,axis=0)[:K,:]
            

            #Save out temporary scores for each class
            temp_X_water_corr[current_class] = np.sum(X_water_dist)/(num_root_pixels*K)
            temp_X_cultivar_corr[current_class] = np.sum(X_cultivar_dist)/(num_root_pixels*K)
            
            #Save distances for plotting
            test_water_dist.append(X_water_dist)
            test_cultivar_dist.append(X_cultivar_dist)
            
        #Create visual of select number of test images
        if visual_img < num_imgs:
            Visualize_differences(temp_test_img,test_water_dist,
                                  test_cultivar_dist,temp_test_indices,folder,
                                  visual_img,water_levels,cultivar) 
            visual_img += 1
        
        #Append scores for each test img
        X_water_corr.append(temp_X_water_corr)
        X_cultivar_corr.append(temp_X_cultivar_corr)
    
    #Change scores to arrays
    X_water_corr = np.stack(X_water_corr,axis=0)
    X_cultivar_corr = np.stack(X_cultivar_corr,axis=0)
    
    #Make predictions and map values to water and cultivar levels, should be min?
    water_preds = np.argmin(X_water_corr,axis=-1)
    cultivar_preds = np.argmin(X_cultivar_corr,axis=-1)
    water_preds = water_levels[water_preds]
    cultivar_preds = cultivar[cultivar_preds]
    
    #Compute probabilities and confidences
    water_prob_scores = softmax(X_water_corr,axis=-1)
    water_conf_scores = expit(X_water_corr)
    cultivar_prob_scores = softmax(X_cultivar_corr,axis=-1)
    cultivar_conf_scores = expit(X_cultivar_corr)
    
    #Return dictionary of scores and model outputs
    scores = {'water_raw_score': X_water_corr, 
              'water_prob_scores': water_prob_scores,
              'water_conf_scores': water_conf_scores,
              'cultivar_raw_score': X_cultivar_corr,
              'cultivar_prob_scores': cultivar_prob_scores,
              'cultivar_conf_scores': cultivar_conf_scores}
    outputs = {'water_GT': test_water_levels, 'water_preds': water_preds,
               'cultivar_GT': test_cultivar, 'cultivar_preds': cultivar_preds}

    time_elapsed = time.time() - start
    
    print('Tested Model in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
    
    return scores, outputs
    
    
