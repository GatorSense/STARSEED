# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:47:10 2020
Test Classifier that uses superpixels and K-NN based distance classifier
@author: jpeeples
"""
import pdb
import numpy as np
from scipy.special import softmax,expit
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
import os
# from scipy.stats import wasserstein_distance
import cv2

from Utils.Visualize_SP_EMD import Visualize_EMD

def compute_EMD(test_img,train_data):
    
    # #Remove SP with zero roots
    # test_img = test_img[test_img[:,-1]!=0]
    
    #Get number of training images
    num_train_imgs = len(train_data)
    
    #Initalize EMD array
    EMD_scores = np.zeros(num_train_imgs)
    
    for img in range(0,num_train_imgs):
        
        #Remove SP with zero roots python
        temp_train_img = train_data[img]
      
        #Compute EMD score, array should be num of superpixels x number of root pixels
        # + spatial coordinates (e.g., 500 x 3)
        # Need to update superpixel generation code to allow for count to be first
        EMD_scores[img], _, flow = cv2.EMD(temp_train_img[:,[2,1,0]].astype(np.float32), 
                                           test_img[:,[2,1,0]].astype(np.float32), 
                                           cv2.DIST_L2)
    
    return EMD_scores, flow
    
def test_SP_classifier(dataset,mode='bw',K=1,num_imgs=1,folder='Test_Imgs_SP/'):

    print('Testing Model...')   
    start = time.time()
    
    #Load training data and test data
    training_imgs = dataset['train']['data']
    train_cultivar = dataset['train']['cultivar_labels']
    train_water_levels = dataset['train']['water_level_labels']
   
    test_imgs = dataset['test']['data']
    test_water_levels = dataset['test']['water_level_labels']
    test_cultivar = dataset['test']['cultivar_labels']
    
    #Delete model and test dataset to clear memory 
    del dataset

   #Initialize array of scores
    X_water_corr = []
    X_cultivar_corr = []
    
    #Get indices of training data
    num_class = len(np.unique(train_water_levels))
    water_level_list = np.unique(train_water_levels)
    cultivar_list = list(set(train_cultivar))
    train_water_imgs = []
    train_cultivar_imgs = []
    
    for current_class in range(0,num_class):
        train_water_imgs.append(training_imgs[water_level_list[current_class]==train_water_levels])
        train_cultivar_imgs.append(training_imgs[cultivar_list[current_class]==train_cultivar])
        
    #Test one image at a time (memory issue with all at once)
    num_test_imgs = len(test_imgs)
    # visual_img = 0
    for img in range(0,num_test_imgs):
        
        temp_test_img = test_imgs[img]
        
        #For each class compute raw score which is the sum of the distance of the
        #nearest non-zero neighbor superpixels over the total number of 
        #superpixels in test img
        temp_X_water_corr = np.zeros(num_class)
        temp_X_cultivar_corr = np.zeros(num_class)
        
        test_water_dist = []
        test_cultivar_dist = []
        flow_water_vis = []
        flow_cultivar_vis = []
        
        for current_class in range(0,num_class):
            
            #Compute EMD for training images in class 
            temp_water_level_dist, flow_water = compute_EMD(temp_test_img,
                                                train_water_imgs[current_class])
            temp_cultivar_dist, flow_cultivar = compute_EMD(temp_test_img,
                                             train_cultivar_imgs[current_class]) 
            
            #Save flow for visual
            flow_water_vis.append(flow_water)
            flow_cultivar_vis.append(flow_cultivar)
            
            #Compute distance for water levels
            #Compute euclidean distance between spatial dimension and return distance of K nearest neighbors
            X_water_dist = temp_water_level_dist
            X_water_dist = np.sort(X_water_dist,axis=0)[:K]
            
            #Compute for cultivar
            X_cultivar_dist = temp_cultivar_dist
            X_cultivar_dist = np.sort(X_cultivar_dist,axis=0)[:K]
            

            #Save out temporary scores for each class
            temp_X_water_corr[current_class] = np.sum(np.sort(X_water_dist,axis=0)[:K])
            temp_X_cultivar_corr[current_class] = np.sum(np.sort(X_water_dist,axis=0)[:K])
            
            #Save distances for plotting
            test_water_dist.append(X_water_dist)
            test_cultivar_dist.append(X_cultivar_dist)
        
        # #Create visual of select number of test images (only for stacked images)
        # if visual_img < num_imgs:
        #     #For water levels
        #     Visualize_EMD(temp_test_img,train_water_imgs, test_water_dist,
        #                   flow_water_vis,folder,num_class=num_class,title='Water Levels') 
        #     #For cultivar
        #     Visualize_EMD(temp_test_img,train_cultivar_imgs, test_cultivar_dist,
        #                   flow_cultivar_vis,folder,num_class=num_class,title='Cultivar')         
            
        #     visual_img += 1
        #Append scores for each test img
        X_water_corr.append(temp_X_water_corr)
        X_cultivar_corr.append(temp_X_cultivar_corr)
    
    #Change scores to arrays
    X_water_corr = np.stack(X_water_corr,axis=0)
    X_cultivar_corr = np.stack(X_cultivar_corr,axis=0)
    
    #Make predictions and map values to water and cultivar levels, should be min?
    water_preds = np.argmin(X_water_corr,axis=-1)
    cultivar_preds = np.argmin(X_cultivar_corr,axis=-1)
    water_preds = train_water_levels[water_preds]
    cultivar_preds = train_cultivar[cultivar_preds]
    
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
    
    
