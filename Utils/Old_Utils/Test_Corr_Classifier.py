# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:47:10 2020
Test Classifier that uses correlation between training and test images
@author: jpeeples
"""
import pdb
import numpy as np
from scipy.special import softmax,expit
import time

#Correlation classifier

def test_corr_classifier(model,test_dataset,mode='bw'):

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
    
    #Expand dimension for test dataset if grayscale/bw
    if mode == 'bw':
        test_imgs = np.expand_dims(test_imgs,axis=-1)
        
    #Expand second dimension of test images and repeat tensor 
    # along second dimension for the number of classes
    # assumes number of cultivar and water levels are the same
    X_water_corr = []
    X_cultivar_corr = []
    
    #Expand first dimension of training data and repeat tensor
    # along first dimension for the number of testing samples
    #X_train_water = np.expand_dims(X_train_water,axis=0)
    #X_train_cultivar = np.expand_dims(X_train_cultivar, axis=0)
    
    #Test one image at a time (memory issue with all at once)
    for img in range(0,test_imgs.shape[0]):
        temp_test_img = np.expand_dims(test_imgs[img],axis=0)
        temp_test_img = np.repeat(temp_test_img,len(cultivar),axis=0)
        
        #X_train_water = np.repeat(X_train_water,test_imgs.shape[0],axis=0)
        #X_train_cultivar = np.repeat(X_train_cultivar,test_imgs.shape[0],axis=0)
        
        #Multiply and sum/average along the spatial dimensions to compute correlation
        # between test and training data
        temp_X_water_corr = np.multiply(X_train_water,temp_test_img)
        temp_X_cultivar_corr = np.multiply(X_train_cultivar,temp_test_img)
        temp_X_water_corr = np.average(temp_X_water_corr,axis=(1,2))
        temp_X_cultivar_corr = np.average(temp_X_cultivar_corr,axis=(1,2))
        
        #Compute average score across channels
        X_water_corr.append(np.average(temp_X_water_corr,axis=-1))
        X_cultivar_corr.append(np.average(temp_X_cultivar_corr,axis=-1))
    
    #Change scores to arrays
    X_water_corr = np.stack(X_water_corr,axis=0)
    X_cultivar_corr = np.stack(X_cultivar_corr,axis=0)
    
    #Make predictions and map values to water and cultivar levels
    water_preds = np.argmax(X_water_corr,axis=-1)
    cultivar_preds = np.argmax(X_cultivar_corr,axis=-1)
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
    
    
    
    
    
    
    
