# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:45:12 2020
Train model
@author: jpeeples
"""
import pdb
import numpy as np
import time


def train_SP_classifier(training_dataset,mode='bw',stack=True):
        
    print('Training Model...')   
    start = time.time()
    
    #Get data and detect if it is RGB or grayscale
    train_data = training_dataset['data']/255
    water_level_labels = training_dataset['water_level_labels']
    cultivar_labels = training_dataset['cultivar_labels']
    
    #Save memory by deleting training_dataset
    del training_dataset
    
        
    #Find number of classes for water level and cultivar
    water_levels = np.unique(water_level_labels)
    cultivar = np.unique(cultivar_labels)
    
    #Compute training data histograms for water levels and cultivar
    X_train_water = []
    X_train_cultivar = []
    
    #Only using one for loop, assuming there are the same number of water
    #levels as cultivar 
    for classes in range(0,len(water_levels)):
        
        #Get indices that correspond to water level and cultivar
        temp_water_indices = np.where(water_level_labels==water_levels[classes])
        temp_cultivar_indices = np.where(cultivar_labels==cultivar[classes])
    
        #Select images that correspond to correct indices and compute average
        # across samples
        
        # X_train_water.append(np.average(train_data[temp_water_indices],axis=0))
        # X_train_cultivar.append(np.average(train_data[temp_cultivar_indices],axis=0))  
        
        #Uncomment when normalization is fixed below
        #If stack images, sum across images
        if stack:
            X_train_water.append(np.sum(train_data[temp_water_indices],axis=0))
            X_train_cultivar.append(np.sum(train_data[temp_cultivar_indices],axis=0))
        else: #Keep images separate
            X_train_water.append(train_data[temp_water_indices])
            X_train_cultivar.append(train_data[temp_cultivar_indices])
    
    #Change list to numpy arrays
    pdb.set_trace()
    #X_train_water = np.stack(X_train_water,axis=0)
    #X_train_cultivar = np.stack(X_train_cultivar,axis=0)
    
    #Normalize images
    #Issue with normalization, dividing by all pixels makes values really small (essentially 0)
    #Also need to update for RGB data
    # for classes in range(0,X_train_water.shape[0]):
    #     water_temp_sum = sum(sum(X_train_water[classes]))
    #     cultivar_temp_sum = sum(sum(X_train_cultivar[classes]))
    #     X_train_water[classes] = np.true_divide(1e6*X_train_water[classes],water_temp_sum)
    #     X_train_cultivar[classes] = np.true_divide(1e6*X_train_cultivar[classes],cultivar_temp_sum)

    #Unsqueeze last dimension if grayscale/bw
    if mode == 'bw':
        for classes in range(0,len(water_levels)):
            X_train_water[classes] = np.expand_dims(X_train_water[classes],axis=-1)
            X_train_cultivar[classes] = np.expand_dims(X_train_cultivar[classes],axis=-1)
    
    #Return dictionary of data and labels
    trained_model = {'X_train_water': X_train_water, 
                     'X_train_cultivar': X_train_cultivar,
                     'cultivar': cultivar,
                     'water_levels': water_levels}
    
    time_elapsed = time.time() - start
    
    print('Trained Model in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return trained_model    
    