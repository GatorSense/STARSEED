# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:10:32 2020

@author: weihuang.xu
"""

import numpy as np
import time
import pdb

# Define function to calculate the histgram along height and width of image.
# Then, do correlation of training data and test data

def hist_corr_classifier(model, test_dataset, mode='bw'):
    
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
    
    #Expand dimension for test dataset if grayscale/bw
    if mode == 'bw':
        test_imgs = np.expand_dims(test_imgs,axis=-1)
    
    #Calculate histgram along each column for training
    train_histcolumn_water = np.sum(X_train_water, 1)
    train_histcolumn_cultivar = np.sum(X_train_cultivar, 1)

    #Calculate histgram along each row for trianing
    train_histrow_water = np.sum(X_train_water, 2)
    train_histrow_cultivar = np.sum(X_train_cultivar, 2)    
    
    #Calculate histgram along each column for test
    test_histcolumn = np.sum(test_imgs, 1)

    #Calculate histgram along each row for test
    test_histrow = np.sum(test_imgs, 2)
    
    #Caluculate column correlation of test image and each training image
    all_score_water = []
    all_score_cultivar = []
    for index in range(X_train_water.shape[0]):
        score_column_water = np.multiply(test_histcolumn, train_histcolumn_water[index]).mean(1)
        score_row_water = np.multiply(test_histrow, train_histrow_water[index]).mean(1)
        score_column_cultivar = np.multiply(test_histcolumn, train_histcolumn_cultivar[index]).mean(1)
        score_row_cultivar = np.multiply(test_histrow, train_histrow_cultivar[index]).mean(1)

        # Calculate overall correlation score    
        all_score_water.append(score_column_water + score_row_water)
        all_score_cultivar.append(score_column_cultivar + score_row_cultivar)
    
    # import pdb; pdb.set_trace()
    if mode == 'bw':
        all_score_water = np.stack(all_score_water,axis=1).squeeze(-1)
        all_score_cultivar = np.stack(all_score_cultivar,axis=1).squeeze(-1)
    elif mode == 'rgb':
        all_score_water = np.stack(all_score_water,axis=1).mean(-1)
        all_score_cultivar = np.stack(all_score_cultivar,axis=1).mean(-1)
        
    #Make predictions and map values to water and cultivar levels
    water_preds = np.argmax(all_score_water,axis=-1)
    cultivar_preds = np.argmax(all_score_cultivar,axis=-1)
    water_preds = water_levels[water_preds]
    cultivar_preds = cultivar[cultivar_preds]
    
    #Return dictionary of scores and model outputs
    scores = {'water_raw_score': all_score_water, 
              'cultivar_raw_score': all_score_cultivar}
    outputs = {'water_GT': test_water_levels, 'water_preds': water_preds,
               'cultivar_GT': test_cultivar, 'cultivar_preds': cultivar_preds}
    
    
    
    time_elapsed = time.time() - start
    
    print('Tested Model in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return scores, outputs