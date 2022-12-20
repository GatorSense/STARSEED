# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:08:19 2020
Load training and test images
@author: jpeeples
"""
import pandas as pd
import os
import natsort
from PIL import Image
import numpy as np
import time
import skimage.measure
from scipy.ndimage import gaussian_filter

def extract_features(X,preprocess = 'avg',mode='rgb'):

    #Perform average pooling (sride and kernel are the same size)
    if preprocess == 'avg':
        if mode == 'rgb':
            X = skimage.measure.block_reduce(X, (7,7,1), np.average)
        else:
            X = skimage.measure.block_reduce(X, (7,7), np.average)
    elif preprocess == 'gauss':
        X = gaussian_filter(X,sigma=1)
    else:
        pass
    return X

#Need to clean up, make one function for DAP variations
def load_data(csv_dir,img_dir,run_nums,train_reps = [1,2,3],test_reps = [4], 
              mode='bw',preprocess = 'avg',downsample=False,ds_factor=16,
              ds_type='No_Pool'):

    print('Loading dataset...')
    start = time.time()
    #Load tube number key csv file
    tube_key = pd.read_csv(csv_dir)

    #Extract data from tube key
    tube_number = tube_key['Tube Number'].tolist()
    tube_rep = tube_key['Rep']
    tube_water_level = tube_key['Water Level']
    tube_cultivar = tube_key['Cultivar']
    
    #For each run, generate training and testing data
    training_data = []
    training_labels_water_level = []
    training_labels_cultivar = []
    training_names = []

    
    testing_data = []
    testing_labels_water_level = []
    testing_names = []
    testing_labels_cultivar = []
   
    for run in run_nums:
        run_dir = os.path.join(img_dir, 'Run' + str(run))
       
        #Grab gray images and labels
        for fname in natsort.natsorted(os.listdir(os.path.join(run_dir,'GT'))):
            img = Image.open(os.path.join(run_dir,'GT', fname)).convert('RGB')
            img = np.array(img)
            
            #For rgb mode, take rgb image and multiply by binary image
            if mode == 'rgb':
                rgb_dir = os.path.join(run_dir,'Images', (fname.split('GT',1)[1][1:-3] + 'jpg'))
                rgb_img = Image.open(rgb_dir).convert('RGB')
                rgb_img = np.array(rgb_img)
                img = np.multiply(rgb_img,img)
                img = extract_features(img,preprocess=preprocess,mode=mode)
                del rgb_img
            elif mode == 'bw':
                img = img[:,:,0]
                img = extract_features(img,preprocess=preprocess,mode=mode)
            else:
                raise RuntimeError('Invalid mode, only bw and rgb supported')
                
            #Find index that corresponds to image to get labels
            temp_tube_number = int(fname.split('DAP',1)[1][:-4])
            temp_index = tube_number.index(temp_tube_number)
            
            #Downsample image
            if downsample:
                if ds_type == 'No_Pool':
                    img = img[::ds_factor,::ds_factor]
                elif ds_type == 'Avg_Pool':
                    if mode == 'rgb':
                        img = skimage.measure.block_reduce(img, (ds_factor,ds_factor,1), np.average)
                    else:
                        img = skimage.measure.block_reduce(img, (ds_factor,ds_factor), np.average)
                elif ds_type == 'Max_Pool':
                    if mode == 'rgb':
                        img = skimage.measure.block_reduce(img,(ds_factor,ds_factor,1), np.max)
                    else:
                        img = skimage.measure.block_reduce(img, (ds_factor,ds_factor), np.max)
                
            #Break up data into training and test
            if tube_rep[temp_index] in train_reps:
                training_data.append(img)
                training_labels_water_level.append(tube_water_level[temp_index])
                training_labels_cultivar.append(tube_cultivar[temp_index])
                training_names.append('Run_' + str(run) + ': ' +fname.split('GT',1)[1][1:-4])
            elif tube_rep[temp_index] in test_reps:
                testing_data.append(img)
                testing_labels_water_level.append(tube_water_level[temp_index])
                testing_labels_cultivar.append(tube_cultivar[temp_index])
                testing_names.append('Run_' + str(run) + ': ' +fname.split('GT',1)[1][1:-4])
            else:
                 raise RuntimeError('Rep not present in train or test data')   
                    
          
    #Return dictionary of dataset
    training_data = np.stack(training_data,axis=0)
    training_labels_water_level = np.stack(training_labels_water_level,axis=0)
    training_labels_cultivar = np.stack(training_labels_cultivar,axis=0)
    training_names = np.stack(training_names,axis=0)
    testing_data = np.stack(testing_data,axis=0)
    testing_labels_water_level = np.stack(testing_labels_water_level,axis=0)
    testing_labels_cultivar = np.stack(testing_labels_cultivar,axis=0)
    testing_names = np.stack(testing_names,axis=0)
    
    train_dataset = {'data': training_data, 
                     'water_level_labels': training_labels_water_level,
                     'cultivar_labels': training_labels_cultivar,
                     'train_names': training_names}
    test_dataset = {'data': testing_data, 
               'water_level_labels': testing_labels_water_level,
               'cultivar_labels': testing_labels_cultivar,
               'test_names': testing_names}
    dataset = {'train': train_dataset,'test': test_dataset}

    time_elapsed = time.time() - start
    
    print('Loaded dataset in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    #Print information about dataset:
    print('Total Number of Images: {}'.format(len(training_data)+len(testing_data)))
    print('Water Levels: {}'.format(np.unique(training_labels_water_level)*100))
    print('Cultivar: {}'.format(np.unique(training_labels_cultivar)))
    return dataset
        
                                       
                    
                    
                    
            
            