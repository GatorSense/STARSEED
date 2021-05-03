# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:08:19 2020
Load training and test images
@author: jpeeples
"""
import pdb
import pandas as pd
import os
import natsort
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import skimage.measure
from scipy.ndimage import gaussian_filter
import pdb
import glob

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
def load_data(csv_dir,img_dir,run_nums, stack_mode='bw', DAP='last',mode='bw',
              preprocess = 'avg',downsample=False,ds_factor=16):

    print('Loading dataset...')
    start = time.time()
    #Load tube number key csv file
    tube_key = pd.read_csv(csv_dir)

    #Extract data from tube key
    tube_number = tube_key['Tube Number'].tolist()
    tube_rep = tube_key['Rep']
    tube_water_level = tube_key['Water Level']
    tube_cultivar = tube_key['Cultivar']
    exps = tube_key[tube_key['Rep']==1]['Tube Number'].tolist() #exp number
    reps = tube_key['Rep'].unique()
    #For each run, generate training and testing data
    training_data = []
    training_labels_water_level = []
    training_labels_cultivar = []
    training_names = []
    
    # testing_data = []
    # testing_labels_water_level = []
    # testing_names = []
    # testing_labels_cultivar = []
    
    # save all the image path to data list
    all_GT_path = []
    for run in run_nums:
        run_dir = os.path.join(img_dir, ('Crop_Run' + str(run)))
        #Select data based on all DAP or just the last day
        root, sub_dirs, _ = next(os.walk(run_dir))
        sub_dirs = natsort.natsorted(sub_dirs)       
        
        if DAP == 'last':
            temp_dir = os.path.join(run_dir, sub_dirs[-1])
            all_GT_path += glob.glob(os.path.join(temp_dir, 'GT','*.png'))
        
        elif DAP == 'all':
            for temp_sub_dirs in sub_dirs:
                temp_dir = run_dir + temp_sub_dirs
                all_GT_path += glob.glob(os.path.join(temp_dir, 'GT','*.png'))
        else:
            raise RuntimeError('Invalid DAP,only all or last supported')

    # get all reps of the same exp from all the data
    for exp in exps:
        sub_exp = str(exp)[-2:]
        sub_GT_path = []
        for path in all_GT_path:
            if sub_exp in path.split('/')[-1].split('DAP')[-1]:
                sub_GT_path.append(path)

        # select images of different reps according to exp
        for i, path in enumerate(sub_GT_path):
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            
            if mode == 'bw':
                img = img[:,:,0]
            
            if mode == 'rgb':
                rgb_path = path.split('GT/')[0]+'Images/'+path.split('/')[-1][3:].replace('.png','.jpg')
                rgb_img = Image.open(rgb_path).convert('RGB')
                rgb_img = np.array(rgb_img)
                img = np.multiply(rgb_img, img)
                
            if i == 0:
                stack_img = np.expand_dims(img, axis=0)
            else:
                stack_img = np.concatenate((stack_img, np.expand_dims(img, axis=0)), axis=0)
        
        # stack images using average mode or keep binary format
        mean_stack = stack_img.mean(0)
        
        #Downsample image
        if downsample:
            mean_stack = mean_stack[::ds_factor,::ds_factor]
        
        if stack_mode == 'avg':
            training_data.append(mean_stack)
            
        elif stack_mode == 'bw':
            bw_stack = np.zeros(mean_stack.shape)
            nonzero_idx = np.where(mean_stack != 0)
            bw_stack[nonzero_idx] = 1
            training_data.append(bw_stack)
        

        # save labels for water level, cultivar and name
        temp_idx = tube_number.index(exp)
        training_names.append(sub_exp)
        training_labels_water_level.append(tube_key['Water Level'].iloc[temp_idx])
        training_labels_cultivar.append(tube_key['Cultivar'].iloc[temp_idx])
        
    training_data = np.stack(training_data,axis=0)
    training_labels_water_level = np.stack(training_labels_water_level,axis=0)
    training_labels_cultivar = np.stack(training_labels_cultivar,axis=0)
    training_names = np.stack(training_names,axis=0)
    
    
    train_dataset = {'data': training_data, 
                  'water_level_labels': training_labels_water_level,
                  'cultivar_labels': training_labels_cultivar,
                  'train_names': training_names}


    dataset = {'train': train_dataset, 'test': None}

    time_elapsed = time.time() - start
    
    print('Loaded dataset in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    return dataset
        

# Test code 
if __name__ == "__main__":
    #Load data
    csv_dir = '/mnt/WD500/Data/TubeNumberKey.csv'
    img_dir = '/mnt/WD500/Data/'
    run_nums = [6,7]
    mode = 'bw' #rgb or bw
    DAP = 'last' #last or all
    train_reps = [1,2,3]
    test_reps = [4]
    preprocess = None
    
    dataset = load_data(csv_dir,img_dir,run_nums, 
                        DAP=DAP,mode=mode,preprocess=None)
                                       
                    
                    
                    
            
            