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
import time
import skimage.measure
from scipy.ndimage import gaussian_filter
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import pdb

def extract_features(X,features = 'avg',mode='rgb'):

    #Perform average pooling (sride and kernel are the same size)
    if features == 'avg':
        if mode == 'rgb':
            X = skimage.measure.block_reduce(X, (7,7,1), np.average)
        else:
            X = skimage.measure.block_reduce(X, (7,7), np.average)
    elif features == 'gauss':
        X = gaussian_filter(X,sigma=1)
    else:
        pass
    return X

def Generate_SP_profile(X,img_dir,numSP=200,lab=False,save_imgs=True):
    
    #Generate SP mask, add 1 to get properties (region properties considers 0 background
    # and ignores)
    SP_mask = slic(X,n_segments=numSP,compactness=10,sigma=1,convert2lab=lab) + 1
    
    #Get spatial coordinates of centroid SP and number of non-zero superpixels
    #SP_profile = np.zeros((numSP,3))
    SP_profile = []
    
    #Repeat SP_mask if lab is True (RGB image)
    if lab:
        SP_mask = np.expand_dims(SP_mask,-1)
        SP_mask = np.repeat(SP_mask,3,axis=-1)
        regions = regionprops(SP_mask,intensity_image=X)
        Root_mask = SP_mask[:,:,0].copy()
        # #Get total number of root pixels for normalization
        # total_root_pixels = np.count_nonzero(X[:,:,0])
    else:
        regions = regionprops(SP_mask,intensity_image=X)
        Root_mask = SP_mask.copy()
        # total_root_pixels = np.count_nonzero(X)

    for props in regions:
        cx, cy = props.centroid[0:2]
        temp_sp_img = props.intensity_image
        num_root_pixels = np.count_nonzero(temp_sp_img) # RGB counts 3
        
        #Assign values to SP_profile
        SP_profile.append([cx,cy,num_root_pixels])
        Root_mask[Root_mask==props.label] = num_root_pixels
    
    #Generate superpixel image and number of non-zero root pixels
    SP_profile = np.stack(SP_profile,axis=0)
    fig, ax = plt.subplots(nrows=1,ncols=2)
    plt.subplots_adjust(wspace=.6)
    
    #Superpixel segmentation
    if lab:
        SP_mask = SP_mask[:,:,0]
    ax[0].imshow(mark_boundaries(X, SP_mask,color=(1,1,0)), aspect="auto")
    ax[0].set_title("SLIC segmentation")
    
    #Show number of non-zero pixels in each superpixel
    mask_values = ax[1].imshow(Root_mask, aspect="auto")
    ax[1].set_title('Number of Root Pixels')
    fig.colorbar(mask_values,ax=ax[1])
    
    for a in ax.ravel():
        a.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig(img_dir,dpi=1000,bbox_inches='tight')
    
    plt.close()
    return SP_profile

#Need to clean up, make one function for DAP variations
def load_data_SP(csv_dir,img_dir,run_nums,numSP=200,train_reps = [1,2,3],test_reps = [4], 
              DAP='last',mode='bw',features = 'avg',folder='SP_Imgs/'):
    
    #Create folder and save figures
    if not os.path.exists(folder):
        os.makedirs(folder)
    
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
    
    testing_data = []
    testing_labels_water_level = []
    testing_labels_cultivar = []
    
    for run in run_nums:
        run_dir = img_dir + 'Crop_Run' + str(run) + '/'
       
        #Select data based on all DAP or just the last day
        root, sub_dirs, _ = next(os.walk(run_dir))
        sub_dirs = natsort.natsorted(sub_dirs)
        if DAP == 'last':
            temp_dir = run_dir + sub_dirs[-1] 
            
            SP_img_folder = folder + 'Crop_Run' + str(run) + '/' + sub_dirs[-1] + '/'
            if not os.path.exists(SP_img_folder):
                os.makedirs(SP_img_folder)
            
            #Grab gray images and labels
            for fname in natsort.natsorted(os.listdir(temp_dir + '/GT/')):
                img = Image.open(temp_dir + '/GT/' + fname).convert('RGB')
                img = np.array(img)
                save_img_fname = SP_img_folder + fname
                idx = save_img_fname.index('.')
                save_img_fname = save_img_fname[:idx] + '_' + mode + save_img_fname[idx:]
                
                #For rgb mode, take rgb image and multiply by binary image
                if mode == 'rgb':
                    rgb_dir = temp_dir + '/Images/' + fname.split('GT',1)[1][1:-3] + 'jpg'
                    rgb_img = Image.open(rgb_dir).convert('RGB')
                    rgb_img = np.array(rgb_img)
                    img = np.multiply(rgb_img,img)
                    #img = extract_features(img,features=features,mode=mode)
                    img = Generate_SP_profile(img,save_img_fname,numSP=numSP,lab=True,
                                              save_imgs=True)
                    del rgb_img
                elif mode == 'bw':
                    img = img[:,:,0]
                    #img = extract_features(img,features=features,mode=mode)
                    img = Generate_SP_profile(img,save_img_fname,numSP=numSP,lab=False,
                                              save_imgs=True)
                else:
                    raise RuntimeError('Invalid mode, only bw and rgb supported')
                    
                #Find index that corresponds to image to get labels
                temp_tube_number = int(fname.split('DAP',1)[1][:-4])
                temp_index = tube_number.index(temp_tube_number)
                
                #Break up data into training and test
                if tube_rep[temp_index] in train_reps:
                    training_data.append(img)
                    training_labels_water_level.append(tube_water_level[temp_index])
                    training_labels_cultivar.append(tube_cultivar[temp_index])
                elif tube_rep[temp_index] in test_reps:
                    testing_data.append(img)
                    testing_labels_water_level.append(tube_water_level[temp_index])
                    testing_labels_cultivar.append(tube_cultivar[temp_index])
                else:
                     raise RuntimeError('Rep not present in train or test data')   
                    
        elif DAP == 'all': #Need to update like last day
            
            for temp_sub_dirs in sub_dirs:
                temp_dir = run_dir + temp_sub_dirs
                
                #Grab gray images and labels
                for fname in natsort.natsorted(os.listdir(temp_dir + '/GT/')):
                    img = Image.open(temp_dir + '/GT/' + fname).convert('RGB')
                    img = np.array(img)
                    
                    #For rgb mode, take rgb image and multiply by binary image
                    if mode == 'rgb':
                        rgb_dir = temp_dir + '/Images/' + fname.split('GT',1)[1][1:-3] + 'jpg'
                        rgb_img = Image.open(rgb_dir).convert('RGB')
                        rgb_img = np.array(rgb_img)
                        img = np.multiply(rgb_img,img)
                        del rgb_img
                    elif mode == 'bw':
                        img = img[:,:,0]
                    else:
                        raise RuntimeError('Invalid mode, only bw and rgb supported')
                        
                    #Find index that corresponds to image to get labels
                    temp_tube_number = int(fname.split('DAP',1)[1][:-4])
                    temp_index = tube_number.index(temp_tube_number)
                    
                    #Break up data into training and test
                    if tube_rep[temp_index] in train_reps:
                        training_data.append(img)
                        training_labels_water_level.append(tube_water_level[temp_index])
                        training_labels_cultivar.append(tube_cultivar[temp_index])
                    elif tube_rep[temp_index] in test_reps:
                        testing_data.append(img)
                        testing_labels_water_level.append(tube_water_level[temp_index])
                        testing_labels_cultivar.append(tube_cultivar[temp_index])
                    else:
                         raise RuntimeError('Rep not present in train or test data') 
        else:
            raise RuntimeError('Invalid DAP,only all or last supported')
          
    #Return dictionary of dataset
    #For RGB, some images do not have the same number of SP (cannot stack)
    if mode == 'bw':
        training_data = np.stack(training_data,axis=0)
        testing_data = np.stack(testing_data,axis=0)
    else:
        training_data = np.array(training_data)
        testing_data = np.array(testing_data)
    training_labels_water_level = np.stack(training_labels_water_level,axis=0)
    training_labels_cultivar = np.stack(training_labels_cultivar,axis=0)
    testing_labels_water_level = np.stack(testing_labels_water_level,axis=0)
    testing_labels_cultivar = np.stack(testing_labels_cultivar,axis=0)  
    
    train_dataset = {'data': training_data, 
                     'water_level_labels': training_labels_water_level,
                     'cultivar_labels': training_labels_cultivar}
    test_dataset = {'data': testing_data, 
               'water_level_labels': testing_labels_water_level,
               'cultivar_labels': testing_labels_cultivar}
    dataset = {'train': train_dataset,'test': test_dataset}
    
    time_elapsed = time.time() - start
    
    print('Loaded dataset in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return dataset
                                       
                    
                    
                    
            
            