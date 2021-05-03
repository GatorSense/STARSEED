# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:08:19 2020
Load training and test images (Pytorch dataloader)
@author: jpeeples
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import pdb
import torch
import numpy as np
import pandas as pd
import natsort
import time
from sklearn import preprocessing
import pdb


#Need to clean up, make one function for DAP variations
class Root_data(Dataset):
    
    def __init__(self, csv_dir, img_dir, run_nums, train_reps = [1,2,3],
                 test_reps = [4], DAP='last',mode = 'bw',
                 data_type='train',label_type = 'water', img_transform=None):
        
        self.img_transform = img_transform
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
                        
            elif DAP == 'all':
                
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
        training_data = np.stack(training_data,axis=0)
        training_labels_water_level = np.stack(training_labels_water_level,axis=0)
        training_labels_cultivar = np.stack(training_labels_cultivar,axis=0)
        testing_data = np.stack(testing_data,axis=0)
        testing_labels_water_level = np.stack(testing_labels_water_level,axis=0)
        testing_labels_cultivar = np.stack(testing_labels_cultivar,axis=0)  
        
        train_dataset = {'data': training_data, 
                          'water_level_labels': training_labels_water_level,
                          'cultivar_labels': training_labels_cultivar}
        test_dataset = {'data': testing_data, 
                    'water_level_labels': testing_labels_water_level,
                    'cultivar_labels': testing_labels_cultivar}
        dataset = {'train': train_dataset,'test': test_dataset}
        
        #Get images
        self.imgs = dataset[data_type]['data']
          
        #Get labels
        if label_type == 'water':
            self.labels = dataset[data_type]['water_level_labels']
        elif label_type == 'cultivar':
            self.labels = dataset[data_type]['cultivar_labels']
        else:
            RuntimeError('Invalid label, only cultivar and water supported')           
        
        #Encode labels for loss function
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        time_elapsed = time.time() - start
        
        print('Loaded dataset in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img = self.imgs[index]
        label = torch.tensor(self.labels[index])

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label,index
            
    
    # Test code 
    # if __name__ == "__main__":
    #     #Load data
    #     csv_dir = 'T:/Fractal_Images/TubeNumberKey.csv'
    #     img_dir = 'T:/Fractal_Images/'
    #     run_nums = [4,5]
    #     mode = 'rgb' #rgb or bw
    #     DAP = 'last' #last or all
    #     train_reps = [1,2,3]
    #     test_reps = [4]
    #     features = None
        
    #     dataset = load_data(csv_dir,img_dir,run_nums,train_reps = train_reps,
    #                         test_reps = test_reps, DAP=DAP,mode=mode,features=None)
                                           
                    
                    
                    
            
            