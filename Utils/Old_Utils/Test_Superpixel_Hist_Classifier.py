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
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import regionprops
import cv2
import ot
from skimage import color
from math import sqrt

from Utils.Visualize_SP_EMD import Visualize_EMD
from Utils.Compute_fractal_dim import fractal_dimension

def compute_EMD(test_img,train_img):
    
    # pdb.set_trace()
    EMD_score, _, flow = cv2.EMD(train_img.astype(np.float32), 
                                           test_img.astype(np.float32), 
                                           cv2.DIST_L2)
    if EMD_score < 0: #Need distance to be positive, negative just indicates direction
        EMD_score = -EMD_score
           
    return EMD_score, flow

def generate_grid(img, numSP):
    
    h = img.shape[0]
    w = img.shape[1]
    SP_mask = np.zeros((h,w))
    ratio = h/w
    w_num = int(sqrt(numSP/ratio))
    h_num = int(ratio*w_num)
    w_int = int(w/w_num)
    h_int = int(h/h_num)
    label = 1
    for i in range(w_int + 1):
        w_start = int(i * w_int)
        if (i+1) * w_int > w:
            w_end = w
        else:
            w_end = int((i+1) * w_int)
            
        for j in range(h_num + 1):
            h_start = int(j * h_int)
            if (j+1) * h_int >h:
                h_end = h
            else:
                h_end = int((j+1) * h_int)
            
            SP_mask[h_start:h_end, w_start:w_end] = label
            label += 1
            
    return SP_mask.astype(int)

def Generate_SP_profile(X,numSP=200,lab=False, seg='slic',features='fractal'):
    
    #Generate SP mask, add 1 to get properties (region properties considers 0 background
    # and ignores)

    if seg == 'slic':
        SP_mask = slic(X,n_segments=numSP,compactness=10,sigma=0,convert2lab=lab,
                       slic_zero=True) + 1
        # print('Slic labels:%d' %(len(np.unique(SP_mask))))
    if seg == 'grid':
        SP_mask = generate_grid(X, numSP)
        # print('grid labels:%d' %(len(np.unique(SP_mask))))
    #Get spatial coordinates of centroid SP and number of non-zero superpixels
    #SP_profile = np.zeros((numSP,3))
    SP_profile = []

    #Repeat SP_mask if lab is True (RGB image)
    if lab:
        SP_mask = np.expand_dims(SP_mask,-1)
        SP_mask = np.repeat(SP_mask,3,axis=-1)
        regions = regionprops(SP_mask,intensity_image=X)
        Root_mask = SP_mask[:,:,0].copy()
    else:
        if X.shape[-1] == 1:
            X = X[:,:,0]
        regions = regionprops(SP_mask,intensity_image=X)
        Root_mask = SP_mask.copy()
        
    for props in regions:
        cx, cy = props.centroid[0:2]
        temp_sp_img = props.intensity_image
        
        if features == 'fractal':
            feat = fractal_dimension(temp_sp_img)
        elif features == 'lacunarity':
            feat = fractal_dimension(temp_sp_img,compute_lac=True)[-1]
        elif features == 'root_pixels':
            #Computing number of root pixels, could also extract features here and
            # aggregate or use features to generate SP segmentation
            if lab:
                # RGB counts 3
                feat = np.count_nonzero(temp_sp_img)/3 
            else:
                feat = np.count_nonzero(temp_sp_img)
        elif features == 'all':
            if lab:
                feat = np.array([np.count_nonzero(temp_sp_img)/3, 
                                 np.array(fractal_dimension(temp_sp_img))])  
            else:
                feat = np.array([np.count_nonzero(temp_sp_img), 
                                 np.array(fractal_dimension(temp_sp_img))])
        else: 
            assert \
                f'Feature not currently supported'
           
        #Assign values to SP_profile
        if lab:
            #Nx6, bin value, spatial, average RGB
            # R,G,B = np.mean(temp_sp_img,axis=(0,1))
            #LAB
            L,A,B = np.mean(color.rgb2lab(temp_sp_img),axis=(0,1))
            SP_profile.append([feat,cx,cy,L,A,B])
        else:
            # SP_profile.append([num_root_pixels,cx,cy])
            SP_profile.append([1,feat,cx,cy])
        # Root_mask[Root_mask==props.label] = SP_profile[-1][0] #Save 
        #Save out mean of cluster centers
        Root_mask[Root_mask==props.label] = np.mean(SP_profile[-1][1:]) #Save 
    #Generate superpixel profile as array: should be 1 x D + 1 where 
    # D is the number of features (e.g., spatial, color,texture) and the first
    # value should be the weight for the bin
    SP_profile = np.stack(SP_profile,axis=0)
 
    return {'SP_profile': SP_profile, 'SP_mask': SP_mask, 'Root_mask': Root_mask,
            'Img': X}

def test_SP_classifier(model,test_dataset,numSP=250,
                       mode='bw',num_imgs=1,folder='Test_Imgs_SP/',
                       downsample=True, seg='slic',features='fractal'):

    print('Testing Model...')   
    start = time.time()
    
    #Load training data and test data
    train_water_imgs = model['X_train_water']
    train_cultivar_imgs = model['X_train_cultivar']
    train_cultivar = model['cultivar']
    train_water_levels = model['water_levels']
   
    test_imgs = test_dataset['data']
    test_water_levels = test_dataset['water_level_labels']
    test_cultivar = test_dataset['cultivar_labels']
    test_names = test_dataset['test_names']
    
    #Delete model and test dataset to clear memory 
    del model,test_dataset
    
    #Downsample training and testing images
    #May use resize instead of throwing away pixels
    if downsample:
        ds = 16
        train_water_imgs = train_water_imgs[:,::ds,::ds]
        train_cultivar_imgs= train_cultivar_imgs[:,::ds,::ds]
        test_imgs = test_imgs[:,::ds,::ds]

   #Initialize array of scores
    X_water_corr = []
    X_cultivar_corr = []
    
    #Get indices of training data
    num_class = len(train_water_levels)
    
    train_water_SP = []
    train_cultivar_SP = []
    
    #For each training image, get SP profile and mask
    if mode == 'bw':
        lab = False
    else:
        lab = True
    for current_class in range(0,num_class):
        
        train_water_SP.append(Generate_SP_profile(train_water_imgs[current_class],
                                                  numSP=numSP,lab=lab,seg=seg,
                                                  features=features))
        train_cultivar_SP.append(Generate_SP_profile(train_cultivar_imgs[current_class],
                                                     numSP=numSP,lab=lab,seg=seg,
                                                     features=features))

    #Test one image at a time (memory issue with all at once) 
    num_test_imgs = len(test_imgs)
    visual_img = 0
    
    for img in range(0,num_test_imgs):
        
        temp_test_img = test_imgs[img]
        
        #Get SP profile and masks
        temp_test_img = Generate_SP_profile(temp_test_img,numSP=numSP,lab=lab,seg=seg,
                                            features=features)
        
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
            
            # if img == 1: 
            #     pdb.set_trace()
            
            #Compute EMD for training images in class 
            temp_water_level_dist, flow_water = compute_EMD(temp_test_img['SP_profile'],
                                                train_water_SP[current_class]['SP_profile'])
            temp_cultivar_dist, flow_cultivar = compute_EMD(temp_test_img['SP_profile'],
                                             train_cultivar_SP[current_class]['SP_profile']) 
            
            #Save flow for visual
            flow_water_vis.append(flow_water)
            flow_cultivar_vis.append(flow_cultivar)
            
            #Compute distance for water levels
            #Compute euclidean distance between spatial dimension and return distance of K nearest neighbors
            X_water_dist = temp_water_level_dist
        
            #Compute for cultivar
            X_cultivar_dist = temp_cultivar_dist
        
            #Save out temporary scores for each class
            temp_X_water_corr[current_class] = X_water_dist
            temp_X_cultivar_corr[current_class] = X_cultivar_dist
            
            #Save distances for plotting
            test_water_dist.append(X_water_dist)
            test_cultivar_dist.append(X_cultivar_dist)
        
        #Create visual of select number of test images
        if visual_img < num_imgs:
            #For water levels
            Visualize_EMD(temp_test_img,train_water_SP, test_water_dist,
                          flow_water_vis,folder,test_water_levels[img],test_names[img],
                          num_class=num_class,title='Water_Levels',
                          class_names=train_water_levels,lab=lab,sp_overlay=True) 
            #For water levels
            Visualize_EMD(temp_test_img,train_water_SP, test_water_dist,
                          flow_water_vis,folder,test_water_levels[img],test_names[img],
                          num_class=num_class,title='Water_Levels',
                          class_names=train_water_levels,lab=lab,sp_overlay=False) 
            
            #For cultivar
            Visualize_EMD(temp_test_img,train_cultivar_SP, test_cultivar_dist,
                          flow_cultivar_vis,folder, test_cultivar[img],test_names[img],
                          num_class=num_class,title='Cultivar',class_names=train_cultivar,
                          lab=lab,sp_overlay=True)         
            Visualize_EMD(temp_test_img,train_cultivar_SP, test_cultivar_dist,
                          flow_cultivar_vis,folder, test_cultivar[img],test_names[img],
                          num_class=num_class,title='Cultivar',class_names=train_cultivar,
                          lab=lab,sp_overlay=False)         
            
            visual_img += 1
        #Append scores for each test img
        X_water_corr.append(temp_X_water_corr)
        X_cultivar_corr.append(temp_X_cultivar_corr)
    
    #Change scores to arrays
    # pdb.set_trace()
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
    
    
