# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:47:10 2020
Relational clustering of images using EMD distance
@author: jpeeples
"""
import pdb
import numpy as np
import time
from skimage.measure import regionprops
from skimage import color
from itertools import combinations
from Utils.EMD_Clustering import Generate_EMD_Clustering
from Utils.Compute_fractal_dim import fractal_dimension
from Utils.Compute_EMD import compute_EMD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torchvision import models


    
def generate_grid(img, numSP):
    h = img.shape[0]
    w = img.shape[1]
    SP_mask = np.zeros((h,w))
    ratio = h/w
    w_num = np.ceil(np.sqrt(numSP/ratio)).astype(int)
    h_num = np.ceil(ratio*np.sqrt(numSP/ratio)).astype(int)
    w_int = np.ceil(w/w_num)
    h_int = np.ceil(h/h_num)
    label = 1
    
    for j in range(h_num+1):
        h_start = int(j * h_int)
        if (j+1) * h_int > h:
            h_end = h
            
        else:
            h_end = int((j+1) * h_int)
        
        for i in range(w_num+1):
            w_start = int(i * w_int)
            if (i+1) * w_int > w:
                w_end = w
            else:
                w_end = int((i+1) * w_int)
            
            SP_mask[h_start:h_end, w_start:w_end] = label
            label += 1      
    return SP_mask.astype(int)
      
def Generate_SP_profile(X,numSP=200,lab=False,features='fractal',equal_weight=False,
                        spatial_wt=1,norm_vals='all_feats',root_only=True,
                        backbone=None):
    
    #Generate SP mask, add 1 to get properties (region properties considers 0 background
    # and ignores)
    SP_mask = generate_grid(X,numSP)

    #Get spatial coordinates of centroid SP and number of non-zero superpixels
    SP_profile = []
    
    #Repeat SP_mask if lab is True (RGB image)
    if lab:
        SP_mask = np.expand_dims(SP_mask,-1)
        SP_mask = np.repeat(SP_mask,3,axis=-1)
        regions = regionprops(SP_mask,intensity_image=X)
        Root_mask = SP_mask[:,:,0].copy().astype(np.float64)
    else:
        if X.shape[-1] == 1:
            X = X[:,:,0]
        regions = regionprops(SP_mask,intensity_image=X)
        Root_mask = SP_mask.copy().astype(np.float64)

    #Get minimimum size
    for props in regions:
        min_size = props.intensity_image.shape
        break
    
    pref_count = 0
    for props in regions:
        cx, cy = props.centroid[0:2]
        temp_sp_img = props.intensity_image
        
        #Increment count of locations without root
        if np.count_nonzero(temp_sp_img) == 0:
            pref_count += 1
        
        if features == 'fractal':
            feat = fractal_dimension(temp_sp_img,min_dim=min_size,root_only=root_only)
        
        elif features == 'lacunarity':
            feat = fractal_dimension(temp_sp_img,min_dim=min_size,compute_lac=True,
                                     root_only=root_only)[-1]
        elif features == 'root_pixels':
            #Computing number of root pixels, could also extract features here and
            # aggregate or use features to generate SP segmentation
            if lab:
                # RGB counts 3
                feat = np.count_nonzero(temp_sp_img)/(3*props.area) 
            else:
                # feat = np.count_nonzero(temp_sp_img)
                feat = np.count_nonzero(temp_sp_img)/props.area
        elif features == 'all':
            if lab:
                feat = [np.count_nonzero(temp_sp_img)/3, 
                                fractal_dimension(temp_sp_img,compute_lac=True)]  
            else:
               feat = fractal_dimension(temp_sp_img,compute_lac=True)
               frac_dim, lac = feat[0],feat[1]
               num_pixels = np.count_nonzero(temp_sp_img)
               
        else: 
            assert \
                f'Feature not currently supported'
                
        #Assign values to SP_profile
        if lab:
            #Nx6, bin value, spatial, average RGB
            L,A,B = np.mean(color.rgb2lab(temp_sp_img),axis=(0,1))
            SP_profile.append([feat,cx,cy,L,A,B])
        else:
            if equal_weight:
                if features == 'all':
                    SP_profile.append([1,num_pixels,frac_dim,lac,cx,cy])
                else:
                    SP_profile.append([1,feat,cx,cy])
            else:
                SP_profile.append(np.concatenate((feat,cx,cy),axis=None))
        
           
        Root_mask[Root_mask==props.label] = SP_profile[-1][0] #Save 
    
    #Generate superpixel profile as array: should be 1 x D + 1 where 
    # D is the number of features (e.g., spatial, color,texture) and the first
    # value should be the weight for the bin
    SP_profile = np.stack(SP_profile,axis=0)
 
   #Normalize values if desired
    if norm_vals == 'all_feats':
        # #Normalize count to be between 0 and 1 for feature values (don't consider weights)
        scaler = MinMaxScaler(feature_range=(0, 1))
        SP_profile[:,1:] = scaler.fit_transform(SP_profile[:,1:])
    
    elif norm_vals == 'texture_feats':
        # #Normalize count to be between 0 and 1 for feature values 
        # (don't consider weights and spatial coordinates)
        scaler = MinMaxScaler(feature_range=(0, 1))
        SP_profile[:,1::-2] = scaler.fit_transform(SP_profile[:,1::-2])
    
    elif norm_vals == 'standardization':
        #Standardization
        scaler = StandardScaler()
        SP_profile[:,1:] = scaler.fit_transform(SP_profile[:,1:])
    
    else: #No normalization
        pass
    
    #Scale spatial weight based on importance
    SP_profile[:,-2::] = spatial_wt*SP_profile[:,-2::]
    
    return {'SP_profile': SP_profile, 'SP_mask': SP_mask, 'Root_mask': Root_mask,
            'Img': X}

def Generate_Pairwise_EMD(data_profiles,root_only=True):
    #Compute pairwise EMD between images for 1) entire dataset or 2) test/train
    #Efficient way to compute pairwise EMD
    #Instead of N^2 computations, N(N-1)/2 computations (still O(N^2))
    num_imgs = len(data_profiles)
    distances = np.zeros((num_imgs,num_imgs))
    img_indices = np.arange(0,num_imgs).tolist()
    img_indices = combinations(img_indices, 2)
    
    for imgs in img_indices:
        temp_dist, _ = compute_EMD(data_profiles[imgs[0]]['SP_profile'],
                                data_profiles[imgs[1]]['SP_profile'],root_only=root_only)
        #EMD is symmetric, use same value for transposed row and column index
        distances[imgs[0],imgs[1]] = temp_dist
        distances[imgs[1],imgs[0]] = temp_dist
     

    return distances
    
def SP_clustering(dataset,numSP=250,mode='bw',num_imgs=1,
                  folder='Cluster_Imgs_SP/', embed='TSNE',split_data=False,
                  features='fractal',seed=42,embed_only=False,equal_weight=False,
                  alpha=1,normalize='all_feats',root_only=True,num_neighbors=15,
                  label_type='Unsupervised', vis_fig_type='Image',
                  score_metric='Scatter'):

    start = time.time()
    
    #Change dataset(s) to SP profile
    train_data = []
    train_water_levels = dataset['train']['water_level_labels'].tolist()
    train_cultivar = dataset['train']['cultivar_labels'].tolist()
    train_names = dataset['train']['train_names'].tolist()
    test_data = []
    test_water_levels = dataset['test']['water_level_labels'].tolist()
    test_cultivar = dataset['test']['cultivar_labels'].tolist()
    test_names = dataset['test']['test_names'].tolist()
    
    print('Generating SP profiles of data')
    temp_start = time.time()
    train_idx = np.arange(0,len(dataset['train']['data']))
    test_idx = np.arange(train_idx[-1]+1,len(dataset['test']['data'])+len(
                                                     dataset['train']['data']))

    #Compute superpixel signature for each image in training and test sets
    for img in range(0,len(dataset['train']['data'])+len(dataset['test']['data'])):
        
        if img in train_idx:
            train_data.append(Generate_SP_profile(dataset['train']['data'][img],
                              numSP=numSP,features=features,equal_weight=equal_weight,
                              spatial_wt=alpha,norm_vals=normalize,root_only=root_only))
        else:
            test_data.append(Generate_SP_profile(dataset['test']['data'][img-train_idx[-1]-1],
                             numSP=numSP,features=features,equal_weight=equal_weight,
                             spatial_wt=alpha,norm_vals=normalize,root_only=root_only))
    
    temp_stop = time.time() - temp_start
    print('Generated SP profiles in {:.0f}m {:.0f}s'.format(temp_stop // 60, 
                                                           temp_stop % 60))
    
    #Delete dataset to clear memory, may look at replacing data with SP profile
    del dataset
    
    #Get pairwise EMD matrix for relational clustering
    print('Computing pairwise distances')
    temp_start = time.time()
    EMD_mat = Generate_Pairwise_EMD(train_data+test_data,root_only=root_only)
    
    temp_stop = time.time() - temp_start
    print('Computed distances in {:.0f}m {:.0f}s'.format(temp_stop // 60, 
                                                           temp_stop % 60))
    
   
    if embed_only:
        embedding = Generate_EMD_Clustering(EMD_mat,
                                       train_data+test_data,
                                       train_cultivar+test_cultivar,
                                       train_water_levels+test_water_levels,
                                       train_names+test_names,embed=embed,
                                       folder=folder,numSP=numSP,seed=seed,
                                       split_data=split_data,train_idx=train_idx,
                                       test_idx=test_idx,
                                       embed_only=embed_only,
                                       root_only=root_only,
                                       num_neighbors=num_neighbors,
                                       label_type=label_type,
                                       vis_fig_type=vis_fig_type,features=features) 
            
        time_elapsed = time.time() - start
        
        print('Embedding finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return (embedding, train_data+test_data, train_water_levels+test_water_levels,
                train_cultivar+test_cultivar,EMD_mat,train_names+test_names,
                train_idx,test_idx)
    
    else:
        #Perform relational clustering
        EMD_scores, labels = Generate_EMD_Clustering(EMD_mat,
                                       train_data+test_data,
                                       train_cultivar+test_cultivar,
                                       train_water_levels+test_water_levels,
                                       train_names+test_names,embed=embed,
                                       folder=folder,numSP=numSP,seed=seed,
                                       split_data=split_data,train_idx=train_idx,
                                       test_idx=test_idx,
                                       num_neighbors=num_neighbors,
                                       label_type=label_type,
                                       vis_fig_type=vis_fig_type,
                                       score_metric=score_metric,
                                       features=features)        

        time_elapsed = time.time() - start
        
        print('EMD Scoring and Visualization finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
        
        # return outputs: EMD score index and labels
        return EMD_scores, labels
    
