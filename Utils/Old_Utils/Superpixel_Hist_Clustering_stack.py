# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:47:10 2020
Relational clustering of images using EMD distance
@author: jpeeples
"""
import pdb
import numpy as np
from scipy.special import softmax,expit
from scipy.spatial import distance
import time
import os
from skimage.segmentation import slic
from skimage.measure import regionprops
import cv2
from skimage import color
from itertools import combinations
from Utils.Relational_Clustering import Generate_Relational_Clustering
from Utils.Compute_fractal_dim import fractal_dimension
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import umap

def compute_EMD(test_img,train_img):
    
    EMD_score, _, flow = cv2.EMD(train_img.astype(np.float32), 
                                           test_img.astype(np.float32), 
                                           cv2.DIST_L2)
    
       
    if EMD_score < 0: #Need distance to be positive, negative just indicates direction
        EMD_score = -EMD_score
        
    return EMD_score, flow
    
def Generate_SP_profile(X,numSP=200,lab=False,features='fractal',equal_weight=False,
                        spatial_wt=1,norm_vals='all_feats'):
    
    #Generate SP mask, add 1 to get properties (region properties considers 0 background
    # and ignores)
    
    SP_mask = slic(X,n_segments=numSP,compactness=10,sigma=0,convert2lab=lab,
                   slic_zero=True) + 1
    
    #Get spatial coordinates of centroid SP and number of non-zero superpixels
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
        
        #Compute weight for EMD
        weight = props.area/(X.shape[0]*X.shape[1])
        
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
                feat = [np.count_nonzero(temp_sp_img)/3, 
                                fractal_dimension(temp_sp_img,compute_lac=True)]  
            else:
               feat = fractal_dimension(temp_sp_img,compute_lac=True)
               frac_dim, lac = feat[0],feat[1]
               num_pixels = np.count_nonzero(temp_sp_img)
        elif features == 'frac_lac':
            feat = fractal_dimension(temp_sp_img,compute_lac=True)
            frac_dim, lac = feat[0],feat[1]
               
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
            #Weight value should be equal
            # SP_profile.append([num_root_pixels,cx,cy])
       
            if equal_weight:
                if features == 'all':
                    SP_profile.append([1,num_pixels,frac_dim,lac,cx,cy])
                elif features == 'frac_lac':
                    SP_profile.append([1,frac_dim,lac,cx,cy])
                else:
                    SP_profile.append([1,feat,cx,cy])
            else:
                if features == 'all':
                    SP_profile.append([weight,num_pixels,frac_dim,lac,cx,cy])
                else:
                    SP_profile.append([weight,feat,cx,cy])
                    
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

def Generate_Pairwise_EMD(data_profiles):
    #Compute pairwise EMD between images for 1) entire dataset or 2) test/train
    #Efficient way to compute pairwise EMD
    #Instead of N^2 computations, N(N-1)/2 computations (still O(N^2))
    num_imgs = len(data_profiles)
    distances = np.zeros((num_imgs,num_imgs))
    img_indices = np.arange(0,num_imgs).tolist()
    img_indices = combinations(img_indices, 2)
    
    for imgs in img_indices:
        temp_dist, _ = compute_EMD(data_profiles[imgs[0]]['SP_profile'],
                                data_profiles[imgs[1]]['SP_profile'])
        #EMD is symmetric, use same value for transposed row and column index
        distances[imgs[0],imgs[1]] = temp_dist
        distances[imgs[1],imgs[0]] = temp_dist
        
    return distances
    
def SP_clustering(dataset,numSP=250,mode='bw',num_imgs=1,
                  folder='Cluster_Imgs_SP/', embed='TSNE', split_data=False,
                  features='fractal',seed=42,embed_only=False,equal_weight=False,
                  alpha=1,normalize='all_feats'):

    start = time.time()
    
    #Change dataset(s) to SP profile
    train_data = []
    train_water_levels = dataset['train']['water_level_labels'].tolist()
    train_cultivar = dataset['train']['cultivar_labels'].tolist()
    train_names = dataset['train']['train_names'].tolist()
    

    # test_data = []
    # test_water_levels = dataset['test']['water_level_labels'].tolist()
    # test_cultivar = dataset['test']['cultivar_labels'].tolist()
    # test_names = dataset['test']['test_names'].tolist()
    
    print('Generating SP profiles of data')
    temp_start = time.time()
    for img in range(0,len(dataset['train']['data'])):
        train_data.append(Generate_SP_profile(dataset['train']['data'][img],
                          numSP=numSP,features=features,equal_weight=equal_weight,
                          spatial_wt=alpha,norm_vals=normalize))

    # for img in range(0,len(dataset['test']['data'])):
    #     test_data.append(Generate_SP_profile(dataset['test']['data'][img],
    #                      numSP=numSP,features=features,equal_weight=equal_weight,
    #                      spatial_wt=alpha,norm_vals=normalize))
     
    temp_stop = time.time() - temp_start
    print('Generated SP profiles in {:.0f}m {:.0f}s'.format(temp_stop // 60, 
                                                           temp_stop % 60))
    #Delete dataset to clear memory, may look at replacing data with SP profile 
    # del dataset
    
    import pdb; pdb.set_trace()
    #Get pairwise EMD matrix for relational clustering
    print('Computing pairwise distances')
    temp_start = time.time()

    EMD_matrix = Generate_Pairwise_EMD(train_data)
    temp_stop = time.time() - temp_start
    print('Computed distances in {:.0f}m {:.0f}s'.format(temp_stop // 60, 
                                                           temp_stop % 60))
    
    #Only return embedding if desired
    if embed_only:
        if embed == 'TSNE':
            embedding = TSNE(n_components=2,verbose=1,random_state=seed,
                              metric='precomputed').fit_transform(EMD_matrix)
        elif embed == 'UMAP':
            embedding = umap.UMAP(random_state=seed,
                                  metric='precomputed').fit_transform(EMD_matrix)
        
        elif embed == 'MDS':
            embedding = MDS(n_components=2,verbose=1,random_state=seed,
                            metric='precomputed').fit_transform(EMD_matrix)
            
        time_elapsed = time.time() - start
        
        print('Embedding finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return (embedding, train_data, train_water_levels,
                train_cultivar ,EMD_matrix,train_names)
    
      
    else:
        EMD_scores, Cluster_scores = Generate_Relational_Clustering(EMD_matrix,
                                       train_data,
                                       train_cultivar,
                                       train_water_levels,
                                       train_names,embed=embed,
                                       folder=folder,numSP=numSP,seed=seed)
    
    
        time_elapsed = time.time() - start
        
        print('Clustering finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
        
        # return outputs: EMD SI index and cluster indices
        return EMD_scores, Cluster_scores
    
