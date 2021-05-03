# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:17:46 2021
Parameters for clustering root architecture images
@author: jpeeples
"""

def Parameters():
    
    #Directory location of images and tube number keys
    csv_dir = './Data/TubeNumberKey.csv'
    img_dir = './Data/'
    
    #Directory desired for results
    results_folder = 'Updated_Results/Percent_Flow_Root_Lac_New/'
    
    #Run numbers for root imagery
    #Runs 4&5 are roots without fertilizer and Runs 6&7 are with fertilizer
    run_nums = [6,7] 
    
    #Pick repetitions to train and test on (total of four runs, 1 to 4)
    train_reps = [1,2,3]
    test_reps = [4]
    
    #Set number of superpixels
    #SP_min: smallest number of super pixels to consider
    #SP_step: incremement between SP_max and SP_min
    #SP_max: largest number of super pixels to consider
    SP_min = 2000
    SP_step = 100
    SP_max = 2000
    
    #Set data type, can run code on 'rgb' or 'bw' ('bw' is default, binary data)
    mode = 'bw' #rgb or bw
    
    #Set days after planting (DAP) to use, either 'last' for last DAP or 'all'
    #for all images after 
    DAP = 'last'
    
    #Preprocessing (smoothing) of image before downsampling
    #Can either use 'gauss' (gaussian blur, default), 'avg' (average filter), or
    # None for no smoothing
    preprocess = 'gauss' 
    
    #Either use 'root_pixels','fractal', 'lacunarity', all' (use all three texture features
    #at once)
    features = ['root_pixels']
    
    #Set labels of interest, currently have for labels:
    # 1) Cultivar, 2) Water Levels, 3) Cross Treatments (Cultivar and Water Levels),
    # 4) Cluster (from Affinity Propagation algorithm)
    labels = ['Cultivar', 'Water Levels', 'Cross Treatments', 'Cluster']
    
    #Normalization options for features (results obtained without normalization): 
    # 1) 'no_norm': no normalziation of features (default)
    # 2) 'texture_feats': normalize texture feature(s) to be between 0 and 1 
    #    (min max normalization)
    # 3) 'all_feats': normalize texture feature
    # 4) 'standardization': standardize (center and unit scale) all features 
    norm_method = 'no_norm'
    
    #Downsampling options for images:
    # 1) 'No_Pool': decimate  
    # ds_methods = ['No_Pool','Avg_Pool','Max_Pool']
    ds_methods = ['Avg_Pool']
    downsample = True #(set to false for 'avg')
    ds_factor = 8
    
    #Adjust clustering measures for chance (set to metrics based on pairwise confusion matrix)
    adjusted=True
    
    #Set to False to use feature value in SP as weights for EMD
    equal_weight = False 
    
    #Set False to use median as preference or True to set to percentage of root SPs
    set_preferences = False 
    
    #Scale on spatial coordinates
    alpha = 1 
    
    #Number of test images to visualize (26 in total)
    num_imgs = 5 
    
    #Seed for Random state for TSNE, Clustering and Metrics (should only affect
    # root visualization and not clustering performance)
    #Embedding method to be used, select either 1) 'TSNE', 2) 'UMAP', 3) 'ISOMAP', 
    # 4) 'LLE' or 5) 'MDS'
    # Note: If 'UMAP' is used, can run supervised version of 'UMAP' for different labels
    # Current options are 1) 'Cluster', 2) 'Cultivar', 3) 'Water', 4) 'Cross_Treatments
    # 5) 'Unsupervised'
    # All other methods are unsupervised (set label_types to 'Unsupervised' 
    # only for these methods)
    seed = 42
    embed = 'MDS' 
    label_types = ['Unsupervised']
    
    #Show raw root images or feature values for embedding results
    vis_fig_type = 'Image' 
    
    # Metric to access embedding of EMD matrix
    #'Silhouette', 'Calinski-Harabasz', 
    # 'Scatter' (compute intra- and inter- scores using EMD matrix)
    score_metric = 'Calinski-Harabasz'
    
    #Number of neighbors for UMAP
    num_neighbors = 15 
    
    #Cluster training and test images separately (if true, will "predict" test labels)
    #For Batch_Cluster_All_Images, set split_data to 'False'
    split_data = False 
    
    #Only compute EMD where root is present (default: True), set to False to 
    #include background
    root_only = True 
    
    
    if root_only:
        results_location = results_folder + '{}/Root_Only/'.format(embed)
    else:
        results_location = results_folder + '{}/Root_Background/'.format(embed)
       

    #Return dictionary of parameters
    return {'csv_dir': csv_dir, 'img_dir': img_dir, 'results_folder': results_folder,
            'run_nums': run_nums, 'train_reps': train_reps, 'test_reps': test_reps,
            'SP_min': SP_min, 'SP_step': SP_step, 'SP_max': SP_max, 'mode': mode,
            'DAP': DAP, 'preprocess': preprocess, 'features': features, 
            'labels': labels, 'norm_method': norm_method, 'ds_methods': ds_methods,
            'downsample': downsample, 'ds_factor': ds_factor, 'adjusted': adjusted,
            'equal_weight': equal_weight, 'set_preferences': set_preferences,
            'alpha': alpha, 'num_imgs': num_imgs, 'seed': seed, 'embed': embed,
            'label_types': label_types, 'vis_fig_type': vis_fig_type, 
            'score_metric': score_metric, 'num_neighbors': num_neighbors, 
            'split_data': split_data, 'root_only': root_only, 
            'results_location': results_location}
        
        
        
    