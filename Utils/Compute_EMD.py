# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:02:36 2021
Compute EMD between two images' superpixel signatures
@author: jpeeples
"""
import cv2
import numpy as np

def compute_EMD(test_img,train_img,root_only=True):
    
    if root_only:
        root_flow = np.zeros((len(train_img),len(test_img)))
        train_idx = np.nonzero(train_img[:,0])[0]
        test_idx = np.nonzero(test_img[:,0])[0]
        train_img = train_img[train_idx,:]
        test_img = test_img[test_idx,:]
        try:
            EMD_score, _, flow = cv2.EMD(train_img.astype(np.float32), 
                                                   test_img.astype(np.float32), 
                                                   cv2.DIST_L2)
        except: #No non-zero root pixels
            EMD_score = 0
            flow = np.zeros((len(train_img),len(test_img)))
        root_flow[np.ix_(train_idx,test_idx)] = flow
        flow = root_flow
        
    else:
        EMD_score, _, flow = cv2.EMD(train_img.astype(np.float32), 
                                               test_img.astype(np.float32), 
                                               cv2.DIST_L2)
    
       
    if EMD_score < 0: #Need distance to be positive, negative just indicates direction
        EMD_score = abs(EMD_score)
        
    return EMD_score, flow