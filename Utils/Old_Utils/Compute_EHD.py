# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:52:40 2020
Function to generate EHD histogram feature maps
@author: jpeeples
"""
import numpy as np
from scipy import signal,ndimage
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import pdb

def Generate_masks(mask_size=3,angle_res=45,normalize=False,rotate=False):
    
    #Make sure masks are appropiate size. Should not be less than 3x3 and needs
    #to be odd size
    if type(mask_size) is list:
        mask_size = mask_size[0]
    if mask_size < 3:
        mask_size = 3
    elif ((mask_size % 2) == 0):
        mask_size = mask_size + 1
    else:
        pass
    
    if mask_size == 3:
        if rotate:
            Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
        else:
            Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
    else:
        if rotate:
            Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
        else:
            Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
        dim = np.arange(5,mask_size+1,2)
        expand_mask =  np.outer(np.array([1,2,1]).T,np.array([1,2,1]))
        for size in dim:
            # Gx = signal.convolve2d(expand_mask,Gx)
            Gy = signal.convolve2d(expand_mask,Gy)
    
    #Generate horizontal masks
    angles = np.arange(0,360,angle_res)
    masks = np.zeros((len(angles),mask_size,mask_size))
    
    #TBD: improve for masks sizes larger than 
    for rot_angle in range(0,len(angles)):
        masks[rot_angle,:,:] = ndimage.rotate(Gy,angles[rot_angle],reshape=False,
                                              mode='nearest')
        
    
    #Normalize masks if desired
    if normalize:
        if mask_size == 3:
            masks = (1/8) * masks
        else:
            masks = (1/8) * (1/16)**len(dim) * masks 
    return masks


def Get_EHD(X,mask_size=3,angle_res=45,normalize=False,threshold=0.9,
            window_size=[5,5],normalize_count=False,stride=1,device='cpu',
            root_only=True):
    
    #Only compute features for images containing roots
    if root_only:
        if np.count_nonzero(X) == 0:
            feats = np.zeros(len(np.arange(0,360,angle_res))+1)
            return feats
        else:
            pass
    #Generate masks based on parameters
    masks = Generate_masks(mask_size=mask_size,
                           angle_res=angle_res,
                           normalize=normalize)
  
    #Convolve input with filters, expand masks to match input channels
    #TBD works for grayscale images (single channel input)
    #Check for multi-image input
    # in_channels = X.shape[1]
    X = torch.tensor(X).float().unsqueeze(0).unsqueeze(1)
    masks = torch.tensor(masks).float()
    masks = masks.unsqueeze(1)
    if device is not None:
        masks = masks.to(device)
  
    edge_responses = F.conv2d(X,masks)
    
    #Find max response
    [value,index] = torch.max(edge_responses,dim=1)
    
    #Set edge responses to "no edge" if not larger than threshold
    index[value<threshold] = masks.shape[0] 
    
    feat_vect = []
    window_scale = np.prod(np.asarray(window_size))
    
    for edge in range(0,masks.shape[0]+1):
        # #Sum count
        if normalize_count:
           #Average count
            feat_vect.append((F.avg_pool2d((index==edge).unsqueeze(1).float(),
                              window_size,stride=stride,
                              count_include_pad=False).squeeze(1)))
        else:
            feat_vect.append(window_scale*F.avg_pool2d((index==edge).unsqueeze(1).float(),
                              window_size,stride=stride,
                              count_include_pad=False).squeeze(1))
        
    
    #Return vector
    feat_vect = torch.stack(feat_vect,dim=1)
    feat_vect = feat_vect.squeeze(0).detach().cpu().numpy()
    # fig, ax = plt.subplots(1,feat_vect.shape[0],figsize=(15,15))
    # angles = np.arange(0,360,angle_res)
    
    # #Remove extra dimension on histogram masks tensor
    # for temp_ang in range(0,feat_vect.shape[0]):
        
    #     if temp_ang == feat_vect.shape[0]-1:
    #          ax[temp_ang].set_title('No Edge')
    #          ax[temp_ang].set_yticks([])
    #          ax[temp_ang].set_xticks([])
    #          im = ax[temp_ang].imshow(feat_vect[temp_ang])
    #     else:
    #         ax[temp_ang].set_title(str(angles[temp_ang])+u'\N{DEGREE SIGN}')
    #         ax[temp_ang].set_yticks([])
    #         ax[temp_ang].set_xticks([])
    #         im = ax[temp_ang].imshow(feat_vect[temp_ang])
            
    #     plt.colorbar(im,ax=ax[temp_ang],fraction=0.046, pad=0.04)
    
    # ax[0].set_ylabel('EHD Outputs',rotation=90,size='small')
    # plt.tight_layout()
    # plt.show()
    
    #Compute average count for across spatial locations
    feat_vect = np.mean(feat_vect,axis=(1,2))
    #Return index of angle corresponding to highest count in image,
    #ignore no edge if root only
    #Need 0 degrees to be non-zero bin weight for EMD calculation (set to 1)
    #No edge will be 0
    # if root_only:
    #     feat_vect = np.argmax(feat_vect[:-1]) + 1
    # else:
    #     feat_vect = np.argmax(feat_vect[:-1]) + 1
    #     if np.count_nonzero(X) == 0:
    #         feats = 0
    # feat_vect = np.max(feat_vect[:-1])
    # feat_vect = F.softmax(feat_vect,dim=0)
        
    return np.float32(feat_vect)
    
    
    
    
    
    
    
    
    
    