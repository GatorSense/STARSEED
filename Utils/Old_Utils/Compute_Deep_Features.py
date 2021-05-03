# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:33:46 2020

@author: jpeeples
"""

import numpy as np
import math
import pdb
import torch
from torchvision import models,transforms

def Deep_Feat(Z,backbone,min_dim=[224,224],root_only=True):

    # Only for 2d image
    assert(len(Z.shape) == 2)
    
    #Only compute features for images containing roots
    if root_only:
        if np.count_nonzero(Z) == 0:
            feats = np.zeros(512)
            return feats
        else:
            pass

    if min_dim is not None:
        #Pad superpixel if not correct size
        Z = np.pad(Z,(abs(Z.shape[0]-min_dim[0]),abs(Z.shape[1]-min_dim[1])))
        #Remove extra padding if necessary
        if not(Z.shape==min_dim):
            Z = Z[0:min_dim[0],0:min_dim[1]]
 
    #Resize root here or just 0 pad?
    transform = transforms.Compose([
                # transforms.Resize(Network_parameters['center_size']),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    #Add small positive value incase mean is 0
    # pdb.set_trace()
    Z = np.repeat(Z[:, :, np.newaxis],3,axis=-1)
    # Z = torch.from_numpy(Z).unsqueeze(0)
    Z = transform(Z).unsqueeze(0).type(torch.FloatTensor)
    backbone = backbone.float()
    backbone.fc = torch.nn.Sequential()
    backbone.eval()
    feats = backbone(Z)
    feats = feats.detach().cpu().numpy()
    # feats = np.mean(abs(feats))
    #Deep lac feat? variance/(mean^2)
    # feats = np.var(feats)/(np.mean(feats)**2)
    return feats


