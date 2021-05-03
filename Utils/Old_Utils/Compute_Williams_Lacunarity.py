# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:33:46 2020

@author: jpeeples
"""
# -----------------------------------------------------------------------------
# https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
# From https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension:
#
# In fractal geometry, the Minkowski–Bouligand dimension, also known as
# Minkowski dimension or box-counting dimension, is a way of determining the
# fractal dimension of a set S in a Euclidean space Rn, or more generally in a
# metric space (X, d).
# -----------------------------------------------------------------------------
import numpy as np
import math
import pdb

def Williams_Lac(Z,root_only=True):

    # Only for 2d image
    assert(len(Z.shape) == 2)
    
    #Only compute features for images containing roots
    if root_only:
        if np.count_nonzero(Z) == 0:
            feats = 0
            return feats
        else:
            pass

    # if min_dim is not None:
    #     #Pad superpixel if not correct size
    #     Z = np.pad(Z,(abs(Z.shape[0]-min_dim[0]),abs(Z.shape[1]-min_dim[1])))
    #     #Remove extra padding if necessary
    #     if not(Z.shape==min_dim):
    #         Z = Z[0:min_dim[0],0:min_dim[1]]
 
 
    #Add small positive value incase mean is 0
    try:
        feats = np.var(Z)/((np.mean(Z)**2))
    except:
        feats = np.var(Z)/((np.mean(Z)**2)+6e-10)
    return feats


