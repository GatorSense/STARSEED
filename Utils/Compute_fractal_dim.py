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

def fractal_dimension(Z, threshold=0.9,compute_lac=False,min_dim=None,root_only=True):

    # Only for 2d image
    assert(len(Z.shape) == 2)
    
    #Only compute features for images containing roots
    if root_only:
        if np.count_nonzero(Z) == 0:
            if compute_lac:
                feats = [0,0]
            else:
                feats = 0
            return feats
        else:
            pass

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        # Each box should have at least one pixel (can't take log of 0, 
        # using eps value results in large value)
        count = len(np.where((S > 0) & (S < k*k))[0])
        if count == 0:
            count = 1
        return count


    if min_dim is not None:
        #Pad superpixel if not correct size
        Z = np.pad(Z,(abs(Z.shape[0]-min_dim[0]),abs(Z.shape[1]-min_dim[1])))
        #Remove extra padding if necessary
        if not(Z.shape==min_dim):
            Z = Z[0:min_dim[0],0:min_dim[1]]
        
    # Transform Z into a binary array (should already be binary for root images)
    # With preprocessing, this will change
    # if Gray/RGB image, need to divide by 255
    # pdb.set_trace()
    Z = (Z < threshold)
   
    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    # Some values returned positive slope (expect negative)
    try:
        if len(sizes)==1: #one data point, slope is zero
            # coeffs = np.polyfit(np.log(sizes), np.log(counts),0)
            coeffs = [0]
        else:
            coeffs = np.polyfit(np.log(sizes),np.log(counts),1)
    except:
        #Area of grid is too small, set value to zero
        coeffs = [0]
        counts = [0]
 
    #Compute lacunarity feature if desired
    if compute_lac:
        lac_feat = lacunarity(counts)
        feats = [abs(coeffs[0]),lac_feat]
    else:
        feats = abs(coeffs[0])
    # return -coeffs[0]
    return feats

def lacunarity(counts):
#Compute lacunarity based on 1st and 2nd moment
# Calculation based on: 
# Keller, J. M., Chen, S., & Crownover, R. M. (1989). Texture description and 
# segmentation through fractal geometry. Computer Vision, Graphics, and image 
# processing, 45(2), 150-166.
#Using Mendelbrot's definition (may use Keller's definition)
    #Convert counts from list to array
    counts = np.array(counts)

    #Compute probabilities of count
    totals = np.sum(counts) + 10e-6 #Avoid division by 0
    probs = counts/totals
    
    #Compute N(L)
    # N = np.sum((1/counts)*probs)
    
    #Compute first and second moments
    M1 = np.sum(counts*probs)
    M2 = np.sum((counts**2)*probs)
    
    #Compute lacunarity feature (Mendelbrot)
    lac_val = (M2-M1**2)/((M1**2)+10e-6)
    
    #Compute lacunarity feature (Keller) (for Runs 4&5, performed slightly less for SI)
    # lac_val = (M1 - N) / (M1 + N)
 
    return lac_val