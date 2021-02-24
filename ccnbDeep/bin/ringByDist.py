#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:32:48 2020

@author: Cell Division
"""
import numpy as np
import skimage.measure as measure
from scipy import ndimage as ndi
import pandas as pd

def get_dist_map(mask):
    """Distance map of a binary object mask
    
    Args:
        mask: bool ndarray, nuclear true, cytoplasm false. dimensionality = 2
    """
    
    contours = measure.find_contours(mask, level=0.5, fully_connected='high')
    if len(contours)!=1:
        print('Warning: two contours in one img, may due to holes/small clusters.')

    contours = np.floor(contours[0]).astype('int')
    contour_coord = (contours[:,0], contours[:,1])
    # compute distance of every point on the image to the contour
    dist_map = np.ones(mask.shape)
    dist_map[contour_coord] = 0
    dist_map = ndi.distance_transform_edt(dist_map)
    
    mask = mask.astype('int')
    mask[mask==1] = -1
    mask[mask==0] = 1
    dist_map = np.multiply(mask, dist_map)
    return(dist_map)


def measure_dist(stack, t0, max_d=float('+inf'), min_d=float('-inf'), remove_out=True):
    """Statistics of each distance relative to NE in a synchrogram
    
    Args:
        stack: time lapse ndarray of 3 channels in order of DIC/GFP/binary mask
        t0: int, time location of the first frame, usually relative to NEBD
        max_d, min_d: int, limit of distance
        rmove_out: bool, whether to remove outliers of intensity values
    
    Return:
        for each object at each frame, mean and s.d. of the GFP signal
    """
    
    mask = stack[:,:,:,2]
    gfp = stack[:,:,:,1]
    
    dic = {"frame":[], "dist":[], "mean":[], "sd":[]}

    for i in range(mask.shape[0]):
        mp = np.round(get_dist_map(mask[i,:,:].astype('bool')))
        dist = np.unique(mp)
        dists = []
        ms = []
        sds = []
        for j in dist:
            if j<=max_d and j>=min_d:
                dists.append(j)
                pix = gfp[i,mp==j]
                if remove_out:
                    pix = pix[np.abs(pix-np.mean(pix))<=np.std(pix)]  # outliers: out of mu + 3sd
                ms.append(np.mean(pix))
                sds.append(np.std(pix))
        
        dic['dist'].extend(dists)
        dic['mean'].extend(ms)
        dic['sd'].extend(sds)
        dic['frame'].extend([i+t0 for k in range(len(dists))])
    
    return pd.DataFrame(dic)


