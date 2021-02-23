#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:32:48 2020

@author: jefft
"""
import numpy as np
import skimage.measure as measure
from scipy import ndimage as ndi

mask = []
img = []

def ring_measure(mask):
    #  # bool, nuclear true, cytoplasm false. dimensionality = 2
    contours = measure.find_contours(mask, level=0.5, fully_connected='high')
    if len(contours)!=1:
        print('Error: two masks in one img')
    else:
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
    
    
#=============================== Testing ======================================
import skimage.io as io
import pandas as pd
mask = ~io.imread('/Users/jefft/Desktop/demo/demo_mask.tif').astype('bool')[:,:,:,0]
img = io.imread('/Users/jefft/Desktop/demo/demo_gfp.tif')
dic = {"dist":[], "img":[], "time":[]}

for i in range(mask.shape[0]):
    dist = np.floor(ring_measure(mask[i,:,:]).ravel())
    dic['dist'] += list(dist)
    dic['img'] += list(img[i,:,:].ravel())
    dic['time'] += [i for k in range(len(dist))]

dic = pd.DataFrame(dic)
dic.to_csv('/Users/jefft/Desktop/test.csv',index=0)
    