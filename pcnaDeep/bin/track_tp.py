#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:31:29 2021

@author: Cell division
"""

import trackpy as tp
import skimage.measure as measure
import numpy as np
import pandas as pd

def track(df, mask, discharge=40, gap_fill=5):
    """Track and relabel mask with trackID

    Args:
        df: pandas data frame with fields:
            Center_of_the_object_0: x location of each object
            Center_of_the_object_1: y location of each object
            frame: time location
            (other optional columns)
        
        mask: ndarray of corresponding table
        discharge: maximum distance an object can move between frames
        gap_fill: temporal filling fo tracks
    
    Return:
        tracked table
        y: mask relabeled with trackID

    """

    f = df[['Center_of_the_object_0', 'Center_of_the_object_1', 'frame']]
    f.columns = ['x','y','frame']
    t = tp.link(f, search_range=discharge, memory=gap_fill)
    t.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'frame', 'trackId']
    out = pd.merge(df, t, on=['Center_of_the_object_0', 'Center_of_the_object_1', 'frame'])
    #  change format for downstream
    out['lineageId'] = out['trackId']
    out['parentTrackId'] = 0
    out = out[['frame','trackId','lineageId','parentTrackId','Center_of_the_object_0','Center_of_the_object_1','phase','Probability of G1/G2','Probability of S','Probability of M','continuous_label']]
    names = list(out.columns)
    names[4] = 'Center_of_the_object_1'
    names[5] = 'Center_of_the_object_0'
    names[6] = 'predicted_class'
    out.columns = names
    out = out.sort_values(by=['trackId','frame'])

    #  relabel mask with trackId
    mask_bin = mask.astype('bool').astype('uint16')
    
    ct = 0
    for i in range(mask_bin.shape[0]):
        sl = mask_bin[i,:,:]
        props = measure.regionprops(mask[i,:,:])
        sub = out[out['frame']==i]
        for p in props:
            y, x = p.centroid
            tg = sub[(sub['Center_of_the_object_0']==x) & (sub['Center_of_the_object_1']==y)]
            assert tg.shape[0] <= 1  #  should only have less than one match
            if tg.shape[0]==0:
                ct += 1

            sl[mask[i,:,:]==int(tg['continuous_label'])] = tg['trackId']
        mask_bin[i,:,:] = sl.copy()

    print('Untracked object: ' + str(ct))
    
    return out.drop('continuous_label', axis=1), mask_bin

