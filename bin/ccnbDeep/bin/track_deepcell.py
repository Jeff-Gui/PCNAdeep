#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:31:29 2021

@author: jefft
"""
import deepcell.applications.cell_tracking as cell_tracking
import skimage.measure as measure
import numpy as np
import pandas as pd

def trackDeepcell(mask, raw):
    tracker = cell_tracking.CellTracking()
    if len(mask.shape)<4:
        mask = np.expand_dims(mask, axis=3)
    if len(raw.shape)<4:
        raw = np.expand_dims(raw, axis=3)
    
    out = tracker.track(raw.copy(), mask.copy())
    y = out['y_tracked']
    out = out['tracks']
    track_table = pd.DataFrame()
    for key in list(out.keys()):
        t = out[key]
        dt = pd.DataFrame()
        dt['frame'] = t['frames']
        dt['trackId'] = t['label'] + 1  # track begins from 2
        dt['lineageId'] = t['label'] + 1
        dt['parentTrackId'] = 0
        
        # search for object center
        lb = t['label']
        c0 = []
        c1 = []
        for i in range(len(t['frames'])):
            f = t['frames'][i]
            slice = y[f,:,:,0].copy()
            slice[slice!=lb] = 0
            obj_x, obj_y = measure.regionprops(measure.label(slice))[0].centroid
            c0.append(obj_x)
            c1.append(obj_y)
        dt['Center_of_the_object_0'] = c0
        dt['Center_of_the_object_1'] = c1
        track_table = track_table.append(dt)
    
    # organize lineages
    for key in list(out.keys()):
        t = out[key]
        if t['daughters']:
            for daug in t['daughters']:
                track_table.iloc[np.flatnonzero(track_table.trackId == daug),3] = int(t['label'])  # parent track ID
                track_table.iloc[np.flatnonzero(track_table.trackId == daug), 2] = int(t['label'])  # lineage ID
                track_table.iloc[np.flatnonzero(track_table.lineageId == daug), 2] = int(t['label'])
    
    return track_table, y

