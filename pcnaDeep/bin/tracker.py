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

def track(df, discharge=40, gap_fill=5):
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
    t = tp.link(f, search_range=discharge, memory=gap_fill, adaptive_stop=0.4*discharge)
    t.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'frame', 'trackId']
    out = pd.merge(df, t, on=['Center_of_the_object_0', 'Center_of_the_object_1', 'frame'])
    #  change format for downstream
    out['trackId'] += 1
    out['lineageId'] = out['trackId']
    out['parentTrackId'] = 0
    out = out[['frame','trackId','lineageId','parentTrackId','Center_of_the_object_0','Center_of_the_object_1','phase','Probability of G1/G2','Probability of S','Probability of M','continuous_label']]
    names = list(out.columns)
    names[4] = 'Center_of_the_object_1'
    names[5] = 'Center_of_the_object_0'
    names[6] = 'predicted_class'
    out.columns = names
    out = out.sort_values(by=['trackId','frame'])

    return out.drop('continuous_label', axis=1)

def track_mask(mask, discharge=40, gap_fill=5, render_phase=False, phase_dic={10:'G1/G2',50:'S',100:'M',200:'G1/G2'}):
    """Track binary mask objects

    Args:
        mask: uint8 nparray, can either be binary or labeled with cell cycle phases
        discharge, gap_fill: see track()
        render_phase: whether to deduce cell cycle phase from the labeled mask
        phase_dic: mapping of object label and cell cycle phase
    """

    p = pd.DataFrame()
    if render_phase:
        for i in range(mask.shape[0]):
            props = measure.regionprops_table(measure.label(mask[i,:,:], connectivity=1), intensity_image=mask[i,:,:], properties=('centroid', 'label', 'max_intensity'))
            props = pd.DataFrame(props)
            props.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'continuous_label', 'mi']
            l = props['mi']
            phase = []
            probG = []
            probS = []
            probM = []
            for k in range(props.shape[0]):
                ps = phase_dic[int(l[k])]
                phase.append(ps)
                if ps=='G1/G2':
                    probG.append(1)
                    probS.append(0)
                    probM.append(0)
                elif ps=='S':
                    probG.append(0)
                    probS.append(1)
                    probM.append(0)
                else:
                    probG.append(0)
                    probS.append(0)
                    probM.append(1)
                
            props['Probability of G1/G2'] = probG
            props['Probability of S'] = probS
            props['Probability of M'] = probM
            props['phase'] = phase
            props['frame'] = i
            probs = probs.drop('mi')
            p = p.append(props)
            
    else:
        for i in range(mask.shape[0]):
            props = measure.regionprops_table(measure.label(mask[i,:,:], connectivity=1), properties=('centroid', 'label'))
            props = pd.DataFrame(props)
            props.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'continuous_label']
            props['Probability of G1/G2'] = 0
            props['Probability of S'] = 0
            props['Probability of M'] = 0
            props['phase'] = 0
            props['frame'] = i
            p = p.append(props)
        
    track_out = track(p, discharge=discharge, gap_fill=gap_fill)
    return track_out
