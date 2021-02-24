#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:39:39 2021

@author: Cell Division
"""
import skimage.io as io
import skimage.measure as measure
import skimage.exposure as exposure
import os, sys, getopt
import pandas as pd
import numpy as np
from ringByDist import measure_dist

def enhance_contrast(stack):
    for i in range(stack.shape[0]):
        stack[i,:,:] = exposure.rescale_intensity(stack[i,:,:], in_range=tuple(np.percentile(stack[i,:,:], (1,99))))

    return stack


def get_synchrogram(mask, intensity_image, track, pad=50):
    """Return time lapse stack of tracked object
    
    Args:
        mask: ndarray with each object labeled
        intensity_image: list[ndarray], images to extract synchrogram from,
                        each channel be one element of the list.
        track: data frame, should have following fields,
            trackId: int, unique ID
            frame: int
            Center_of_the_object_0: int, y location
            Center_of_the_object_1: int, x location
        pad: int, pixel amount surround bounding box of the object in output
    
    Return:
        image stack, time axis being the first
    """
    h, w = mask.shape[1], mask.shape[2]
    b0_min = h
    b1_min = w
    b2_max = 0
    b3_max = 0
    labels = []
    for f in list(track['frame']):
        sub = track[track['frame']==f]
        target_x = int(np.round(sub['Center_of_the_object_0']))
        target_y = int(np.round(sub['Center_of_the_object_1']))
        
        assert sub.shape[0]==1
        mask_frame = mask[f,:,:].copy()
        props = measure.regionprops(mask_frame)
        for p in props:
            y,x = np.round(p.centroid)
            if target_x==int(x) and target_y==int(y):
                labels.append(p.label)
                mask_frame[mask_frame!=p.label]=0
                b = p.bbox
                if (b[0]-pad) < b0_min:
                    b0_min = b[0]-pad
                if (b[1]-pad) < b1_min:
                    b1_min = b[1]-pad
                if (b[2]+pad) > b2_max:
                    b2_max = b[2]+pad
                if (b[3]+pad) > b3_max:
                    b3_max = b[3]+pad
                break
    
    assert len(labels) == track.shape[0]
    
    if b0_min < 0:
        b0_min = 0
    if b1_min < 0:
        b1_min = 0
    if b2_max > h:
        b2_max = h
    if b3_max > w:
        b3_max = w
    
    sls = []
    for i in intensity_image:
        sl = i[list(track['frame']), b0_min:b2_max, b1_min:b3_max].copy()
        sls.append(sl)
    
    new_mask = mask[list(track['frame']), b0_min:b2_max, b1_min:b3_max].copy()
    for i in range(new_mask.shape[0]):
        fr = new_mask[i,:,:]
        fr[fr!=labels[i]] = 0
        new_mask[i,:,:] = fr
    
    new_mask = new_mask.astype('bool').astype('uint16')
    new_mask = new_mask * 65535
    sls.append(new_mask)
    
    return np.stack(sls, axis=3)

def main(table, dic, gfp, mask):
    out = {}
    for i in np.unique(table['trackId']):
        track = table[table['trackId']==i]
        entry = int(track[track['is_entry']==1]['frame'])
        begin = int(np.min(track['frame'])) - int(entry)
        s = get_synchrogram(mask, [dic, gfp], track, pad=50)
        out[i] = (s, begin)
        
    return out


if __name__ == "__main__":
    en = False
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hg:d:m:o:t:e", ["gfp_dir=","dic_dir", "mask=", "outdir=", "track="])
        # h: switch-type parameter, help
        # i: / o: parameter must with some values
        # m: mask dir
    except getopt.GetoptError:
        print('analysis_mitosis.py -g <gfp input> -d <dic input> -o <outputfile> -m <mask> -t <track table> -e <enhance contrast>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('analysis_mitosis.py -g <gfp input> -d <dic input> -o <outputfile> -m <mask> -t <track table> -e <enhance contrast>')
            sys.exit()
        elif opt in ("-g", "--gfp_dir"):
            fp_gfp = arg
        elif opt in ("-o", "--outdir"):
            out = arg
        elif opt in ("-m", "--mask"):
            fp_mask = arg
        elif opt in ("-d", "--dic_dir"):
            fp_dic = arg
        elif opt in ("-t", "--track"):
            fp_track = arg
        elif opt in ("-e"):
            en = True
    
    df = pd.read_csv(fp_track)
    dic = io.imread(fp_dic)
    gfp = io.imread(fp_gfp)
    mask = io.imread(fp_mask)
    mask = measure.label(mask)  # relabel each object
    if en:
        print('Enhance contrast...')
        dic = enhance_contrast(dic)
        gfp = enhance_contrast(gfp)
    
    prefix = os.path.basename(fp_track)[:-4]
    rt = main(df, dic, gfp, mask)
    ds = pd.DataFrame()
    for i in list(rt.keys()):
        io.imsave(os.path.join(out, prefix+'_'+str(i)+'.tif'), rt[i][0])
        d = measure_dist(rt[i][0],t0=rt[i][1],max_d=10,min_d=-10)
        d['trackId'] = i
        ds = ds.append(d)
    ds.to_csv(os.path.join(out, 'measure.csv'), index=0)
    print('Finish.')
    

