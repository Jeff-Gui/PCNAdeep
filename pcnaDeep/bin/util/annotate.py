# -*- coding: utf-8 -*-
"""
Created on Mon Feb  22 09:03:20 2021

@author: Yifan Gui
"""
import pandas as pd
import sys, getopt, re, os, tarfile, tempfile, json
from io import BytesIO
import skimage.io as io
import skimage.measure as measure
import numpy as np
from skimage.util import img_as_float64

def relabel_trackID(label_table):
    """Relabel trackID in tracking table, starting from 1

    Args:
        label_table: track table
    """

    dic = {}
    ori = list(np.unique(label_table['trackId']))
    for i in range(1, len(ori)+1):
        dic[ori[i-1]] = i
    dic[0] = 0
    for i in range(label_table.shape[0]):
        label_table.loc[i, 'trackId'] = dic[label_table['trackId'][i]]
        label_table.loc[i, 'parentTrackId'] = dic[label_table['parentTrackId'][i]]
        label_table.loc[i, 'lineageId'] = dic[label_table['lineageId'][i]]
    
    return label_table
    
def label_by_track(mask, label_table):
    """Label objects in mask with track ID

    Args:
        mask: uint8 np array
        label_table: track table
    
    Return:
        unit8/uint16 np array, dtype based on track count
    """

    assert mask.shape[0] == np.max(label_table['frame']+1)
    assert mask.dtype == np.dtype('uint8')
    
    mask = mask.astype('bool').astype('uint8')
    if np.max(label_table['trackId']) > 254:
        mask = mask.astype('bool').astype('uint16')
        
    for i in np.unique(label_table['frame']):
        sub_table = label_table[label_table['frame']==i]
        sl = mask[i,:,:].copy()
        sl = measure.label(sl)
        props = measure.regionprops(sl)
        for p in props:
            y, x = np.floor(p.centroid)
            tar = sub_table[(np.floor(sub_table['Center_of_the_object_0'])==x) & (np.floor(sub_table['Center_of_the_object_1'])==y)]
            assert tar.shape[0] <= 1
            if tar.shape[0] == 0:
                mask[i,:,:][sl==p.label] = 0  # untracked, should not exist
            else:
                mask[i,:,:][sl==p.label] = int(tar['trackId'])
    
    return mask

def get_lineage_dict(label_table):
    """Generate lineage dictionary in deepcell tracking format
    """

    out = {}
    for i in list(np.unique(label_table['trackId'])):
        i = int(i)
        sub_table = label_table[label_table['trackId']==i]
        out[i] = {'capped':False, 'daughters':[], 'frame_div':None, 'frames':list(sub_table['frame']), 'label':i, 'parent':None}
        if list(sub_table['parentTrackId'])[0] != 0:
            out[i]['parent'] = list(sub_table['parentTrackId'])[0]
    
    for i in list(np.unique(label_table['trackId'])):
        i = int(i)
        if out[i]['parent'] is not None:
            par = out[i]['parent']
            out[par]['daughters'].append(i)
            out[par]['capped'] = True
            if out[par]['frame_div'] is None:
                out[par]['frame_div'] = int(np.min(out[i]['frames']) - 1)
            else:
                out[par]['frame_div'] = int(np.min((np.min(out[i]['frames']) - 1, out[par]['frame_div'])))
            
    return out
    
def get_lineage_txt(label_table):
    """Generate txt table in Cell Tracking Challenge format

    Return:
        pandas dataframe, remove index and col name before output.
    """

    dic = {'id':[], 'appear':[], 'disappear':[], 'parent':[]}
    for i in np.unique(label_table['trackId']):
        sub = label_table[label_table['trackId']==i]
        begin = np.min(sub['frame'])
        end = np.max(sub['frame'])
        parent = np.unique(sub['parentTrackId'])

        dic['id'].append(i)
        dic['appear'].append(int(begin))
        dic['disappear'].append(int(end))
        dic['parent'].append(int(parent))

    return pd.DataFrame(dic)

def save_trks(filename, lineages, raw, tracked):
    """Copied from deepcell_tracking.utils, version 0.3.1. Author Van Valen Lab
        ! Changed trks to trk to fit caliban labeler
    """
    """Saves raw, tracked, and lineage data into one trk_file.

    Args:
        filename (str): full path to the final trk files.
        lineages (dict): a list of dictionaries saved as a json.
        raw (np.array): 4D raw images data. THWC
        tracked (np.array): 4D annotated image data. THWC

    Raises:
        ValueError: filename does not end in ".trk".
    """
    if not str(filename).lower().endswith('.trk'):
        raise ValueError('filename must end with `.trk`. Found %s' % filename)

    with tarfile.open(filename, 'w') as trks:
        with tempfile.NamedTemporaryFile('w', delete=False) as lineages_file:
            json.dump(lineages, lineages_file, indent=4)
            lineages_file.flush()
            lineages_file.close()
            trks.add(lineages_file.name, 'lineage.json')
            os.remove(lineages_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as raw_file:
            np.save(raw_file, raw)
            raw_file.flush()
            raw_file.close()
            trks.add(raw_file.name, 'raw.npy')
            os.remove(raw_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as tracked_file:
            np.save(tracked_file, tracked)
            tracked_file.flush()
            tracked_file.close()
            trks.add(tracked_file.name, 'tracked.npy')
            os.remove(tracked_file.name)

def load_trks(filename):
    """Copied from deepcell_tracking.utils, version 0.3.1. Author Van Valen Lab
    """
    """Load a trk/trks file.

    Args:
        filename (str): full path to the file including .trk/.trks.

    Returns:
        dict: A dictionary with raw, tracked, and lineage data.
    """
    with tarfile.open(filename, 'r') as trks:

        # numpy can't read these from disk...
        array_file = BytesIO()
        array_file.write(trks.extractfile('raw.npy').read())
        array_file.seek(0)
        raw = np.load(array_file)
        array_file.close()

        array_file = BytesIO()
        array_file.write(trks.extractfile('tracked.npy').read())
        array_file.seek(0)
        tracked = np.load(array_file)
        array_file.close()

        # trks.extractfile opens a file in bytes mode, json can't use bytes.
        _, file_extension = os.path.splitext(filename)

        if file_extension == '.trks':
            trk_data = trks.getmember('lineages.json')
            lineages = json.loads(trks.extractfile(trk_data).read().decode())
            # JSON only allows strings as keys, so convert them back to ints
            for i, tracks in enumerate(lineages):
                lineages[i] = {int(k): v for k, v in tracks.items()}

        elif file_extension == '.trk':
            trk_data = trks.getmember('lineage.json')
            lineage = json.loads(trks.extractfile(trk_data).read().decode())
            # JSON only allows strings as keys, so convert them back to ints
            lineages = []
            lineages.append({int(k): v for k, v in lineage.items()})

    return {'lineages': lineages, 'X': raw, 'y': tracked}

def lineage_dic2txt(lineage_dic):
    """Convert deepcell .trk lineage format to CTC txt format

    Args:
        lineage_dic: [index:dict], extracted from deepcell .trk file 
    """
    
    lineage_dic = lineage_dic[0]
    dic = {'id':[], 'appear':[], 'disappear':[]}
    pars = []
    for d in lineage_dic.values():
        i = d['label']
        begin = np.min(d['frames'])
        end = np.max(d['frames'])

        dic['id'].append(i)
        dic['appear'].append(int(begin))
        dic['disappear'].append(int(end))
        if d['daughters']:
            pars.append(d)

    dic = pd.DataFrame(dic)
    dic['parents'] = 0

    # resolve parents
    for p in pars:
        for d in d['daughters']:
            dic.loc[dic.index[dic['id']==d], 'parents'] = d
    
    return dic

'''
    # 2021/3/4
    # 1. From detection and tracking output, generate RES folder files
    mask = io.imread('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/pcnaDeep/examples/10A_20200902_s1_cpd_trackPy/mask_tracked.tif')
    mask.dtype
    track = pd.read_csv('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/pcnaDeep/examples/10A_20200902_s1_cpd_trackPy/output/tracks-refined.csv')
    track
    track_new = relabel_trackID(track.copy())
    tracked_mask = label_by_track(mask.copy(), track_new.copy())
    txt = get_lineage_txt(track_new)
    # write out processed files for RES folder
    io.imsave('/Users/jefft/Desktop/mask_tracked.tif', tracked_mask.astype('uint16'))
    txt.to_csv('/Users/jefft/Desktop/res_track.txt', sep=' ', index=0, header=False)
    
    # 2. From ground truth mask, generate Caliban files for annotating tracks, eventually for GT folder files
    # Ground truth mask may be annotated by VIA2
    mask = io.imread('/Users/jefft/Desktop/mask_GT.tif')
    raw = io.imread('/Users/jefft/Desktop/raw.tif')
    from tracker import track_mask
    out = track_mask(mask, dis)
    track_new = relabel_trackID(out.copy())
    tracked_mask = label_by_track(mask.copy(), track_new.copy())
    #txt = get_lineage_txt(track_new.copy())
    dic = get_lineage_dict(track_new.copy())
    save_trks('/Users/jefft/Desktop/001.trk', dic, np.expand_dims(raw, axis=3), np.expand_dims(tracked_mask, axis=3))

'''