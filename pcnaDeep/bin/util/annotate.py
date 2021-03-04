# -*- coding: utf-8 -*-
"""
Created on Mon Feb  22 09:03:20 2021

@author: Yifan Gui
"""
import pandas as pd
import sys, getopt, re, os, tarfile, tempfile, json
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
    ori = np.unique(label_table['trackId'])
    for i in range(1, len(ori)+1):
        dic[ori[i-1]] = i
    dic[0] = 0
    label_table['trackId'] = list(map(lambda x:dic[x], label_table['trackId']))
    label_table['parentTrackId'] = list(map(lambda x:dic[x], label_table['parentTrackId']))
    label_table['lineageId'] = list(map(lambda x:dic[x], label_table['lineageId']))
    
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
            y, x = np.round(p.centroid)
            tar = sub_table[(sub_table['Center_of_the_object_0']==int(x)) & (sub_table['Center_of_the_object_1']==int(y))]
            assert tar.shape[0] <= 1
            if tar.shape[0] == 0:
                mask[i,:,:][sl==p.label] = 0  # untracked
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
        raw (np.array): raw images data.
        tracked (np.array): annotated image data.

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


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:r", ["indir=", "mask=", "outdir="])
        # h: switch-type parameter, help
        # i: / o: parameter must with some values
        # m: mask dir
    except getopt.GetoptError:
        print('generateCalibanNPZ.py -i <inputfile> -o <outputfile> -m <mask> -r <relabel mask> -trk <generate .trk file instead of npz>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('generateCalibanNPZ.py -i <inputfile> -o <outputfile> -m <mask> -r <relabel mask> -trk <generate .trk file instead of npz>')
            sys.exit()
        elif opt == '-r':
            relabel = True
        elif opt in ("-i", "--indir"):
            ip = arg
        elif opt in ("-o", "--outdir"):
            out = arg
        elif opt in ("-m", "--mask"):
            mask = arg
    
    X = io.imread(ip)
    if len(X.shape)<4:
        X = np.expand_dims(X, axis=3)
    y = io.imread(mask).astype('bool').astype('uint16')
    if len(y.shape)<4:
        y = np.expand_dims(y, axis=3)
    
    for i in range(y.shape[0]):
        y[i,:,:,:] = measure.label(y[i,:,:,:])
        
    np.savez(out, X=X, y=y)
    
    
    # generate trk files from tracks
    df = pd.read_csv('/Users/jefft/Downloads/20200902-MCF10A-dual_extended/track/trks-refined-relabeled.csv')
    mask_fp = '/Users/jefft/Downloads/20200902-MCF10A-dual_extended/mask/'
    raw_fp = '/Users/jefft/Downloads/20200902-MCF10A-dual_extended/'
    out = '/Users/jefft/Desktop/test.npz'
    mask_dic = {}
    raw_dic = {}  # raw should have same file name as mask
    l = os.listdir(mask_fp)
    for i in l:
        if re.search('.png$', i) or re.search('.tif$', i):
            mask_dic[int(re.search('.*-(\d+).[pngtif]', i).group(1))] = os.path.join(mask_fp, i)
            raw_dic[int(re.search('.*-(\d+).[pngtif]', i).group(1))] = os.path.join(raw_fp, i)
            
    frame = sorted(list(mask_dic.keys()))
    frames = list(map(lambda x:io.imread(mask_dic[x]), frame))
    raw_frames = list(map(lambda x:io.imread(raw_dic[x]), frame))
    mask = np.stack(frames, axis=0)
    raw = np.stack(raw_frames, axis=0)
    #raw = img_as_float64(raw)
    
    df_new = relabel_trackID(df)
    tracked = label_by_track(mask, df_new)
    lin = get_lineage_dict(df_new)
    save_trks(out, lin, np.expand_dims(raw, axis=3), np.expand_dims(tracked, axis=3))
    
    # generate npz instead
    X = np.expand_dims(raw, axis=3)
    y = np.expand_dims(tracked, axis=3)
    np.savez(out, X=X, y=y)


    # 2021/3/4
    mask = io.imread('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/pcnaDeep/examples/10A_20200902_s1_cpd_trackPy/mask_tracked.tif')
    mask.dtype
    track = pd.read_csv('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/pcnaDeep/examples/10A_20200902_s1_cpd_trackPy/output/tracks-refined.csv')
    track
    track_new = relabel_trackID(track)
    tracked_mask = label_by_track(mask, track_new)
    txt = get_lineage_txt(track_new)
    # write out processed files for RES folder
    io.imsave('/Users/jefft/Desktop/mask_tracked.tif', tracked_mask)
    
