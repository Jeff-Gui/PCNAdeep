# -*- coding: utf-8 -*-
"""
Created on Mon Feb  22 09:03:20 2021

@author: Yifan Gui
"""
import pandas as pd
import os, tarfile, tempfile, json
from io import BytesIO
import skimage.io as io
import numpy as np

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
        mask: uint8 np array, output from main model
        label_table: track table
    
    Return:
        unit8/uint16 np array, dtype based on track count
    """

    assert mask.shape[0] == np.max(label_table['frame']+1)
    assert mask.dtype == np.dtype('uint8')
    
    if np.max(label_table['trackId']) * 2 > 254:
        mask = mask.astype('uint16')
    
    for i in np.unique(label_table['frame']):
        sub_table = label_table[label_table['frame']==i]
        sl = mask[i,:,:].copy()
        ori_labels = set(np.unique(sl)) - set([0])
        untracked = list(ori_labels - set(list(sub_table['continuous_label'])))
        #  remove untracked
        for j in untracked:
            sl[mask[i,:,:]==j] = 0
        #  update tracked
        for j in sub_table.index:
            sl[mask[i,:,:]==sub_table.loc[j, 'continuous_label']] = sub_table.loc[j, 'trackId']
        mask[i,:,:] = sl.copy()
    
    return mask

def get_lineage_dict(label_table, rel):
    """Generate lineage dictionary in deepcell tracking format
    
    Args:
        label_table: table processed
        rel: mitosis relationship, use to determine frame of division
    """

    out = {}
    for i in list(np.unique(label_table['trackId'])):
        i = int(i)
        sub_table = label_table[label_table['trackId']==i]
        out[i] = {'capped':False, 'daughters':[], 'frame_div':None, 'frames':list(sub_table['frame']), 'label':i, 'parent':None}
        if list(sub_table['parentTrackId'])[0] != 0:
            out[i]['parent'] = list(sub_table['parentTrackId'])[0]
    
    for i in list(rel.keys()):
        out[i]['capped'] = True
        out[i]['frame_div'] = rel[i][1]
        out[i]['daughters'].extend(rel[i][0])
    
    for i in list(out.keys()):
        if out[i]['parent'] is not None:
            par = out[i]['parent']
            if i not in out[par]['daughters']:
                out[par]['daughters'].append(i)
            
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
    pars = {}
    for d in lineage_dic.values():
        i = d['label']
        begin = np.min(d['frames'])
        end = np.max(d['frames'])

        dic['id'].append(i)
        dic['appear'].append(int(begin))
        dic['disappear'].append(int(end))
        if d['daughters']:
            for dg in d['daughters']:
                pars[dg] = i

    dic = pd.DataFrame(dic)
    dic['parents'] = 0

    # resolve parents
    for dg in list(pars.keys()):
        dic.loc[dic.index[dic['id']==dg], 'parents'] = pars[dg]
    
    return dic

def break_track(label_table):
    """Break tracks in a lineage table into single tracks, where
    No gapped tracks allowed. All gap must be transferred into parent-daughter
    relationship.
    
    Algorithm:
        Rename raw parentTrackId to mtParTrk
        Initiate new parentTrackId column with 0
        Separate all tracks individually
    
    (In original lineage table, single track can be gapped, lineage only associates
     mitosis tracks, not gapped tracks.)
    
    Returns:
        processed table
        {parentId:[[daughter ids], mitosis frame]}
    """
    
    max_trackId = np.max(label_table['trackId'])
    label_table['mtParTrk'] = label_table['parentTrackId']
    label_table['parentTrackId'] = 0
    label_table['ori_trackId'] = label_table['trackId']
    new_table = pd.DataFrame()
    
    for l in np.unique(label_table['trackId']):
        tr = label_table[label_table['trackId']==l].copy()
        
        if np.max(tr['frame']) - np.min(tr['frame']) + 1 != tr.shape[0]:
            sep, max_trackId = separate(list(tr['frame']).copy(), list(tr['mtParTrk']).copy(), l, base=max_trackId)
            tr.loc[:,'frame'] = sep['frame']
            tr.loc[:,'trackId'] = sep['trackId']
            tr.loc[:,'parentTrackId'] = sep['parentTrackId']
            tr.loc[:,'mtParTrk'] = sep['mtParTrk']
            
        new_table = new_table.append(tr)
          
    # For tracks that have mitosis parents, find new ID of their parents
    # Currently, we do not handle mitosis track loss daughter, 
    # though the mitosis duration is count by cell cycle resolver ResolveClass.R

    rel = {}  # relation lookup, parent:([daug], div frame)
    for l in np.unique(new_table['trackId']):
        tr = new_table[new_table['trackId']==l].copy()
        
        ori_par = list(tr['mtParTrk'])[0]
        if ori_par != 0:
            app = np.min(tr['frame'])
            search = new_table[new_table['ori_trackId']==ori_par]
            new_par = search.iloc[np.argmin(abs(search['frame']-app))]['trackId']
            new_table.loc[tr.index,'mtParTrk'] = new_par
            if new_par in rel.keys():
                rel[new_par][0].append(l)
            else:
                rel[new_par] = [[l], None]
                
    # break parent if the parent have only one daughter
    # also deduce mitosis frame
    for parent in list(rel.keys()):
        if len(rel[parent][0])==1:
            m_entry = np.min(new_table[new_table['trackId']==rel[parent][0][0]]['frame'])
            rel[parent][1] = m_entry
            # break parent at m_entry
            idx = new_table[(new_table['trackId']==parent) & (new_table['frame']>=m_entry)].index
            if len(idx):
                new_table.loc[idx, 'trackId'] = max_trackId + 1
                new_table.loc[idx, 'parentTrackId'] = parent
                new_table.loc[idx, 'mtParTrk'] = parent
                rel[parent][0].append(max_trackId+1)
                max_trackId += 1
                # for non-mitosis tracks that are daughters of the parent, point the first daughter
                # to new track
                idx = new_table[(new_table['parentTrackId']==parent) & (new_table['trackId'] != max_trackId)].index
                new_table.loc[idx, 'parentTrackId'] = max_trackId
            else:
                # if mitosis happens at the end of a parent track, any track following the parent will be mitosis daughter
                idx = new_table[(new_table['parentTrackId']==parent) & (new_table['frame'] >=m_entry)].index
                if len(idx):
                    rel[parent][0].append(np.unique(new_table.loc[idx, 'trackId'])[0])
                    new_table.loc[idx, 'mtParTrk'] = parent
         
        else:
            assert len(rel[parent][0])==2
            m_entry = np.max(new_table[new_table['trackId']==parent]['frame'])
            rel[parent][1] = m_entry
    
    for i in range(new_table.shape[0]):
        new_table.loc[i,'parentTrackId'] = np.max(((new_table['parentTrackId'][i]), new_table['mtParTrk'][i]))  # merge mitosis information in to parent
    
    return new_table, rel

def separate(frame_list, mtPar_list, ori_id, base):
    """For single track, separate into all complete tracks
    
    Args:
        frame_list: frames of exist, length equalt to label table
        mtPar_list: mitosis parent list, for solving mitosis relationship
        ori_id: original track ID
        base: base track ID, will assign new track ID sequentially from base + 1
    """
    
    trackId = [ori_id for i in range(len(frame_list))]
    parentTrackId = [0 for i in range(len(frame_list))]
    for i in range(1,len(frame_list)):
        if frame_list[i] - frame_list[i-1] != 1:
            trackId[i:] = [base + 1 for s in range(i,len(trackId))]
            parentTrackId[i:] = [trackId[i-1] for s in range(i, len(trackId))]
            mtPar_list[i:] = [0 for s in range(i, len(trackId))]
            base += 1
    rt = {'frame':frame_list, 'trackId':trackId, 'parentTrackId':parentTrackId, 'mtParTrk':mtPar_list}
    return rt, base

def save_seq(stack, out_dir, prefix, dig_num=3, dtype='uint16', base=0):
    """Save image stack and label sequentially
    
    Args:
        stack: nparray, THW
        out_dir: output directory
        prefix: prefix of single slice
        dig_num = digit number (3 -> 00x) for labeling image sequentially
        dtype: data type to save
        base: base number of the label (starting from)
    """
    if len(stack.shape)==4:
        stack = stack[:,:,:,0]

    for i in range(stack.shape[0]):
        fm = ("%0" + str(dig_num) + "d") % (i + base)
        name = os.path.join(out_dir, prefix + fm + '.tif')
        io.imsave(name, stack[i,:,:].astype(dtype))
    
    return