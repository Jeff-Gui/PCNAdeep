# -*- coding: utf-8 -*-

import json
import os
import tarfile
import tempfile
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import skimage.io as io
from skimage.util import img_as_uint
from skimage.util import img_as_ubyte
import skimage.measure as measure
from pcnaDeep.tracker import track_mask


def relabel_trackID(label_table):
    """Relabel trackID in tracking table, starting from 1

    Args:
        label_table (pandas.DataFrame): track table
    """

    dic = {}
    ori = list(np.unique(label_table['trackId']))
    for i in range(1, len(ori) + 1):
        dic[ori[i - 1]] = i
    dic[0] = 0
    for i in range(label_table.shape[0]):
        label_table.loc[i, 'trackId'] = dic[label_table['trackId'][i]]
        label_table.loc[i, 'parentTrackId'] = dic[label_table['parentTrackId'][i]]
        label_table.loc[i, 'lineageId'] = dic[label_table['lineageId'][i]]

    return label_table


def label_by_track(mask, label_table):
    """Label objects in mask with track ID

    Args:
        mask (numpy.array): uint8 np array, output from main model
        label_table (pandas.DataFrame): track table
    
    Return:
        unit8/uint16 numpy array, dtype based on track count
    """

    assert mask.shape[0] == np.max(label_table['frame'] + 1)

    if np.max(label_table['trackId']) * 2 > 254:
        mask = mask.astype('uint16')

    for i in np.unique(label_table['frame']):
        sub_table = label_table[label_table['frame'] == i]
        sl = mask[i, :, :].copy()
        lbs = np.unique(sl).tolist()

        if lbs[-1] + 1 != len(lbs):
            raise ValueError('Mask is not continuously or wrongly labeled.')

        ori_labels = set(lbs) - {0}
        untracked = list(ori_labels - set(list(sub_table['continuous_label'])))
        #  remove untracked
        for j in untracked:
            sl[mask[i, :, :] == j] = 0
        #  update tracked
        for j in sub_table.index:
            sl[mask[i, :, :] == sub_table.loc[j, 'continuous_label']] = sub_table.loc[j, 'trackId']
        mask[i, :, :] = sl.copy()

    return mask


def get_lineage_dict(label_table):
    """Generate lineage dictionary in deepcell tracking format
    
    Args:
        label_table (pandas.DataFrame): table processed
    """

    out = {}
    for i in list(np.unique(label_table['trackId'])):
        i = int(i)
        sub_table = label_table[label_table['trackId'] == i]
        out[i] = {'capped': False, 'daughters': [], 'frame_div': None, 'frames': list(sub_table['frame']), 'label': i,
                  'parent': None}
        if list(sub_table['parentTrackId'])[0] != 0:
            out[i]['parent'] = list(sub_table['parentTrackId'])[0]

    return out


def get_lineage_txt(label_table):
    """Generate txt table in Cell Tracking Challenge format

    Args:
        label_table (pandas.DataFrame): table processed, should not has gaped tracks

    Return:
        pandas.DataFrame, remove index and col name before output.
    """

    dic = {'id': [], 'appear': [], 'disappear': [], 'parent': []}
    for i in np.unique(label_table['trackId']):
        sub = label_table[label_table['trackId'] == i]
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

    Saves raw, tracked, and lineage data into one trk_file.

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

    Load a trk/trks file.

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
            lineages = [{int(k): v for k, v in lineage.items()}]

    return {'lineages': lineages, 'X': raw, 'y': tracked}


def lineage_dic2txt(lineage_dic):
    """Convert deepcell .trk lineage format to CTC txt format

    Args:
        lineage_dic (list[dict]): [index:dict], extracted from deepcell .trk file
    """

    lineage_dic = lineage_dic[0]
    dic = {'id': [], 'appear': [], 'disappear': []}
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
        dic.loc[dic.index[dic['id'] == dg], 'parents'] = pars[dg]

    return dic


def break_track(label_table):
    """Break tracks in a lineage table into single tracks, where
    No gapped tracks allowed. All gap must be transferred into parent-daughter
    relationship.

    Args:
        label_table (pandas.DataFrame): table to process

    Algorithm:
        Rename raw parentTrackId to mtParTrk
        Initiate new parentTrackId column with 0
        Separate all tracks individually
    
    (In original lineage table, single track can be gapped, lineage only associates
     mitosis tracks, not gapped tracks.)
    
    Returns:
        processed tracked object table
    """

    # For parent track that have one daughter extrude into the parent frame, 
    #       e.g. parent: t1-10; daughter1: t8-20; daughter2: t11-20.
    # re-organize the track by triming parent and add to daughter,
    #       i.e. parent: t1-7; daughter1: t8-20; daughter2: t8-10, t11-20
    # If both daughter extrude, e.g. daughter2: t9-20, then trim parent directly
    # to t1-8. Since this indicages faulty track, warning shown

    for l in np.unique(label_table['trackId']):
        daugs = np.unique(label_table[label_table['parentTrackId'] == l]['trackId'])
        if len(daugs) == 2:
            daug1 = label_table[label_table['trackId'] == daugs[0]]['frame'].iloc[0]
            daug2 = label_table[label_table['trackId'] == daugs[1]]['frame'].iloc[0]
            par = label_table[label_table['trackId'] == l]
            par_frame = par['frame'].iloc[-1]
            if par_frame >= daug1 and par_frame >= daug2:
                raise UserWarning('Faluty mitosis, check parent: ' + str(l) +
                                  ', daughters: ' + str(daugs[0]) + '/' + str(daugs[1]))
                label_table.drop(par[(par['frame'] >= daug1) | (par['frame'] >= daug2)].index, inplace=True)
            elif par_frame >= daug1:
                # migrate par to daug2
                label_table.loc[par[par['frame'] >= daug1].index, 'trackId'] = daugs[1]
                label_table.loc[par[par['frame'] >= daug1].index, 'parentTrackId'] = l
            elif par_frame >= daug2:
                # migrate par to daug1
                label_table.loc[par[par['frame'] >= daug2].index, 'trackId'] = daugs[0]
                label_table.loc[par[par['frame'] >= daug2].index, 'parentTrackId'] = l

    label_table = label_table.sort_values(by=['trackId', 'frame'])

    # break tracks individually
    max_trackId = np.max(label_table['trackId'])
    label_table['mtParTrk'] = label_table['parentTrackId']
    label_table['parentTrackId'] = 0
    label_table['ori_trackId'] = label_table['trackId']
    new_table = pd.DataFrame()

    for l in np.unique(label_table['trackId']):
        tr = label_table[label_table['trackId'] == l].copy()

        if np.max(tr['frame']) - np.min(tr['frame']) + 1 != tr.shape[0]:
            sep, max_trackId = separate(list(tr['frame']).copy(), list(tr['mtParTrk']).copy(), l, base=max_trackId)
            tr.loc[:, 'frame'] = sep['frame']
            tr.loc[:, 'trackId'] = sep['trackId']
            tr.loc[:, 'parentTrackId'] = sep['parentTrackId']
            tr.loc[:, 'mtParTrk'] = sep['mtParTrk']

        new_table = new_table.append(tr)

        # For tracks that have mitosis parents, find new ID of their parents
    for l in np.unique(new_table['trackId']):
        tr = new_table[new_table['trackId'] == l].copy()

        ori_par = list(tr['mtParTrk'])[0]
        if ori_par != 0:
            app = np.min(tr['frame'])
            search = new_table[new_table['ori_trackId'] == ori_par]
            new_par = search.iloc[np.argmin(abs(search['frame'] - app))]['trackId']
            new_table.loc[tr.index, 'mtParTrk'] = new_par

    for i in range(new_table.shape[0]):
        new_table.loc[i, 'parentTrackId'] = np.max(
            ((new_table['parentTrackId'][i]), new_table['mtParTrk'][i]))  # merge mitosis information in to parent

    return new_table


def separate(frame_list, mtPar_list, ori_id, base):
    """For single track, separate into all complete tracks
    
    Args:
        frame_list (list): frames of exist, length equals to label table
        mtPar_list (list): mitosis parent list, for solving mitosis relationship
        ori_id (int): original track ID
        base (int): base track ID, will assign new track ID sequentially from base + 1
    """

    trackId = [ori_id for i in range(len(frame_list))]
    parentTrackId = [0 for i in range(len(frame_list))]
    for i in range(1, len(frame_list)):
        if frame_list[i] - frame_list[i - 1] != 1:
            trackId[i:] = [base + 1 for s in range(i, len(trackId))]
            parentTrackId[i:] = [trackId[i - 1] for s in range(i, len(trackId))]
            mtPar_list[i:] = [0 for s in range(i, len(trackId))]
            base += 1
    rt = {'frame': frame_list, 'trackId': trackId, 'parentTrackId': parentTrackId, 'mtParTrk': mtPar_list}
    return rt, base


def save_seq(stack, out_dir, prefix, dig_num=3, dtype='uint16', base=0, img_format='.tif', keep_chn=True, sep='-'):
    """Save image stack and label sequentially
    
    Args:
        stack (numpy array) : nparray, THW
        out_dir (str) : output directory
        prefix (str) : prefix of single slice, output will be prefix-000x.tif/png
            (see sep below for separater)
        dig_num (int) : digit number (3 -> 00x) for labeling image sequentially
        dtype (numpy.dtype) : data type to save, either 'uint8' or 'uint16'
        base (int) : base number of the label (starting from)
        img_format (str): image format, '.tif' or '.png', remind the dot
        keep_chn (bool): whether to keep full channel or not
        sep (str): separater between file name and id, default '-'
    """
    if len(stack.shape) == 4 and not keep_chn:
        stack = stack[:, :, :, 0]

    for i in range(stack.shape[0]):
        fm = ("%0" + str(dig_num) + "d") % (i + base)
        name = os.path.join(out_dir, prefix + sep + fm + img_format)
        if dtype == 'uint16':
            img = img_as_uint(stack[i, :])
        elif dtype == 'uint8':
            img = img_as_ubyte(stack[i, :])
        else:
            raise ValueError("Seq save only accepts uint8 or uint16 format.")
        io.imsave(name, img)

    return


def generate_calibanTrk(raw, mask, out_dir, dt_id, digit_num=3, displace=100, gap_fill=3, track=None,
                        render_phase=False):
    """Generate caliban .trk format for annotation
    """
    fm = ("%0" + str(digit_num) + "d") % dt_id
    if track is None:
        track, mask = track_mask(mask, displace=displace, gap_fill=gap_fill, render_phase=render_phase)
    track_new = relabel_trackID(track.copy())
    track_new = break_track(track_new.copy())
    tracked_mask = label_by_track(mask.copy(), track_new.copy())
    dic = get_lineage_dict(track_new.copy())
    if len(raw.shape) < 4:
        raw = np.expand_dims(raw, axis=3)
    save_trks(os.path.join(out_dir, fm + '.trk'), dic, raw, np.expand_dims(tracked_mask, axis=3))
    return track_new


def findM(gt_cls, direction='begin'):
    """Find M exit/entry from ground truth classification

    Args:
        gt_cls (list): list of classifications
        direction (str): begin/end, search M from
    """
    i = 0
    if direction == 'begin':
        if gt_cls[0] != 'M':
            return None
        while gt_cls[i] == 'M':
            i += 1
        return i - 1
    else:
        gt_cls = gt_cls[::-1]
        if gt_cls[0] != 'M':
            return None
        while gt_cls[i] == 'M':
            i += 1
        return -(i + 1)


def check_continuous_track(table):
    """Check if every track is continuous (no gap)
    """
    out = []
    for i in np.unique(table['trackId']).tolist():
        f = table[table['trackId'] == i]['frame'].tolist()
        if f[-1] - f[0] != len(f) - 1:
            out.append(i)
    return out


def mergeTrkAndTrack(trk_path, table_path, return_mask=False):
    """Merge ground truth .trk and tracked table
    
    Args:
        trk_path (str): path to deepcell-label .trk file
        table_path (str): path to tracked object table
        return_mask (bool): whether to return mask
    
    Returns:
        tracked object table, trackId, parentTrackId, mtParTrk
            corrected by ground truth
        mt_dic: standard table for initializing pcnaDeep.resolver.Resolver()
        mask: tracked mask
    """
    trk = load_trks(trk_path)
    lin = trk['lineages'][0]
    mask = trk['y'][:, :, :, 0]
    del (trk)
    table = pd.read_csv(table_path)
    new_table = pd.DataFrame()
    for i in range(mask.shape[0]):
        props = measure.regionprops_table(label_image=mask[i, :, :],
                                          intensity_image=mask[i, :, :],
                                          properties=('centroid', 'label'))
        props = pd.DataFrame(props)
        props['frame'] = i
        new_table = new_table.append(props)

    new_table['centroid-1'] = np.floor(new_table['centroid-1'])
    new_table['centroid-0'] = np.floor(new_table['centroid-0'])
    new_table['uid'] = range(new_table.shape[0])

    table['Center_of_the_object_0'] = np.floor(table['Center_of_the_object_0'])
    table['Center_of_the_object_1'] = np.floor(table['Center_of_the_object_1'])
    out = table.merge(new_table,
                      left_on=['frame', 'Center_of_the_object_0', 'Center_of_the_object_1'],
                      right_on=['frame', 'centroid-1', 'centroid-0'])

    ret = [i for i in new_table['uid'].tolist() if i not in out['uid'].tolist()]
    for j in ret:
        lb = new_table[new_table['uid'] == j]['label'].values[0]
        frame = new_table[new_table['uid'] == j]['frame'].values[0]
        mask[frame, mask[frame, :, :] == lb] = 0

    out['trackId'] = out['label']
    out['lineageId'] = out['lineageId']
    out['parentTrackId'] = 0
    out['mtParTrk'] = 0

    out = out.drop(columns=['centroid-0', 'centroid-1', 'uid', 'label'])
    out.sort_values(by=['trackId', 'frame'], inplace=True)
    e = check_continuous_track(out)
    if e:
        raise ValueError('Gapped tracks found: ' + str(e))

    # resolve lineage and mt_dic
    # deepcell-label does not automatically assign parent ID to daughter if 
    # the parent ID has been changed. However, it does assign original daughter 
    # ID to a changed parent. Therefore, we deduce relationship from daughter 
    # field under parent information, and ignore all daughters that do not exist.
    mt_dic = {}
    # if a daughter appears as 'G1/G2' then m_exit is not accurate
    imprecise_m = []
    all_tracks = out['trackId'].tolist()
    for i in list(lin.keys()):
        if lin[i]['daughters']:
            daugs = lin[i]['daughters']
            for k in daugs:
                if k not in all_tracks:
                    daugs.remove(k)
            if len(daugs) == 1:
                out.loc[out['trackId'] == daugs[0], 'parentTrackId'] = i
                out.loc[out['lineageId'] == daugs[0], 'lineageId'] = i
            else:
                sub_par = out.loc[out['trackId'] == i]
                m_entry = findM(sub_par['predicted_class'].tolist(), 'end')
                if m_entry is None:
                    warnings.warn('Mitosis parent no M class, check ' + str(i))
                    continue
                sub_daug1 = out.loc[out['trackId'] == daugs[0]]
                sub_daug2 = out.loc[out['trackId'] == daugs[1]]

                out.loc[out['trackId'] == daugs[0], 'parentTrackId'] = i
                out.loc[out['trackId'] == daugs[0], 'mtParTrk'] = i
                out.loc[out['lineageId'] == daugs[0], 'lineageId'] = i
                out.loc[out['trackId'] == daugs[1], 'parentTrackId'] = i
                out.loc[out['trackId'] == daugs[1], 'mtParTrk'] = i
                out.loc[out['lineageId'] == daugs[1], 'lineageId'] = i

                m_exit1 = findM(sub_daug1['predicted_class'].tolist(), 'begin')
                m_exit2 = findM(sub_daug2['predicted_class'].tolist(), 'begin')

                if m_exit1 is None:
                    m_exit1 = 0
                    imprecise_m.append(daugs[0])
                if m_exit2 is None:
                    m_exit2 = 0
                    imprecise_m.append(daugs[1])

                mt_dic[i] = {'div': sub_par['frame'].iloc[m_entry],
                             'daug': {daugs[0]: {'m_exit': sub_daug1['frame'].iloc[m_exit1], 'dist': 0},
                                      daugs[1]: {'m_exit': sub_daug2['frame'].iloc[m_exit2], 'dist': 0}}}

    if return_mask:
        return out, mask, mt_dic, imprecise_m
    else:
        return out, mt_dic, imprecise_m
