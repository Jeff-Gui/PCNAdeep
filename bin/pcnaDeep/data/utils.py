# -*- coding: utf-8 -*-

import copy
import json
import os
import re
import numpy as np
import pandas as pd
import skimage.exposure as exposure
import skimage.io as io
import skimage.measure as measure
from PIL import Image, ImageDraw
from skimage.util import img_as_ubyte


def json2mask(ip, out, height, width, label_phase=False, mask_only=False):
    """Draw mask according to VIA2 annotation and summarize information

    Args:
        ip (str): input directory of the json file
        out (str): output directory of the image and summary table
        height (int): image height
        width (int): image width
        label_phase (bool): whether to label the mask with values corresponding to cell cycle classification or not. 
            If true, will label as the following values: 'G1/G2':10, 'S':50, 'M':100;
            If false, will output binary masks
        mask_only (bool): whether to suppress file output and return mask only

    Outputs:
        .png files of object masks
    """

    OUT_PHASE = label_phase
    PHASE_DIS = {"G1/G2": 10, "S": 50, "M": 100, "E": 200}
    stack = []
    with open(ip, 'r', encoding='utf8')as fp:
        j = json.load(fp)
        if '_via_img_metadata' in list(j.keys()):
            j = j['_via_img_metadata']
        for key in list(j.keys()):
            img = Image.new('L', (height, width))
            dic = j[key]
            objs = dic['regions']  # containing all object areas
            draw = ImageDraw.Draw(img)
            for o in objs:
                x = o['shape_attributes']['all_points_x']
                y = o['shape_attributes']['all_points_y']
                xys = [0 for i in range(len(x) + len(y))]
                xys[::2] = x
                xys[1::2] = y
                phase = o['region_attributes']['phase']
                draw.polygon(xys, fill=PHASE_DIS[phase], outline=0)
            img = np.array(img)

            if not OUT_PHASE:
                img = img_as_ubyte(img.astype('bool'))
            if mask_only:
                stack.append(img)
            else:
                io.imsave(os.path.join(out, dic['filename']), img)
        if mask_only:
            return np.stack(stack, axis=0)

    return


def mask2json(in_dir, out_dir, phase_labeled=False, phase_dic={10: "G1/G2", 50: "S", 100: "M", 200: 'E'},
              prefix='object_info'):
    """Generate VIA2-readable json file from masks

    Args:
        in_dir (str): input directory of mask slices in .png format. Stack input is not implemented.
        out_dir (str): output directory for .json output
        phase_labeled (bool): whether cell cycle phase has already been labeled. 
            If true, a phase_dic variable should be supplied to resolve phase information.
        phase_dic (dic): lookup dictionary of cell cycle phase labeling on the mask.
        prefix (str): prefix of .json output.
    
    Outputs:
        prefix.json in VIA2 format. Note the output is not a VIA2 project, so default image directory
            must be set for the first time of labeling.
    """
    out = {}
    region_tmp = {"shape_attributes": {"name": "polygon", "all_points_x": [], "all_points_y": []},
                  "region_attributes": {"phase": "G1/G2"}}

    imgs = os.listdir(in_dir)
    for i in imgs:
        if re.search('.png', i):

            img = io.imread(os.path.join(in_dir, i))
            # img = binary_erosion(binary_erosion(img.astype('bool')))
            img = img.astype('bool')
            tmp = {"filename": os.path.join(i), "size": img.size, "regions": [], "file_attributes": {}}
            regions = measure.regionprops(measure.label(img, connectivity=1), img)
            for region in regions:
                if region.image.shape[0] < 2 or region.image.shape[1] < 2:
                    continue
                # register regions
                cur_tmp = copy.deepcopy(region_tmp)
                if phase_labeled:
                    cur_tmp['region_attributes']['phase'] = phase_dic[int(region.mean_intensity)]
                bbox = list(region.bbox)
                bbox[0], bbox[1] = bbox[1], bbox[0]  # swap x and y
                bbox[2], bbox[3] = bbox[3], bbox[2]
                ct = measure.find_contours(region.image, 0.5)
                if len(ct) < 1:
                    continue
                ct = ct[0]
                if ct[0][0] != ct[-1][0] or ct[0][1] != ct[-1][1]:
                    # non connected
                    ct_image = np.zeros((bbox[3] - bbox[1] + 2, bbox[2] - bbox[0] + 2))
                    ct_image[1:-1, 1:-1] = region.image.copy()
                    ct = measure.find_contours(ct_image, 0.5)[0]
                    # edge = measure.approximate_polygon(ct, tolerance=0.001)
                    edge = ct
                    for k in range(len(edge)):  # swap x and y
                        x = edge[k][0] - 1
                        if x < 0:
                            x = 0
                        elif x > region.image.shape[0] - 1:
                            x = region.image.shape[0] - 1
                        y = edge[k][1] - 1
                        if y < 0:
                            y = 0
                        elif y > region.image.shape[1] - 1:
                            y = region.image.shape[1] - 1
                        edge[k] = [y, x]
                    edge = edge.tolist()
                    elements = list(map(lambda x: tuple(x), edge))
                    edge = list(set(elements))
                    edge.sort(key=elements.index)
                    edge = np.array(edge)
                    edge[:, 0] += bbox[0]
                    edge[:, 1] += bbox[1]
                    edge = list(edge.ravel())
                    edge += edge[0:2]
                else:
                    # edge = measure.approximate_polygon(ct, tolerance=0.4)
                    edge = ct
                    for k in range(len(edge)):  # swap x and y
                        edge[k] = [edge[k][1], edge[k][0]]
                    edge[:, 0] += bbox[0]
                    edge[:, 1] += bbox[1]
                    edge = list(edge.ravel())
                cur_tmp['shape_attributes']['all_points_x'] = edge[::2]
                cur_tmp['shape_attributes']['all_points_y'] = edge[1::2]
                tmp['regions'].append(cur_tmp)
            out[i] = tmp

    with(open(os.path.join(out_dir, prefix + '.json'), 'w', encoding='utf8')) as fp:
        json.dump(out, fp)
    return


def getDetectInput(pcna, dic, sat=2):
    """Generate pcna-mScarlet and DIC channel to RGB format for detectron2 model prediction

    Args:
        pcna (numpy.ndarray): uint16 PCNA-mScarlet image stack (T*H*W)
        dic (numpy.ndarray): uint16 DIC or phase contrast image stack
        sat (int): percent saturation, 0~100, default 0.

    Returns:
        (numpy.ndarray): uint8 composite image (T*H*W*C)
    """
    stack = pcna
    dic_img = dic
    if stack.dtype != np.dtype('uint16') or dic_img.dtype != np.dtype('uint16'):
        raise ValueError('Input image must be in uint16 format.')
    if sat < 0 or sat > 100:
        raise ValueError('Saturated pixel should not be negative or exceeds 100')

    print("Input shape: " + str(stack.shape))
    if len(stack.shape) < 3:
        stack = np.expand_dims(stack, axis=0)
        dic_img = np.expand_dims(dic_img, axis=0)

    outs = []
    sat = sat//2
    rg = (sat, 100-sat)
    for f in range(stack.shape[0]):
        # rescale mCherry intensity
        fme = exposure.rescale_intensity(stack[f, :, :], in_range=tuple(np.percentile(stack[f, :, :], rg)))
        dic_img[f, :, :] = exposure.rescale_intensity(dic_img[f, :, :],
                                                      in_range=tuple(np.percentile(dic_img[f, :, :], rg)))

        # save two-channel image for downstream
        fme = fme / 65535 * 255
        dic_slice = dic_img[f, :, :] / 65535 * 255
        fme = fme.astype('uint8')
        dic_slice = dic_slice.astype('uint8')

        slice_list = [fme, fme, dic_slice]

        s = np.stack(slice_list, axis=2)
        outs.append(s)

    final_out = np.stack(outs, axis=0)
    print("Output shape: ", final_out.shape)
    return final_out


def retrieve(table, mask, image, rp_fields=[], funcs=[]):
    """Retrieve extra skimage.measure.regionprops fields of every object;
        Or apply customized functions to extract features form the masked object.
        
    Args:
        table (pandas.DataFrame): object table tracked or untracked, 
            should have 2 fields:
            1. frame: time location; 
            2. continuous label: region label on mask
        mask (numpy.ndarray): labeled mask corresponding to table
        image (numpy.ndarray): intensity image, only the first channel allowed
        rp_fields (list(str)): skimage.measure.regionpprops allowed fields
        funcs (list(function)): customized function that outputs one value from
            an array input
            
    Returns:
        labeled object table with additional columns
    """
    track = table
    if rp_fields == [] and funcs == []:
        return track

    new_track = pd.DataFrame()
    track = track.sort_values(by=['frame', 'continuous_label'])
    for f in np.unique(track['frame']).tolist():
        sl = mask[f, :, :]
        img = image[f, :, :]
        sub = track[track['frame'] == f]

        if rp_fields:
            if 'label' not in rp_fields:
                rp_fields.append('label')
            props = pd.DataFrame(measure.regionprops_table(sl, img, properties=tuple(rp_fields)))
            new = pd.merge(sub, props, left_on='continuous_label', right_on='label')
            new = new.drop(['label'], axis=1)

        if funcs:
            p = measure.regionprops(sl, img)
            out = {'label': []}
            for fn in funcs:
                out[fn.__name__] = []
            for i in p:
                out['label'].append(i.label)
                i_img = img.copy()
                i_img[sl != i.label] = 0
                for fn in funcs:
                    out[fn.__name__].append(fn(i_img))
            new2 = pd.DataFrame(out)
            if rp_fields:
                new2 = pd.merge(new, new2, left_on='continuous_label', right_on='label')
                new2 = new2.drop(['label'], axis=1)
            else:
                new2 = pd.merge(sub, new2, left_on='continuous_label', right_on='label')

        if rp_fields:
            if funcs:
                new_track = new_track.append(new2)
            else:
                new_track = new_track.append(new)
        elif funcs:
            new_track = new_track.append(new2)

    return new_track


def mt_dic2mt_lookup(mt_dic):
    """Convert mt_dic to mitosis lookup
    
    Args:
        mt_dic (dict): standard mitosis info dictionary in pcnaDeep
    
    Returns:
        mt_lookup (pd.DataFrame): mitosis lookup table with 3 columns:
            trackA (int) | trackB (int) | Mitosis? (int, 0/1)
    """
    out = {'par': [], 'daug': [], 'mitosis': []}
    for i in list(mt_dic.keys()):
        for j in list(mt_dic[i]['daug'].keys()):
            out['par'].append(i)
            out['daug'].append(j)
            out['mitosis'].append(1)
    return pd.DataFrame(out)


def get_outlier(array, col_ids=None):
    """Get outlier index in an array, specify target column
    
    Args:
        array (numpy.ndarray): original array
        col_ids ([int]): target columns to remove outliers. Default all
        
    Returns:
        index of row containing at least one outlier
    """
    
    if col_ids is None:
        col_ids = list(range(array.shape[1]))
    
    idx = []
    for c in col_ids:
        col = array[:,c]
        idx.extend(list(np.where(np.abs(col - np.mean(col)) > 3 * np.std(col))[0]))
    
    idx = list(set(idx))
    idx.sort()
    return idx

