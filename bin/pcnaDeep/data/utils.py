# -*- coding: utf-8 -*-

import re, json, os
import skimage.io as io
import skimage.measure as measure
import skimage.exposure as exposure
from skimage.util import img_as_ubyte
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def json2mask(ip, out, height, width, label_phase=False):
    """Draw mask according to VIA2 annotation and summarize information

    Args:
        in_dir (str): input directory of the json file
        out_dir (str): output directory of the image and summary table
        height (int): image height
        width (int): image width
        label_phase (bool): whether to label the mask with values corresponding to cell cycle classification or not. 
            If true, will label as the following values: 'G1/G2':10, 'S':50, 'M':100;
            If false, will output binary masks

    Outputs:
        .png files of object masks
        .csv file of object information in json file
    """

    OUT_PHASE = label_phase
    PHASE_DIS = {"G1/G2":10, "S":50, "M":100, "E":10}
    PHASE_TRANS = {10:"G1/G2", 50:"S", 100:"M"}
    
    dt = pd.DataFrame()
    with open(ip,'r',encoding='utf8')as fp:
        j = json.load(fp)
        if '_via_img_metadata' in list(j.keys()):
            j = j['_via_img_metadata']
        for key in list(j.keys()):
            img = Image.new('L',(height,width))
            dic = j[key]
            objs = dic['regions'] # containing all object areas
            draw = ImageDraw.Draw(img)
            for o in objs:
                x = o['shape_attributes']['all_points_x']
                y = o['shape_attributes']['all_points_y']
                xys = [0 for i in range(len(x)+len(y))]
                xys[::2] = x
                xys[1::2] = y
                phase = o['region_attributes']['phase']
                draw.polygon(xys, fill=PHASE_DIS[phase], outline=0)
            img = np.array(img)
            prop_dt = measure.regionprops_table(measure.label(img, connectivity=1), intensity_image=img, properties=('centroid', 'mean_intensity'))
            prop_dt = pd.DataFrame(prop_dt)
            prop_dt['mean_intensity'] = list(map(lambda x:PHASE_TRANS[x], prop_dt['mean_intensity']))
            prop_dt.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'phase']
            frame = int(re.search('.*-(\d+).png', key).group(1)) - 1  # frame begins from 0, file name begins from 1
            prop_dt['frame'] = frame
            prop_dt['Probability of G1/G2'] = list(map(lambda x:int(x=='G1/G2'), prop_dt['phase']))
            prop_dt['Probability of S'] = list(map(lambda x:int(x=='S'), prop_dt['phase']))
            prop_dt['Probability of M'] = list(map(lambda x:int(x=='M'), prop_dt['phase']))
            prop_dt['Center_of_the_object_0'] = np.round(prop_dt['Center_of_the_object_0'])
            prop_dt['Center_of_the_object_1'] = np.round(prop_dt['Center_of_the_object_1'])
            dt = dt.append(prop_dt)
            if not OUT_PHASE:
                img = img_as_ubyte(img.astype('bool'))
            io.imsave(os.path.join(out, dic['filename']), img)
        dt = dt[['Center_of_the_object_0','Center_of_the_object_1','frame','phase','Probability of G1/G2', 'Probability of S', 'Probability of M']]
        dt.to_csv(os.path.join(out, 'cls.csv'), index=0)
    return


def mask2json(in_dir, out_dir, phase_labeled=False, phase_dic={10:"G1/G2", 50:"S", 100:"M"}, prefix='object_info'):
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
    region_tmp = {"shape_attributes":{"name":"polygon","all_points_x":[],"all_points_y":[]}, "region_attributes":{"phase":"G1/G2"}}

    imgs = os.listdir(in_dir)
    for i in imgs:
        if re.search('.png',i):
            
            img = io.imread(os.path.join(in_dir, i))
            #img = binary_erosion(binary_erosion(img.astype('bool')))
            img = img.astype('bool')
            tmp = {"filename":os.path.join(i),"size":img.size,"regions":[],"file_attributes":{}}
            regions = measure.regionprops(measure.label(img, connectivity=1), img)
            for region in regions:
                if region.image.shape[0]<2 or region.image.shape[1]<2:
                    continue
                # register regions
                cur_tmp = copy.deepcopy(region_tmp)
                if phase_labeled:
                    cur_tmp['region_attributes']['phase'] = phase_dic[int(region.mean_intensity)]
                bbox = list(region.bbox)
                bbox[0],bbox[1] = bbox[1], bbox[0] # swap x and y
                bbox[2],bbox[3] = bbox[3], bbox[2]
                ct = measure.find_contours(region.image, 0.5)
                if len(ct)<1:
                    continue
                ct = ct[0]
                if ct[0][0] != ct[-1][0] or ct[0][1] != ct[-1][1]:
                    # non connected
                    ct_image = np.zeros((bbox[3]-bbox[1]+2, bbox[2]-bbox[0]+2))
                    ct_image[1:-1,1:-1] = region.image.copy()
                    ct = measure.find_contours(ct_image, 0.5)[0]
                    # edge = measure.approximate_polygon(ct, tolerance=0.001)
                    edge = ct
                    for k in range(len(edge)): # swap x and y
                        x = edge[k][0] - 1
                        if x<0: 
                            x=0
                        elif x>region.image.shape[0]-1:
                            x = region.image.shape[0]-1
                        y = edge[k][1] - 1
                        if y<0:
                            y=0
                        elif y> region.image.shape[1]-1:
                            y = region.image.shape[1]-1
                        edge[k] = [y,x]
                    edge = edge.tolist()
                    elements = list(map(lambda x:tuple(x), edge))
                    edge = list(set(elements))
                    edge.sort(key=elements.index)
                    edge = np.array(edge)
                    edge[:,0] += bbox[0]
                    edge[:,1] += bbox[1]
                    edge = list(edge.ravel())
                    edge += edge[0:2]
                else:
                    # edge = measure.approximate_polygon(ct, tolerance=0.4)
                    edge = ct
                    for k in range(len(edge)): # swap x and y
                        edge[k] = [edge[k][1], edge[k][0]]   
                    edge[:,0] += bbox[0]
                    edge[:,1] += bbox[1]
                    edge = list(edge.ravel())
                cur_tmp['shape_attributes']['all_points_x'] = edge[::2]
                cur_tmp['shape_attributes']['all_points_y'] = edge[1::2]
                tmp['regions'].append(cur_tmp)
            out[i] = tmp
        
    with(open(os.path.join(out_dir, prefix+'.json'), 'w', encoding='utf8')) as fp:
        json.dump(out,fp)
    return


def getModelInput(pcna, dic):
    """Generate pcna-mScarlet and DIC channel to RGB format for model prediction

    Args:
        pcna (numpy.array): uint16 PCNA-mScarlet image stack (T*H*W)
        dic (numpy.array): uint16 DIC or phase contrast image stack

    Returns:
        (numpy.array): uint8 composite image (T*H*W*C)
    """
    stack = pcna
    dic_img = dic
    print("Input shape: " + str(stack.shape))
    if len(stack.shape) < 3:
        stack = np.expand_dims(stack, axis=0)
        dic_img = np.expand_dims(dic_img, axis=0)

    outs = []
    for f in range(stack.shape[0]):
        # rescale mCherry intensity
        fme = exposure.rescale_intensity(stack[f, :, :], in_range=tuple(np.percentile(stack[f, :, :], (2, 98))))
        dic_img[f, :, :] = exposure.rescale_intensity(dic_img[f, :, :],
                                                      in_range=tuple(np.percentile(dic_img[f, :, :], (2, 98))))

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
    return outs
