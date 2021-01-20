#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:41:27 2021

@author: jefft
"""
import os
import re
import random

def get_files(dir):
    files = []
    for filepath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if re.match('man_seg', filename):
                files.append(os.path.join(filepath, filename))
    return files

def resolve_cellLine(col, filepath):
    for cellline in col:
        if re.search(cellline, filepath):
            return cellline 
    return None

def load_DeepcellNucleus(root, subset='all'):
    col = ['3T3NIH', 'HEK293', 'HeLa', '264.7']
    import skimage.io as io
    import skimage.measure as measure
    import numpy as np
    import detectron2.structures as st

    fullset_flag = False
    if subset=='all':
        fullset_flag = True
    
    if fullset_flag:
        imgs = []
        for cellline in col:
            imgs += get_files(root +'/'+cellline)
    else:
        if subset not in col:
            raise ValueError
            return
        imgs = get_files(root + '/' + subset)
    random.shuffle(imgs)

    outs = []
    count = 0
    for i in imgs:
        count += 1
        #if i == '.DS_Store':
        #    continue
        image_info = re.search(r'(.*\/\d+)_GT\/SEG\/man_seg(\d+.tif)', i)
        image_fp = os.path.join(image_info.group(1), 't'+image_info.group(2))
        cell_line = resolve_cellLine(col, image_fp)
        uid = '-'.join([cell_line, image_info.group(1)[-3:], 't'+image_info.group(2)[0:3]])
        ori_image = io.imread(image_fp)
        ori_image = ori_image[:,:,0]
        masks = io.imread(i)
        height = ori_image.shape[0]
        width = ori_image.shape[1]
        
        # output initialize
        out = {'file_name':image_fp, 'height':height, 'width':width, 'image_id':uid, 'annotations':[]}
        regions = measure.regionprops(masks)
        for region in regions:
            if region.image.shape[0]<2 or region.image.shape[1]<2:
                continue
            # register regions
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
                edge = measure.approximate_polygon(ct, tolerance=0.4)
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
                edge = measure.approximate_polygon(ct, tolerance=0.4)
                for k in range(len(edge)): # swap x and y
                    edge[k] = [edge[k][1], edge[k][0]]   
                edge[:,0] += bbox[0]
                edge[:,1] += bbox[1]
                edge = list(edge.ravel())
            out['annotations'].append({'bbox':list(bbox), 'bbox_mode':st.BoxMode.XYXY_ABS, 'category_id':0, 'segmentation':[edge.copy()]})
        outs.append(out)
        if count%100==0:
            print("Loaded " + str(count) + " images.")
    return outs


def inspect_DeepCell_data(root,out_dir='../inspect/deepcell'):
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register("deepcell", lambda d:load_DeepcellNucleus(root, '3T3NIH'))
    metadata = MetadataCatalog.get("deepcell").set(thing_classes=['cell'])

    dataset_dicts = load_DeepcellNucleus(root, '3T3NIH')
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(out_dir + d["image_id"] + '.png', vis.get_image())

