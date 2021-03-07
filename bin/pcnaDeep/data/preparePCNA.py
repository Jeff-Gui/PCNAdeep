#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:41:27 2021

@author: jefft
"""
import os
import re
import random
import json

def get_files(dir):
    files = []
    for filepath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if re.search('_mask', filename):
                files.append(os.path.join(filepath, filename))
    return files

def load_PCNA(root):
    """
    File structure:
        |------Root-------
            |____Image1______
                |__image.png__
                |__mask.png___
            |____Image2______
            ...

    """

    import skimage.io as io
    import skimage.measure as measure
    import numpy as np
    import detectron2.structures as st

    imgs = get_files(root)
    random.shuffle(imgs)

    outs = []
    count = 0
    for i in imgs:
        count += 1
        #if i == '.DS_Store':
        #    continue
        image_info = re.search(r'(.*\/)(\d+)_mask.png', i)
        image_fp = image_info.group(1)+image_info.group(2)+'.png'
        uid = image_info.group(2)
        ori_image = io.imread(image_fp)
        masks = io.imread(i)
        height = ori_image.shape[0]
        width = ori_image.shape[1]
        
        # output initialize
        out = {'file_name':image_fp, 'height':height, 'width':width, 'image_id':uid, 'annotations':[]}
        regions = measure.regionprops(measure.label(masks))
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

def load_PCNA_from_json(json_path, image_path, width=1200, height=1200):
    import numpy as np
    import detectron2.structures as st
    import math

    cc_stageDic = {"G1/G2":0, "S":1, "M":2, "E":3}

    with open(json_path,'r', encoding='utf8') as fp:
        ann = json.load(fp)
    count = 1
    outs = []
    for key in list(ann.keys()):
        ann_img = ann[key]
        fn = ann_img['filename']
        regions = ann_img['regions']
        id = re.search('(.+)\.\w*',fn).group(1)
        out = {'file_name':os.path.join(image_path, fn), 'height':height, 'width':width, 'image_id':id, 'annotations':[]}

        for r in regions:
            phase = r['region_attributes']['phase']
            shape = r['shape_attributes']
            x = shape['all_points_x']
            y = shape['all_points_y']
            bbox = [math.floor(np.min(x)), math.floor(np.min(y)), math.ceil(np.max(x)), math.ceil(np.min(y))]
            edge = [0 for i in range(len(x)+len(y))]
            edge[::2] = x
            edge[1::2] = y
            # register output
            out['annotations'].append({'bbox':bbox, 'bbox_mode':st.BoxMode.XYXY_ABS, 'category_id':cc_stageDic[phase], 'segmentation':[edge.copy()]})
        
        outs.append(out)
        if count%100==0:
            print("Loaded " + str(count) + " images.")
        count += 1
    return outs

def load_PCNAs_json(json_paths, image_paths):
    """Load multiple training dataset
    """
    import random
    assert len(json_paths) == len(image_paths)
    out = []
    for i in range(len(json_paths)):
        print('Loading dataset from: '+image_paths[i])
        dic = load_PCNA_from_json(json_paths[i], image_paths[i])
        out += dic
    random.shuffle(out)
    return out


def inspect_PCNA_simple_data(root,out_dir='../inspect/pcna'):
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register("pcna", lambda d:load_PCNA(root))
    metadata = MetadataCatalog.get("pcna").set(thing_classes=['cell'])

    dataset_dicts = load_PCNA(root)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(out_dir + d["image_id"] + '.png', vis.get_image())

def inspect_PCNA_data(json_path, image_path, out_dir='../inspect/pcna'):
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register("pcna", lambda d:load_PCNA_from_json(json_path, image_path))
    metadata = MetadataCatalog.get("pcna").set(thing_classes=['G1/G2', 'S', 'M'])

    dataset_dicts = load_PCNA_from_json(json_path, image_path)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(out_dir + d["image_id"] + '.png', vis.get_image())

