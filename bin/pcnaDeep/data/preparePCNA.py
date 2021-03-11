# -*- coding: utf-8 -*-

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
    metadata = MetadataCatalog.get("pcna").set(thing_classes=['G1/G2', 'S', 'M', 'E'])

    dataset_dicts = load_PCNA_from_json(json_path, image_path)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(out_dir + d["image_id"] + '.png', vis.get_image())

