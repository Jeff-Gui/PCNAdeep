# -*- coding: utf-8 -*-

import os
import re
import random
import json
import numpy as np
import detectron2.structures as st
import math
import skimage.io as io
import skimage.exposure as exposure
from skimage.util import img_as_ubyte


def load_PCNA_from_json(json_path, image_path, width=1200, height=1200):
    """Load PCNA training data and ground truth from json.

    Args:
        json_path (str): path to .json ground truth in VIA2 format.
        image_path (str): path to raw image.
        width (int): width of the image.
        height (int): height of the image.

    """
    cc_stageDic = {"G1/G2": 0, "S": 1, "M": 2, "E": 3}

    with open(json_path, 'r', encoding='utf8') as fp:
        ann = json.load(fp)
    count = 1
    outs = []
    for key in list(ann.keys()):
        ann_img = ann[key]
        fn = ann_img['filename']
        regions = ann_img['regions']
        id = re.search('(.+)\.\w*', fn).group(1)
        out = {'file_name': os.path.join(image_path, fn), 'height': height, 'width': width, 'image_id': id,
               'annotations': []}

        for r in regions:
            phase = r['region_attributes']['phase']
            shape = r['shape_attributes']
            x = shape['all_points_x']
            y = shape['all_points_y']
            bbox = [math.floor(np.min(x)), math.floor(np.min(y)), math.ceil(np.max(x)), math.ceil(np.min(y))]
            edge = [0 for i in range(len(x) + len(y))]
            edge[::2] = x
            edge[1::2] = y
            # register output
            out['annotations'].append(
                {'bbox': bbox, 'bbox_mode': st.BoxMode.XYXY_ABS, 'category_id': cc_stageDic[phase],
                 'segmentation': [edge.copy()]})

        outs.append(out)
        if count % 100 == 0:
            print("Loaded " + str(count) + " images.")
        count += 1
    return outs


def load_PCNAs_json(json_paths, image_paths):
    """Load multiple training dataset.
    """
    import random
    assert len(json_paths) == len(image_paths)
    out = []
    for i in range(len(json_paths)):
        print('Loading dataset from: ' + image_paths[i])
        dic = load_PCNA_from_json(json_paths[i], image_paths[i])
        out += dic
    random.shuffle(out)
    return out


def read_PCNA_training(filename, rescale_sat=0.2, gamma=0.5, base_pcna='mcy', base_dic='dic'):
    """Read PCNA/DIC image, make composite and perform pre-processing

    Args:
        filename (str): File name conjugated with parent directory.
        rescale_sat (float): pixel saturation for rescaling intensity (0~100).
        gamma (float): gamma correction factor (>0) Adjust PCNA only.
        base_pcna (str): folder name storing pcna channel.
        base_dic (str): folder name storing bright field channel.
    """
    if sat < 0 or sat > 100:
        raise ValueError('Saturated pixel should not be negative or exceeds 100.')
    if gamma <= 0:
        raise ValueError('Gamma factor should not be negative or zero.')

    img_name = os.path.basename(filename)
    pcna_name = os.path.join(os.path.dirname(filename), base_pcna, img_name)
    dic_name = os.path.join(os.path.dirname(filename), base_dic, img_name)
    pcna = io.imread(pcna_name)
    dic = io.imread(dic_name)

    if pcna.dtype != np.dtype('uint16') or dic.dtype != np.dtype('uint16'):
        raise ValueError('Input image must be in uint16 format.')

    pcna = exposure.adjust_gamma(pcna, gamma=gamma)
    rg = (sat, 100-sat)
    pcna = exposure.rescale_intensity(pcna, in_range=tuple(np.percentile(pcna, rg)))
    dic = exposure.rescale_intensity(dic, in_range=tuple(np.percentile(dic, rg)))

    pcna = img_as_ubyte(pcna)
    dic = img_as_ubyte(dic)
    slice_list = [pcna, pcna, dic]
    image = np.stack(slice_list, axis=2)

    return image


def inspect_PCNA_data(json_path, image_path, out_dir='../../../inspect/test'):
    """Inspect PCNA training data.
    """
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    prefix = os.path.basename(image_path)
    DatasetCatalog.register("pcna", lambda d: load_PCNA_from_json(json_path, image_path))
    metadata = MetadataCatalog.get("pcna").set(thing_classes=['G1/G2', 'S', 'M', 'E'])

    dataset_dicts = load_PCNA_from_json(json_path, image_path)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(out_dir, prefix + d["image_id"] + '.png'), vis.get_image())
