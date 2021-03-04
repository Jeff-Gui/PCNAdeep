#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:22:40 2021

@author: Yifan Gui
"""
import re, json, os, sys, getopt
import skimage.io as io
import skimage.measure as measure
from skimage.util import img_as_ubyte
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def json2mask(ip, out, height, width, out_phase=False):
    OUT_PHASE = out_phase
    PHASE_DIS = {"G1/G2":10, "S":50, "M":100, "E":10}
    PHASE_TRANS = {10:"G1/G2", 50:"S", 100:"M"}
    
    dt = pd.DataFrame()
    with open(ip,'r',encoding='utf8')as fp:
        j = json.load(fp)
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
            prop_dt = measure.regionprops_table(measure.label(img), intensity_image=img, properties=('centroid', 'mean_intensity'))
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

def mask2json(home, picture_home, outpath):
    out = {}
    region_tmp = {"shape_attributes":{"name":"polygon","all_points_x":[],"all_points_y":[]}, "region_attributes":{"phase":"G1/G2"}}

    #home = '/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/Mask_RCNN/datasets/pcna/processing/ground_truth_refined/'
    #picture_home = '/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/Mask_RCNN/datasets/pcna/processing/raw_contrast_refined/'
    imgs = os.listdir(home)
    for i in imgs:
        if re.search('.png',i):
            
            img = io.imread(os.path.join(home, i))
            #img = binary_erosion(binary_erosion(img.astype('bool')))
            img = img.astype('bool')
            tmp = {"filename":os.path.join(i),"size":img.size,"regions":[],"file_attributes":{}}
            regions = measure.regionprops(measure.label(img))
            for region in regions:
                
                if region.image.shape[0]<2 or region.image.shape[1]<2:
                    continue
                # register regions
                cur_tmp = copy.deepcopy(region_tmp)
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
        
    #with(open('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/Mask_RCNN/datasets/pcna/processing/refined_new.json', 'w', encoding='utf8')) as fp:
    with(open(outpath, 'w', encoding='utf8')) as fp:
        json.dump(out,fp)
    return

if __name__ == "__main__":
    argv = sys.argv[1:]
    rev = False
    try:
        opts, args = getopt.getopt(argv, "hri:m:o:", ["indir=", "mask=", "outdir="])
        # h: switch-type parameter, help
        # i: / o: parameter must with some values
        # m: mask dir
    except getopt.GetoptError:
        print('mask2jsonVIA2.py -i <inputfile> -o <outputfile> -m <mask>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('mask2jsonVIA2.py -i <inputfile> -o <outputfile> -m <mask>')
            sys.exit()
        elif opt == '-r':
            rev = True
        elif opt in ("-i", "--indir"):
            ip = arg
        elif opt in ("-o", "--outdir"):
            out = arg
        elif opt in ("-m", "--mask"):
            mask = arg
    
    if not rev:
        mask2json(mask, ip, out)
    else:
        json2mask(ip, out, 1200, 1200)
