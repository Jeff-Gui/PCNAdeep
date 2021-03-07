# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:03:20 2021

@author: Yifan Gui
"""

from skimage import io
import sys, getopt
import numpy as np
import skimage.exposure as exposure
import skimage.filters as filters

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:d:o:", ["indir=", "DICdir=","outdir="])
        # h: switch-type parameter, help
        # i: / o: parameter must with some values
        # m: mask dir
    except getopt.GetoptError:
        print('transFrame.py -i <input dir> -d <DIC dir> -o <output dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('transFrame.py -i <input dir> -d <DIC dir> -o <output dir>')
            sys.exit()
        elif opt in ("-i", "--indir"):
            ip = arg
        elif opt in ("-d", "--DICdir"):
            dic = arg
        elif opt in ('-o',"--outdir"):
            out = arg

    stack = io.imread(ip) # shape: thwc
    dic_img = io.imread(dic)
    print("Input shape: " + str(stack.shape))
    if len(stack.shape)<3:
        stack = np.expand_dims(stack, axis=0)
        dic_img = np.expand_dims(dic_img, axis=0)
    
    outs = []
    for f in range(stack.shape[0]):
        
        # rescale mCherry intensity
        fme = exposure.rescale_intensity(stack[f,:,:], in_range=tuple(np.percentile(stack[f,:,:], (2, 98))))
        dic_img[f,:,:] = exposure.rescale_intensity(dic_img[f,:,:], in_range=tuple(np.percentile(dic_img[f,:,:], (2, 98))))
        
        # perform otsu thresholding as the third channel
        fme_gau = filters.gaussian(fme, 2)
        thresh = filters.threshold_otsu(fme_gau)
        dst = (fme_gau > thresh-0.15) * 1.0
       
        # save two-channel image for downstream
        fme = fme/65535*255
        dic_slice = dic_img[f,:,:]/65535*255
        dst = dst*255
        fme = fme.astype('uint8')
        dic_slice = dic_slice.astype('uint8')
        dst = dst.astype('uint8')
        
        slice_list = []
        slice_list.append(fme)
        slice_list.append(fme)
        slice_list.append(dic_slice)
        #slice_list.append(dst)
        
        s = np.stack(slice_list, axis=2)
        outs.append(s)
    
    final_out = np.stack(outs, axis=0)
    print("Output shape: ", final_out.shape)
    io.imsave(out, final_out)


     