import skimage.exposure as exposure
import skimage.filters as filters
import numpy as np


def feature_extraction(mcy, dic, offset=0.1):
    """
    Args:
        mcy: mCherry channel PCNA signal, should be uint16, no contrast enhanced, HW
        dic: DIC channel, no contrast enhanced, HW, uint16
        offset: how much lower threshold is during masking (to ensure all foreground is in the mask), [0,1)
    
    Return:
        HWC image, channel: mcy, dic, prior_mask, uint8
    """

    fme = exposure.rescale_intensity(mcy, in_range=tuple(np.percentile(mcy, (2, 98))))
    dic_img = exposure.rescale_intensity(dic, in_range=tuple(np.percentile(dic, (2, 98))))
    # perform otsu thresholding as the third channel
    fme_gau = filters.gaussian(fme, 2)
    thresh = filters.threshold_otsu(fme_gau)
    dst = (fme_gau > thresh-offset) * 1.0
    fme = fme/65535*255
    dic_slice = dic_img/65535*255
    dst = dst*255
    fme = fme.astype('uint8')
    dic_slice = dic_slice.astype('uint8')
    dst = dst.astype('uint8')
    s = np.stack([fme, dic_slice, dst], axis=2)
    return s