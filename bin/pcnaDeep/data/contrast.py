# -*- coding: utf-8 -*-
"""
@author: Yifan Gui
"""

import skimage.exposure as exposure
from skimage.util import img_as_float32
import numpy as np


def enhance_contrast(img):
    """Enhance contrast by rescaling
    
    Args:
        img: input, should be single slice
    Return:
        ndarray
    """
    return exposure.rescale_intensity(img, in_range=tuple(np.percentile(img, (0.5, 99.5))))


def enhance_contrast_stack(stack):
    """Enhance contrast of a stack
    """

    for i in range(stack.shape[0]):
        stack[i,:] = enhance_contrast(stack[i,:])
    return stack


def normalize(image, epsilon=1e-07):
    """Normalize image data by dividing by the maximum pixel value

    *** Adapted from deepcell_toolbox.processing.normalize()

    Args:
        image (numpy.array): numpy array of image data
        epsilon (float): fuzz factor used in numeric expressions.

    Returns:
        numpy.array: normalized image data, in float32
    """
    
    image = img_as_float32(image)

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = (img - img.mean()) / (img.std() + epsilon)
            image[batch, ..., channel] = normal_image
    return image
