import os
import cv2
import torch
import numpy as np
from .demo import Predictor
from .aligh import CENTER_MODEL

def detect_line(im):
    """

    Args:
        im (np.array): input image 

    Returns:
        dict: 
            {
                'address_line_1':   [left,top,right,bottom]
                'address_line_2':   [left,top,right,bottom]
                'birthday':         [left,top,right,bottom]
                'hometown_line_1':  [left,top,right,bottom]
                'hometown_line_2':  [left,top,right,bottom]
                'id':               [left,top,right,bottom]
                'name':             [left,top,right,bottom]
                'nation':           [left,top,right,bottom]
                'sex':              [left,top,right,bottom]
            }

    """

    predictor = Predictor()
    _, res = predictor.inference(im)
    
    center_model = CENTER_MODEL()
    img_aligh, point = center_model.aligh(im, res)
    if not point:
        return img_aligh, False
    return img_aligh, True