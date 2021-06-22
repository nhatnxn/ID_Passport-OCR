import os
import cv2
import torch
import numpy as np
from .detector import Predictor
from .detector import detect_box

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
    
    info = detect_box[res]
    
    return info