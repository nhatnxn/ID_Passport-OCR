import os
import cv2
import torch
import numpy as np
from .detector import Predictor
from .detector import CENTER_MODEL

def detect_card(im):
    """detect 4 corners of a table

    Args:
        im (np.array): input image 

    Returns:
        list: 
            - case 1: table detected: [top-left, top-right, bottom-right, bottom-left]
            - case 2: no table detected: []
    """
    
    predictor = Predictor()
    _, res = predictor.inference(im)
    """dewarped table with conner

    Args:
        im (np.array): input image 

    Returns:
        case1: None, False
        case2: dewarped image, True
    """
    center_model = CENTER_MODEL()
    img_aligh, point = center_model.aligh(im, res)
    if not point:
        return img_aligh, False
    return img_aligh, True

if __name__ == '__main__':
    im = cv2.imread('')
    image_aligh, have_card = detect_card(im)