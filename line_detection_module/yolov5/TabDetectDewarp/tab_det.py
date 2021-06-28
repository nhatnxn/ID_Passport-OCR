import cv2
import numpy as np
from .utils import increase_border
from .yolov5 import inference, polygon_from_corners

def detect_bbox(im):
    """detect 9 information box

    Args:
        im (np.array): input image 

    Returns:
        dict:
            {
                'address_line_1':   [x,y,w,h]
                'address_line_2':   [x,y,w,h]
                'birthday':         [x,y,w,h]
                'hometown_line_1':  [x,y,w,h]
                'hometown_line_2':  [x,y,w,h]
                'id':               [x,y,w,h]
                'name':             [x,y,w,h]
                'nation':           [x,y,w,h]
                'sex':              [x,y,w,h]
            } 

    """
    PADDING_SIZE    = 0
    info            = {}
    list_dict       = ['address_line_1','address_line_2', 'birthday', 'hometown_line_1', 'hometown_line_2', 'id', 'name', 'nation', 'sex', 'passport']

    # check input
    if im is None:
        return []

    target = inference(im)
        
    pts = polygon_from_corners(target)
    # pts = pts.astype(int)
    
    # infs = increase_border(pts, PADDING_SIZE)
    
    for i, inf in enumerate(pts):
        info[list_dict[i]] = inf

    # in = [(int(p[0]), int(p[1])) for p in corners]
    
    return info

    