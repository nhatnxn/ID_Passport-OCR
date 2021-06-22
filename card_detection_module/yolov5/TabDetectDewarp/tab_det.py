import cv2
import numpy as np
from .utils import increase_border
from .yolov5 import inference, polygon_from_corners

def detect_table_corners(im):
    """detect 4 corners of a table

    Args:
        im (np.array): input image 

    Returns:
        list: 
            - case 1: table detected: [top-left, top-right, bottom-right, bottom-left]
            - case 2: no table detected: []
    """
    PADDING_SIZE    = 8
    corners         = []

    # check input
    if im is None:
        return []

    target = inference(im)
    if target is None:
        return []
        
    pts = polygon_from_corners(target)
    if pts is None:
        return []

    pts = pts.astype(int)
    
    if pts is None:
        return []
    else:
        corners = increase_border(pts, PADDING_SIZE)
        corners = [(int(p[0]), int(p[1])) for p in corners]
    
        return corners

    