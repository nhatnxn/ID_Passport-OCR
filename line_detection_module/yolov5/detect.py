from logging import info
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .TabDetectDewarp import detect_bbox, get_dewarped_table

def detect_line(im):
    '''
    
    Args:
        im (np.array): input image 

    Returns:
        dict:
            { 
                'address_line_1':   [left,right,top,bottom]
                'address_line_2':   [left,right,top,bottom]
                'birthday':         [left,right,top,bottom]
                'hometown_line_1':  [left,right,top,bottom]
                'hometown_line_2':  [left,right,top,bottom]
                'id':               [left,right,top,bottom]
                'name':             [left,right,top,bottom]
                'nation':           [left,right,top,bottom]
                'sex':              [left,right,top,bottom]
            }
    '''
    info = detect_bbox(im)

    return info

if __name__ == '__main__':
    im = cv2.imread('')
    info = detect_line(im)
    print(info)