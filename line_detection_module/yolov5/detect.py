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
    '''
    info = detect_bbox(im)

    return info

if __name__ == '__main__':
    im = cv2.imread('')
    info = detect_line(im)
    print(info)