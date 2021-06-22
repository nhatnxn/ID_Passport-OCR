import cv2
import numpy as np
import matplotlib.pyplot as plt
from .TabDetectDewarp import detect_table_corners, get_dewarped_table

def detect_card(im):
    # call api 1: detect corners
    # Returns a list: 
    #       - case 1: table detected: [top-left, top-right, bottom-right, bottom-left]
    #                 e.g. [(1,2), (3, 4), (5, 6), (7, 8)]
    #       - case 2: no table detected: []
    corners = detect_table_corners(im)
    # call api 2: dewarp table with corners:
        # return:
        # - case 1: dewarped image
        # - case 2: None
    if len(corners) == 4:
        image_aligh = get_dewarped_table(im,corners)
        return image_aligh, True
    else:
        return im, False


if __name__ == '__main__':
    im = cv2.imread('')
    img, status = detect_card(im)
    print(img, status)