import cv2
import numpy as np
import matplotlib.pyplot as plt
from TabDetectDewarp import detect_table_corners, get_dewarped_table

if __name__ == '__main__':
    sample = 'samples/image3.jpg'
    im = cv2.imread(sample)

    # call api 1: detect corners
    # Returns a list: 
    #       - case 1: table detected: [top-left, top-right, bottom-right, bottom-left]
    #                 e.g. [(1,2), (3, 4), (5, 6), (7, 8)]
    #       - case 2: no table detected: []
    corners = detect_table_corners(im)
    print(corners)

    if len(corners) == 4:
        # call api 2: dewarp table with corners:
        # return:
        # - case 1: dewarped image
        # - case 2: None
        out = get_dewarped_table(im, corners)
        plt.figure(figsize=(10,10))
        cv2.imwrite('out.jpg', out)
        print('dewarp image success! please check out file: out.jpg')
    else:
        print('Table not found!')