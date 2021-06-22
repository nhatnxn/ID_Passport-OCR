import numpy as np
import cv2

from .utils import distance

def get_dewarped_table(im, corners):
    
    # check input 
    if im is None:
        return None
    if len(corners) != 4:
        return None
    
    target_w = int(max(distance(corners[0], corners[1]), distance(corners[2], corners[3])))
    target_h = int(max(distance(corners[0], corners[3]), distance(corners[1], corners[2])))
    target_corners = [[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]]

    pts1 = np.float32(corners)
    pts2 = np.float32(target_corners)
    transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dewarped = cv2.warpPerspective(im, transform_matrix, (target_w, target_h))
    
    return dewarped