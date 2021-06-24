import numpy as np
from torch._C import device
import cv2
import argparse
import time

class CENTER_MODEL(object):
    def __init__(self):
        super(CENTER_MODEL).__init__()

    def aligh(self, image, list_points):
        source_points = self.get_cornor_point(list_points)
        if not source_points:
            return image, False
        # print('source_points: ', source_points)
        point = np.float32([source_points[0],source_points[1],source_points[2],source_points[3]])
        # image = cv2.imread(image_path)
        max_width = max(abs(source_points[1][0]-source_points[0][0]), abs(source_points[2][0]-source_points[3][0]))
        max_height = max(abs(source_points[3][1]-source_points[0][1]), abs(source_points[2][1]-source_points[1][1]))
        dest_points = np.float32([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])
        
        M = cv2.getPerspectiveTransform(point, dest_points)
        dst = cv2.warpPerspective(image, M, (int(max_width), int(max_height)))
        return dst, True
    
    def get_cornor_point(self, res):
        points = res[0].copy()
        missing_point = []

        for label in points:
            if points[label] is not None:
                # print('='*20)
                # print(points[label][0])
                xmin, ymin, xmax, ymax, score =  points[label][0]
                x_center = (xmin+xmax)/2
                y_center = (ymin+ymax)/2
                points[label] = (x_center,y_center)
            else:
                missing_point.append(label)
        if len(missing_point) == 0:
            return points
        if len(missing_point) == 1:
            points = self.calculate_missed_coord_corner(missing_point[0], points)
            return points
        else:
            print('cannot detect id card')
            return 0

    def calculate_missed_coord_corner(self, missing_point, points):

        thresh = 0
        if missing_point == 0:
            midpoint = np.add(points[1], points[3]) / 2
            y = 2 * midpoint[1] - points[2][1] - thresh
            x = 2 * midpoint[0] - points[2][0] - thresh
            points[0] = (x, y)
        elif missing_point == 1:  # "top_right"
            midpoint = np.add(points[0], points[2]) / 2
            y = 2 * midpoint[1] - points[3][1] - thresh
            x = 2 * midpoint[0] - points[3][0] - thresh
            points[1] = (x, y)
        elif missing_point == 2:  # "bottom_left"
            midpoint = np.add(points[0], points[2]) / 2
            y = 2 * midpoint[1] - points[1][1] - thresh
            x = 2 * midpoint[0] - points[1][0] - thresh
            points[2] = (x, y)
        elif missing_point == 3:  # "bottom_right"
            midpoint = np.add(points[3], points[1]) / 2
            y = 2 * midpoint[1] - points[0][1] - thresh
            x = 2 * midpoint[0] - points[0][0] - thresh
            points[3] = (x, y)

        return points
