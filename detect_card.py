import os
from numpy.lib.type_check import imag
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "center"))

import numpy as np
from torch._C import device
import cv2
import argparse
import time

class CardDetection(object):
    def __init__(self, model='yolov5'):
        self.model = model
    
    def predict_card(self, im):
        '''
            
        '''
        
        if self.model == 'nanodet':
            from card_detection_module.nanodet import detect_card
        
        else:
            from card_detection_module.yolov5 import detect_card
        
        img_aligh, point = detect_card(im)

        return img_aligh, point

if __name__ == '__main__':
    im = cv2.imread('')
    card_detection = CardDetection()
    img_aligh, have_card = card_detection.predict_card(im)
    print(img_aligh, have_card)