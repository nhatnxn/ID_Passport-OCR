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
from PIL import Image

# import matplotlib.pylot as plt

# from nanodet.util import cfg, config, load_config, Logger, logger
# from nanodet.model.arch import build_model
# from nanodet.util import load_model_weight
# from nanodet.data.transform import Pipeline
# from nanodet.util.path import mkdir

class LineDetection(object):
    def __init__(self, model='yolov5'):
        self.model = model
        self.colors = [(255, 0, 0), (0, 0, 255), (123, 2, 190),
                        (253, 124, 98) ,  (255, 251, 134), (128,256,0),
                        (123,182,111), (21,234,56), (235,123,45)]
        self.min_size = 1
        
    def predict_box(self, img):
        '''

        @param img: cv2 image: BGR
        @return: list PIL image
        '''
        
        
        if self.model == 'nanodet':
            from line_detection_module.nanodet import detect_line

        else:
            from line_detection_module.yolov5 import detect_line
        
        

        # # Preprocessing image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #convert to rgb (pretrained model RGB)
        # image, meta = pre_process(img, self.scale)
        # if torch.cuda.is_available():
        #     image = image.cuda()

        # # Predict box
        # with torch.no_grad():
        #     start = time.time()
        #     output = self.model(image)[-1]
        #     # print(time.time() - start)
        #     hm = output['hm'].sigmoid_()
        #     wh = output['wh']
        #     reg = output['reg']
        #     dets = ctdet_decode(hm, wh, reg=reg, K=100)

        # dets = post_process(dets, meta, self.num_classes)
        # #print(dets)
        # detections = [dets]
        # results = merge_outputs(detections, self.num_classes, self.max_obj_per_img)

        # Get boxes with score larger threshold
        info = detect_line(img)
        h_extend_size = 0.06
        w_extend_size = 0.04
        list_box = {}
        list_label       = ['address_line_1','address_line_2', 'birthday', 'hometown_line_1', 'hometown_line_2', 'id', 'name', 'nation', 'sex']
        
        # processing
        for inf in list_label:
            bbox = info[inf]
            if len(bbox) == 4:
                if bbox[3]>self.min_size:
                    xmin, ymin, xmax, ymax = max(int(bbox[0]), 0), max(0, int(bbox[1])), \
                                                    min(int(bbox[2]),img.shape[1]), min(int(bbox[3]), img.shape[0])

                    # Extend
                    xmin_extend, ymin_extend, xmax_extend, ymax_extend = max(0, xmin - int((xmax - xmin) * w_extend_size)), \
                                                max(0, ymin - int((ymax - ymin) * h_extend_size)), \
                                                xmax + int((xmax - xmin) * w_extend_size), \
                                                ymax + int((ymax - ymin) * h_extend_size)

                    info[inf] = [xmin_extend, ymin_extend, xmax_extend, ymax_extend]

        # for j in range(1, self.num_classes + 1):
        #     if self.list_label[j - 1] not in list_box.keys():
        #         list_box[self.list_label[j - 1]] = []
        #     for bbox in results[j]:
        #         if bbox[4] >= self.threshold:
        #             print(bbox[4])
        #             xmin, ymin, xmax, ymax = max(int(bbox[0]), 0), max(0, int(bbox[1])), \
        #                                      min(int(bbox[2]),img.shape[1]), min(int(bbox[3]), img.shape[0])

        #             # Extend
        #             xmin_extend, ymin_extend, xmax_extend, ymax_extend = max(0, xmin - int((xmax - xmin) * w_extend_size)), \
        #                                      max(0, ymin - int((ymax - ymin) * h_extend_size)), \
        #                                      xmax + int((xmax - xmin) * w_extend_size), \
        #                                      ymax + int((ymax - ymin) * h_extend_size)

        #             list_box[self.list_label[j-1]].append([xmin_extend, ymin_extend, xmax_extend, ymax_extend])
        # print("List box: ", list_box)

        # if return_line_draw:
        img_res = Image.fromarray(img)
        img_res = np.ascontiguousarray(img_res)
        # box - [xmin, ymin, xmax, ymax]
        for idx, label in enumerate(list_label):
            if len(info[label]) == 4:
                box = info[label]
                cv2.rectangle(img_res, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), self.colors[idx], 2, 2)

        # Crop line
        result_line_img = {}
        img_for_crop = Image.fromarray(img)
        for idx, label in enumerate(list_label):
            if len(info[label]) == 4:
                box = info[label]
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                
                print(box)

                line_cropped = img_for_crop.copy().crop((xmin, ymin, xmax, ymax))
                result_line_img[label] = line_cropped
        return result_line_img, img_res

if __name__ == '__main__':
    im = cv2.imread('')
    line_detect = LineDetection()
    result_line_img, img_res = line_detect.predict_box(im)
    print(result_line_img)

