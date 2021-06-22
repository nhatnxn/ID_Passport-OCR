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
import matplotlib.pylot as plt

from nanodet.util import cfg, config, load_config, Logger, logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.util.path import mkdir

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == 'RepVGG':
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({'deploy': True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {'id': 0}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))
        return result_img


class CENTER_MODEL(object):
    def __init__(self, config):
        super().__init__(CENTER_MODEL)
        self.weight_path = config['detect_card_model']
        self.scale = config['scale']
        self.threshold = config['detect_card_threshold']
        self.config_path = config['detect_card_config']
    
    def detect_obj(self, img, show=False, save_res=False):
        load_config(cfg, self.config_path)
        logger = Logger(-1, use_tensorboard=False)
        predictor = Predictor(cfg, self.weight_path, logger, device=device)
        _, res = predictor.inference(img)
        img_aligh, point = self.aligh(img,res)
        if not point:
            return img, False

        if show:
            plt.imshow(cv2.cvtColor(img_aligh))
            plt.show()
        
        if save_res:
            cv2.imwrite("../insightface/deploy/imgs/res.jpg", img_aligh)
        return img_aligh, True

    def aligh(self, image, list_points):
        source_points = self.get_cornor_point(list_points)
        if not source_points:
            return image, False
        point = np.float32([source_points[0],source_points[1],source_points[2],source_points[3]])
        # image = cv2.imread(image_path)
        max_width = max(abs(source_points[1][0]-source_points[0][0]), abs(source_points[2][0]-source_points[3][0]))
        max_height = max(abs(source_points[3][1]-source_points[0][1]), abs(source_points[2][1]-source_points[1][1]))
        dest_points = np.float32([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])
        
        M = cv2.getPerspectiveTransform(point, dest_points)
        dst = cv2.warpPerspective(image, M, (int(max_width), int(max_height)))
        return dst, True
    
    def get_cornor_point(self, res):
        points = res.copy()
        missing_point = []
        for label in points:
            if points[label] is not None:
                xmin, ymin, xmax, ymax, score = points[label][0]
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./center/config/cmnd.yml')
    parser.add_argument("--image_path", type=str,
                        default='C:\\Users\\hoanglv10\\PycharmProjects\\Object_Corner_Detection\\demo\\cmnd_hoang.jpg')
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    # print(config)
    model = CENTER_MODEL(config)
    # for i in range(10):
    img = cv2.imread(args.image_path)
    model.detect_obj(img, show=False, save_res=False)
