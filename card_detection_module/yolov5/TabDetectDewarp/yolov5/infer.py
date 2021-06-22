import os
import sys
import cv2
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from .models.experimental import attempt_load
from .utils.torch_utils import select_device
from .utils.general import check_img_size, non_max_suppression, xywh2xyxy, xyxy2xywh, scale_coords

MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'weights', 'yolov5s_thalas.pt')
IMG_SIZE    = 640
CONF_THES   = 0.3

# Select device
device = select_device(device='cpu', batch_size=1)
# Load model
model = attempt_load(MODEL_PATH, map_location=device)
# Configure
model.eval()
classes = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)

def preprocess(image, img_size = IMG_SIZE, stride = 32, pad = 0.5):
    # Padded resize
    img = letterbox(image, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2, color, tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def plot_image(image, target, names):
    tl = 1  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if len(target) > 0:
        boxes = xywh2xyxy(target[:, 2:6]).T
        classes = target[:, 1]
        labels = target.shape[1] == 6  # labels if no conf column
        conf = None if labels else target[:, 6]  # check for confidence presence (label vs pred)
        for j, box in enumerate(boxes.T):
            cls = int(classes[j])
            cls = names[cls] if names else cls
            if labels or conf[j] > 0.25:  # 0.25 conf thresh
                label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                plot_one_box(box, image, label=label, color=(255, 0, 0), line_thickness=tl)
    
    return image

def inference(image):    
    """detect table corners

    Args:
        image (np.array): input image 

    Returns:
        np.array: yolo detection result
    """

    if image is None:
        return None
        
    w0, h0 = image.shape[:2]

    # preprocess image
    img = preprocess(image)
    w, h = img.shape[1:3]

    batch = torch.from_numpy(np.expand_dims(img, axis=0))
    batch = batch/255.0  # 0 - 255 to 0.0 - 1.0

    with torch.no_grad():
        # Run model
        out, train_out = model(batch, augment=False)  # inference and training outputs
        out = non_max_suppression(out, conf_thres=CONF_THES, iou_thres=0.5, multi_label=True)

    target = output_to_target(out)
    target = torch.from_numpy(target)
    target[:, 2:6] = scale_coords((w, h), target[:, 2:6], (w0, h0)).round()
    
    return target.numpy()

def best_point(t):
    return t[t[:,-1].argsort()[-1]]

def polygon_from_corners(t):
    t0 = t[t[:,1] == 0] # top-left points
    t1 = t[t[:,1] == 1] # top-right points
    t2 = t[t[:,1] == 2] # bottom-right points
    t3 = t[t[:,1] == 3] # bottom-left points
    
    if t0.shape[0] == 0 or t1.shape[0] == 0 or t2.shape[0] == 0 or t3.shape[0] == 0:
        return None
    
    A = best_point(t0)
    B = best_point(t1)
    C = best_point(t2)
    D = best_point(t3)
    
    return np.stack((A, B, C, D))[:,2:4]




if __name__ == '__main__':
    
    img = cv2.imread('2.png')
    target = inference(img)
    pts = polygon_from_corners(target).astype(int)
    if not pts is None:
        pts = pts.reshape((-1, 1, 2))
        # draw corners
        r = plot_image(img, target, classes)
        # draw box
        r = cv2.polylines(r, [pts], True, (255, 0, 0), 2)

        cv2.imwrite('result.png', r)
    else:
        print('Not found!')
    