from time import time
import cv2

# from center_DCNv2.utils.config import Cfg
# from line_detection_module.model_box import LineDetection
# from detect import CENTER_MODEL
# from center.nanodet.detector.dectect import detect_card
from detect_card import CardDetection
from detect_line import LineDetection
import time


class TEXT_IMAGES(object):

    def __init__(self, latency=False, line_model='yolov5', card_model='yolov5', text_model='vietocr', reg_model='vgg_seq2seq'):
        print("Loading TEXT_MODEL...")
        self.card_detect_module = CardDetection(card_model)
        self.line_detect_module = LineDetection(line_model)
        if text_model == 'paddleocr':
            from ocr_engine.paddleocr import OcrModel
        else:
            from ocr_engine.vietocr import OcrModel
        self.recognition_text_module = OcrModel()
        self.latency = latency

    def get_content_image(self, image):
        # cv image
        # return image_drawed, texts, boxes
        
        t1 = time.time()
        img_detected, have_card = self.card_detect_module.predict_card(image)
        t_card = time.time() - t1
        if self.latency:
            t1 = time.time()
            for i in range(20):
                img_detected, have_card = self.card_detect_module.predict_card(image)
            t_card = (time.time() - t1)/20
        if not have_card:
            print("Không phát hiện ID card")
            return None, None, None, None, None
        
        t2 = time.time()
        result_line_img, img_draw_box = self.line_detect_module.predict_box(img_detected)
        t_line = time.time() - t2
        if self.latency:
            t2 = time.time()
            for i in range(20):
                result_line_img, img_draw_box = self.line_detect_module.predict_box(img_detected)
            t_line = (time.time() - t2)/20

        # result_ocr = {}
        # for key, value in result_line_img.items():
        #     label = key
        #     img = value
        #     result_ocr[label] = []

            # for img in imgs:
        t3 = time.time()
        res_str = self.recognition_text_module.get_ocr(result_line_img)
        t_ocr = time.time() - t3
        if self.latency:
            t3 = time.time()
            for i in range(20):
                res_str = self.recognition_text_module.get_ocr(result_line_img)
            t_ocr = (time.time() - t3)/20

            # result_ocr[label].append(res_str)

        print(res_str)
        return res_str, img_draw_box, t_card, t_line, t_ocr



if __name__ == "__main__":
    app = TEXT_IMAGES(line_model='yolov5', card_model='nanodet', reg_model='vgg_seq2seq', ocr_weight_path='weights/seq2seqocr_best.pth')
    # app = TEXT_IMAGES(reg_model='vgg_transformer', ocr_weight_path='weights/transformerocr.pth')

    img_path ="/content/000886.jpg"
    img = cv2.imread(img_path)
    app.get_content_image(img, show_line=True)
    # print(text_boxes)
    # print(res)

