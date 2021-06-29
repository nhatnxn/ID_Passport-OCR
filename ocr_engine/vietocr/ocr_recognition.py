from .tool.config import Cfg_reg
from .tool.predictor import Predictor
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__),'weights/transformerocr.pth')
# MODEL_PATH = os.path.join(os.path.dirname(__file__),'weights/vgg-seq2seq.pth')


class OcrModel(object):
    
    def __init__(self, reg_model='vgg_transformer', ocr_weight_path=MODEL_PATH):
        print("Loading TEXT_MODEL...")

        config = Cfg_reg.load_config_from_name(reg_model)
        config['weights'] = ocr_weight_path
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.recognition_text_module = Predictor(config)

    def get_ocr(self, result_line_img):

        result_ocr = {}
        for key, value in result_line_img.items():
            label = key
            img = value
            result_ocr[label] = []

            res_str = self.recognition_text_module.predict(img)
            result_ocr[label].append(res_str)

        return result_ocr

if __name__ == '__main__':
    result_line_img = {}
    ocr_model = OcrModel()
    result_ocr = ocr_model.get_ocr(result_line_img)
