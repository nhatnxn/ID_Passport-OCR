from .tool.config import Cfg_reg
from .tool.predictor import Predictor

MODEL_PATH = 'weights/vgg-seq2seq.pth'

class OcrModel(object):
    
    def __init__(self, reg_model='vgg_seq2seq', ocr_weight_path=MODEL_PATH):
        print("Loading TEXT_MODEL...")

        config = Cfg_reg.load_config_from_name(reg_model)
        config['weights'] = ocr_weight_path
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.recognition_text_module = Predictor(config)

    def get_content_image(self, result_line_img):

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
    result_ocr = ocr_model.get_content_image(result_line_img)
