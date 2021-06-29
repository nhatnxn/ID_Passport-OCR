from .models.paddleocr.ppocr_rec import load_ppocr_model
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH      = 'models/paddleocr/weights'
DICT_PATH       = 'models/paddleocr/vi_dict.txt'
BATCH_SIZE      = 20
MAX_TEXT_LENGTH = 43

class OcrModel(object):
    print("Loading TEXT_MODEL...")
    def __init__(self, lang='latin', weights=MODEL_PATH, dict_path=DICT_PATH, batch_size=BATCH_SIZE):
        self.model = load_ppocr_model(lang = lang, 
                                weights = weights, 
                                dict_path = dict_path, 
                                use_gpu = False, 
                                batch_size = batch_size, 
                                max_text_lenth = MAX_TEXT_LENGTH)
  
    def get_ocr(self, result_line_img):
        result_ocr = {}
        labels  = []
        imgs    = []
        for key, value in result_line_img.items():
            labels.append(key)
            imgs.append(value)
        
        res_ocr = self.model(imgs)
        for label, res in zip(labels, res_ocr):
            result_ocr[label] = res

        return result_ocr

        
if __name__ == '__main__':
    result_line_img = {}
    ocr_model = OcrModel()
    result_ocr = ocr_model.get_ocr(result_line_img)