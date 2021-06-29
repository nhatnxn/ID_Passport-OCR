from .models.paddleocr.ppocr_rec import load_ppocr_model
import os
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':

    # load model
    paddleocr = load_ppocr_model(lang = "latin", 
                                 weights = "models/paddleocr/weights", 
                                 dict_path = "models/paddleocr/vi_dict.txt", 
                                 use_gpu = False, 
                                 batch_size = 20, 
                                 max_text_lenth = 43)
  
    # inference batch 
    imgs = [cv2.imread(f"samples/{img}") for img in os.listdir("samples")]
    res = paddleocr(imgs)
    print(res)

