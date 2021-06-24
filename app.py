import time
import cv2
import streamlit as st
import SessionState
import numpy as np
from PIL import Image
from ocr_engine import TEXT_IMAGES
state = SessionState.get(result_text="", res="", prob_positive=0.0, prob_negative= 0.0, initial=True, img_drawed=None, img_cropped=None, reg_text_time=None)


def main():
    
    st.title("Demo eKYC")
    # Load model

    pages = {
        'ID card': page_eKYC

    }

    st.sidebar.title("Application")
    page = st.sidebar.radio("Demo application:", tuple(pages.keys()))
    
    # st.title('Model Options')
    st.sidebar.subheader('Model Options')
    
    latency = st.sidebar.selectbox('Latency mesurement', [False,True])
    card_model = st.sidebar.selectbox('Card detection', ['yolov5','nanodet'])
    line_model = st.sidebar.selectbox('Fields detection', ['yolov5', 'nanodet'])
    text_model = st.sidebar.selectbox('Text recognition', ['VietOCR', 'PaddleOCR'])
    
    model = load_model(latency, card_model, line_model, text_model)

    pages[page](state, model)

    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model(latency, card_model, line_model, text_model):
    print("Loading model ...")
    model = TEXT_IMAGES(latency=latency, line_model=line_model, card_model=card_model, text_model=text_model, reg_model='vgg_seq2seq', ocr_weight_path='weights/seq2seqocr_best.pth')
    return model


def page_eKYC(state, model):
    st.header("Identification of ID card information ")

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        pil_image = Image.open(img_file_buffer)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # print(cv_image.shape)

        # CMND detection
        t1 = time.time()

        result_text, img_drawed_box, t_card, t_line, t_ocr= model.get_content_image(cv_image)

        state.result_text = result_text
        state.img_drawed = img_drawed_box
        state.t_card = t_card
        state.t_line = t_line
        state.t_ocr  = t_ocr
        state.reg_text_time = time.time() - t1

        col1, col2 = st.beta_columns(2)
        with col2:

            if state.result_text is not None:
                # result_text_format = []
                # for texts in state.result_text:
                #     result_text_format.append(" ".join(texts))
                st.json(state.result_text)
                st.success("Card detection time: %2f"%(state.t_card))
                st.success("Fields detection time: %2f"%(state.t_line))
                st.success("OCR recognition time: %2f"%(state.t_ocr))
                st.success("Total time: %.2f"%(state.reg_text_time))
            else:
                st.error("Not detected ID card")
        with col1:
            if state.img_drawed is not None:
                st.image(state.img_drawed, use_column_width=True)

        # if state.img_cropped is not None:
        #     st.title("Chi tiết:")
        #     for idx, img in enumerate(state.img_cropped):
        #         st.image(img, caption=state.result_text[idx])
        #         st.empty()

if __name__ == "__main__":
    main()