import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pywt
from autogluon.tabular import TabularPredictor
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# CLASS DICTIONARY

class_dict = {
    "maria_sharapoa": 0,
    "ms_dhoni": 1,
    "pv_sindhu": 2,
    "rohit_sharma": 3,
    "virat_kohli": 4
}
class_dict_inv = {v:k for k,v in class_dict.items()}


# WAVELET TRANSFORM

def w2d(img, mode='db1', level=5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    img_gray /= 255

    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    img_H = pywt.waverec2(coeffs_H, mode)
    img_H *= 255
    img_H = np.uint8(img_H)
    return img_H


# LOAD MODEL

predictor = TabularPredictor.load("./AutogluonModels/ag-20260307_184830")
best_model = predictor.model_best


# LOAD FACE CASCADE

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.set_page_config(page_title="Celebrity Classifier", layout="centered")
st.title("Celebrity Face Recognition App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", width="stretch")

    # FACE DETECTION
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please upload a clear face image.")
    else:
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]

            
            # PREPROCESSING
           
            scalled_raw_img = cv2.resize(face_img, (32,32))
            img_har = w2d(face_img,'db1',5)
            scalled_img_har = cv2.resize(img_har,(32,32))

            combined_img = np.vstack((
                scalled_raw_img.reshape(32*32*3,1),
                scalled_img_har.reshape(32*32,1)
            ))

            feature_vector = combined_img.reshape(1,4096)
            df = pd.DataFrame(feature_vector)

            
            # PREDICTION
         
            prediction = predictor.predict(df, model=best_model)[0]
            proba = predictor.predict_proba(df, model=best_model)

            celebrity_name = class_dict_inv[int(prediction)]

            st.success(f"Predicted Celebrity: **{celebrity_name}**")

           
            # TOP 3 PROBABILITIES
           
            proba_series = proba.iloc[0]
            top3 = proba_series.sort_values(ascending=False).head(3)

            st.write("Top 3 Predictions:")
            for idx, score in top3.items():
                st.write(f"{class_dict_inv[int(idx)]} → {score*100:.2f}%")