import streamlit as st
import keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model (pastikan model sudah dilatih dan disimpan sebagai 'model.h5')
@st.cache(allow_output_mutation=True)
def load_model_keras():
    model = load_model('final_model_Adam.h5')
    return model

model = load_model_keras()

# Daftar kelas (sesuaikan dengan kelas yang Anda miliki)
kelas = ['Kue 1', 'Kue 2', 'Kue 3', 'Kue 4', 'Kue 5', 'Kue 6', 'Kue 7', 'Kue 8']

st.title("Klasifikasi Gambar Kue dengan EfficientNetB0")
st.write("Upload gambar kue Anda dan model akan memprediksi kelasnya.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar kue...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca dan menampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Preprocessing gambar
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension

    # Prediksi
    pred = model.predict(img_array)
    pred_prob = pred[0]
    pred_idx = np.argmax(pred_prob)
    pred_label = kelas[pred_idx]
    pred_confidence = pred_prob[pred_idx]

    # Tampilkan hasil prediksi
    st.write(f"Prediksi: **{pred_label}**")
    st.write(f"Kepercayaan: {pred_confidence*100:.2f}%")
