import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Muat model yang sudah disimpan
import os

model_path = 'snake_pred1.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path) 

# Daftar kelas yang digunakan selama pelatihan
# Pastikan sesuai dengan class_indices dari data generator Anda
class_labels = ['kue1', 'kue2', 'kue3', 'kue4', 'kue5', 'kue6', 'kue7', 'kue8']

# Fungsi preprocess gambar
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Judul aplikasi
st.title('Aplikasi Klasifikasi Citra Digital')

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar untuk diklasifikasi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Proses prediksi
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    pred_label = class_labels[pred_idx]
    confidence = predictions[0][pred_idx]

    # Tampilkan hasil prediksi
    st.write(f"Prediksi: **{pred_label}**")
    st.write(f"Kepercayaan: {confidence:.2%}")
