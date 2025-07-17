import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    models = {
        'Adam': load_model('model_klasifikasi_kue.h5'),
    }
    return models

models = load_models()

st.title("Klasifikasi Kue menggunakan EfficientNetB0")

# Pilih model
model_choice = st.selectbox("Pilih model untuk prediksi:", list(models.keys()))
model = models[model_choice]

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar kue", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Proses prediksi
    if st.button("Prediksi"):
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred_probs = model.predict(img_array)
        pred_class = np.argmax(pred_probs, axis=1)[0]

        class_indices = {v: k for k, v in train_generator.class_indices.items()}
        # Pastikan class_indices sesuai dengan label yang digunakan saat training
        # Jika tidak, buat dictionary manual sesuai label
        class_labels = list(train_generator.class_indices.keys())

        predicted_label = class_labels[pred_class]
        confidence = pred_probs[0][pred_class]

        st.write(f"Prediksi: **{predicted_label}** dengan kepercayaan {confidence*100:.2f}%")
