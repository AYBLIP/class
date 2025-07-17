import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('best_model_Adam.h5')

# Daftar kelas
kelas = ['Kue A', 'Kue B', 'Kue C', 'Kue D', 'Kue E', 'Kue F', 'Kue G', 'Kue H']

st.title("Klasifikasi Kue dengan Streamlit")

uploaded_file = st.file_uploader("Unggah gambar kue Anda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Pra-pemrosesan gambar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    pred = model.predict(img_array)
    pred_kelas = np.argmax(pred, axis=1)[0]
    kelas_terpilih = kelas[pred_kelas]
    confidence = np.max(pred) * 100

    st.write(f"Prediksi: **{kelas_terpilih}**")
    st.write(f"Kepercayaan: {confidence:.2f}%")
