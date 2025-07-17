import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('model_Adam.h5')

# Daftar label kelas
kelas = [
    'Kue Coklat',
    'Kue Keju',
    'Kue Stroberi',
    'Kue Lapis',
    'Kue Brownies',
    'Kue Nastar',
    'Kue Putu',
    'Kue Apem'
]

st.title("Aplikasi Klasifikasi Citra Digital Kue")
st.write("Unggah gambar kue untuk diklasifikasikan ke dalam salah satu dari 8 kelas.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Preprocessing gambar sesuai model
    # Misalkan model membutuhkan input 128x128
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0  # normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # bentuk (1, 128, 128, 3)
    
    # Prediksi
    pred = model.predict(img_array)
    pred_kelas = np.argmax(pred, axis=1)[0]
    confidence = pred[0][pred_kelas]
    
    # Tampilkan hasil
    st.write(f"Prediksi: **{kelas[pred_kelas]}**")
    st.write(f"Kepercayaan: {confidence*100:.2f}%")
