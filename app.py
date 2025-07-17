import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Muat model yang sudah dilatih (pastikan model ini sudah dilatih dan menyimpan arsitektur + bobot)
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

st.title("Aplikasi Klasifikasi Citra Digital Kue dengan EfficientNetB0")
st.write("Unggah gambar kue untuk diklasifikasikan ke dalam salah satu dari 8 kelas.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Resize sesuai input model (misalnya 224x224 untuk EfficientNetB0)
    image_resized = image.resize((224, 224))
    
    # Konversi ke array numpy
    img_array = np.array(image_resized)
    
    # Preprocessing sesuai EfficientNetB0
    img_preprocessed = preprocess_input(img_array)
    
    # Tambahkan batch dimension
    img_input = np.expand_dims(img_preprocessed, axis=0)
    
    # Prediksi
    pred = model.predict(img_input)
    pred_kelas = np.argmax(pred, axis=1)[0]
    confidence = pred[0][pred_kelas]
    
    # Tampilkan hasil
    st.write(f"Prediksi: **{kelas[pred_kelas]}**")
    st.write(f"Kepercayaan: {confidence*100:.2f}%")
