import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('snake_pred1.h5')

# Daftar kelas kue
labels = ['Kue A', 'Kue B', 'Kue C', 'Kue D', 'Kue E', 'Kue F', 'Kue G', 'Kue H']

# Fungsi preprocessing gambar
def preprocess_image(image):
    size = (224, 224)  # Sesuaikan dengan input model
    image = image.resize(size)
    image_array = np.array(image) / 255.0  # normalisasi
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("Klasifikasi Kue - Deteksi 8 Kelas")

uploaded_file = st.file_uploader("Upload gambar kue", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    if st.button('Prediksi'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        confidence = np.max(prediction)

        st.write(f'Kelas Prediksi: **{predicted_label}**')
        st.write(f'Probabilitas: {confidence:.2f}')
