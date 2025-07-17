import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load model yang sudah dilatih
@st.cache(allow_output_mutation=True)
def load_models():
    model_adam = load_model('model_Adam.h5')
    model_sgd = load_model('model_SGD.h5')
    model_rmsprop = load_model('model_RMSprop.h5')
    return {
        'Adam': model_adam,
        'SGD': model_sgd,
        'RMSprop': model_rmsprop
    }

model_dict = load_models()

# Judul Aplikasi
st.title('Klasifikasi Kue Indonesia')

# Pilih optimizer yang ingin digunakan
optimizer_choice = st.selectbox('Pilih optimizer untuk prediksi:', ['Adam', 'SGD', 'RMSprop'])

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar kue", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Gambar Unggahan', use_column_width=True)

    # Konversi gambar ke array dan preprocess
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    model = model_dict[optimizer_choice]
    pred_probs = model.predict(img_array)
    pred_class = np.argmax(pred_probs, axis=1)[0]

    # Tampilkan hasil prediksi
    class_labels = list(test_generator.class_indices.keys())  # Pastikan ini sesuai dengan data
    predicted_label = class_labels[pred_class]
    confidence = pred_probs[0][pred_class]

    st.write(f'Prediksi: **{predicted_label}** dengan kepercayaan {confidence:.2f}')
