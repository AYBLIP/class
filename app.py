import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Path model
model_path = f'model_{optimizer_choice}.h5'

# Muat model
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'FixedDropout': tf.keras.layers.Dropout,
            'swish': tf.nn.swish
        }
    )
    st.success(f"Model {optimizer_choice} berhasil dimuat.")
except:
    model = None
    st.error(f"Gagal memuat model dari {model_path}.")

# Daftar kelas
kelas = ['Kue Dadar Gulung', 'Kue Kastengel', 'Kue Klepon', 'Kue Lapis', 'Kue Lumpur', 'Kue Putri Salju', 'Kue Risoles', 'Kue Serabi']

uploaded_files = st.file_uploader("Unggah beberapa gambar kue Anda", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Buka gambar asli
        img = Image.open(uploaded_file)
        # Resize agar lebih kecil
        max_width = 150
        aspect_ratio = img.height / img.width
        new_width = max_width
        new_height = int(max_width * aspect_ratio)
        img_resized = img.resize((new_width, new_height))
        # Tampilkan gambar kecil
        st.image(img_resized, caption=uploaded_file.name)

        # Pra-pemrosesan untuk prediksi
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        if model:
            pred = model.predict(img_array)
            pred_kelas = np.argmax(pred, axis=1)[0]
            kelas_terpilih = kelas[pred_kelas]
            confidence = np.max(pred) * 100

            st.write(f"Prediksi: **{kelas_terpilih}**")
            st.write(f"Kepercayaan: {confidence:.2f}%")
        else:
            st.warning("Model belum berhasil dimuat.")
