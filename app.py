import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Daftar kelas
labels = ['Kue A', 'Kue B', 'Kue C', 'Kue D', 'Lumpur', 'Kue F', 'Kue G', 'Kue H']

def preprocess_image(image):
    size = (224, 224)
    image = image.resize(size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

st.title("Klasifikasi Kue - Deteksi 8 Kelas")

optimizer_choice = st.selectbox(
    "Pilih optimizer yang digunakan saat pelatihan model:",
    ("Adam", "SGD", "RMSprop")
)

# Inisialisasi model sebagai None
model = None

# Coba muat model sesuai pilihan optimizer
model_path = f'model_{optimizer_choice}.h5'
try:
    model = tf.keras.models.load_model(model_path)
    st.success(f"Model {optimizer_choice} berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model dari {model_path}. Pesan: {str(e)}")

uploaded_file = st.file_uploader("Upload gambar kue", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    if st.button('Prediksi'):
        if model is not None:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]
            confidence = np.max(prediction)

            st.write(f'Kelas Prediksi: **{predicted_label}**')
            st.write(f'Probabilitas: {confidence:.2f}')
        else:
            st.error("Model belum berhasil dimuat. Periksa file model dan coba lagi.")
