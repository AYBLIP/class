import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
import numpy as np
import requests
import tempfile
import os
from PIL import Image

# URL model weights di GitHub (pastikan file publik)
MODEL_WEIGHTS_URL = 'model_Adam.h5'

@st.cache(allow_output_mutation=True)
def load_model_from_github():
    # Unduh bobot dari GitHub
    response = requests.get(MODEL_WEIGHTS_URL)
    temp_dir = tempfile.mkdtemp()
    weights_path = os.path.join(temp_dir, 'model_weights.h5')
    with open(weights_path, 'wb') as f:
        f.write(response.content)
    # Bangun model arsitektur
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    outputs = Dense(10, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    # Muat bobot
    model.load_weights(weights_path)
    return model

st.title("Deploy EfficientNetB0 dengan Pilihan Optimizer dan Model dari GitHub")

# Pilihan optimizer
optimizer_choice = st.selectbox('Pilih Optimizer', ['Adam', 'SGD', 'RMSprop'])

# Tombol untuk memuat model
if st.button('Muat Model'):
    model = load_model_from_github()
    st.success("Model berhasil dimuat dari GitHub!")

    # Tampilkan ringkasan model
    st.subheader("Ringkasan Model")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

    # Simulasi prediksi
    st.subheader("Unggah Gambar untuk Prediksi")
    uploaded_file = st.file_uploader("Pilih gambar (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_resized = image.resize((224, 224))
        st.image(image, caption='Gambar Input', use_column_width=True)

        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        class_labels = [f'Kelas {i}' for i in range(10)]
        top_idx = np.argmax(predictions[0])
        top_prob = predictions[0][top_idx]
        st.write(f"Prediksi Terbaik: {class_labels[top_idx]} dengan probabilitas {top_prob:.2f}")
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {predictions[0][i]:.2f}")
else:
    st.info("Tekan tombol 'Muat Model' untuk memulai.")
