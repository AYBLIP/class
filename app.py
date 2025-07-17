import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import register_keras_serializable

# Daftarkan fungsi swish agar bisa dikenali saat load
@register_keras_serializable()
def swish(x):
    return x * tf.nn.sigmoid(x)

@st.cache(allow_output_mutation=True)
def load_model_keras():
    return load_model('final_model_Adam.H5')  # ganti path sesuai model Anda

model = load_model_keras()

kelas = ['Kue 1', 'Kue 2', 'Kue 3', 'Kue 4', 'Kue 5', 'Kue 6', 'Kue 7', 'Kue 8']

st.title("Klasifikasi Gambar Kue dengan EfficientNetB0")
st.write("Upload gambar kue dan prediksi kelasnya.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_idx = np.argmax(pred[0])
    pred_label = kelas[pred_idx]
    pred_confidence = pred[0][pred_idx]

    st.write(f"Prediksi: **{pred_label}**")
    st.write(f"Kepercayaan: {pred_confidence*100:.2f}%")
