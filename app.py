import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

st.title("Deploy Model dengan Pilihan Optimizer dan File H5")

# Upload file model .h5
model_file = st.file_uploader("Upload Model .h5", type=["h5"])

# Pilihan optimizer
optimizer = st.selectbox(
    "Pilih optimizer:",
    ("Adam", "SGD", "RMSProp")
)

if model_file is not None:
    # Load model
    model = load_model(model_file)
    st.success("Model berhasil dimuat!")

    # Input fitur untuk prediksi
    input_feature = st.number_input("Masukkan nilai fitur untuk prediksi", value=0.0)

    # Lakukan prediksi
    if st.button("Prediksi"):
        # Pastikan input dalam bentuk array
        input_array = np.array([[input_feature]])
        # Jika model membutuhkan proses tertentu sebelum prediksi, lakukan di sini
        pred = model.predict(input_array)
        st.write(f"Hasil prediksi: {pred[0][0]}")
else:
    st.info("Silakan upload file model .h5 terlebih dahulu.")
