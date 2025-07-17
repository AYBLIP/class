import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

st.title("Deploy Model dengan Pilihan Optimizer")

# Pilihan optimizer
optimizer_choice = st.selectbox(
    "Pilih optimizer:",
    ("Adam", "RMSProp", "SGD")
)

# Load model sesuai pilihan optimizer
@st.cache(allow_output_mutation=True)
def load_selected_model(opt):
    if opt == "Adam":
        model_path = "model_Adam.h5"
    elif opt == "RMSProp":
        model_path = "model_RMSProp.h5"
    elif opt == "SGD":
        model_path = "model_SGD.h5"
    else:
        model_path = None
    
    if model_path:
        return load_model(model_path)
    else:
        return None

model = load_selected_model(optimizer_choice)

# Input data dari user (sesuaikan dengan bentuk input model)
st.write("Masukkan fitur untuk prediksi:")

# Contoh input satu fitur
input_feature = st.number_input("Masukkan fitur input:", value=0.0)

if st.button("Prediksi"):
    # Pastikan input berbentuk array 2D
    input_array = np.array([[input_feature]])
    prediction = model.predict(input_array)
    st.write(f"Hasil prediksi: {prediction[0][0]:.4f}")
