import streamlit as st
import pickle
import numpy as np

# Muat model yang sudah dilatih
model = pickle.load(open('model.pkl', 'rb'))

# Judul aplikasi
st.title("Klasifikasi 8 Kelas")

# Input fitur (sesuaikan dengan fitur model Anda)
# Misalnya, jika model menerima 4 fitur numerik
input_features = []

# Contoh: input fitur numerik
for i in range(1, 5):
    feature = st.number_input(f'Fitur {i}', min_value=0.0, max_value=100.0, step=0.1)
    input_features.append(feature)

# Tombol untuk prediksi
if st.button('Prediksi'):
    # Mengubah input menjadi numpy array dan bentuk yang sesuai
    input_array = np.array([input_features])
    # Prediksi
    prediction = model.predict(input_array)
    # Jika output berupa label numerik, bisa di-mapping ke nama kelas
    kelas = prediction[0]
    st.write(f'Kelas Prediksi: {kelas}')
