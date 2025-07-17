import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Judul aplikasi
st.title("Model Deploy dengan Pilihan Optimizer (Adam, SGD, RMSprop)")

# Pilihan optimizer
optimizer_option = st.selectbox(
    "Pilih optimizer:",
    ("Adam", "SGD", "RMSprop")
)

# Parameter learning rate
learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)

# Fungsi untuk mendapatkan optimizer sesuai pilihan
def get_optimizer(name, lr):
    if name == "Adam":
        return Adam(learning_rate=lr)
    elif name == "SGD":
        return SGD(learning_rate=lr)
    elif name == "RMSprop":
        return RMSprop(learning_rate=lr)

# Tombol untuk memulai training
if st.button("Train Model"):
    with st.spinner("Training model..."):
        # Membuat data dummy
        import numpy as np
        X = np.random.rand(1000, 10)
        y = (np.sum(X, axis=1) > 5).astype(int)

        # Membuat model sederhana
        model = Sequential([
            Dense(16, activation='relu', input_shape=(10,)),
            Dense(1, activation='sigmoid')
        ])

        # Kompilasi model dengan optimizer yang dipilih
        optimizer = get_optimizer(optimizer_option, learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Melatih model
        history = model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        # Menampilkan hasil training
        st.success("Training selesai!")
        st.write("Loss terakhir:", history.history['loss'][-1])
        st.write("Akurasi terakhir:", history.history['accuracy'][-1])
