import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing import image
import numpy as np

# Definisikan dan daftarkan fungsi aktivasi dan lapisan kustom jika diperlukan
@register_keras_serializable()
def swish(x):
    return x * tf.nn.sigmoid(x)

@register_keras_serializable()
class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# Fungsi untuk membangun dan melatih model (contoh)
def build_and_train_model(optimizer_name):
    # Buat model sederhana sebagai contoh
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        FixedDropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])

    # Pilih optimizer berdasarkan input pengguna
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD()
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop()
    else:
        optimizer = tf.keras.optimizers.Adam()  # default

    # Kompilasi model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Data dummy untuk pelatihan (sebagai contoh)
    # Ganti dengan data nyata jika diperlukan
    x_train = np.random.rand(100, 224, 224, 3)
    y_train = np.random.randint(0, 8, 100)

    # Latih model (sebagai contoh)
    model.fit(x_train, y_train, epochs=1, verbose=0)
    return model

# Streamlit UI
st.title("Pelatihan Model dengan Pilihan Optimizer")

# Pilihan optimizer
optimizer_option = st.selectbox(
    "Pilih optimizer yang ingin digunakan:",
    ("adam", "sgd", "rmsprop")
)

# Tombol untuk melatih
if st.button("Latih Model"):
    with st.spinner(f"Melatih dengan optimizer {optimizer_option}..."):
        model = build_and_train_model(optimizer_option)
    st.success("Model telah dilatih!")

# Jika ingin memuat model dan melakukan prediksi
# (kode sebelumnya tetap sama, gunakan model yang sudah dilatih)
