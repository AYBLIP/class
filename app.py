import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models

# Judul aplikasi
st.title("Model Deploy dengan Pilihan Optimizer (Adam, SGD, RMSprop)")

# Pilihan optimizer dari pengguna
optimizer_choice = st.selectbox(
    "Pilih optimizer:",
    ("Adam", "SGD", "RMSprop")
)

# Fungsi untuk mendapatkan optimizer sesuai pilihan
def get_optimizer(name):
    if name == "Adam":
        return tf.keras.optimizers.Adam()
    elif name == "SGD":
        return tf.keras.optimizers.SGD()
    elif name == "RMSprop":
        return tf.keras.optimizers.RMSprop()

# Tampilkan ketika tombol ditekan
if st.button("Latih Model"):
    st.write(f"Melatih model dengan optimizer: {optimizer_choice}")

    # Load dataset MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisasi data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Membuat model sederhana
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model dengan optimizer yang dipilih
    optimizer = get_optimizer(optimizer_choice)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Melatih model
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Evaluasi model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"Akurasi pada data test: {accuracy*100:.2f}%")
    
    # Optional: tampilkan grafik training
    import matplotlib.pyplot as plt

    # Plot akurasi
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Akurasi')
    ax.legend()
    st.pyplot(fig)
