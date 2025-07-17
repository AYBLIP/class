import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing import image
import numpy as np

# Definisikan dan daftarkan fungsi aktivasi dan lapisan kustom
@register_keras_serializable()
def swish(x):
    return x * tf.nn.sigmoid(x)

@register_keras_serializable()
class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        # Implementasi khusus jika diperlukan
        return super().call(inputs, training=True)
st.title("Klasifikasi Kue dengan Streamlit")

# Pilihan optimizer
optimizer_options = ['Adam', 'SGD', 'RMSprop']
optimizer_choice = st.selectbox("Optimizer", optimizer_options)

# Tentukan path model secara dinamis
model_path = f'model_{optimizer_choice}.h5'


# Muat model dengan penanganan error
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'FixedDropout': FixedDropout,
            'swish': swish
        }
    )
    st.success(f"Model {optimizer_choice} berhasil dimuat.")
except:
    st.error(f"Gagal memuat model dari {model_path}. Pastikan file model tersedia.")

# Daftar kelas
kelas = ['Kue Dadar Gulung', 'Kue Kastengel', 'Kue Klepon', 'Kue Lapis', 'Kue Lumpur', 'Kue Putri Salju', 'Kue Risoles', 'Kue Serabi']


uploaded_file = st.file_uploader("Unggah gambar kue Anda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, use_container_width=True)

    # Pra-pemrosesan gambar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi jika model berhasil dimuat
    if 'model' in locals():
        pred = model.predict(img_array)
        pred_kelas = np.argmax(pred, axis=1)[0]
        kelas_terpilih = kelas[pred_kelas]
        confidence = np.max(pred) * 100

        st.write(f"Prediksi: **{kelas_terpilih}**")
        st.write(f"Kepercayaan: {confidence:.2f}%")
    else:
        st.warning("Model belum berhasil dimuat. Harap pilih optimizer dan pastikan file model tersedia.")
