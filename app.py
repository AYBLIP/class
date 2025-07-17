import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Path ke model yang sudah dilatih
model_paths = [
    'model_Adam.h5',
    'model_SGD.h5',
    'model_RMSprop.h5'
]

# Muat semua model
models = [load_model(path) for path in model_paths]

# Fungsi prediksi menggunakan ketiga model
def predict_with_models(img_path):
    # Load dan preprocess gambar
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # buat batch dimension

    # Prediksi dari tiap model
    predictions = []
    for i, model in enumerate(models):
        pred = model.predict(x)
        predictions.append(pred)
        print(f"Prediksi dari Model {i+1} ({model_paths[i]}): {pred}")

    # Jika ingin menggabungkan prediksi, bisa dilakukan rata-rata
    avg_prediction = np.mean(predictions, axis=0)
    print(f"\nPrediksi gabungan (rata-rata): {avg_prediction}")

    # Tentukan kelas prediksi tertinggi
    predicted_class = np.argmax(avg_prediction)
    print(f"Prediksi kelas: {predicted_class}")

# Contoh penggunaan
predict_with_models('path/ke/gambar_anda.jpg')
