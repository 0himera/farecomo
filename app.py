from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import joblib


app = Flask(__name__)

# Загрузка обученной модели
model = tf.keras.models.load_model("facedetectmo.keras")

# Загрузка LabelEncoder
label_encoder = joblib.load("label_encoder.joblib")
@app.route('/predict', methods=['POST'])
def predict():
    # Получение изображения из запроса
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = image.resize((128, 128))  # Изменение размера
    image = np.array(image) / 255.0  # Нормализация
    image = np.expand_dims(image, axis=0)  # Добавление батча

    # Предсказание
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_name = label_encoder.inverse_transform([predicted_class])[0]

    # Возврат результата
    return jsonify({'predicted_name': predicted_name })

if __name__ == '__main__':
    app.run(debug=True)