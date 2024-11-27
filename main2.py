from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import os
import sys
from flask_cors import CORS
import dlib
import pymysql
import base64

app = Flask(__name__)
CORS(app)

DB_CONFIG = {
    'host': 'autorack.proxy.rlwy.net',
    'port': 58330,
    'user': 'root',
    'password': 'rqSlIlxswhTrVEEdgwzoLIXFAjWGFkbx',
    'database': 'pruebaApi'
}

# Asegurarse de que existe la carpeta "uploads"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('NUEVONUEVONUEVO2222222222.h5')
sys.stdout.reconfigure(encoding='utf-8')

# Inicializar el detector de rostros
detector = dlib.get_frontal_face_detector()


@app.route('/asistencias', methods=['GET'])
def get_asistencias():
    try:
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        query = "SELECT * FROM asistenciaPersonal"
        cursor.execute(query)
        asistencias = cursor.fetchall()
        return jsonify(asistencias), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   

@app.route('/api/recognize', methods=['POST'])
def recognize():
    # Obtener la imagen desde el JSON
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decodificar la imagen base64
    image_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convertir el frame a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    faces = detector(gray)

    results = []  # Lista para almacenar los resultados
    for face in faces:
        # Extraer coordenadas del rectángulo del rostro
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Recortar el área del rostro para la predicción
        face_crop = frame[y:y + h, x:x + w]

        # Preprocesar la imagen para el modelo
        face_resized = cv2.resize(face_crop, (100, 100))  # Asegúrate de que coincida con el tamaño de entrada
        face_normalized = face_resized.astype('float32') / 255.0  # Normalizar
        face_expanded = np.expand_dims(face_normalized, axis=0)  # Añadir dimensión de batch

        # Realizar la predicción
        predictions = model.predict(face_expanded)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Obtener el índice de clase

        # Almacenar el resultado
        results.append({'class': int(predicted_class), 'coordinates': {'x': x, 'y': y, 'w': w, 'h': h}})
        print(results)
    # Retornar los resultados como JSON
    
    return jsonify({'faces': results})

if __name__ == '__main__':
    app.run(debug=True)
