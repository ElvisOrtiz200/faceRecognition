from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import sys
from flask_cors import CORS
import pymysql
import base64

app = Flask(__name__)
CORS(app)

DB_CONFIG = {
    'host': 'autorack.proxy.rlwy.net',
    'port': 58330,
    'user': 'root',
    'password': 'rqSlIlxswhTrVEEdgwzoLIXFAjWGFkbx',
    'database': 'pruebaApi',
    'cursorclass': pymysql.cursors.DictCursor,  # Esto asegura que los resultados sean diccionarios
    'connect_timeout': 10  # Establece un tiempo de espera de conexión para evitar bloqueos largos
}

@app.route('/asistencias', methods=['GET'])
def get_asistencias():
    try:
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()
        query = "SELECT * FROM asistenciaPersonal"
        cursor.execute(query)
        asistencias = cursor.fetchall()
        return jsonify(asistencias), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   

@app.route('/api/recognize', methods=['POST'])
def recognize():
    sys.stdout.reconfigure(encoding='utf-8')
    model = load_model('NUEVONUEVONUEVO2222222222.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []  # Lista para almacenar los resultados
    for face in faces:
        # Extraer coordenadas del rectángulo del rostro
        x, y, w, h = face

        # Recortar el área del rostro para la predicción
        face_crop = frame[y:y + h, x:x + w]

        # Preprocesar la imagen para el modelo
        face_resized = cv2.resize(face_crop, (100, 100))  # Asegúrate de que coincida con el tamaño de entrada
        face_normalized = face_resized.astype('float32') / 255.0  # Normalizar
        face_expanded = np.expand_dims(face_normalized, axis=0)  # Añadir dimensión de batch

        # Realizar la predicción
        predictions = model.predict(face_expanded)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Obtener el índice de clase
        results = [{'class': 0, 'coordinates': {'x': 230, 'y': 269, 'w': 204, 'h': 204}}]
            
            # Convert any np.int32 in the result to a native Python int
        for result in results:
                result['class'] = int(result['class'])  # ensure 'class' is a regular int
                coords = result.get('coordinates', {})
                for key, value in coords.items():
                    coords[key] = int(value)        # convert coordinates values to regular ints
    
    return jsonify({'faces': results})
    # Retornar los resultados como JSON



@app.route('/api/asistencia', methods=['POST'])
def insertar_asistencia():
  
        # Obtener los datos del cuerpo de la solicitud
        data = request.json
        fecha = data.get('fecha')
        estado = data.get('estado')  # Entrada o Salida
        personal = data.get('personal')
        descripcion = data.get('descripcion')  # Descripción adicional (Entrada/Salida)

        # Validación de que los datos necesarios están presentes
        if not all([ fecha, estado, personal, descripcion]):
            return jsonify({'error': 'Faltan datos en la solicitud'}), 400

        # Conexión a la base de datos
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Insertar en la tabla asistenciaPersonal
        query = """
        INSERT INTO asistenciaPersonal ( fecha, estado, personal, descripcion)
        VALUES ( %s, %s, %s, %s)
        """
        cursor.execute(query, (fecha, estado, personal, descripcion))
        connection.commit()

        return jsonify({'message': 'Asistencia registrada correctamente'}), 200




    




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)