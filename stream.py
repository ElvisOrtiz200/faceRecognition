import streamlit as st
import requests
import base64
import cv2
import numpy as np

# Función para cargar la imagen y convertirla a base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Título de la página
st.title("Sistema de Reconocimiento Facial para Asistencia - Colegio Nikola Tesla")

# Subir la imagen
uploaded_image = st.file_uploader("Sube una imagen para tomar la asistencia", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convertir la imagen a un arreglo de numpy
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Mostrar la imagen cargada
    st.image(image, channels="BGR", caption="Imagen cargada para reconocimiento")

    # Convertir la imagen a base64
    image_base64 = image_to_base64(image)

    # Llamada a la API Flask para reconocimiento facial
    api_url = "http://localhost:5000/api/recognize"  # URL de tu API Flask
    response = requests.post(api_url, json={'image': image_base64})

    if response.status_code == 200:
        faces = response.json().get("faces", [])
        if faces:
            st.write(f"Se han reconocido {len(faces)} rostros")
            for face in faces:
                st.write(f"Rostro detectado en las coordenadas: {face['coordinates']}")
                st.write(f"Clase predicha: {face['class']}")
        else:
            st.write("No se han detectado rostros.")
    else:
        st.write("Error en la comunicación con la API.")
