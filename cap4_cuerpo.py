import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

def run():
    st.header("Capítulo 4: Detección de Rostros con Máscara 😷")
    st.markdown("""
    Este ejemplo aplica una **detección de rostros y ojos** usando el clasificador Haar Cascade de OpenCV.
    Puedes:
    - Usar la cámara en tiempo real 🧠  
    - O subir una imagen para analizar 👇
    """)

    # Cargar clasificadores Haar
    face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')

    if face_cascade.empty() or eye_cascade.empty():
        st.error("No se pudieron cargar los clasificadores Haar Cascade. Verifica las rutas.")
        return

    # --- OPCIÓN 1: Cargar imagen ---
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image.convert('RGB'))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                radius = int(0.3 * (w_eye + h_eye))
                cv2.circle(roi_color, center, radius, (0, 255, 0), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Imagen procesada", use_column_width=True)

    # --- OPCIÓN 2: Cámara en tiempo real ---
    st.markdown("### O usa la cámara para detección en vivo:")
    start_camera = st.checkbox("Activar cámara en tiempo real")

    if start_camera:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        stop = st.button("Detener cámara")
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.warning("No se pudo acceder a la cámara.")
                break

            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (x_eye, y_eye, w_eye, h_eye) in eyes:
                    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                    radius = int(0.3 * (w_eye + h_eye))
                    cv2.circle(roi_color, center, radius, (0, 255, 0), 3)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        st.info("Cámara detenida.")
