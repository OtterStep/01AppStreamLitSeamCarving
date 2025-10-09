import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image

# ============================================================
# Clase procesadora de video (detección en vivo)
# ============================================================
class FaceMaskProcessor(VideoProcessorBase):
    def __init__(self):
        # Cargar clasificadores Haar Cascade
        self.face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
        self.eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')

        if self.face_cascade.empty() or self.eye_cascade.empty():
            st.error("❌ No se pudieron cargar los clasificadores Haar Cascade.")
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                radius = int(0.3 * (w_eye + h_eye))
                cv2.circle(roi_color, center, radius, (0, 255, 0), 3)
        return frame

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.detect_faces(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================
# Función principal Streamlit
# ============================================================
def run():
    st.header("Capítulo 4: Detección de Rostros y Ojos 😷")
    st.markdown("""
    Este ejemplo aplica detección de **rostros y ojos** usando los clasificadores Haar Cascade de OpenCV.  
    Puedes:
    - 🖼️ Subir una **imagen** para analizar  
    - 🎥 Usar tu **cámara** en tiempo real con WebRTC
    """)

    # -----------------------------
    # OPCIÓN 1: Subir imagen
    # -----------------------------
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image.convert('RGB'))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                radius = int(0.3 * (w_eye + h_eye))
                cv2.circle(roi_color, center, radius, (0, 255, 0), 3)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Resultado de detección", use_container_width=True)

    # -----------------------------
    # OPCIÓN 2: Cámara en tiempo real con WebRTC
    # -----------------------------
    st.markdown("### O usa tu cámara para detección en vivo 👇")

    webrtc_streamer(
        key="face-detection",
        video_processor_factory=FaceMaskProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

