import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Color Tracker", page_icon="🎨")

st.title("🎨 Seguimiento de color con Streamlit + WebRTC")

st.markdown("""
Selecciona un color para rastrear en el video en tiempo real.  
El sistema resaltará las áreas que coincidan con el color elegido.
""")

# Selector de color
color_option = st.selectbox(
    "Selecciona el color a rastrear:",
    ("Rojo", "Verde", "Azul")
)

# Umbrales HSV para distintos colores
COLOR_RANGES = {
    "Rojo": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],  # dos rangos para rojo
    "Verde": [(36, 100, 100), (86, 255, 255)],
    "Azul": [(94, 80, 2), (126, 255, 255)]
}


class ColorTracker(VideoTransformerBase):
    def __init__(self):
        self.color = color_option

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color = self.color

        if color == "Rojo":
            lower1, upper1, lower2, upper2 = COLOR_RANGES[color]
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = mask1 + mask2
        else:
            lower, upper = COLOR_RANGES[color]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Limpieza de la máscara
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Encuentra los contornos del color
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return img


webrtc_streamer(
    key="color-tracker",
    video_processor_factory=ColorTracker,
    media_stream_constraints={"video": True, "audio": False},
)
