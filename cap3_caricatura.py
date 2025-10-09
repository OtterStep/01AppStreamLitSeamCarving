import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ============================================================
# Clase de procesamiento del video
# ============================================================
class CartoonProcessor(VideoProcessorBase):
    def __init__(self):
        self.mode = "Normal"

    def cartoonize_image(self, img, ksize=5, sketch_mode=False):
        num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4

        # Escala de grises y detección de bordes
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 7)
        edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
        _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

        # Sketch (blanco y negro)
        if sketch_mode:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Filtro bilateral repetido para suavizar colores
        img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
        for _ in range(num_repetitions):
            img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
        img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)

        # Combinar con bordes
        dst = cv2.bitwise_and(img_output, img_output, mask=mask)
        return dst

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.mode == "Sketch":
            img = self.cartoonize_image(img, sketch_mode=True)
        elif self.mode == "Cartoon":
            img = self.cartoonize_image(img, sketch_mode=False)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================
# Función principal Streamlit
# ============================================================
def run():
    st.header("🎨 Capítulo 3: Filtros Cartoon en Video en Vivo")

    st.markdown("""
    Aplica efectos en **tiempo real** directamente a tu cámara.  
    Usa los filtros disponibles:
    - 🎞 **Normal** → sin efecto  
    - 🖍 **Sketch** → dibujo en blanco y negro  
    - 🎨 **Cartoon** → caricatura en color  
    """)

    # Selección de filtro
    mode = st.radio("Selecciona el filtro:", ["Normal", "Sketch", "Cartoon"], horizontal=True)

    # Stream de cámara en vivo
    ctx = webrtc_streamer(
        key="cartoon-video",
        video_processor_factory=CartoonProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Actualizar modo de filtro en tiempo real
    if ctx.video_processor:
        ctx.video_processor.mode = mode
