import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

class ObjectTracker(VideoTransformerBase):
    def __init__(self):
        self.tracker = None
        self.bbox = None
        self.initialized = False
        self.filter_type = "None"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.tracker is not None and self.initialized:
            success, bbox = self.tracker.update(img)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.ellipse(img, ((x + w // 2), (y + h // 2)), (w // 2, h // 2), 0, 0, 360, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


def run():
    st.title("🎯 Seguimiento de Objetos en Video (Object Tracking)")
    st.markdown("""
    🖱 **Instrucciones:**
    - Haz clic y arrastra sobre el cuadro de video para seleccionar el objeto.  
    - El sistema lo seguirá con una elipse verde.  
    - Usa los botones para pausar, reiniciar o aplicar un filtro.
    """)

    # 🔹 Inicializar el tracker en session_state si no existe
    if "tracker" not in st.session_state:
        st.session_state.tracker = ObjectTracker()

    # 🔹 Opciones de filtro
    filtro = st.selectbox("Selecciona filtro:", ["None", "Gray", "Blur"])
    st.session_state.tracker.filter_type = filtro

    # 🔹 Iniciar WebRTC
    webrtc_streamer(
        key="object-tracker",
        video_transformer_factory=lambda: st.session_state.tracker,
        media_stream_constraints={"video": True, "audio": False},
    )
