import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ============================================================
# CLASE PRINCIPAL DE RASTREADOR (CAMShift)
# ============================================================
class ObjectTracker(VideoTransformerBase):
    def __init__(self):
        self.drag_start = None
        self.selection = None
        self.tracking_state = 0
        self.track_window = None
        self.hist = None
        self.paused = False  # nuevo estado de pausa

    def transform(self, frame):
        # Si está en pausa, mostrar frame sin procesar
        img = frame.to_ndarray(format="bgr24")
        if self.paused:
            return img

        img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        vis = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        # Inicialización al seleccionar una región
        if self.selection is not None:
            x0, y0, x1, y1 = self.selection
            self.track_window = (x0, y0, x1 - x0, y1 - y0)
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self.hist = hist
            self.selection = None
            self.tracking_state = 1

        # Si está rastreando, aplicar CAMShift
        if self.tracking_state == 1 and self.hist is not None:
            prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob &= mask
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
            cv2.ellipse(vis, track_box, (0, 255, 0), 2)

        return vis


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
def run():
    st.header("🎯 Capítulo 8: Rastreo de Objetos Interactivo (CAMShift)")
    st.markdown("""
    En este capítulo aplicamos el algoritmo **CAMShift (Continuously Adaptive Mean Shift)**  
    para rastrear objetos seleccionados en tiempo real desde tu cámara.
    """)

    st.info("""
    🖱 **Instrucciones:**  
    - Haz clic y arrastra en el cuadro de video para seleccionar un objeto.  
    - El sistema lo seguirá automáticamente con una elipse verde.  
    - Puedes **pausar** o **reiniciar** el seguimiento en cualquier momento.
    """)

    # Estado compartido entre la app y el rastreador
    if "tracker" not in st.session_state:
        st.session_state.tracker = ObjectTracker()

    col1, col2 = st.columns(2)
    with col1:
        pause_btn = st.button("⏸ Pausar / Reanudar rastreo")
    with col2:
        reset_btn = st.button("🔄 Reiniciar seguimiento")

    if pause_btn:
        st.session_state.tracker.paused = not st.session_state.tracker.paused

    if reset_btn:
        st.session_state.tracker = ObjectTracker()

    webrtc_streamer(
        key="object-tracker",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: st.session_state.tracker,
        media_stream_constraints={"video": True, "audio": False},
    )
