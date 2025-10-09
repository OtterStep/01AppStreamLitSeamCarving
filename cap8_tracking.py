import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ============================================================
# CLASE PRINCIPAL DE RASTREADOR (CAMShift + filtros)
# ============================================================
class ObjectTracker(VideoTransformerBase):
    def __init__(self):
        self.drag_start = None
        self.selection = None
        self.tracking_state = 0
        self.track_window = None
        self.hist = None
        self.paused = False
        self.apply_filter = False  # nuevo: activar/desactivar filtro
        self.filter_type = "Normal"  # tipo de filtro

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.paused:
            return img

        img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        vis = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        # Inicialización del seguimiento (si se selecciona un área)
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

        # Rastreo con CAMShift
        if self.tracking_state == 1 and self.hist is not None:
            prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob &= mask
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
            cv2.ellipse(vis, track_box, (0, 255, 0), 2)

        # Filtro opcional
        if self.apply_filter:
            if self.filter_type == "Escala de grises":
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            elif self.filter_type == "Color resaltado":
                hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)
                mask_color = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))  # verde
                vis = cv2.bitwise_and(vis, vis, mask=mask_color)

        return vis


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
def run():
    st.header("🎯 Capítulo 8: Rastreo de Objetos Interactivo (CAMShift)")
    st.markdown("""
    Este capítulo aplica el algoritmo **CAMShift (Continuously Adaptive Mean Shift)**  
    para rastrear objetos seleccionados en tiempo real desde tu cámara.  
    Además, puedes aplicar **filtros visuales en vivo**.
    """)

    # Estado del rastreador
    if "tracker" not in st.session_state:
        st.session_state.tracker = ObjectTracker()

    # Controles de interfaz
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏸ Pausar / Reanudar"):
            st.session_state.tracker.paused = not st.session_state.tracker.paused
    with col2:
        if st.button("🔄 Reiniciar"):
            st.session_state.tracker = ObjectTracker()
    with col3:
        st.session_state.tracker.apply_filter = st.checkbox("🎨 Activar filtro en vivo")

    # Selector de tipo de filtro
    if st.session_state.tracker.apply_filter:
        st.session_state.tracker.filter_type = st.radio(
            "Selecciona un filtro:",
            ["Normal", "Escala de grises", "Color resaltado"],
            index=["Normal", "Escala de grises", "Color resaltado"].index(
                st.session_state.tracker.filter_type
            ),
        )

    st.info("""
    🖱 **Instrucciones:**  
    - Haz clic y arrastra sobre el cuadro de video para seleccionar el objeto.  
    - El sistema lo seguirá con una elipse verde.  
    - Usa los botones para pausar, reiniciar o aplicar un filtro.
    """)

    # Muestra el video en vivo
    webrtc_streamer(
        key="object-tracker",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: st.session_state.tracker,
        media_stream_constraints={"video": True, "audio": False},
    )
