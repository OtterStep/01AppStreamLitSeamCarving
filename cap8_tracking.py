import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="Seguimiento de Objetos", layout="wide")

st.title("🎯 Seguimiento de Objetos con CAMShift (interactivo)")
st.markdown("""
Selecciona con el mouse el objeto que deseas rastrear en el video.  
El algoritmo **CAMShift (Continuously Adaptive Mean Shift)** ajustará automáticamente el área de seguimiento.
""")

class ObjectTracker(VideoTransformerBase):
    def __init__(self):
        self.drag_start = None
        self.selection = None
        self.tracking_state = 0
        self.track_window = None
        self.hist = None
        self.frame_hsv = None
        self.mask = None

    def detect_object(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        return hsv, mask

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        vis = img.copy()

        hsv, mask = self.detect_object(img)

        # Si el usuario seleccionó un área, inicializamos el rastreador
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

        # Si ya está rastreando, aplicar CAMShift
        if self.tracking_state == 1 and self.hist is not None:
            prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob &= mask
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
            cv2.ellipse(vis, track_box, (0, 255, 0), 2)

        return vis

    def mouse_event(self, x, y, event, flags, params):
        # Detectar arrastre y clics
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            xo, yo = self.drag_start
            x0, y0 = np.minimum((x, y), (xo, yo))
            x1, y1 = np.maximum((x, y), (xo, yo))
            self.selection = (x0, y0, x1, y1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None

# Lanzar el streamer WebRTC
webrtc_streamer(
    key="camshift-tracker",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=ObjectTracker,
    media_stream_constraints={"video": True, "audio": False},
)
