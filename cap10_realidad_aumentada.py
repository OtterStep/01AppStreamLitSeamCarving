import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# -----------------------------
# Clase Tracker simplificada
# -----------------------------
class PoseEstimator:
    def __init__(self):
        flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.min_matches = 10
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    def detect_features(self, frame):
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors


# -----------------------------
# Procesador de video WebRTC
# -----------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.color_lines = (0, 255, 0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # espejo

        # Detectar características
        keypoints, descriptors = self.pose_estimator.detect_features(img)
        img = cv2.drawKeypoints(img, keypoints, None, color=self.color_lines)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Interfaz Streamlit
# -----------------------------
def run():
    st.set_page_config(page_title="Realidad Aumentada (Streamlit + WebRTC)", layout="wide")
    st.title("🎯 Seguimiento Automático con Cámara Web")

    st.markdown("Este demo usa **streamlit-webrtc** para acceder a la cámara desde el navegador.")

    webrtc_streamer(
        key="pose-tracker",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == "__main__":
    run()
