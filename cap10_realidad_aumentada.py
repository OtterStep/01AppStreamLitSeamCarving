import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from collections import namedtuple

# --- Clase PoseEstimator (igual que tu versión base) ---
class PoseEstimator:
    def __init__(self):
        flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.min_matches = 10
        self.cur_target = namedtuple('Current', 'image, rect, keypoints, descriptors, data')
        self.tracked_target = namedtuple('Tracked', 'target, points_prev, points_cur, H, quad')

        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    def add_target(self, image, rect, data=None):
        x0, y0, x1, y1 = rect
        keypoints, descriptors = [], []
        kp, desc = self.detect_features(image)
        for k, d in zip(kp, desc):
            x, y = k.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                keypoints.append(k)
                descriptors.append(d)

        if len(descriptors) == 0:
            st.warning("⚠️ No se encontraron características en esa región.")
            return

        descriptors = np.array(descriptors, dtype='uint8')
        self.feature_matcher.add([descriptors])
        target = self.cur_target(image=image, rect=rect, keypoints=keypoints, descriptors=descriptors, data=data)
        self.tracking_targets.append(target)

    def track_target(self, frame):
        kp, desc = self.detect_features(frame)
        if len(kp) < self.min_matches or desc is None:
            return []

        try:
            matches_all = self.feature_matcher.knnMatch(desc, k=2)
        except:
            return []

        good_matches = [m[0] for m in matches_all if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(good_matches) < self.min_matches:
            return []

        matches_by_index = [[] for _ in range(len(self.tracking_targets))]
        for m in good_matches:
            matches_by_index[m.imgIdx].append(m)

        tracked = []
        for idx, matches in enumerate(matches_by_index):
            if len(matches) < self.min_matches:
                continue

            target = self.tracking_targets[idx]
            pts_prev = [target.keypoints[m.trainIdx].pt for m in matches]
            pts_cur = [kp[m.queryIdx].pt for m in matches]
            pts_prev, pts_cur = np.float32(pts_prev), np.float32(pts_cur)

            H, status = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, 3.0)
            if H is None:
                continue
            status = status.ravel() != 0

            if status.sum() < self.min_matches:
                continue

            pts_prev, pts_cur = pts_prev[status], pts_cur[status]
            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = self.tracked_target(target=target, points_prev=pts_prev, points_cur=pts_cur, H=H, quad=quad)
            tracked.append(track)
        return tracked

    def detect_features(self, frame):
        kp, desc = self.feature_detector.detectAndCompute(frame, None)
        if desc is None:
            desc = []
        return kp, desc

    def clear_targets(self):
        self.feature_matcher.clear()
        self.tracking_targets = []


# --- Procesador de video para Streamlit WebRTC ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_tracker = PoseEstimator()
        self.rect = None
        self.drawing = False
        self.start_point = None
        self.select_mode = False
        self.frame = None

    def toggle_select_mode(self):
        self.select_mode = not self.select_mode

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_display = img.copy()

        # Dibuja rectángulo de selección manual (si está activado)
        if self.select_mode and self.drawing and self.start_point:
            x0, y0 = self.start_point
            x1, y1 = self.end_point
            cv2.rectangle(img_display, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Si hay rect y objetivo, hacer tracking
        if self.rect is not None:
            tracked = self.pose_tracker.track_target(img)
            for t in tracked:
                cv2.polylines(img_display, [np.int32(t.quad)], True, (0, 255, 0), 2)
                for (x, y) in np.int32(t.points_cur):
                    cv2.circle(img_display, (x, y), 2, (0, 0, 255), -1)

        self.frame = img
        return av.VideoFrame.from_ndarray(img_display, format="bgr24")

    def set_roi(self, x0, y0, x1, y1):
        if self.frame is None:
            return
        rect = (x0, y0, x1, y1)
        self.rect = rect
        self.pose_tracker.add_target(self.frame, rect)


# --- Interfaz Streamlit ---
st.title("🧠 AR Surface Tracker - Detección de Superficie Plana")

webrtc_ctx = webrtc_streamer(
    key="tracker",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    vp = webrtc_ctx.video_processor
    st.markdown("### 🎨 Selección de superficie")
    x0 = st.slider("x0", 0, 640, 100)
    y0 = st.slider("y0", 0, 480, 100)
    x1 = st.slider("x1", 0, 640, 200)
    y1 = st.slider("y1", 0, 480, 200)

    if st.button("📍 Seleccionar superficie"):
        vp.set_roi(x0, y0, x1, y1)
        st.success("Superficie añadida para seguimiento.")

    if st.button("❌ Limpiar objetivos"):
        vp.pose_tracker.clear_targets()
        vp.rect = None
        st.info("Superficies limpiadas.")
