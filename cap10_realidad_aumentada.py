import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ------------------------------
#   TrackerColorStream
# ------------------------------
class TrackerColorStream(VideoTransformerBase):
    def __init__(self, color='green'):
        self.tracker = PoseEstimator()
        self.color = color
        self.rect = None
        self.overlay_vertices = np.float32([[0, 0, 0], [0, 1, 0],
                                            [1, 1, 0], [1, 0, 0], [0.5, 0.5, 4]])
        self.overlay_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                              (0, 4), (1, 4), (2, 4), (3, 4)]
        self.color_lines = (0, 0, 0)
        self.graphics_counter = 0
        self.time_counter = 0
        self.paused = False

    # ------------------------------
    # Detecta región por color (HSV)
    # ------------------------------
    def detect_color_region(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.color == 'green':
            lower = np.array([40, 50, 50])
            upper = np.array([80, 255, 255])
        elif self.color == 'blue':
            lower = np.array([90, 50, 50])
            upper = np.array([130, 255, 255])
        elif self.color == 'red':
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([160, 100, 100])
            upper2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 | mask2
            return self.get_largest_contour_rect(mask)
        else:
            return None

        mask = cv2.inRange(hsv, lower, upper)
        return self.get_largest_contour_rect(mask)

    def get_largest_contour_rect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, x + w, y + h)

    # ------------------------------
    # Dibuja plano 3D
    # ------------------------------
    def overlay_graphics(self, img, tracked):
        x_start, y_start, x_end, y_end = tracked.target.rect
        quad_3d = np.float32([
            [x_start, y_start, 0],
            [x_end, y_start, 0],
            [x_end, y_end, 0],
            [x_start, y_end, 0]
        ])
        h, w = img.shape[:2]
        K = np.float64([
            [w, 0, 0.5*(w-1)],
            [0, w, 0.5*(h-1)],
            [0, 0, 1.0]
        ])
        dist_coef = np.zeros(4)
        _, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, K, dist_coef)

        self.time_counter += 1
        if not self.time_counter % 20:
            self.graphics_counter = (self.graphics_counter + 1) % 8

        self.overlay_vertices = np.float32([
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0.5, 0.5, self.graphics_counter]
        ])

        verts = self.overlay_vertices * [
            (x_end - x_start),
            (y_end - y_start),
            -(x_end - x_start) * 0.3
        ] + (x_start, y_start, 0)

        verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
        verts_floor = np.int32(verts).reshape(-1, 2)

        cv2.drawContours(img, [verts_floor[:4]], -1, (0, 255, 0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[:2], verts_floor[4:5]))], -1, (0, 255, 0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[1:3], verts_floor[4:5]))], -1, (255, 0, 0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[2:4], verts_floor[4:5]))], -1, (0, 0, 150), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[3:4], verts_floor[0:1], verts_floor[4:5]))], -1, (255, 255, 0), -3)

        for i, j in self.overlay_edges:
            (x1, y1), (x2, y2) = verts[i], verts[j]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), self.color_lines, 2)

    # ------------------------------
    # Procesa cada frame
    # ------------------------------
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if not self.paused:
            rect = self.detect_color_region(img)
            if rect and self.rect is None:
                self.rect = rect
                self.tracker.add_target(img, rect)

            if self.rect is not None:
                tracked = self.tracker.track_target(img)
                for item in tracked:
                    cv2.polylines(img, [np.int32(item.quad)], True, (0, 0, 0), 2)
                    self.overlay_graphics(img, item)

        if self.rect:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img


# ------------------------------
#   Función pública para Streamlit
# ------------------------------
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ======================
# 1. Funciones auxiliares
# ======================
def get_color_range(color_name):
    """Devuelve los rangos HSV para un color dado"""
    color_ranges = {
        "rojo": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                 (np.array([170, 120, 70]), np.array([180, 255, 255]))],
        "verde": [(np.array([35, 60, 60]), np.array([85, 255, 255]))],
        "azul": [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
        "amarillo": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
    }
    return color_ranges.get(color_name, [])


# ======================
# 2. Procesador de video
# ======================
class ColorPlaneTracker(VideoProcessorBase):
    def __init__(self):
        self.selected_colors = ["verde"]

    def set_colors(self, colors):
        self.selected_colors = colors

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color in self.selected_colors:
            for (lower, upper) in get_color_range(color):
                mask = cv2.inRange(hsv, lower, upper)
                mask_total = cv2.bitwise_or(mask_total, mask)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 1000:
                # Polígono aproximado (plano detectado)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

                # Centro del contorno
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                    # Proyección de pirámide 3D
                    h, w = img.shape[:2]
                    K = np.float64([[w, 0, 0.5 * (w - 1)],
                                    [0, w, 0.5 * (h - 1)],
                                    [0, 0, 1.0]])
                    dist_coef = np.zeros(4)

                    side = 100
                    quad_3d = np.float32([
                        [cx - side, cy - side, 0],
                        [cx + side, cy - side, 0],
                        [cx + side, cy + side, 0],
                        [cx - side, cy + side, 0],
                        [cx, cy, -side * 2]
                    ])

                    rvec = np.zeros((3, 1))
                    tvec = np.zeros((3, 1))
                    verts = cv2.projectPoints(quad_3d, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
                    verts = np.int32(verts)

                    # Dibujar pirámide
                    cv2.drawContours(img, [verts[:4]], -1, (0, 255, 0), -3)
                    for i in range(4):
                        cv2.line(img, tuple(verts[i]), tuple(verts[4]), (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ======================
# 3. Interfaz Streamlit
# ======================
def run():
    st.title("🎨 Realidad Aumentada con Selector de Color")
    st.markdown(
        "Selecciona uno o varios colores. "
        "Se detectará la superficie más grande con ese color y se proyectará una pirámide 3D sobre ella."
    )

    colors = st.multiselect(
        "Selecciona colores a rastrear:",
        ["rojo", "verde", "azul", "amarillo"],
        default=["verde"],
    )

    ctx = webrtc_streamer(
        key="color-plane-tracker",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ColorPlaneTracker,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.set_colors(colors)


if __name__ == "__main__":
    run()


import cv2
import numpy as np
from collections import namedtuple

class PoseEstimator(object):
    def __init__(self): 
        flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.min_matches = 10
        self.cur_target = namedtuple('Current', 'image, rect, keypoints, descriptors, data')
        self.tracked_target = namedtuple('Tracked', 'target, points_prev, points_cur, H, quad')
        self.feature_detector = cv2.ORB_create()
        self.feature_detector.setMaxFeatures(1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    def add_target(self, image, rect, data=None):
        x_start, y_start, x_end, y_end = rect
        keypoints, descriptors = [], []
        for keypoint, descriptor in zip(*self.detect_features(image)):
            x, y = keypoint.pt
            if x_start <= x <= x_end and y_start <= y <= y_end:
                keypoints.append(keypoint)
                descriptors.append(descriptor)

        descriptors = np.array(descriptors, dtype='uint8')
        self.feature_matcher.add([descriptors])
        target = self.cur_target(image=image, rect=rect, keypoints=keypoints, descriptors=descriptors, data=None)
        self.tracking_targets.append(target)

    def track_target(self, frame):
        self.cur_keypoints, self.cur_descriptors = self.detect_features(frame)
        if len(self.cur_keypoints) < self.min_matches:
            return []

        try:
            matches = self.feature_matcher.knnMatch(self.cur_descriptors, k=2)
        except Exception as e:
            print('Invalid target, please select another with features to extract')
            return []

        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < self.min_matches:
            return []

        matches_by_index = [[] for _ in range(len(self.tracking_targets))]
        for m in matches:
            matches_by_index[m.imgIdx].append(m)

        tracked = []
        for image_index, matches in enumerate(matches_by_index):
            if len(matches) < self.min_matches:
                continue

            target = self.tracking_targets[image_index]
            points_prev = [target.keypoints[m.trainIdx].pt for m in matches]
            points_cur = [self.cur_keypoints[m.queryIdx].pt for m in matches]
            points_prev, points_cur = np.float32((points_prev, points_cur))

            H, status = cv2.findHomography(points_prev, points_cur, cv2.RANSAC, 3.0)
            if H is None:
                continue

            status = status.ravel() != 0
            if status.sum() < self.min_matches:
                continue

            points_prev, points_cur = points_prev[status], points_cur[status]
            x_start, y_start, x_end, y_end = target.rect
            quad = np.float32([[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            track = self.tracked_target(target=target, points_prev=points_prev, points_cur=points_cur, H=H, quad=quad)
            tracked.append(track)

        tracked.sort(key=lambda x: len(x.points_prev), reverse=True)
        return tracked

    def detect_features(self, frame):
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors

    def clear_targets(self):
        self.feature_matcher.clear()
        self.tracking_targets = []
