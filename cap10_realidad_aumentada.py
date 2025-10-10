import streamlit as st
import cv2
import numpy as np
import time
from collections import namedtuple 

# -----------------------------
# Clase Tracker simplificada para Streamlit
# -----------------------------
class PoseEstimator(object): 
    def __init__(self): 
        # Use locality sensitive hashing algorithm 
        flann_params = dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1) 
 
        self.min_matches = 10 
        self.cur_target = namedtuple('Current', 'image, rect, keypoints, descriptors, data')
        self.tracked_target = namedtuple('Tracked', 'target, points_prev, points_cur, H, quad') 
 
        self.feature_detector = cv2.ORB_create()
        self.feature_detector.setMaxFeatures(1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {}) 
        self.tracking_targets = [] 
 
    # Function to add a new target for tracking 
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
 
    # To get a list of detected objects 
    def track_target(self, frame): 
        self.cur_keypoints, self.cur_descriptors = self.detect_features(frame) 

        if len(self.cur_keypoints) < self.min_matches: return []
        try: matches = self.feature_matcher.knnMatch(self.cur_descriptors, k=2)
        except Exception as e:
            print('Invalid target, please select another with features to extract')
            return []
        matches = [match[0] for match in matches if len(match) == 2 and match[0].distance < match[1].distance * 0.75] 
        if len(matches) < self.min_matches: return [] 
 
        matches_using_index = [[] for _ in range(len(self.tracking_targets))] 
        for match in matches: 
            matches_using_index[match.imgIdx].append(match) 
 
        tracked = [] 
        for image_index, matches in enumerate(matches_using_index): 
            if len(matches) < self.min_matches: continue 
 
            target = self.tracking_targets[image_index] 
            points_prev = [target.keypoints[m.trainIdx].pt for m in matches]
            points_cur = [self.cur_keypoints[m.queryIdx].pt for m in matches]
            points_prev, points_cur = np.float32((points_prev, points_cur))
            H, status = cv2.findHomography(points_prev, points_cur, cv2.RANSAC, 3.0) 
            status = status.ravel() != 0

            if status.sum() < self.min_matches: continue 
 
            points_prev, points_cur = points_prev[status], points_cur[status] 
 
            x_start, y_start, x_end, y_end = target.rect 
            quad = np.float32([[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            track = self.tracked_target(target=target, points_prev=points_prev, points_cur=points_cur, H=H, quad=quad) 
            tracked.append(track) 
 
        tracked.sort(key = lambda x: len(x.points_prev), reverse=True) 
        return tracked 
 
    # Detect features in the selected ROIs and return the keypoints and descriptors 
    def detect_features(self, frame): 
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None) 
        if descriptors is None: descriptors = [] 
        return keypoints, descriptors 
 
    # Function to clear all the existing targets 
    def clear_targets(self): 
        self.feature_matcher.clear() 
        self.tracking_targets = []  
 
class TrackerAuto:
    def __init__(self, capId=0, scaling_factor=0.8):
        self.cap = cv2.VideoCapture(capId)
        self.scaling_factor = scaling_factor
        self.tracker = PoseEstimator()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo acceder a la cámara o al video.")

        self.frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                                interpolation=cv2.INTER_AREA)

        h, w = self.frame.shape[:2]
        box_size = min(h, w) // 4
        x1, y1 = w//2 - box_size//2, h//2 - box_size//2
        x2, y2 = x1 + box_size, y1 + box_size
        self.rect = (x1, y1, x2, y2)

        self.tracker.add_target(self.frame, self.rect)

        # Parámetros del gráfico 3D
        self.overlay_vertices = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0],
                                            [1, 0, 0], [0.5, 0.5, 4]])
        self.overlay_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                              (0, 4), (1, 4), (2, 4), (3, 4)]
        self.color_lines = (0, 0, 0)
        self.color_base = (0, 255, 0)
        self.graphics_counter = 0
        self.time_counter = 0

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

        # Caras del plano
        cv2.drawContours(img, [verts_floor[:4]], -1, self.color_base, -3)
        cv2.drawContours(img, [np.vstack((verts_floor[:2], verts_floor[4:5]))], -1, (0, 255, 0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[1:3], verts_floor[4:5]))], -1, (255, 0, 0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[2:4], verts_floor[4:5]))], -1, (0, 0, 150), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[3:4], verts_floor[0:1], verts_floor[4:5]))],
                         -1, (255, 255, 0), -3)

        for i, j in self.overlay_edges:
            (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
            cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), self.color_lines, 2)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, None, fx=self.scaling_factor, fy=self.scaling_factor,
                           interpolation=cv2.INTER_AREA)
        self.frame = frame.copy()
        img = self.frame.copy()

        tracked = self.tracker.track_target(self.frame)
        for item in tracked:
            cv2.polylines(img, [np.int32(item.quad)], True, self.color_lines, 2)
            for (x, y) in np.int32(item.points_cur):
                cv2.circle(img, (x, y), 2, self.color_lines, -1)
            self.overlay_graphics(img, item)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


# -----------------------------
# Interfaz Streamlit
# -----------------------------
def run():
    st.set_page_config(page_title="Pose Estimation Tracker", layout="wide")
    st.title("🎯 Seguimiento Automático con Plano 3D (Streamlit)")
    st.write("Ejemplo adaptado del ejercicio original, sin selección manual.")

    start_button = st.button("Iniciar Cámara")

    if start_button:
        tracker = TrackerAuto(0, 0.8)
        stframe = st.empty()

        while True:
            frame = tracker.get_frame()
            if frame is None:
                st.warning("No se pudo leer más frames.")
                break

            stframe.image(frame, channels="RGB", use_container_width=True)
            time.sleep(0.03)  # control de FPS (~30fps aprox)


