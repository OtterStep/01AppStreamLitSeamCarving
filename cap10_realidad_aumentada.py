import cv2
import numpy as np
from pose_estimation import PoseEstimator
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
def run(color='green'):
    """Ejecuta el rastreador de color con proyección 3D en Streamlit."""
    webrtc_streamer(
        key="color-tracker",
        video_transformer_factory=lambda: TrackerColorStream(color),
        media_stream_constraints={"video": True, "audio": False},
    )
