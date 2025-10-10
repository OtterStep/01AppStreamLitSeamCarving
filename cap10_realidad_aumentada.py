import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Pirámide sobre superficie blanca 🤍", page_icon="🧱")

def run():
    st.title("🧱 Realidad aumentada sobre superficie blanca (Pirámide 3D)")
    st.markdown("""
    Este demo detecta una **superficie blanca** en la cámara y proyecta una **pirámide 3D virtual** sobre ella.  
    Asegúrate de tener buena iluminación y una hoja o fondo blanco visible en cámara.
    """)

    mostrar_mascara = st.sidebar.checkbox("🔍 Mostrar máscara de detección", False)

    class WhiteSurfacePyramid(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Convertir a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # --- Rango para detectar superficies blancas ---
            # H: 0-179 (cualquiera), S: baja (0-50), V: alta (200-255)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([179, 50, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # --- Filtrar ruido ---
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            if mostrar_mascara:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                return cv2.hconcat([img, mask_bgr])

            # --- Buscar contornos ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if area > 4000:  # evitar ruido
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int32(cv2.convexHull(box))
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

                    # --- Proyección de la pirámide 3D ---
                    h_img, w_img = img.shape[:2]
                    K = np.float64([
                        [w_img, 0, w_img / 2],
                        [0, w_img, h_img / 2],
                        [0, 0, 1]
                    ])
                    dist_coef = np.zeros(4)

                    # Orden consistente
                    box = np.array(sorted(box, key=lambda p: (p[1], p[0])))

                    obj_pts = np.float32([
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0]
                    ])
                    img_pts = np.float32(box[:4])

                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coef)
                    if success:
                        # Pirámide 3D: base cuadrada + vértice
                        pyramid = np.float32([
                            [0, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [0.5, 0.5, -1.2]
                        ])
                        proj, _ = cv2.projectPoints(pyramid, rvec, tvec, K, dist_coef)
                        proj = np.int32(proj).reshape(-1, 2)

                        # Base
                        cv2.drawContours(img, [proj[:4]], -1, (0, 255, 0), -3)

                        # Caras laterales
                        faces = [
                            [proj[0], proj[1], proj[4]],
                            [proj[1], proj[2], proj[4]],
                            [proj[2], proj[3], proj[4]],
                            [proj[3], proj[0], proj[4]],
                        ]
                        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                        for face, color in zip(faces, colors):
                            cv2.drawContours(img, [np.int32(face)], -1, color, -3)

                        # Aristas
                        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0),
                                       (0, 4), (1, 4), (2, 4), (3, 4)]:
                            cv2.line(img, tuple(proj[i]), tuple(proj[j]), (0, 0, 0), 2)

            return img

    # Iniciar cámara
    webrtc_streamer(
        key="white-pyramid",
        video_processor_factory=WhiteSurfacePyramid,
        media_stream_constraints={"video": True, "audio": False},
    )


