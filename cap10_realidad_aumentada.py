import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Pirámide AR por color 🎨", page_icon="🧱")

def run():
    st.title("🧱 Detección de color con pirámide 3D (Streamlit + OpenCV + WebRTC)")

    st.markdown("""
    Este demo rastrea una **superficie de color específica** en la cámara y proyecta una **pirámide virtual** sobre ella.  
    Ajusta el color objetivo y el rango HSV hasta obtener la superficie deseada.
    """)

    # --- Controles de color
    st.sidebar.header("🎛️ Control de color (HSV)")
    h = st.sidebar.slider("Tono (H)", 0, 179, 60)
    s = st.sidebar.slider("Saturación mínima (S)", 0, 255, 100)
    v = st.sidebar.slider("Brillo mínimo (V)", 0, 255, 100)
    rango_h = st.sidebar.slider("Rango de H ±", 0, 50, 20)
    color_bgr = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    st.sidebar.markdown(f"Color aproximado: <div style='width:50px;height:25px;background-color:rgb({color_bgr[2]},{color_bgr[1]},{color_bgr[0]});'></div>", unsafe_allow_html=True)

    # Clase procesadora del video
    class ColorPyramid(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # --- Crear máscara del color seleccionado
            lower = np.array([max(0, h - rango_h), s, v])
            upper = np.array([min(179, h + rango_h), 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            # --- Filtrar ruido
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # --- Buscar contornos grandes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if area > 3000:  # evitar ruido
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

                    # --- Proyección de la pirámide 3D
                    h_img, w_img = img.shape[:2]
                    K = np.float64([
                        [w_img, 0, w_img / 2],
                        [0, w_img, h_img / 2],
                        [0, 0, 1]
                    ])
                    dist_coef = np.zeros(4)

                    # Coordenadas base (plano)
                    obj_pts = np.float32([
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0]
                    ])
                    # Coordenadas de la imagen (puntos detectados)
                    img_pts = np.float32(box)

                    # Resolver pose
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coef)
                    if success:
                        # Pirámide: 4 base + 1 vértice
                        pyramid = np.float32([
                            [0, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [0.5, 0.5, -0.8]
                        ])
                        proj, _ = cv2.projectPoints(pyramid, rvec, tvec, K, dist_coef)
                        proj = np.int32(proj).reshape(-1, 2)

                        # Base
                        cv2.drawContours(img, [proj[:4]], -1, (0, 255, 0), -3)

                        # Caras
                        faces = [
                            [proj[0], proj[1], proj[4]],
                            [proj[1], proj[2], proj[4]],
                            [proj[2], proj[3], proj[4]],
                            [proj[3], proj[0], proj[4]],
                        ]
                        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                        for face, color in zip(faces, colors):
                            cv2.drawContours(img, [np.int32(face)], -1, color, -3)

                        # Líneas de aristas
                        for p in proj:
                            cv2.circle(img, tuple(p), 3, (255, 255, 255), -1)

                        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0),
                                       (0, 4), (1, 4), (2, 4), (3, 4)]:
                            cv2.line(img, tuple(proj[i]), tuple(proj[j]), (0, 0, 0), 2)

            return img

    # Iniciar cámara
    webrtc_streamer(
        key="pyramid-color-tracker",
        video_processor_factory=ColorPyramid,
        media_stream_constraints={"video": True, "audio": False},
    )

# Llamar a la app
if __name__ == "__main__":
    run()
