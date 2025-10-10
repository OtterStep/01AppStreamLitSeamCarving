import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Pirámide sobre superficie blanca 🤍", page_icon="🧱")


def run():
    st.title("🧱 Realidad aumentada sobre superficie blanca (Pirámide 3D)")
    st.markdown(
        "Coloca una hoja o fondo blanco frente a la cámara. La pirámide aparecerá sobre la superficie blanca detectada."
    )

    # --- Controles de calibración ---
    st.sidebar.header("🔧 Calibración")
    v_min = st.sidebar.slider("V mínimo (brillo) para considerar blanco", 150, 255, 200)
    s_max = st.sidebar.slider("S máxima (saturación) para considerar blanco", 0, 120, 60)
    area_thresh = st.sidebar.slider("Área mínima (px) para activar pirámide", 500, 20000, 4000, step=500)
    mostrar_mascara = st.sidebar.checkbox("🔍 Mostrar máscara de detección (lado a lado)", False)

    st.sidebar.markdown("Si no aparece la pirámide: baja `V mínimo` o reduce `Área mínima`.")

    # Helper: ordenar puntos en el orden (tl, tr, br, bl)
    def order_points(pts):
        # pts: (4,2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    class WhiteSurfacePyramid(VideoProcessorBase):
        def __init__(self, v_min, s_max, area_thresh, mostrar_mascara):
            self.v_min = int(v_min)
            self.s_max = int(s_max)
            self.area_thresh = int(area_thresh)
            self.mostrar_mascara = bool(mostrar_mascara)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Convertir a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Rango para blanco: H libre, S bajo, V alto
            lower_white = np.array([0, 0, self.v_min])
            upper_white = np.array([179, self.s_max, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Operaciones morfológicas para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Si el usuario quiere ver la máscara, concatena y devuelve
            if self.mostrar_mascara:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                side_by_side = cv2.hconcat([img, mask_bgr])
                return av.VideoFrame.from_ndarray(side_by_side, format="bgr24")

            # Encontrar contornos y seleccionar la mayor área
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if area >= self.area_thresh:
                    # Caja mínima rotada
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)  # 4 puntos (float)
                    box = np.array(box, dtype=np.float32)

                    # Asegurar que los puntos estén ordenados: tl, tr, br, bl
                    try:
                        img_pts_ordered = order_points(box)
                    except Exception:
                        img_pts_ordered = box.copy()

                    # Dibujar la base detectada
                    cv2.drawContours(img, [np.int32(img_pts_ordered)], -1, (0, 255, 0), 2)

                    # Cámara / matriz intrínseca (aprox.)
                    h_img, w_img = img.shape[:2]
                    f = w_img  # focal aproximada
                    K = np.array([[f, 0, w_img / 2],
                                  [0, f, h_img / 2],
                                  [0, 0, 1]], dtype=np.float64)
                    dist_coef = np.zeros((4, 1))  # suponer sin distorsión

                    # Puntos 3D del objeto: cuadrado unitario en plano z=0 (tl,tr,br,bl)
                    obj_pts = np.array([
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0]
                    ], dtype=np.float32)

                    img_pts = np.array(img_pts_ordered, dtype=np.float32)

                    # Intentar resolver la pose
                    try:
                        retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
                    except cv2.error:
                        retval = False

                    if retval:
                        # Definir pirámide (base 4 puntos + vértice por encima del centro)
                        pyramid = np.array([
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.5, 0.5, -1.2]  # vértice (altura en unidades del lado)
                        ], dtype=np.float32)

                        proj, _ = cv2.projectPoints(pyramid, rvec, tvec, K, dist_coef)
                        proj = proj.reshape(-1, 2).astype(int)

                        # Dibujar base rellena
                        cv2.drawContours(img, [proj[:4]], -1, (0, 255, 0), -3)

                        # Caras laterales con colores diferentes
                        faces = [
                            [proj[0], proj[1], proj[4]],
                            [proj[1], proj[2], proj[4]],
                            [proj[2], proj[3], proj[4]],
                            [proj[3], proj[0], proj[4]],
                        ]
                        colors = [(220, 30, 30), (30, 30, 220), (220, 220, 30), (30, 220, 220)]
                        for face, color in zip(faces, colors):
                            cv2.drawContours(img, [np.array(face, dtype=np.int32)], -1, color, -3)

                        # Aristas en negro
                        edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]
                        for i,j in edges:
                            cv2.line(img, tuple(proj[i]), tuple(proj[j]), (0,0,0), 2)

                        # Opcional: mostrar área detectada
                        cv2.putText(img, f"Area: {int(area)} px", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            # Devolver frame final
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Iniciar WebRTC con la fábrica que pasa los parámetros actuales
    webrtc_streamer(
        key="white-pyramid",
        video_processor_factory=lambda: WhiteSurfacePyramid(v_min, s_max, area_thresh, mostrar_mascara),
        media_stream_constraints={"video": True, "audio": False},
    )


if __name__ == "__main__":
    run()
