import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Clases de detectores
# -----------------------------
class DenseDetector:
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(y), float(x), self.initXyStep))
        return keypoints


class SIFTDetector:
    def __init__(self):
        self.detector = cv2.SIFT_create()

    def detect(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_image, None)


# -----------------------------
# Función principal de la app
# -----------------------------
def run():
    st.title("🔍 Comparador de Detectores de Características")
    st.write("Compara los puntos detectados con **DenseDetector** y **SIFT**.")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Parámetros del DenseDetector
        st.sidebar.header("⚙️ Parámetros del DenseDetector")
        step = st.sidebar.slider("Step Size", 5, 40, 20)
        scale = st.sidebar.slider("Feature Scale", 5, 40, 20)
        bound = st.sidebar.slider("Image Bound", 0, 50, 10)

        # Detectar características
        dense_kp = DenseDetector(step, scale, bound).detect(img_bgr)
        sift_kp = SIFTDetector().detect(img_bgr)

        # Dibujar los puntos
        dense_img = cv2.drawKeypoints(img_bgr, dense_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        sift_img = cv2.drawKeypoints(img_bgr, sift_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Convertir a RGB
        dense_img = cv2.cvtColor(dense_img, cv2.COLOR_BGR2RGB)
        sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB)

        # Mostrar lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dense Feature Detector")
            st.image(dense_img, caption=f"{len(dense_kp)} puntos detectados")

        with col2:
            st.subheader("SIFT Detector")
            st.image(sift_img, caption=f"{len(sift_kp)} puntos detectados")
    else:
        st.info("⬆️ Sube una imagen para comenzar.")


