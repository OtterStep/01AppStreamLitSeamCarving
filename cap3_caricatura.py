import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ============================================================
# Función de cartoonización
# ============================================================
def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4

    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Filtro bilateral repetido (color suave)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    for _ in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)

    # Combinar con bordes
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst


# ============================================================
# Interfaz Streamlit con cámara en vivo
# ============================================================
def run():
    st.header("Capítulo 3: Filtros Cartoon en Video 🎨📸")
    st.markdown("""
    Visualiza el efecto cartoon en **tiempo real** desde tu cámara.  
    Usa los botones para cambiar el filtro:
    - 🎞 **Normal**: sin efecto  
    - 🖍 **Sketch**: contornos en blanco y negro  
    - 🎨 **Cartoon**: caricatura en color
    """)

    # Iniciar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara.")
        return

    # Selección de filtro
    filter_mode = st.radio("Selecciona el filtro:", ["Normal", "Sketch", "Cartoon"], horizontal=True)

    # Contenedor de imagen
    frame_container = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo leer el fotograma de la cámara.")
            break

        # Aplicar filtro
        if filter_mode == "Sketch":
            frame = cartoonize_image(frame, sketch_mode=True)
        elif filter_mode == "Cartoon":
            frame = cartoonize_image(frame, sketch_mode=False)

        # Convertir a RGB y mostrar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_container.image(frame_rgb, channels="RGB", use_container_width=True)

        # Si el usuario presiona el botón, detener cámara
    cap.release()
    st.success("Cámara detenida ✅")


