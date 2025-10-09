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
# Interfaz Streamlit con cámara del navegador
# ============================================================
def run():
    st.header("Capítulo 3: Filtros Cartoon con Cámara del Navegador 🎨📸")
    st.markdown("""
    Visualiza el efecto cartoon aplicando filtros a una **foto capturada desde tu cámara**.  
    Usa los botones para elegir el tipo de efecto:
    - 🎞 **Normal**: sin efecto  
    - 🖍 **Sketch**: contornos en blanco y negro  
    - 🎨 **Cartoon**: caricatura en color
    """)

    # Selección de filtro
    filter_mode = st.radio("Selecciona el filtro:", ["Normal", "Sketch", "Cartoon"], horizontal=True)

    # Componente de cámara
    img_file = st.camera_input("Captura una foto con tu cámara")

    if img_file is not None:
        # Convertir a formato OpenCV
        image = Image.open(img_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Aplicar filtro
        if filter_mode == "Sketch":
            processed_img = cartoonize_image(img_bgr, sketch_mode=True)
        elif filter_mode == "Cartoon":
            processed_img = cartoonize_image(img_bgr, sketch_mode=False)
        else:
            processed_img = img_bgr

        # Mostrar resultado
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(processed_img_rgb, caption="Resultado", use_container_width=True)

