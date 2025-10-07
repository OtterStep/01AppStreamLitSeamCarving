import streamlit as st
import cv2
import numpy as np

def run():
    st.header("Capítulo 1: Transformaciones Geométricas")
    st.markdown("""
    En este capítulo aprenderás sobre **transformaciones geométricas** en imágenes:
    - **Traslación**: mover la imagen en el plano.
    - **Rotación**: girar la imagen un ángulo específico.
    - **Escalado**: aumentar o reducir el tamaño.
    - **Reflexión**: voltear la imagen horizontal o verticalmente.
    Estas operaciones son la base del procesamiento de imágenes en visión computacional.
    """)

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="cap1")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Imagen Original", use_column_width=True)

        # --- Ejemplo: rotación ---
        angle = st.slider("Ángulo de rotación", -180, 180, 45)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))

        st.image(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), caption="Imagen Rotada", use_column_width=True)
    else:
        st.info("Sube una imagen para aplicar transformaciones.")
