import streamlit as st
import cv2
import numpy as np

def run():
    st.header("Capítulo 2: Detección de Bordes y Filtros")
    st.markdown("""
    La **detección de bordes** es clave en visión computacional: nos permite resaltar los contornos
    y transiciones de intensidad en una imagen.  
    Algunos filtros comunes:
    - **Canny** → Detecta bordes precisos.
    - **Sobel** → Resalta bordes en dirección horizontal/vertical.
    - **Laplacian** → Detecta cambios bruscos en intensidad.
    """)

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","png","jpeg"], key="cap2")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        st.image([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), edges], 
                 caption=["Original","Bordes (Canny)"], width=300)
