import streamlit as st
import cv2
import numpy as np

def run():
    st.header("Capítulo 3: Convertir en Caricatura")
    st.markdown("""
    El efecto caricatura se logra combinando:
    1. **Suavizado bilateral** → para reducir el ruido y mantener bordes.
    2. **Detección de bordes** → para marcar los contornos.
    3. **Combinación** → superponer bordes sobre la imagen suavizada.
    """)

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","png","jpeg"], key="cap3")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)

        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        st.image([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)], 
                 caption=["Original","Caricatura"], width=300)
