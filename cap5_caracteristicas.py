import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detectar_esquinas(img_cv):
    """Devuelve dos imágenes: una con esquinas fuertes y otra con esquinas suaves."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # --- Detección de esquinas fuertes ---
    dst_sharp = cv2.cornerHarris(gray, blockSize=4, ksize=5, k=0.04)
    dst_sharp = cv2.dilate(dst_sharp, None)
    img_sharp = img_cv.copy()
    img_sharp[dst_sharp > 0.01 * dst_sharp.max()] = [0, 0, 255]  # rojo

    # --- Detección de esquinas suaves ---
    dst_soft = cv2.cornerHarris(gray, blockSize=14, ksize=5, k=0.04)
    dst_soft = cv2.dilate(dst_soft, None)
    img_soft = img_cv.copy()
    img_soft[dst_soft > 0.01 * dst_soft.max()] = [0, 255, 0]  # verde

    return img_sharp, img_soft


def run():
    st.header("Capítulo 5: Extracción de Características / Extracting Features from an Image 🧩")
    st.markdown("""
    En este capítulo exploramos cómo **detectar puntos característicos (esquinas o “features”)** en una imagen,  
    que son esenciales para tareas como:
    - Detección de objetos  
    - Seguimiento de movimiento  
    - Reconocimiento de patrones o comparación de imágenes  

    💡 En este ejemplo se usa el **Detector de esquinas de Harris**, que identifica regiones con alto cambio de intensidad.
    """)

    st.subheader("🔹 Subir imagen o usar la predeterminada")
    uploaded_file = st.file_uploader("Sube una imagen (formato JPG o PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
    else:
        st.info("Usando la imagen predeterminada: `box.png`")
        img_cv = cv2.imread("./images/box.png")

    if img_cv is None:
        st.error("❌ No se pudo cargar la imagen. Verifica la ruta o el formato.")
        return

    # Procesamiento
    img_sharp, img_soft = detectar_esquinas(img_cv)

    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB),
                 caption="Harris Corners (solo esquinas fuertes)", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(img_soft, cv2.COLOR_BGR2RGB),
                 caption="Harris Corners (también esquinas suaves)", use_container_width=True)

    st.markdown("""
    ---
    🔍 **Conclusión:**  
    Las esquinas (features) son puntos clave en una imagen donde hay un cambio abrupto en la intensidad.
    Detectarlas permite identificar estructuras relevantes para análisis visual o visión computacional.
    """)
