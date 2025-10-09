import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

# ============================================================
# Función para extraer todos los contornos de la imagen
# ============================================================
def get_all_contours(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

# ============================================================
# Función principal para segmentación y censura
# ============================================================
def detectar_formas(img):
    img_orig = np.copy(img)
    input_contours = get_all_contours(img)
    solidity_values = []

    # Calcular la solidez de cada contorno
    for contour in input_contours:
        area_contour = cv2.contourArea(contour)
        convex_hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(convex_hull)
        if area_hull > 0:
            solidity = float(area_contour) / area_hull
            solidity_values.append(solidity)
        else:
            solidity_values.append(0)

    # Agrupar por K-Means según la solidez
    if len(solidity_values) < 2:
        return img, img_orig  # No hay suficientes formas

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    solidity_values = np.array(solidity_values).reshape((-1, 1)).astype("float32")
    compactness, labels, centers = cv2.kmeans(solidity_values, 2, None, criteria, 10, flags)

    # Elegir la clase con menor centroide (formas más sólidas)
    closest_class = np.argmin(centers)
    output_contours = []
    for i in range(len(solidity_values)):
        if labels[i] == closest_class:
            output_contours.append(input_contours[i])

    # Dibujar los contornos detectados
    img_contornos = np.copy(img)
    cv2.drawContours(img_contornos, output_contours, -1, (0, 0, 0), 3)

    # Censurar las regiones detectadas
    img_censurado = np.copy(img_orig)
    for contour in output_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)   # ✅ compatible con NumPy >= 1.24
        cv2.drawContours(img_censurado, [box], 0, (0, 0, 0), -1)


    return img_contornos, img_censurado

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
def run():
    st.header("Capítulo 7: Detección de Formas y Segmentación / Detecting Shapes and Segmenting")
    st.markdown("""
    En este capítulo se estudia la **segmentación de imágenes**, proceso clave para:
    - Identificar regiones de interés.
    - Detectar formas geométricas (círculos, rectángulos, contornos).
    - Separar el fondo del objeto principal.

    En este ejercicio se usa **solidez (convexidad)** y **K-Means** para agrupar las formas y censurar algunas.
    """)

    st.info("👉 Puedes usar la imagen base `random_shapes.png` o subir una propia.")

    # Cargar imagen base o subida
    uploaded_file = st.file_uploader("Sube una imagen (opcional)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread("./images/random_shapes.png")

    if img is None:
        st.error("⚠️ No se pudo cargar la imagen. Verifica la ruta o sube otra imagen.")
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

    if st.button("Procesar imagen"):
        img_contornos, img_censurado = detectar_formas(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB), caption="Formas detectadas")
        with col2:
            st.image(cv2.cvtColor(img_censurado, cv2.COLOR_BGR2RGB), caption="Regiones censuradas")

        st.success("✅ Segmentación y detección completadas con éxito.")

