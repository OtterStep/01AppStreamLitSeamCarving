import streamlit as st
import cv2
import numpy as np
import math
from io import BytesIO
from PIL import Image

# Función: Rotación centrada
def rotar_imagen(image, angle):
    """Rota la imagen alrededor del centro manteniendo el lienzo original."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

# Asegurar formato RGB
def ensure_rgb(img):
    """Convierte la imagen a formato RGB (compatible con Streamlit y PIL)."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img

# Aplicar transformaciones geométricas
def aplicar_transformacion(image, transform_type, scale=1.0, tx=0, ty=0):
    """Aplica la transformación geométrica indicada."""
    h, w = image.shape[:2]

    if transform_type == "Escalado":
        # Escalado centrado: agrandar o reducir respecto al centro del lienzo
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        scaled = cv2.warpAffine(image, M, (w, h))
        return scaled

    elif transform_type == "Traslación":
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (w, h))

    elif transform_type == "Reflexión Horizontal":
        return cv2.flip(image, 1)

    elif transform_type == "Reflexión Vertical":
        return cv2.flip(image, 0)

    elif transform_type == "Onda Vertical":
        output = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                offset_x = int(25.0 * math.sin(2 * math.pi * i / 180))
                if 0 <= j + offset_x < w:
                    output[i, j] = image[i, (j + offset_x) % w]
        return output

    elif transform_type == "Onda Horizontal":
        output = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                offset_y = int(16.0 * math.sin(2 * math.pi * j / 150))
                if 0 <= i + offset_y < h:
                    output[i, j] = image[(i + offset_y) % h, j]
        return output

    elif transform_type == "Onda Multidireccional":
        output = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                offset_x = int(20.0 * math.sin(2 * math.pi * i / 150))
                offset_y = int(20.0 * math.cos(2 * math.pi * j / 150))
                if 0 <= i + offset_y < h and 0 <= j + offset_x < w:
                    output[i, j] = image[(i + offset_y) % h, (j + offset_x) % w]
        return output

    elif transform_type == "Efecto Cóncavo":
        output = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                offset_x = int(128.0 * math.sin(2 * math.pi * i / (2 * w)))
                if 0 <= j + offset_x < w:
                    output[i, j] = image[i, (j + offset_x) % w]
        return output

    return image


# Interfaz Principal Streamlit

def run():
    st.header("Capítulo 1: Transformaciones Geométricas 🌀")

    st.markdown("""
    En este capítulo aprenderás sobre **transformaciones geométricas**:
    - **Rotación**: girar la imagen.  
    - **Escalado**: aumentar o reducir tamaño (centrado).  
    - **Traslación**: mover la imagen.  
    - **Reflexión**: voltear horizontal o verticalmente.  
    - **Ondas y efectos especiales**: deformaciones no lineales.  
    """)

    uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leer imagen (manteniendo canales)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is None:
            st.error("❌ No se pudo leer la imagen.")
            return

        st.subheader("Imagen Original")
        st.image(ensure_rgb(image), use_container_width=True)

        # Selector de transformación
        st.subheader("🔧 Elige una transformación:")
        transform_type = st.selectbox(
            "Tipo de transformación",
            ["Rotación", "Escalado", "Traslación", "Reflexión Horizontal", "Reflexión Vertical",
             "Onda Vertical", "Onda Horizontal", "Onda Multidireccional", "Efecto Cóncavo"]
        )

        # Parámetros dinámicos
        angle, scale, tx, ty = 0, 1.0, 0, 0
        if transform_type == "Rotación":
            angle = st.slider("Ángulo de rotación (°)", -180, 180, 45)
            transformed = rotar_imagen(image, angle)

        elif transform_type == "Escalado":
            scale = st.slider("Factor de escalado", 0.1, 3.0, 1.0, step=0.1)
            transformed = aplicar_transformacion(image, "Escalado", scale=scale)

        elif transform_type == "Traslación":
            tx = st.slider("Desplazamiento X (px)", -300, 300, 50)
            ty = st.slider("Desplazamiento Y (px)", -300, 300, 50)
            transformed = aplicar_transformacion(image, "Traslación", tx=tx, ty=ty)

        else:
            transformed = aplicar_transformacion(image, transform_type)

        # Mostrar resultado reducido al 25 % para visualización
        display_img = ensure_rgb(transformed)
        if transform_type == "Escalado":
            display_resized = cv2.resize(display_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            st.subheader(f"Resultado — {transform_type} (Vista reducida al 25%)")
            st.image(display_resized, use_container_width=True)
        else:
            st.subheader(f"Resultado — {transform_type}")
            st.image(display_img, use_container_width=True)

        # Descarga en tamaño original (no reducido)
        result_pil = Image.fromarray(display_img)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        safe_name = transform_type.replace(" ", "_").replace("á", "a").replace("ó", "o")
        st.download_button(
            label="💾 Descargar imagen transformada (Tamaño completo)",
            data=byte_im,
            file_name=f"transformada_{safe_name}.png",
            mime="image/png"
        )

    else:
        st.info("📥 Sube una imagen para aplicar las transformaciones.")
