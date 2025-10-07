import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# ============================================================
# 🔧 FUNCIONES DE UTILIDAD
# ============================================================
def ensure_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def download_button(image, filename):
    """Botón de descarga para una imagen procesada."""
    pil_img = Image.fromarray(ensure_rgb(image))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label=f"💾 Descargar {filename}",
        data=byte_im,
        file_name=f"{filename}.png",
        mime="image/png",
    )

# ============================================================
# 🎨 FILTROS Y EFECTOS
# ============================================================
def filtro_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def filtro_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    combined = cv2.convertScaleAbs(np.sqrt(sobelx ** 2 + sobely ** 2))
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

def filtro_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def filtro_blur(img, k=3):
    return cv2.filter2D(img, -1, np.ones((k, k), np.float32) / (k * k))

def filtro_motion_blur(img, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel /= size
    return cv2.filter2D(img, -1, kernel)

def filtro_sharpen_1(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def filtro_sharpen_2(img):
    kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
    return cv2.filter2D(img, -1, kernel)

def filtro_sharpen_3(img):
    kernel = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1],
        [-1,-1,-1,-1,-1]
    ]) / 8.0
    return cv2.filter2D(img, -1, kernel)

def filtro_emboss(img):
    kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
    return cv2.filter2D(img, -1, kernel)

def filtro_erosion(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def filtro_dilatacion(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def filtro_vignette(img):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(int(1.5*cols), 200)
    kernel_y = cv2.getGaussianKernel(int(1.5*rows), 200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask[int(0.5*rows):, int(0.5*cols):]
    output = np.copy(img)
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    return output

def filtro_hist_eq(img):
    """Ecualización del histograma (canal Y en YUV)."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# ============================================================
# 🚀 INTERFAZ STREAMLIT
# ============================================================
def run():
    st.header("Capítulo 2: Detección de Bordes y Filtros ✨")
    st.markdown("""
    En este capítulo exploramos varios **filtros y efectos** aplicados mediante convolución 2D y transformaciones de intensidad.
    """)

    uploaded = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"], key="cap2")
    if not uploaded:
        st.info("Sube una imagen para continuar.")
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None:
        st.error("Error al leer la imagen.")
        return

    # Diccionario de efectos
    efectos = {
        "Original": img,
        "Bordes (Canny)": filtro_canny(img),
        "Sobel": filtro_sobel(img),
        "Laplacian": filtro_laplacian(img),
        "Blur 3x3": filtro_blur(img, 3),
        "Blur 5x5": filtro_blur(img, 5),
        "Motion Blur": filtro_motion_blur(img),
        "Sharpen 1": filtro_sharpen_1(img),
        "Sharpen 2": filtro_sharpen_2(img),
        "Edge Enhancement": filtro_sharpen_3(img),
        "Emboss": filtro_emboss(img),
        "Erosión": filtro_erosion(img),
        "Dilatación": filtro_dilatacion(img),
        "Vignette": filtro_vignette(img),
        "Histograma Ecualizado": filtro_hist_eq(img)
    }

    st.subheader("🎨 Galería de efectos")
    st.markdown("*(Miniaturas al 30% del tamaño original, descargables en tamaño completo)*")

    keys = list(efectos.keys())
    for i in range(0, len(keys), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(keys):
                nombre = keys[i + j]
                resultado = efectos[nombre]
                mini = cv2.resize(ensure_rgb(resultado), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
                with col:
                    st.image(mini, caption=nombre, use_container_width=True)
                    download_button(resultado, nombre.replace(" ", "_").lower())

