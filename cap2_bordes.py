import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================
def ensure_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def download_button(image, filename):
    """Botón para descargar imagen procesada"""
    pil_img = Image.fromarray(ensure_rgb(image))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label=f"📥 Descargar {filename}",
        data=byte_im,
        file_name=f"{filename}.png",
        mime="image/png",
    )

# ============================================================
# FUNCIONES DE FILTROS
# ============================================================
def filtro_identidad(img):
    kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
    return cv2.filter2D(img, -1, kernel)

def filtro_3x3(img):
    kernel = np.ones((3,3), np.float32)/9.0
    return cv2.filter2D(img, -1, kernel)

def filtro_5x5(img):
    kernel = np.ones((5,5), np.float32)/25.0
    return cv2.filter2D(img, -1, kernel)

def filtro_sharpen_basico(img):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def filtro_sharpen_excesivo(img):
    kernel = np.array([[1,1,1],[1,-7,1],[1,1,1]])
    return cv2.filter2D(img, -1, kernel)

def filtro_realce_bordes_5x5(img):
    kernel = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1],
        [-1,-1,-1,-1,-1]
    ]) / 8.0
    return cv2.filter2D(img, -1, kernel)

def filtro_desenfoque_movimiento(img):
    size = 15
    kernel = np.zeros((size,size))
    kernel[int((size-1)/2),:] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)

def filtro_emboss_suroeste(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    output = cv2.filter2D(gray, -1, kernel) + 128
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def filtro_emboss_sureste(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
    output = cv2.filter2D(gray, -1, kernel) + 128
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def filtro_emboss_noroeste(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1,0,0],[0,0,0],[0,0,-1]])
    output = cv2.filter2D(gray, -1, kernel) + 128
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def filtro_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    output = cv2.convertScaleAbs(sobel)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def filtro_laplaciano(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    output = cv2.convertScaleAbs(laplacian)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def filtro_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 240)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

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

def filtro_hist_eq_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histeq = cv2.equalizeHist(gray)
    return cv2.cvtColor(histeq, cv2.COLOR_GRAY2BGR)

def filtro_hist_eq_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# ============================================================
#  INTERFAZ STREAMLIT
# ============================================================
def run():
    st.header("Capítulo 2: Filtros y Realce de Imágenes 🧩")
    st.markdown("Explora cómo los distintos **filtros espaciales y realces** modifican la imagen original. 👇")

    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="cap2")
    if not uploaded:
        st.info("Sube una imagen para aplicar los filtros.")
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    filtros = {
        "Filtro identidad": filtro_identidad,
        "Filtro 3x3": filtro_3x3,
        "Filtro 5x5": filtro_5x5,
        "Sharpen básico": filtro_sharpen_basico,
        "Sharpen excesivo": filtro_sharpen_excesivo,
        "Realce de bordes (5x5)": filtro_realce_bordes_5x5,
        "Desenfoque por movimiento": filtro_desenfoque_movimiento,
        "Emboss - Suroeste": filtro_emboss_suroeste,
        "Emboss - Sureste": filtro_emboss_sureste,
        "Emboss - Noroeste": filtro_emboss_noroeste,
        "Detección de bordes - Sobel": filtro_sobel,
        "Detección de bordes - Laplaciano": filtro_laplaciano,
        "Detección de bordes - Canny": filtro_canny,
        "Vignette": filtro_vignette,
        "Ecualización de histograma (gris)": filtro_hist_eq_gray,
        "Ecualización de histograma (color)": filtro_hist_eq_color,
    }

    st.subheader("🎨 Galería de filtros (3 por fila)")
    st.markdown("Cada filtro genera su propia versión. Haz clic para descargar la imagen procesada.")

    keys = list(filtros.keys())
    for i in range(0, len(keys), 3):  # 3 por fila
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(keys):
                nombre = keys[i + j]
                resultado = filtros[nombre](img)
                mini = cv2.resize(ensure_rgb(resultado), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
                with col:
                    st.image(mini, caption=nombre, use_container_width=True)
                    download_button(resultado, nombre.replace(" ", "_"))
