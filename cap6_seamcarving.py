import streamlit as st
import cv2
import numpy as np
from PIL import Image

# =====================================================
# FUNCIONES DE PROCESAMIENTO
# =====================================================

def overlay_vertical_seam(img, seam):
    """Dibuja la costura detectada sobre la imagen."""
    img_overlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    img_overlay[x_coords, y_coords] = (0, 255, 0)  # verde
    return img_overlay


def compute_energy_matrix(img):
    """Calcula el mapa de energía con filtros Sobel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def find_vertical_seam(img, energy):
    """Encuentra la costura vertical de menor energía."""
    rows, cols = img.shape[:2]
    seam = np.zeros(rows)
    dist_to = np.zeros((rows, cols)) + float("inf")
    edge_to = np.zeros((rows, cols))

    dist_to[0, :] = energy[0, :]

    # Programación dinámica
    for row in range(1, rows):
        for col in range(cols):
            min_val = dist_to[row - 1, col]
            offset = 0
            if col > 0 and dist_to[row - 1, col - 1] < min_val:
                min_val = dist_to[row - 1, col - 1]
                offset = -1
            if col < cols - 1 and dist_to[row - 1, col + 1] < min_val:
                min_val = dist_to[row - 1, col + 1]
                offset = 1
            dist_to[row, col] = energy[row, col] + min_val
            edge_to[row, col] = offset

    # Retroceso
    seam[-1] = np.argmin(dist_to[-1])
    for row in range(rows - 2, -1, -1):
        seam[row] = seam[row + 1] + edge_to[row + 1, int(seam[row + 1])]
    return seam


def remove_vertical_seam(img, seam):
    """Elimina una costura vertical de la imagen."""
    rows, cols = img.shape[:2]
    output = np.zeros((rows, cols - 1, 3), dtype=np.uint8)
    for r in range(rows):
        c = int(seam[r])
        output[r, :, :] = np.delete(img[r, :, :], c, axis=0)
    return output


def add_vertical_seam(img, seam, offset):
    """Agrega una costura vertical interpolando valores vecinos."""
    seam = seam + offset
    rows, cols = img.shape[:2]
    extended = np.zeros((rows, cols + 1, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols + 1):
            if col < seam[row]:
                extended[row, col] = img[row, col]
            elif col == seam[row]:
                v1 = img[row, max(col - 1, 0)]
                v2 = img[row, min(col, cols - 1)]
                extended[row, col] = ((v1.astype(np.int32) + v2.astype(np.int32)) // 2).astype(np.uint8)
            else:
                extended[row, col] = img[row, col - 1]
    return extended


def seam_carving(img, num_seams, mode="reduce"):
    """Aplica seam carving para reducir o agrandar el ancho."""
    img_proc = img.copy()
    img_overlay = img.copy()
    img_output = img.copy()

    for i in range(num_seams):
        energy = compute_energy_matrix(img_proc)
        seam = find_vertical_seam(img_proc, energy)
        img_overlay = overlay_vertical_seam(img_overlay, seam)

        if mode == "reduce":
            img_proc = remove_vertical_seam(img_proc, seam)
        elif mode == "expand":
            img_output = add_vertical_seam(img_output, seam, i)
            img_proc = remove_vertical_seam(img_proc, seam)

    return img_overlay, img_proc if mode == "reduce" else img_output


# =====================================================
# INTERFAZ STREAMLIT
# =====================================================

def run():
    st.header("Capítulo 6: Seam Carving / Reducción y Expansión Basada en Contenido 🧩")
    st.markdown("""
    El **Seam Carving** es una técnica de *retargeting de imágenes* que **elimina o agrega costuras (seams)**,  
    preservando las partes más importantes de una imagen.

    👉 Este ejercicio te permite **reducir** o **agrandar** el ancho de una imagen sin distorsionarla.
    """)

    # Cargar imagen
    uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        st.info("Usando imagen predeterminada: `beach.jpg`")
        img = cv2.imread("./images/beach.jpg")

    if img is None:
        st.error("❌ No se pudo cargar la imagen. Verifica la ruta o formato.")
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

    # Parámetros
    mode = st.radio("Selecciona el modo:", ["Reducir", "Aumentar"])
    num_seams = st.slider("Número de costuras a procesar:", 1, 100, 30)

    if st.button("Aplicar Seam Carving"):
        with st.spinner("Procesando... ⏳"):
            mode_key = "reduce" if mode == "Reducir" else "expand"
            seams_img, result_img = seam_carving(img, num_seams, mode=mode_key)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(seams_img, cv2.COLOR_BGR2RGB), caption="Costuras detectadas", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                     caption=f"Resultado ({mode} {num_seams} costuras)",
                     use_container_width=True)

        st.success("✅ Proceso completado con éxito.")

    st.markdown("""
    ---
    🧠 **Notas:**
    - Las zonas con **alta energía (bordes)** se preservan.  
    - Las regiones **planas o uniformes** son las que se eliminan o expanden.  
    - Ideal para cambiar el tamaño de una imagen sin deformar objetos principales.
    """)


# Para ejecutar directamente
if __name__ == "__main__":
    run()
