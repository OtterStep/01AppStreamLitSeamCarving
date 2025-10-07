import streamlit as st

# Importar módulos de capítulos
import cap1_transformaciones as cap1
import cap2_bordes as cap2
import cap3_caricatura as cap3
import cap4_cuerpo as cap4
import cap5_caracteristicas as cap5
import cap6_seamcarving as cap6
import cap7_segmentacion as cap7
import cap8_tracking as cap8
import cap9_reconocimiento as cap9
import cap10_realidad_aumentada as cap10
import cap11_red_neuronal as cap11

st.set_page_config(page_title="Demostraciones de Procesamiento de Imágenes", layout="wide")

st.title("📖 Demostraciones de Procesamiento de Imágenes")
st.markdown("Ejercicios prácticos por capítulo con breve teoría.")

# --- Menú lateral ---
menu = st.sidebar.selectbox(
    "Selecciona un capítulo",
    [
        "1. Transformaciones Geométricas / Applying Geometric Transformations",
        "2. Detección de Bordes y Filtros / Detecting Edges & Filters",
        "3. Convertir en Caricatura / Cartoonizing",
        "4. Seguimiento de Partes del Cuerpo / Body Parts Tracking",
        "5. Extracción de Características / Feature Extraction",
        "6. Seam Carving",
        "7. Detección de Formas y Segmentación / Shapes & Segmentation",
        "8. Seguimiento de Objetos / Object Tracking",
        "9. Reconocimiento de Objetos / Object Recognition",
        "10. Realidad Aumentada / Augmented Reality",
        "11. Redes Neuronales Artificiales / ANN"
    ]
)

# --- Router del menú ---
if "Transformaciones" in menu: cap1.run()
elif "Bordes" in menu: cap2.run()
elif "Caricatura" in menu: cap3.run()
elif "Cuerpo" in menu: cap4.run()
elif "Características" in menu: cap5.run()
elif "Seam Carving" in menu: cap6.run()
elif "Segmentación" in menu: cap7.run()
elif "Objetos / Object Tracking" in menu: cap8.run()
elif "Reconocimiento" in menu: cap9.run()
elif "Realidad Aumentada" in menu: cap10.run()
elif "Neuronales" in menu: cap11.run()
