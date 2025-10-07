import streamlit as st

def run():
    st.header("Capítulo 6: Seam Carving / Reducción de Imagen Basada en Contenido")
    st.markdown("""
    **Seam Carving** permite redimensionar imágenes preservando las áreas más importantes.
    - Calcula un **mapa de energía**.
    - Encuentra la ruta de menor energía (seam).
    - Elimina el seam y repite hasta lograr el tamaño deseado.
    """)
    st.info("👉 Aquí implementa el ejercicio de seam carving.")
