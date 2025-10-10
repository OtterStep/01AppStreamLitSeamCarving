# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Cargar modelo preentrenado
model = MobileNetV2(weights="imagenet")

# Función para procesar imagen y predecir
def predict_dog(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (224, 224))
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # Índices de ImageNet que corresponden a perros: 151–268
    if np.argmax(preds[0]) in range(151, 269):
        return True
    else:
        return False

def run():
    st.header("Capítulo 11: Redes Neuronales Artificiales / Artificial Neural Networks")
    st.markdown("""
    En este capítulo se estudia el uso de **redes neuronales artificiales** para procesamiento de imágenes:
    - Neuronas y capas densas.
    - Entrenamiento con retropropagación.
    - Aplicaciones en clasificación de imágenes.
    """)

    st.subheader("Clasificador de Perros 🐶")
    st.markdown("Sube una imagen y la red neuronal determinará si es un perro o no.")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen cargada", use_column_width=True)
        
        if st.button("Clasificar"):
            result = predict_dog(img)
            if result:
                st.success("✅ Es un perro")
            else:
                st.error("❌ No es un perro")


