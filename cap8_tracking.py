import streamlit as st
import cv2
import numpy as np
from streamlit.components.v1 import html

def run():
    st.header("🎯 Seguimiento de objetos con dos clics en la imagen")

    st.markdown("""
    **Instrucciones:**
    1️⃣ Captura un cuadro de la cámara.  
    2️⃣ Haz **dos clics en la imagen** (inicio y fin del rectángulo).  
    3️⃣ Se inicia el seguimiento del objeto con **CamShift**.
    """)

    # --- Captura un solo frame de la cámara ---
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("No se pudo acceder a la cámara.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape

    # Mostrar la imagen con HTML para capturar clics
    st.markdown("### 📸 Haz dos clics sobre la imagen:")
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = f"data:image/jpeg;base64,{buffer.tobytes().hex()}"

    # JavaScript para capturar los clics
    html_code = f"""
    <div style="position: relative; display: inline-block;">
      <img id="target" src="data:image/jpeg;base64,{buffer.tobytes().hex()}" width="640">
      <canvas id="overlay" width="640" height="{int(640*h/w)}"
              style="position: absolute; top: 0; left: 0;"></canvas>
    </div>

    <script>
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('target');
    let clicks = [];

    canvas.addEventListener('click', (event) => {{
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        clicks.push([x, y]);
        if (clicks.length === 2) {{
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            const [x1, y1] = clicks[0];
            const [x2, y2] = clicks[1];
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const data = JSON.stringify({{x1, y1, x2, y2}});
            window.parent.postMessage({{type: 'coords', data: data}}, '*');
        }}
    }});
    </script>
    """

    # Escuchar los clics desde JS
    coords = st.experimental_get_query_params().get("coords")

    html(html_code, height=int(640*h/w) + 10)

    # Mostrar resultados del clic
    if coords:
        st.write("📍 Coordenadas seleccionadas:", coords)

    st.write("⚙️ Próximo paso: usar esas coordenadas para inicializar CamShift.")

