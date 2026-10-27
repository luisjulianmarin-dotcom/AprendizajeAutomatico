import streamlit as st
import pandas as pd
import numpy as np

# Título de la aplicación
st.title("Mi Primera App en Google Colab 🚀")
st.write("Esta aplicación corre en la nube de Google y se visualiza a través de un túnel público.")

# Sección interactiva: Control deslizante
num_puntos = st.slider("Selecciona el número de puntos de datos:", 10, 500, 100)

# Generar datos aleatorios
df = pd.DataFrame(
    np.random.randn(num_puntos, 2),
    columns=['Eje X', 'Eje Y']
)

# Mostrar una tabla y un gráfico
st.subheader("Visualización de Datos")
st.line_chart(df)

if st.button("Saludar"):
    st.success("¡Hola! Streamlit y Colab funcionan a la perfección.")
