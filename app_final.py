import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Phishing Websites",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Dashboard Interactivo: Phishing Websites Dataset")
st.markdown("Aplicación de análisis exploratorio de datos basada en el repositorio de la UCI (ID: 327).")

# Función para cargar y cachear los datos
@st.cache_data
def cargar_datos():
    phishing_websites = fetch_ucirepo(id=327)
    df_inicial = phishing_websites.data.features
    y_target = phishing_websites.data.targets

    if isinstance(y_target, pd.DataFrame):
        y_target = y_target.iloc[:, 0]

    df_inicial['target'] = y_target

    # Tratamiento de valores nulos (relleno con la mediana según el flujo del taller)
    for col in df_inicial.select_dtypes(include=[np.number]).columns:
        if df_inicial[col].isnull().sum() > 0:
            median_val = df_inicial[col].median()
            df_inicial[col].fillna(median_val, inplace=True)

    return df_inicial

# Carga de datos con indicador de progreso
with st.spinner("Cargando dataset desde el repositorio UCI..."):
    df = cargar_datos()

# Menú lateral de navegación
st.sidebar.header("Panel de Control")
menu = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Vista General", "Resumen Estadístico", "Control de Nulos", "Visualizaciones"]
)

target_col = df.columns[-1]

if menu == "Vista General":
    st.subheader("📄 Primeras filas del DataFrame")
    st.dataframe(df.head())

    st.subheader("📊 Distribución de la Variable Objetivo (Target)")
    conteo_target = df[target_col].value_counts()
    st.bar_chart(conteo_target)
    st.write(f"Dimensiones del dataset: **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")

elif menu == "Resumen Estadístico":
    st.subheader("📈 Resumen Estadístico Descriptivo")
    st.dataframe(df.describe())

elif menu == "Control de Nulos":
    st.subheader("🔍 Verificación de Valores Nulos")
    nulos = df.isnull().sum()
    df_nulos = pd.DataFrame({"Columna": nulos.index, "Nulos Encontrados": nulos.values})
    st.dataframe(df_nulos, use_container_width=True)
    st.success("¡Tratamiento de nulos aplicado correctamente (imputación por mediana)!")

elif menu == "Visualizaciones":
    st.subheader("📊 Distribución de la Variable Objetivo")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[target_col].dropna(), bins=30, kde=True, ax=ax, color="teal")
    ax.set_title('Distribución de la Variable Objetivo / Característica Principal')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

    st.subheader("📈 Pairplot de Muestra (Primeras 5 características)")
    if st.checkbox("Generar Pairplot (puede tardar unos segundos)"):
        columnas_muestra = list(df.columns[:5]) + [target_col]
        fig_pair = sns.pairplot(df[columnas_muestra], hue=target_col, palette="Set2")
        st.pyplot(fig_pair.fig)


# Cargar modelos y columnas
rf_model = joblib.load("modelo_rf.joblib")
nn_optimized = tf.keras.models.load_model("modelo_nn_optimizado.keras")
feature_columns = joblib.load("columnas_features.joblib")

st.title("Detección de Sitios Web de Phishing")
st.markdown("""
**Descripción del problema:** Esta aplicación utiliza modelos de Machine Learning para clasificar si un sitio web es legítimo o si representa una amenaza de *phishing* basándose en características de su URL y estructura técnica.
**Fuente del dataset:** [UCI Machine Learning Repository - Phishing Websites](https://archive.ics.uci.edu/dataset/327)
""")

# Selector de modelo con nombres reales
opcion_modelo = st.selectbox("Seleccione el Algoritmo", ["Random Forest", "Red Neuronal Optimizada"])

st.subheader("Ingrese las características de la URL / Sitio Web")

# Crear entradas dinámicas para las características
input_data = {}
col1, col2 = st.columns(2)

with col1:
    input_data['having_ip_address'] = st.selectbox("¿Tiene dirección IP en la URL?", [0.0, 1.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
    input_data['url_length'] = st.selectbox("Longitud de la URL", [1.0, 0.5, 0.0], format_func=lambda x: "Larga (0.0)" if x==0.0 else ("Moderada (0.5)" if x==0.5 else "Corta (1.0)"))
    input_data['shortining_service'] = st.selectbox("Usa acortador de URL", [1.0, 0.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
    input_data['having_at_symbol'] = st.selectbox("Presencia del símbolo '@'", [1.0, 0.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")

with col2:
    input_data['double_slash_redirecting'] = st.selectbox("Redirección con doble barra '//'", [1.0, 0.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
    input_data['prefix_suffix'] = st.selectbox("Uso de guion (-) en dominio", [1.0, 0.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
    input_data['having_sub_domain'] = st.selectbox("Cantidad de subdominios", [1.0, 0.5, 0.0], format_func=lambda x: "Múltiples (0.0)" if x==0.0 else ("Uno (0.5)" if x==0.5 else "Ninguno (1.0)"))
    input_data['sslfinal_state'] = st.selectbox("Estado del certificado SSL", [0.0, 0.5, 1.0], format_func=lambda x: "Inseguro (0.0)" if x==0.0 else ("Dudoso (0.5)" if x==0.5 else "Seguro (1.0)"))

# Completar las características restantes con valores neutrales (0.5 o 0.0)
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0.5

# Botón de predicción
if st.button("Predecir Sitio Web"):
    # Convertir a DataFrame manteniendo estrictamente el orden original de entrenamiento
    df_input = pd.DataFrame([input_data])[feature_columns]

    # Seleccionar modelo
    if opcion_modelo == "Random Forest":
        modelo = rf_model
        prediccion = modelo.predict(df_input)[0]
        probabilidad = modelo.predict_proba(df_input).max()
    else:
        # Para la Red Neuronal
        prob_nn = nn_optimized.predict(df_input)[0][0]
        prediccion = 1.0 if prob_nn > 0.5 else 0.0
        probabilidad = prob_nn if prediccion == 1.0 else (1.0 - prob_nn)

    # Interpretar resultado (Asumiendo 1.0 = Legítimo, 0.0 = Phishing)
    if prediccion == 1.0:
        st.success(f"**Resultado:** ¡Sitio Web Legítimo! (Modelo: {opcion_modelo})")
    else:
        st.error(f"**Resultado:** ¡Alerta! Es un sitio potencial de **Phishing** (Modelo: {opcion_modelo})")

    st.info(f"Confianza de la predicción: {probabilidad * 100:.2f}%")
