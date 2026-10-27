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
