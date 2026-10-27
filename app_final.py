import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import re
from urllib.parse import urlparse
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Phishing Websites",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
**Trabajo efectuado por:** - Mayra Alejandra Castañeda
                           - Luis Julián Marin Noreña
""")
st.title("🛡️ Detección de Sitios Web de Phishing")
st.markdown("""
**Descripción del problema:** Esta aplicación utiliza modelos de Machine Learning para clasificar si un sitio web es legítimo o si representa una amenaza de *phishing* basándose en características de su URL y estructura técnica.
**Fuente del dataset:** [UCI Machine Learning Repository - Phishing Websites](https://archive.ics.uci.edu/dataset/327)
""")

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

# Función heurística para extraer características a partir de una URL
def analizar_url(url):
    if not url.startswith(('http://', 'https://')):
        url_parseada = urlparse('http://' + url)
    else:
        url_parseada = urlparse(url)

    domain = url_parseada.netloc
    path = url_parseada.path
    query = url_parseada.query
    analisis = {}

    # 1. having_ip_address
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    analisis['having_ip_address'] = 0.0 if ip_pattern.search(domain) else 1.0

    # 2. url_length
    longitud = len(url)
    if longitud < 10:
        analisis['url_length'] = 1.0
    elif longitud <= 15:
        analisis['url_length'] = 0.5
    else:
        analisis['url_length'] = 0.0

    # 3. shortining_service
    acortadores = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'is.gd', 'rebrand.ly', 'buff.ly','logi.php','logi.html']
    analisis['shortining_service'] = 0.0 if any(ac in domain for ac in acortadores) else 1.0

    # 4. having_at_symbol
    analisis['having_at_symbol'] = 0.0 if '@' in url else 1.0

    # 5. double_slash_redirecting
    analisis['double_slash_redirecting'] = 0.0 if url.rfind('//') > 1 else 1.0

    # 6. prefix_suffix (Guión '-' en dominio)
    analisis['prefix_suffix'] = 0.0 if '-' in domain else 1.0

    # 7. having_sub_domain (Cantidad de puntos en el dominio)
    dots = domain.count('.')
    if dots <= 1:
        analisis['having_sub_domain'] = 1.0  # Seguro
    elif dots == 2:
        analisis['having_sub_domain'] = 0.5  # Moderado
    else:
        analisis['having_sub_domain'] = 0.0  # Sospechoso

    # 8. sslfinal_state
    if url_parseada.scheme == 'https':
        analisis['sslfinal_state'] = 1.0
    else:
        analisis['sslfinal_state'] = 0.0

    # 9. domain_registration_length
    if url_parseada.scheme == 'https' and len(domain) > 4:
         analisis['domain_registration_length'] = 1.0
    else:
         analisis['domain_registration_length'] = 0.0

    # 10. favicon
    analisis['favicon'] = 1.0 if url_parseada.scheme == 'https' else 0.0

    # 11. port
    puertos_sospechosos = [':8080', ':8443', ':3000', ':8000',':80']
    analisis['port'] = 0.0 if any(port in domain for port in puertos_sospechosos) else 1.0

    # 12. https_token
    analisis['https_token'] = 0.0 if 'https' in domain else 1.0

    # 13. request_url
    if 'wp-content' in url or 'assets' in url or 'uploads' in url:
        analisis['request_url'] = 0.5
    else:
        analisis['request_url'] = 1.0

    # 14. url_of_anchor
    if '#' in url or 'javascript' in url.lower() or 'void' in url.lower():
        analisis['url_of_anchor'] = 0.0
    else:
        analisis['url_of_anchor'] = 1.0

    # 15. links_in_tags
    if '?' in url or '&' in url or len(query) > 0:
        analisis['links_in_tags'] = 0.0
    else:
        analisis['links_in_tags'] = 1.0

    # 16. sfh
    if ip_pattern.search(domain) or domain == "":
        analisis['sfh'] = 0.0
    else:
        analisis['sfh'] = 0.5

    # 17. submitting_to_email
    analisis['submitting_to_email'] = 0.0 if 'mail' in url or 'mailto' in url else 0.5

    # 18. abnormal_url
    analisis['abnormal_url'] = 0.5 if domain in url else 0.0

    # 19. redirect
    analisis['redirect'] = 0.0 if url.count('//') > 1 else 1.0

    # 20. on_mouseover
    analisis['on_mouseover'] = 0.5

    # 21. rightclick
    analisis['rightclick'] = 0.5

    # 22. popupwindow
    analisis['popupwindow'] = 0.5

    # 23. iframe
    analisis['iframe'] = 0.5

    # 24. age_of_domain
    if len(domain) > 4 and url_parseada.scheme == 'https':
        analisis['age_of_domain'] = 0.5
    else:
        analisis['age_of_domain'] = 0.0

    # 25. dnsrecord
    analisis['dnsrecord'] = 0.5

    # 26. web_traffic
    dominios_populares = ['google.com', 'facebook.com', 'youtube.com', 'twitter.com', 'instagram.com', 'netflix.com', 'amazon.com', 'microsoft.com', 'github.com']
    if any(dom in domain.lower() for dom in dominios_populares):
        analisis['web_traffic'] = 1.0
    else:
        analisis['web_traffic'] = 0.0

    # 27. page_rank
    if any(dom in domain.lower() for dom in dominios_populares):
        analisis['page_rank'] = 1.0
    else:
        analisis['page_rank'] = 0.0

    # 28. google_index
    analisis['google_index'] = 1.0

    # 29. links_pointing_to_page
    if len(path) > 30:
        analisis['links_pointing_to_page'] = 0.0
    elif len(path) > 10:
        analisis['links_pointing_to_page'] = 0.5
    else:
        analisis['links_pointing_to_page'] = 1.0

    # 30. statistical_report
    api_url = "https://checkurl.phishtank.com/checkurl/"
    data = {
        'url': url,  # Corregido de url_parseada (un objeto) a url (string)
        'format': 'json'
    }

    try:
        headers = {'User-Agent': 'phishing-detector-app'}
        response = requests.post(api_url, data=data, headers=headers, timeout=5)

        if response.status_code == 200:
            resultado = response.json()
            in_database = resultado.get('results', {}).get('in_database', False)
            verified = resultado.get('results', {}).get('verified', False)

            if in_database and verified:
                analisis['statistical_report'] = 0.0
            else:
                analisis['statistical_report'] = 1.0
        else:
            analisis['statistical_report'] = 0.5

    except Exception as e:
        analisis['statistical_report'] = 0.5
        
    return analisis  # Corregido: Se debe retornar obligatoriamente el diccionario al final

# Menú lateral de navegación
st.sidebar.header("Panel de Control")
menu = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Predicción por URL Directa","Vista de Modelos","Vista General", "Resumen Estadístico", "Control de Nulos", "Visualizaciones"]
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

elif menu == "Predicción por URL Directa":
    st.subheader("🔗 Análisis Inteligente y Automatizado de URL")
    st.markdown("Introduce una dirección web y los modelos extraerán de forma automática sus características para determinar el nivel de riesgo de Phishing.")
    st.markdown("Ejemplos sospechosos: http://103.119.17.126/?d6o, www.bancolombia.olimpia.co, http://paypal-secure-auth.net, https://www.google.com")
    st.markdown("Ejemplos sospechosos: http://bancolombia.seguridad-actualizacion.co, https://sesja-1c4b-pkobiznes-firma.wabze.net/pl-PL")

    url_input = st.text_input("Ingrese la dirección URL:", "https://www.google.com")

    # Selector de modelo
    opcion_modelo = st.selectbox("Algoritmo de Clasificación", ["Random Forest", "Red Neuronal"])

    if st.button("Analizar y Predecir URL"):
        if url_input.strip() == "":
            st.warning("Por favor ingrese una URL válida.")
        else: 
            # Cargar modelos y mapeos
            rf_model = joblib.load("modelo_rf.joblib")
            nn_optimized = tf.keras.models.load_model("modelo_nn_optimizado.keras")
            feature_columns = joblib.load("columnas_features.joblib")

            # Extraer heurística de la URL para todas las 30 variables
            valores_extraidos = analizar_url(url_input)

            # Crear el DataFrame con el orden original exacto de las variables
            input_completo = {}
            for col in feature_columns:
                input_completo[col] = valores_extraidos.get(col, 0.5)

            df_input = pd.DataFrame([input_completo])[feature_columns]

            # Mostrar visualmente las 30 variables estimadas divididas en tres columnas
            st.info("🔗 Características extraídas de la URL por el extractor heurístico integrado (30 Variables Predictoras):")

            # Crear tres columnas en Streamlit
            col1, col2, col3 = st.columns(3)

            # Dividir la lista de las 30 variables en tres grupos de 10 de manera equitativa
            grupo_1 = feature_columns[:10]
            grupo_2 = feature_columns[10:20]
            grupo_3 = feature_columns[20:]

            with col1:
                st.markdown("**Grupo 1 (Características de URL / Redirecciones)**")
                for col in grupo_1:
                    valor = input_completo[col]
                    st.write(f"🔹 **{col}:** `{valor}`")

            with col2:
                st.markdown("**Grupo 2 (Servicios, Protocolos y Puertos)**")
                for col in grupo_2:
                    valor = input_completo[col]
                    st.write(f"🔹 **{col}:** `{valor}`")

            with col3:
                st.markdown("**Grupo 3 (Métricas Externas e Identificadores)**")
                for col in grupo_3:
                    valor = input_completo[col]
                    st.write(f"🔹 **{col}:** `{valor}`")

            # Predecir
            if opcion_modelo == "Random Forest":
                prediccion = rf_model.predict(df_input)[0]
                probabilidad = rf_model.predict_proba(df_input).max()
            else:
                prob_nn = nn_optimized.predict(df_input)[0][0]
                prediccion = 1.0 if prob_nn > 0.5 else 0.0
                probabilidad = prob_nn if prediccion == 1.0 else (1.0 - prob_nn)

            # Mostrar Diagnóstico
            st.markdown("--- ")
            if prediccion == 1.0:
                st.success(f"✅ **Resultado:** ¡Sitio Web Legítimo! (Modelo: {opcion_modelo})")
            else:
                st.error(f"🚨 **Resultado:** ¡Alerta! Es un sitio potencial de **Phishing** (Modelo: {opcion_modelo})")

            st.info(f"Confianza de la predicción: {probabilidad * 100:.2f}%")

elif menu == "Vista de Modelos":
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
    opcion_modelo = st.selectbox("Seleccione el Algoritmo", ["Random Forest", "Red Neuronal"])

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

    # Completar las características restantes con valores neutrales (0.5 o 1.0 según la heurística por defecto)
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
            prediccion = 1.0 if prob_nn > 0.90 else 0.0
            probabilidad = prob_nn if prediccion == 1.0 else (1.0 - prob_nn)

        # Interpretar resultado (Asumiendo 1.0 = Legítimo, 0.0 = Phishing)
        if prediccion == 1.0:
            st.success(f"**Resultado:** ¡Sitio Web Legítimo! (Modelo: {opcion_modelo})")
        else:
            st.error(f"**Resultado:** ¡Alerta! Es un sitio potencial de **Phishing** (Modelo: {opcion_modelo})")

        st.info(f"Confianza de la predicción: {probabilidad * 100:.2f}%")
