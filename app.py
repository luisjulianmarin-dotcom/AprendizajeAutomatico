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
import requests
from bs4 import BeautifulSoup
import random

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


@st.cache_data
def cargar_base_datos_local(ruta_csv="URL_Phishing.csv"):
    urls = set()
    try:
        with open(ruta_csv, 'r', encoding='utf-8') as f:
            for linea in f:
                url_limpia = linea.strip().lower()
                if url_limpia and url_limpia != 'url':
                    urls.add(url_limpia)
        return urls
    except FileNotFoundError:
        return set()
    except Exception as e:
        st.error(f"Error al leer el archivo local: {e}")
        return set()

def evaluar_url_of_anchor(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        soup = BeautifulSoup(response.text, 'html.parser')
        anchors = soup.find_all('a')
        
        total_anchors = len(anchors)
        if total_anchors == 0:
            print("La página no contiene etiquetas <a>.")
            return 1
        
        suspicious_count = 0
      
        for a in anchors:
            href = a.get('href')
            if not href or href.startswith('#') or href.startswith('javascript:'):
                suspicious_count += 1
            else:
                parsed_href = urlparse(href)
                if parsed_href.netloc and parsed_href.netloc != base_domain:
                    suspicious_count += 1
                  
        percentage = (suspicious_count / total_anchors) * 100
        
        if percentage < 31:
            resultado = "Legítimo / Seguro (Clase baja o 1)"
            return 1.0
        elif 31 <= percentage <= 67:
            resultado = "Sospechoso (Clase neutral 0.5)"
            return 0.5
        else:
            resultado = "Phishing / Alto riesgo (Clase alta o 0)"
            return 0.0
        
    except requests.exceptions.RequestException as e:
        return 0

# Función heurística para extraer características a partir de una URL
def analizar_url(url):
    if not url.startswith(('http://', 'https://')):
        url_parseada = urlparse('http://' + url)
    else:
        url_parseada = urlparse(url)

    st.write(f"URL: {url_parseada}")
    domain = url_parseada.netloc
    path = url_parseada.path
    query = url_parseada.query
    analisis = {}

    # 30. statistical_report (Se revisa primero esta variable para afectar las otras del modelo)
    urls_maliciosas = cargar_base_datos_local()
    url_limpia = url.strip().lower()

    if url_limpia in urls_maliciosas:
        analisis['statistical_report'] = 0.0
    else:
        analisis['statistical_report'] = 1.0

    # 1. having_ip_address
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    if analisis['statistical_report'] != 1.0:
      analisis['having_ip_address'] = 0.5
    else:
      analisis['having_ip_address'] = 0.0 if ip_pattern.search(domain) else 1.0

    # 2. url_length
    longitud = len(url)
    if analisis['statistical_report'] != 1.0:
      analisis['url_length'] = 0.5
    else:
      if longitud < 10:
          analisis['url_length'] = 1.0
      elif longitud <= 15:
          analisis['url_length'] = 0.5
      else:
          analisis['url_length'] = 0.0

    # 3. shortining_service
    acortadores = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'is.gd', 'rebrand.ly', 'buff.ly','logi.php','logi.html','d6o']
    if analisis['statistical_report'] != 1.0:
        analisis['shortining_service'] = 0.5
    else:
       analisis['shortining_service'] = 0.0 if any(ac in domain for ac in acortadores) else 0.5

    # 4. having_at_symbol
    if analisis['statistical_report'] != 1.0:
      analisis['having_at_symbol'] = 0.5
    else:
      analisis['having_at_symbol'] = 0.0 if '@' in url else 1.0

    # 5. double_slash_redirecting
    if analisis['statistical_report'] != 1.0:
      analisis['double_slash_redirecting'] = 0.5
    else:
      analisis['double_slash_redirecting'] = 0.0 if url.rfind('//') > 2 else 1.0

    # 6. prefix_suffix (Guión '-' en dominio)
    if analisis['statistical_report'] != 1.0:
      analisis['double_slash_redirecting'] = 0.0
    else:
      analisis['prefix_suffix'] = 0.0 if '-' in domain else 0.5

    # 7. having_sub_domain (Cantidad de puntos en el dominio)
    dots = domain.count('.')
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['having_sub_domain'] = 0.0  # Sospechoso
    else:
      if dots <= 2:
          analisis['having_sub_domain'] = 1.0  # Seguro
      elif dots == 3:
          analisis['having_sub_domain'] = 0.5  # Moderado
      else:
          analisis['having_sub_domain'] = 0.0  # Sospechoso

    # 8. sslfinal_state
    if url_parseada.scheme == 'https':
        if analisis['statistical_report'] != 1.0:
          analisis['sslfinal_state'] = 0.5
        else:
          analisis['sslfinal_state'] = 1.0
    else:
        analisis['sslfinal_state'] = 0.0


    # 9. domain_registration_length
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['domain_registration_length'] = 0.0
    else:
      if url_parseada.scheme == 'https' and len(domain) > 4:
          analisis['domain_registration_length'] = 1.0
      else:
          analisis['domain_registration_length'] = 0.0

    # 10. favicon
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['favicon'] = 0.0
    else:
      analisis['favicon'] = 1.0 if url_parseada.scheme == 'https' else 0.0

    # 11. port
    puertos_sospechosos = [':8080', ':8443', ':3000', ':8000',':80']
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['port'] = 0.0
    else:
      analisis['port'] = 0.0 if any(port in domain for port in puertos_sospechosos) else 1.0

    # 12. https_token
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['https_token'] = 0.0
    else:
      analisis['https_token'] = 0.0 if 'https' in domain else 1.0

    # 13. request_url
    if 'wp-content' in url or 'assets' in url or 'uploads' in url:
        analisis['request_url'] = 0.5
    else:
        analisis['request_url'] = 1.0

    # 14. url_of_anchor
    analisis['url_of_anchor']= evaluar_url_of_anchor(url_limpia)
    
    # 15. links_in_tags
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['links_in_tags'] = 0.0
    else:
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
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['submitting_to_email'] = 0.0
    else:
      analisis['submitting_to_email'] = 0.0 if 'mail' in url or 'mailto' in url else 0.5

    # 18. abnormal_url
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['abnormal_url']  = 0.0
    else:
      analisis['abnormal_url'] = 0.5 if domain in url else 0.0

    # 19. redirect
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['redirect']  = 0.0
    else:
      analisis['redirect'] = 0.0 if url.count('//') > 1 else 1.0

    # 20. on_mouseover
    analisis['on_mouseover'] = 0.0

    # 21. rightclick
    analisis['rightclick'] = 0.0

    # 22. popupwindow
    analisis['popupwindow'] = 0.0

    # 23. iframe
    analisis['iframe'] = 0.0

    # 24. age_of_domain
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['age_of_domain']  = 0.0
    else:
      if len(domain) > 4 and url_parseada.scheme == 'https':
          analisis['age_of_domain'] = 0.5
      else:
          analisis['age_of_domain'] = 0.0

    # 25. dnsrecord
    analisis['dnsrecord'] = 0.0

    # 26. web_traffic
    dominios_populares = ['google.com', 'facebook.com', 'youtube.com', 'twitter.com', 'instagram.com', 'netflix.com', 'amazon.com', 'microsoft.com', 'github.com']
    if ip_pattern.search(domain) or analisis['statistical_report'] != 1.0:
      analisis['web_traffic']  = 0.0
    else:
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
    analisis['google_index'] = 0.0

    # 29. links_pointing_to_page
    if len(path) > 30:
        analisis['links_pointing_to_page'] = 0.0
    elif len(path) > 10:
        analisis['links_pointing_to_page'] = 0.5
    else:
        analisis['links_pointing_to_page'] = 1.0

    return analisis

# Menú lateral de navegación
st.sidebar.header("Panel de Control")
menu = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Predicción por URL Directa","Variables Random x Modelo","Vista de Modelos","Vista General", "Resumen Estadístico", "Control de Nulos", "Visualizaciones"]
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
    st.success("¡Tratamiento de nulos aplicado correctamente (imputación por moda)!")

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
    st.write("Introduce una dirección web y los modelos extraerán de forma automática sus características para determinar el nivel de riesgo de Phishing.")
    st.write("Ejemplos de links o sitios web que son sospechosos: ")
    st.write("                          http://103.119.17.126/?d6o")
    st.write("                          http://paypal-secure-auth.net")
    st.write("                          https://www.google.com")
    st.write("                          http://bancolombia.seguridad-actualizacion.co")
    st.write("                          https://sesja-1c4b-pkobiznes-firma.wabze.net/pl-PL")

    url_input = st.text_input("Ingrese la dirección URL:", "https://www.google.com")

    # Selector de modelo
    opcion_modelo = st.selectbox("Algoritmo de Clasificación", ["Random Forest", "Red Neuronal"])

    if st.button("Analizar y Predecir URL"):
        if url_input.strip() == "":
            st.warning("Por favor ingrese una URL válida.")
        else:
            # Cargar modelos y mapeos
            rf_model = joblib.load("modelo_rf.joblib")
            nn_optimized = tf.keras.models.load_model("modelo_nnrl_optimizado.keras")
            feature_columns = joblib.load("columnas_features.joblib")

            valores_extraidos = analizar_url(url_input) # Extraer heurística de la URL para todas las 30 variables

            input_completo = {}   # Crear el DataFrame con el orden original exacto de las variables
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
            if prediccion == 1.0 and probabilidad >=0.80:
                st.success(f"✅ **Resultado:** ¡Sitio Web Legítimo! (Modelo: {opcion_modelo})")
            else:
                st.error(f"🚨 **Resultado:** ¡Alerta! Es un sitio potencial de **Phishing** (Modelo: {opcion_modelo})")

            st.info(f"Confianza de la predicción: {probabilidad * 100:.2f}%")

elif menu == "Vista de Modelos":
    # Cargar modelos y columnas
    rf_model = joblib.load("modelo_rf.joblib")
    nn_optimized = tf.keras.models.load_model("modelo_nnrl_optimizado.keras")
    feature_columns = joblib.load("columnas_features.joblib")

    # Selector de modelo con nombres reales
    opcion_modelo = st.selectbox("Seleccione el Algoritmo", ["Random Forest", "Red Neuronal"])
    st.subheader("Ingrese las 15 características más significativas de la URL / Sitio Web")
    input_data = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data['sslfinal_state'] = st.selectbox("sslfinal_state (Certificado SSL)", [0.0, 0.5, 1.0], format_func=lambda x: "Inseguro (0.0)" if x==0.0 else ("Dudoso (0.5)" if x==0.5 else "Seguro (1.0)"))
        input_data['url_of_anchor'] = st.selectbox("url_of_anchor (Anclas de URL)", [0.0, 0.5, 1.0], format_func=lambda x: "Sospechoso (0.0)" if x==0.0 else ("Moderado (0.5)" if x==0.5 else "Seguro (1.0)"))
        input_data['prefix_suffix'] = st.selectbox("prefix_suffix (Guión en Dominio)", [0.0, 1.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
        input_data['having_sub_domain'] = st.selectbox("having_sub_domain (Subdominios)", [0.0, 0.5, 1.0], format_func=lambda x: "Múltiples (0.0)" if x==0.0 else ("Uno (0.5)" if x==0.5 else "Ninguno (1.0)"))
        input_data['web_traffic'] = st.selectbox("web_traffic (Tráfico de Red)", [0.0, 0.5, 1.0], format_func=lambda x: "Bajo/Nulo (0.0)" if x==0.0 else ("Medio (0.5)" if x==0.5 else "Alto (1.0)"))
        
    with col2:
        input_data['request_url'] = st.selectbox("request_url (Recursos Externos)", [0.0, 0.5, 1.0], format_func=lambda x: "Excesivo (0.0)" if x==0.0 else ("Moderado (0.5)" if x==0.5 else "Seguro (1.0)"))
        input_data['links_in_tags'] = st.selectbox("links_in_tags (Enlaces en Metaetiquetas)", [0.0, 0.5, 1.0], format_func=lambda x: "Sospechoso (0.0)" if x==0.0 else ("Incompleto (0.5)" if x==0.5 else "Seguro (1.0)"))
        input_data['sfh'] = st.selectbox("sfh (Manejo de Formularios)", [0.0, 0.5, 1.0], format_func=lambda x: "Vacío/Inseguro (0.0)" if x==0.0 else ("Diferente Dominio (0.5)" if x==0.5 else "Mismo Dominio (1.0)"))
        input_data['age_of_domain'] = st.selectbox("age_of_domain (Edad del Dominio)", [0.0, 0.5, 1.0], format_func=lambda x: "Reciente (0.0)" if x==0.0 else ("Moderado (0.5)" if x==0.5 else "Antiguo (1.0)"))
        input_data['having_ip_address'] = st.selectbox("having_ip_address (Dirección IP)", [0.0, 1.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
        
    with col3:
        input_data['dnsrecord'] = st.selectbox("dnsrecord (Registro DNS)", [0.0, 1.0], format_func=lambda x: "No Registrado (0.0)" if x==0.0 else "Registrado (1.0)")
        input_data['statistical_report'] = st.selectbox("statistical_report (API / PhishTank)", [0.0, 0.5, 1.0], format_func=lambda x: "Phishing (0.0)" if x==0.0 else ("No Identificado (0.5)" if x==0.5 else "Legítimo (1.0)"))
        input_data['shortining_service'] = st.selectbox("shortining_service (Acortadores)", [0.0, 1.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
        input_data['having_at_symbol'] = st.selectbox("having_at_symbol (Arroba @)", [0.0, 1.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
        input_data['double_slash_redirecting'] = st.selectbox("double_slash_redirecting (Doble barra)", [0.0, 1.0], format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)")
          
    # Completar el resto de variables con el valor neutral por defecto (0.5)
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0.0

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
            prediccion = 1.0 if prob_nn > 0.50 else 0.0
            probabilidad = prob_nn if prediccion == 1.0 else (1.0 - prob_nn)

        # Interpretar resultado (Asumiendo 1.0 = Legítimo, 0.0 = Phishing)
        if prediccion == 1.0 and probabilidad >=0.80:
            st.success(f"**Resultado:** ¡Sitio Web Legítimo! (Modelo: {opcion_modelo})")
        else:
            st.error(f"**Resultado:** ¡Alerta! Es un sitio potencial de **Phishing** (Modelo: {opcion_modelo})")

        st.info(f"Confianza de la predicción: {probabilidad * 100:.2f}%")

elif menu == "Variables Random x Modelo":
    # Cargar modelos y columnas
    rf_model = joblib.load("modelo_rf.joblib")
    nn_optimized = tf.keras.models.load_model("modelo_nnrl_optimizado.keras")
    feature_columns = joblib.load("columnas_features.joblib")

    # Selector de modelo con nombres reales
    opcion_modelo = st.selectbox("Seleccione el Algoritmo", ["Random Forest", "Red Neuronal"])
    st.subheader("Ingrese las 15 características más significativas de la URL / Sitio Web")

    # Definir las listas de opciones permitidas para cada variable específica
    opciones_3_niveles = [0.0, 0.5, 1.0]
    opciones_2_niveles = [0.0, 1.0]

    # Diccionario de mapeo de opciones para cada variable
    mapping_variables = {
        'sslfinal_state': opciones_3_niveles,
        'url_of_anchor': opciones_3_niveles,
        'prefix_suffix': opciones_2_niveles,
        'having_sub_domain': opciones_3_niveles,
        'web_traffic': opciones_3_niveles,
        'request_url': opciones_3_niveles,
        'links_in_tags': opciones_3_niveles,
        'sfh': opciones_3_niveles,
        'age_of_domain': opciones_3_niveles,
        'having_ip_address': opciones_2_niveles,
        'dnsrecord': opciones_2_niveles,
        'statistical_report': opciones_3_niveles,
        'shortining_service': opciones_2_niveles,
        'having_at_symbol': opciones_2_niveles,
        'double_slash_redirecting': opciones_2_niveles
    }

    # Botón para asignar valores aleatorios a todas las variables usando session_state
    if st.button("🎲 Llenar valores aleatoriamente"):
        for var_name, opciones in mapping_variables.items():
            st.session_state[f"input_{var_name}"] = random.choice(opciones)
        st.success("¡Se han asignado valores aleatorios a los 15 campos!")
        st.rerun()

    input_data = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data['sslfinal_state'] = st.selectbox(
            "sslfinal_state (Certificado SSL)", opciones_3_niveles, 
            key="input_sslfinal_state",
            format_func=lambda x: "Inseguro (0.0)" if x==0.0 else ("Dudoso (0.5)" if x==0.5 else "Seguro (1.0)")
        )
        input_data['url_of_anchor'] = st.selectbox(
            "url_of_anchor (Anclas de URL)", opciones_3_niveles, 
            key="input_url_of_anchor",
            format_func=lambda x: "Sospechoso (0.0)" if x==0.0 else ("Moderado (0.5)" if x==0.5 else "Seguro (1.0)")
        )
        input_data['prefix_suffix'] = st.selectbox(
            "prefix_suffix (Guión en Dominio)", opciones_2_niveles, 
            key="input_prefix_suffix",
            format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)"
        )
        input_data['having_sub_domain'] = st.selectbox(
            "having_sub_domain (Subdominios)", opciones_3_niveles, 
            key="input_having_sub_domain",
            format_func=lambda x: "Múltiples (0.0)" if x==0.0 else ("Uno (0.5)" if x==0.5 else "Ninguno (1.0)")
        )
        input_data['web_traffic'] = st.selectbox(
            "web_traffic (Tráfico de Red)", opciones_3_niveles, 
            key="input_web_traffic",
            format_func=lambda x: "Bajo/Nulo (0.0)" if x==0.0 else ("Medio (0.5)" if x==0.5 else "Alto (1.0)")
        )
        
    with col2:
        input_data['request_url'] = st.selectbox(
            "request_url (Recursos Externos)", opciones_3_niveles, 
            key="input_request_url",
            format_func=lambda x: "Excesivo (0.0)" if x==0.0 else ("Moderado (0.5)" if x==0.5 else "Seguro (1.0)")
        )
        input_data['links_in_tags'] = st.selectbox(
            "links_in_tags (Enlaces en Metaetiquetas)", opciones_3_niveles, 
            key="input_links_in_tags",
            format_func=lambda x: "Sospechoso (0.0)" if x==0.0 else ("Incompleto (0.5)" if x==0.5 else "Seguro (1.0)")
        )
        input_data['sfh'] = st.selectbox(
            "sfh (Manejo de Formularios)", opciones_3_niveles, 
            key="input_sfh",
            format_func=lambda x: "Vacío/Inseguro (0.0)" if x==0.0 else ("Diferente Dominio (0.5)" if x==0.5 else "Mismo Dominio (1.0)")
        )
        input_data['age_of_domain'] = st.selectbox(
            "age_of_domain (Edad del Dominio)", opciones_3_niveles, 
            key="input_age_of_domain",
            format_func=lambda x: "Reciente (0.0)" if x==0.0 else ("Moderado (0.5)" if x==0.5 else "Antiguo (1.0)")
        )
        input_data['having_ip_address'] = st.selectbox(
            "having_ip_address (Dirección IP)", opciones_2_niveles, 
            key="input_having_ip_address",
            format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)"
        )
        
    with col3:
        input_data['dnsrecord'] = st.selectbox(
            "dnsrecord (Registro DNS)", opciones_2_niveles, 
            key="input_dnsrecord",
            format_func=lambda x: "No Registrado (0.0)" if x==0.0 else "Registrado (1.0)"
        )
        input_data['statistical_report'] = st.selectbox(
            "statistical_report (API / PhishTank)", opciones_3_niveles, 
            key="input_statistical_report",
            format_func=lambda x: "Phishing (0.0)" if x==0.0 else ("No Identificado (0.5)" if x==0.5 else "Legítimo (1.0)")
        )
        input_data['shortining_service'] = st.selectbox(
            "shortining_service (Acortadores)", opciones_2_niveles, 
            key="input_shortining_service",
            format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)"
        )
        input_data['having_at_symbol'] = st.selectbox(
            "having_at_symbol (Arroba @)", opciones_2_niveles, 
            key="input_having_at_symbol",
            format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)"
        )
        input_data['double_slash_redirecting'] = st.selectbox(
            "double_slash_redirecting (Doble barra)", opciones_2_niveles, 
            key="input_double_slash_redirecting",
            format_func=lambda x: "Sí (0.0)" if x==0.0 else "No (1.0)"
        )
         
    # Completar el resto de variables con el valor por defecto
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0.0

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
            prediccion = 1.0 if prob_nn > 0.50 else 0.0
            probabilidad = prob_nn if prediccion == 1.0 else (1.0 - prob_nn)

        # Interpretar resultado (Asumiendo 1.0 = Legítimo, 0.0 = Phishing)
        if prediccion == 1.0 and probabilidad >= 0.80:
            st.success(f"**Resultado:** ¡Sitio Web Legítimo! (Modelo: {opcion_modelo})")
        else:
            st.error(f"**Resultado:** ¡Alerta! Es un sitio potencial de **Phishing** (Modelo: {opcion_modelo})")

        st.info(f"Confianza de la predicción: {probabilidad * 100:.2f}%")
