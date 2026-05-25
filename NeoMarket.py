import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="NeoMarket", page_icon="IMG/NeoMarket - Icono.png", layout="wide")

# ==========================================
# 1. FUNCIONES DE LIMPIEZA (ETL)
# ==========================================
def limpiar_nombre(nombre):
    if pd.isna(nombre) or str(nombre).strip() == "": return "No Name (NN)"
    nombre = str(nombre)
    nombre = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', nombre)
    nombre = re.sub(r'\s+', ' ', nombre).strip()
    return nombre.title()

def inferir_genero(nombre):
    if pd.isna(nombre) or not isinstance(nombre, str) or nombre == "No Name (NN)": return "Otro"
    nombre = nombre.strip().split()[0].lower()
    if nombre.endswith('a') or nombre in ['maria', 'ana', 'luisa', 'camila', 'laura', 'andrea']: return 'Femenino'
    elif nombre.endswith('o') or nombre in ['juan', 'carlos', 'pedro', 'luis', 'sergio', 'andres']: return 'Masculino'
    return 'Otro'

def limpiar_localidad(valor):
    if not isinstance(valor, str) or valor.strip() == '': return "(Loc. Desconocida)"
    valor = valor.lower().strip()
    if "chap" in valor: return "Chapinero"
    elif "usaq" in valor or "usaquen" in valor: return "Usaquén"
    elif "eng" in valor: return "Engativá"
    elif "sub" in valor: return "Suba"
    elif "ken" in valor: return "Kennedy"
    return "(Loc. Desconocida)"

def limpiar_nivel(valor):
    if pd.isna(valor): return "NS. Desconocido"
    valor = str(valor).strip().capitalize()
    if "bajo" in valor.lower(): return "Bajo"
    elif "medio" in valor.lower(): return "Medio"
    elif "alto" in valor.lower(): return "Alto"
    return "NS. Desconocido"

# ==========================================
# 2. MOTOR DEL ETL AUTOMÁTICO
# ==========================================
@st.cache_data
def ejecutar_etl_automatico():
    cambios_log = [] 
    dataframes = {}
    
    # NUEVO: Contadores de métricas reales
    metricas = {"nulos": 0, "textos": 0, "fechas": 0}
    
    ruta_datos = "datos/"
    archivos = ["Cliente.csv", "Producto.csv", "Tiempo.csv", "Tienda.csv", "Ventas.csv"]
    
    for arch in archivos:
        try:
            nombre_tabla = arch.replace(".csv", "")
            dataframes[nombre_tabla] = pd.read_csv(os.path.join(ruta_datos, arch), sep=None, engine='python', encoding='utf-8-sig')
            cambios_log.append(f"✅ {nombre_tabla}: Se cargaron {len(dataframes[nombre_tabla])} registros iniciales.")
        except Exception as e:
            st.error(f"Error al cargar {arch}. Asegúrate de que está en la carpeta 'datos/'. Error: {e}")
            return None, None, None # Modificado para devolver 3 elementos en caso de error

    clientes = dataframes['Cliente']
    ventas = dataframes['Ventas']
    tiempo = dataframes['Tiempo']
    tiendas = dataframes['Tienda']
    productos = dataframes['Producto']

    # Transformaciones globales
    for nombre, df in dataframes.items():
        df.drop(columns=df.columns[df.columns.str.contains('^Unnamed')], inplace=True, errors='ignore')
        df.columns = df.columns.str.strip().str.replace(' ', '_')
    cambios_log.append("🔧 Todas las tablas: Se eliminaron espacios en nombres de columnas y columnas 'Unnamed'.")

    # --- Limpieza Clientes ---
    # Contar nulos reales antes de imputar
    if 'Edad' in clientes.columns:
        metricas["nulos"] += int(clientes['Edad'].isna().sum())
        clientes['Edad'] = pd.to_numeric(clientes['Edad'], errors='coerce').fillna(clientes['Edad'].median()).astype(int)
    
    if 'Genero' in clientes.columns:
        metricas["nulos"] += int(clientes['Genero'].isna().sum())
    
    # Contar textos a capitalizar (Nombres, Localidades, Niveles)
    metricas["textos"] += len(clientes) * 3 
        
    clientes['Nombre'] = clientes['Nombre'].apply(limpiar_nombre)
    clientes['Genero'] = clientes['Genero'].fillna(clientes['Nombre'].apply(inferir_genero)).replace('', 'Otro')
    clientes['Localidad'] = clientes['Localidad'].apply(limpiar_localidad)
    clientes['Nivel_Socioeconomico'] = clientes['Nivel_Socioeconomico'].apply(limpiar_nivel)
    cambios_log.append("🧑 Clientes: Edades nulas imputadas. Nombres, géneros y localidades estandarizadas.")

    # --- Limpieza Ventas ---
    # Contar nulos reales antes de rellenar con ceros
    if 'Cantidad' in ventas.columns: metricas["nulos"] += int(ventas['Cantidad'].isna().sum())
    if 'Precio_Unitario' in ventas.columns: metricas["nulos"] += int(ventas['Precio_Unitario'].isna().sum())
    if 'Descuento' in ventas.columns: metricas["nulos"] += int(ventas.get('Descuento', pd.Series([0])).isna().sum())
    
    ventas['Cantidad'] = pd.to_numeric(ventas['Cantidad'], errors='coerce').fillna(0)
    ventas['Precio_Unitario'] = pd.to_numeric(ventas['Precio_Unitario'], errors='coerce').fillna(0).astype(int)
    ventas['Descuento'] = pd.to_numeric(ventas.get('Descuento', 0), errors='coerce').fillna(0).astype(int)
    ventas.loc[ventas['Descuento'] < 0, 'Descuento'] = 0
    ventas['Total_Venta'] = ((ventas['Precio_Unitario'] - ventas['Descuento']) * ventas['Cantidad']).apply(lambda x: max(x, 0)).astype(int)
    cambios_log.append("💰 Ventas: Precios convertidos a enteros. Descuentos negativos corregidos a 0.")

    # --- Limpieza Tiempo ---
    metricas["fechas"] += len(tiempo) # Sumamos las fechas formateadas
    tiempo['Fecha'] = pd.to_datetime(tiempo['Fecha'], errors='coerce', dayfirst=True)
    tiempo['Mes'] = tiempo['Fecha'].dt.month.fillna(0).astype(int)
    tiempo['Dia'] = tiempo['Fecha'].dt.day.fillna(0).astype(int)
    tiempo['Anio'] = tiempo['Fecha'].dt.year.fillna(0).astype(int)
    tiempo = tiempo[tiempo['Fecha'].notna()]
    cambios_log.append("📅 Tiempo: Fechas parseadas correctamente.")

    # --- Limpieza Tiendas ---
    metricas["fechas"] += len(tiendas) # Sumamos las fechas de apertura formateadas
    tiendas['Fecha_Apertura'] = pd.to_datetime(tiendas['Fecha_Apertura'], errors='coerce', dayfirst=True)
    tiendas = tiendas[tiendas['Fecha_Apertura'].notna()]

    # Limpieza final
    for nombre, df in dataframes.items():
        df.drop_duplicates(inplace=True)
        df.dropna(how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)
    cambios_log.append("🧹 Todas las tablas: Se eliminaron filas 100% nulas y duplicadas.")

    # RETORNAMOS LAS MÉTRICAS JUNTO CON LOS DATOS
    return dataframes, cambios_log, metricas

# ==========================================
# 3. MOTOR DE MACHINE LEARNING (K-Means)
# ==========================================
@st.cache_data
def aplicar_kmeans(datos_limpios):
    clientes = datos_limpios['Cliente'].copy()
    ventas = datos_limpios['Ventas'].copy()
    tiempo = datos_limpios['Tiempo'].copy()
    
    df_completo = pd.merge(ventas, tiempo[['ID_Tiempo', 'Fecha']], on='ID_Tiempo', how='inner')
    fecha_actual = df_completo['Fecha'].max() + pd.Timedelta(days=1)
    
    rfm = df_completo.groupby('ID_Cliente').agg({
        'Fecha': lambda x: (fecha_actual - x.max()).days,
        'ID_Venta': 'count',
        'Total_Venta': 'sum'
    }).reset_index()
    
    rfm.rename(columns={'Fecha': 'Recencia', 'ID_Venta': 'Frecuencia', 'Total_Venta': 'Monetario'}, inplace=True)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recencia', 'Frecuencia', 'Monetario']])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    def definir_perfil(c):
        if c == 0: return "Cliente Bronce (Bajo Consumo)"
        if c == 1: return "Cliente VIP (Alto Valor)"
        if c == 2: return "Cliente Frecuente"
        return "Cliente en Riesgo (Inactivo)"
    
    rfm['Perfil_Cliente'] = rfm['Cluster'].apply(definir_perfil)
    
    clientes_enriquecidos = pd.merge(clientes, rfm[['ID_Cliente', 'Cluster', 'Perfil_Cliente', 'Recencia', 'Frecuencia', 'Monetario']], on='ID_Cliente', how='left')
    clientes_enriquecidos['Perfil_Cliente'] = clientes_enriquecidos['Perfil_Cliente'].fillna("Sin Compras")
    clientes_enriquecidos['Cluster'] = clientes_enriquecidos['Cluster'].fillna(-1)
    
    return clientes_enriquecidos, rfm

# ==========================================
# 4. INTERFAZ WEB (SIDEBAR Y RUTAS)
# ==========================================

# Ejecutar procesos core en segundo plano (Ahora recibe 3 variables)
with st.spinner('Cargando motor de datos y entrenando IA...'):
    datos_limpios, log_cambios, metricas_etl = ejecutar_etl_automatico()

if datos_limpios:
    clientes_ml, rfm_ml = aplicar_kmeans(datos_limpios)
    datos_limpios['Cliente'] = clientes_ml 

    # --- BARRA LATERAL (SIDEBAR) ---
    st.sidebar.image("IMG/NeoMarket - Logo.png", use_container_width=True)
    st.sidebar.markdown("---")
    
    st.sidebar.title("Navegación")
    pagina = st.sidebar.radio("Módulos del Sistema:", [
        "⚙️ ETL y Datos Crudos", 
        "🚀 Transformación Digital", 
        "📊 Minería - Dashboards", 
        "🎲 Modelado y Simulación"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.title("🔍 Filtrar por tienda")
    
    fechas_apertura = {
        "NeoMarket Nororiental": datetime.date(2023, 3, 15),
        "NeoMarket Norte": datetime.date(2022, 7, 20),
        "NeoMarket Noroccidental": datetime.date(2020, 1, 10),
        "NeoMarket Noroccidental Alto": datetime.date(2021, 9, 5),
        "NeoMarket Suroccidental": datetime.date(2022, 5, 1)
    }
    
    lista_tiendas = ["Todas las Tiendas"] + list(fechas_apertura.keys())
    tienda_seleccionada = st.sidebar.selectbox("🛒 Seleccionar Tienda", lista_tiendas)
    
    if tienda_seleccionada == "Todas las Tiendas":
        min_date = datetime.date(2020, 1, 10) 
    else:
        min_date = fechas_apertura[tienda_seleccionada]
        
    max_date = datetime.date.today() 
    
    rango_fechas = st.sidebar.date_input(
        "📅 Fecha de Operación",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # --- ENRUTADOR DE PÁGINAS ---
    
    # MÓDULO 1: ETL
    if pagina == "⚙️ ETL y Datos Crudos":
        st.title("⚙️ Motor de Integración de Datos (ETL)")
        st.write("Visualización del proceso de extracción, transformación y carga (automático).")
        
        # KPIs CON DATOS REALES EXTRAÍDOS DEL MOTOR
        st.subheader("Indicadores de Calidad de Datos")
        c1, c2, c3 = st.columns(3)
        c1.metric("🕳️ Datos Vacíos Corregidos", f"{metricas_etl['nulos']:,}")
        c2.metric("🔠 Textos Estandarizados", f"{metricas_etl['textos']:,}")
        c3.metric("📅 Fechas Formateadas", f"{metricas_etl['fechas']:,}")
        
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["📝 Resumen de Cambios", "📊 Explorador de Datos"])
        
        with tab1:
            st.subheader("Bitácora de Transformación")
            for log in log_cambios:
                st.info(log)
                
        with tab2:
            st.subheader("Bases de Datos Transformadas")
            tabla_seleccionada = st.selectbox("Selecciona la tabla a inspeccionar:", list(datos_limpios.keys()))
            df_mostrar = datos_limpios[tabla_seleccionada]
            
            col1, col2 = st.columns(2)
            col1.metric("Total de Filas", f"{df_mostrar.shape[0]:,}")
            col2.metric("Total de Columnas", f"{df_mostrar.shape[1]:,}")
            st.dataframe(df_mostrar, use_container_width=True)

    # MÓDULO 2: TRANSFORMACIÓN DIGITAL
    elif pagina == "🚀 Transformación Digital":
        st.title("🚀 Estrategia de Transformación Digital")
        st.write("Evaluación de madurez digital, plan de acción, roadmap tecnológico y métricas estratégicas para NeoMarket.")
        
        # Pestañas del módulo
        tab_madurez, tab_roadmap, tab_kpis, tab_roi = st.tabs([
            "📊 Madurez Digital", 
            "🗺️ Roadmap y Plan de Acción", 
            "🎯 OKRs y KPIs", 
            "💰 Retorno de Inversión (ROI)"
        ])
        
        # ---------------------------------------------------------
        # PESTAÑA 1: MADUREZ DIGITAL
        # ---------------------------------------------------------
        with tab_madurez:
            st.subheader("Evaluación Actual de NeoMarket")
            st.info("**Diagnóstico:** NeoMarket se posiciona actualmente en un **Nivel 3 (Definido)**. Cuenta con una bodega de datos, herramientas de Business Intelligence, minería de datos y modelos de simulación aplicados a procesos clave. Sin embargo, existen oportunidades de mejora hacia la automatización total y la consolidación de una cultura 100% data-driven.")
            
            st.markdown("### Matriz de Madurez Digital")
            # Convertimos la tabla del PDF en un DataFrame
            datos_madurez = {
                "Dimensión": ["Estructura organizacional", "Datos", "Tecnología", "Procesos", "Analítica", "Modelado y simulación", "Cultura digital", "Seguridad", "Toma de decisiones", "Integración"],
                "Nivel 3 - Definido (Actual)": ["Roles definidos (gerente, analista)", "Bodega de datos implementada", "BI, bodega de datos, seguridad", "Procesos definidos y estandarizados", "Análisis descriptivo y segmentación", "Modelos básicos de simulación", "Cultura en desarrollo", "Control de accesos y protección", "Decisiones basadas en dashboards", "Integración parcial"],
                "Nivel 4 - Gestionado (Meta a mediano plazo)": ["Equipos integrados y colaborativos", "Datos integrados en toda la org.", "Integración y automatización", "Procesos automatizados", "Analítica predictiva", "Simulación avanzada de escenarios", "Cultura data-driven consolidada", "Monitoreo y auditorías constantes", "Decisiones predictivas", "Integración total"],
                "Nivel 5 - Optimizado (Visión a futuro)": ["Organización ágil orientada a datos", "Datos explotados estratégicamente", "Ecosistema inteligente con IA", "Procesos optimizados y adaptativos", "Analítica prescriptiva", "Optimización automática en tiempo real", "Innovación basada en datos", "Seguridad proactiva e inteligente", "Decisiones automatizadas", "Ecosistema completamente conectado"]
            }
            df_madurez = pd.DataFrame(datos_madurez)
            st.dataframe(df_madurez, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # PESTAÑA 2: ROADMAP Y PLAN DE ACCIÓN
        # ---------------------------------------------------------
        with tab_roadmap:
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Plan de Acción")
                datos_plan = {
                    "Fase": ["Inicio", "Planeación", "Ejecución", "Ejecución", "Ejecución", "Control", "Cierre"],
                    "Actividad": ["Diagnóstico digital", "Diseño de arquitectura (bodega)", "Implementación de bodega de datos", "Aplicación de minería de datos", "Desarrollo de simulaciones", "Creación de dashboards y KPIs", "Evaluación y mejora continua"],
                    "Tiempo": ["Semana 1", "Semanas 2-3", "Semanas 4-6", "Semanas 6-8", "Semanas 8-10", "Semana 12", "Semana 13"]
                }
                st.dataframe(pd.DataFrame(datos_plan), use_container_width=True, hide_index=True)
                
            with c2:
                st.subheader("RoadMap Tecnológico (3 a 5 años)")
                datos_roadmap = {
                    "Fase": ["1. Fundamentos Digitales", "2. Analítica y Visualización", "3. Optimización y Simulación", "4. Automatización e Integración", "5. Transformación Total"],
                    "Tiempo": ["0-6 meses", "6-12 meses", "1-2 años", "2-3 años", "3-5 años"],
                    "Objetivo": ["Consolidar base tecnológica", "Generar valor con datos", "Mejorar procesos", "Escalar el sistema", "Alcanzar nivel máximo con IA"]
                }
                st.dataframe(pd.DataFrame(datos_roadmap), use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # PESTAÑA 3: OKRs y KPIs
        # ---------------------------------------------------------
        with tab_kpis:
            st.subheader("🎯 Objetivo Principal (OKR)")
            st.success("Implementar una estrategia de transformación digital en NeoMarket mediante una bodega de datos, minería de datos, modelado y simulación e inteligencia de negocios para optimizar la toma de decisiones, mejorar la eficiencia operativa y fortalecer la competitividad organizacional.")
            
            st.markdown("### 📌 Resultados Clave (Key Results)")
            kr_col1, kr_col2 = st.columns(2)
            with kr_col1:
                st.markdown("""
                * **KR1:** Centralizar y mejorar la calidad de la información.
                * **KR2:** Optimizar el análisis y la toma de decisiones.
                * **KR3:** Mejorar la eficiencia operativa del negocio.
                """)
            with kr_col2:
                st.markdown("""
                * **KR4:** Fortalecer la seguridad y confiabilidad de la información.
                * **KR5:** Impulsar la transformación digital y adopción tecnológica.
                * **KR6:** Mejorar el desempeño del proyecto (Enfoque PMI).
                """)
                
            st.markdown("### 📈 Indicadores Clave (KPIs)")
            datos_kpis = {
                "KPI": ["Cumplimiento del alcance digital", "Índice de desempeño del cronograma (SPI)", "Índice de desempeño de costos (CPI)", "Calidad de datos", "Adopción del sistema", "Precisión analítica", "Nivel de automatización", "Incidentes de seguridad"],
                "Categoría": ["Alcance", "Tiempo", "Costos", "Calidad", "Recursos", "Calidad", "Integración", "Riesgos"],
                "Mide": ["% de funcionalidades implementadas", "Avance vs plan", "Control del presupuesto", "Nivel de datos limpios y confiables", "Uso por parte de usuarios", "Exactitud de modelos de minería", "Procesos automatizados", "Nivel de vulnerabilidad"],
                "Meta": ["≥ 95%", "≥ 1", "≥ 1", "≥ 95%", "≥ 80%", "≥ 85%", "≥ 70%", "≤ 2 mensual"]
            }
            st.dataframe(pd.DataFrame(datos_kpis), use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # PESTAÑA 4: RETORNO DE INVERSIÓN (ROI)
        # ---------------------------------------------------------
        with tab_roi:
            st.subheader("💰 Proyección del Retorno de Inversión (ROI)")
            st.write("Cálculo proyectado a 5 años basado en los beneficios generados por la optimización de inventarios, automatización de procesos y reducción de pérdidas operativas.")
            
            col_roi1, col_roi2, col_roi3 = st.columns(3)
            col_roi1.metric("Inversión Total Proyectada", "$322,000,000 COP")
            col_roi2.metric("Beneficio Neto Esperado", "$480,000,000 COP")
            col_roi3.metric("Retorno de Inversión (ROI)", "49.0%", delta="Rentable", delta_color="normal")
            
            st.markdown("---")
            st.latex(r"ROI = \left( \frac{\text{Beneficio Neto} - \text{Inversión Total}}{\text{Inversión Total}} \right) \times 100")
            st.write("Esto significa que el proyecto NeoMarket generará un retorno aproximado del **49% sobre la inversión realizada** durante los cinco años de implementación.")

    # MÓDULO 3: MINERÍA DE DATOS Y DASHBOARDS
    elif pagina == "📊 Minería - Dashboards":
        st.title("📊 Analítica Avanzada y Dashboards")
        
        # 1. Crear un DataFrame Maestro cruzando todas las tablas necesarias
        df_ventas = datos_limpios['Ventas']
        df_tiempo = datos_limpios['Tiempo']
        df_tienda = datos_limpios['Tienda']
        df_producto = datos_limpios['Producto']
        df_cliente = datos_limpios['Cliente']
        
        # 🛠️ SOLUCIÓN AL BUG: Eliminar espacios en blanco invisibles en los nombres de las tiendas
        df_tienda['Nombre_Tienda'] = df_tienda['Nombre_Tienda'].astype(str).str.strip()
        
        # Uniones (Merges)
        df_master = df_ventas.merge(df_tiempo[['ID_Tiempo', 'Fecha']], on='ID_Tiempo', how='inner')
        df_master = df_master.merge(df_tienda[['ID_Tienda', 'Nombre_Tienda']], on='ID_Tienda', how='inner')
        df_master = df_master.merge(df_producto[['ID_Producto', 'Nombre_Producto', 'Categoria']], on='ID_Producto', how='inner')
        df_master = df_master.merge(df_cliente[['ID_Cliente', 'Perfil_Cliente', 'Localidad', 'Nivel_Socioeconomico']], on='ID_Cliente', how='left')
        # 2. Aplicar los Filtros Globales (Sidebar)
        if len(rango_fechas) == 2:
            start_date, end_date = rango_fechas
        else:
            start_date, end_date = rango_fechas[0], rango_fechas[0]
            
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filtrar por fecha y tienda
        mask_fechas = (df_master['Fecha'] >= start_date) & (df_master['Fecha'] <= end_date)
        df_filtrado = df_master[mask_fechas]
        
        if tienda_seleccionada != "Todas las Tiendas":
            df_filtrado = df_filtrado[df_filtrado['Nombre_Tienda'] == tienda_seleccionada.strip()]
            
        # 3. Construir las pestañas (Tabs)
        tab_ml, tab_dash = st.tabs(["🤖 Modelo K-Means ", "📈 Dashboards Interactivos"])
        
        with tab_ml:
            st.subheader("Segmentación Espacial de Clientes")
            st.write("Nota: Este modelo 3D muestra el comportamiento histórico total.")
            
            # Gráfico 3D
            fig_3d = px.scatter_3d(
                rfm_ml, x='Recencia', y='Frecuencia', z='Monetario',
                color='Perfil_Cliente', opacity=0.8,
                color_discrete_sequence=['#08529B', '#E53935', '#6EBA42', '#FF9900'],
                title='Distribución de Clusters RFM (3D)'
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=500)
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with tab_dash:
            if df_filtrado.empty:
                st.warning(f"⚠️ No hay datos registrados para '{tienda_seleccionada}' en el rango de fechas seleccionado.")
            else:
                # --- MÉTRICAS PRINCIPALES (KPIs) ---
                ingresos_totales = df_filtrado['Total_Venta'].sum()
                productos_vendidos = df_filtrado['Cantidad'].sum()
                ticket_promedio = ingresos_totales / len(df_filtrado) if len(df_filtrado) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("💰 Ingresos Totales", f"${ingresos_totales:,.0f}")
                col2.metric("📦 Unidades Vendidas", f"{productos_vendidos:,.0f}")
                col3.metric("🧾 Ticket Promedio", f"${ticket_promedio:,.0f}")
                
                st.markdown("---")
                
                # --- FILA 1: TENDENCIA Y COMPOSICIÓN ---
                c1_full = st.container()
                with c1_full:
                    ventas_tiempo = df_filtrado.groupby('Fecha')['Total_Venta'].sum().reset_index()
                    fig_tiempo = px.line(
                        ventas_tiempo, x='Fecha', y='Total_Venta', 
                        title="Evolución de Ingresos",
                        color_discrete_sequence=['#08529B']
                    )
                    st.plotly_chart(fig_tiempo, use_container_width=True)

                st.markdown("---")

                # --- FILA 2: CLUSTERING 2D Y BIGOTES (OUTLIERS) ---
                c3, c4 = st.columns(2)
                with c3:
                    # Gráfico de Dispersión 2D (Frecuencia vs Monetario)
                    rfm_filtrado = df_filtrado.groupby(['ID_Cliente', 'Perfil_Cliente']).agg({'Total_Venta': 'sum', 'ID_Venta': 'count'}).reset_index()
                    fig_2d = px.scatter(
                        rfm_filtrado, x='ID_Venta', y='Total_Venta', color='Perfil_Cliente',
                        title="Clustering 2D: Frecuencia vs Gasto",
                        labels={'ID_Venta': 'Frecuencia (Num. Compras)', 'Total_Venta': 'Gasto Monetario ($)'},
                        color_discrete_sequence=['#08529B', '#E53935', '#6EBA42', '#FF9900']
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                with c4:
                    # Mostramos los top 15 para que el gráfico de barras no se sature y sea legible
                    ventas_perfil = df_filtrado.groupby('Perfil_Cliente')['Total_Venta'].sum().reset_index()
                    fig_perfil = px.pie(
                        ventas_perfil, names='Perfil_Cliente', values='Total_Venta', hole=0.4,
                        title="Aportación de Ingresos por Segmento",
                        color_discrete_sequence=['#FF9900', '#E53935', '#08529B', '#6EBA42']
                    )
                    st.plotly_chart(fig_perfil, use_container_width=True)

                st.markdown("---")

                # --- FILA 3: CATEGORÍAS Y PRODUCTOS ESTRELLA ---
                c5, c6 = st.columns(2)
                with c5:
                    ventas_cat = df_filtrado.groupby('Categoria')['Total_Venta'].sum().reset_index().sort_values(by='Total_Venta', ascending=True)
                    fig_cat = px.bar(
                        ventas_cat, x='Total_Venta', y='Categoria', orientation='h',
                        title="Ventas por Categoría de Producto",
                        color_discrete_sequence=['#08529B']
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

                with c6:
                    top_productos = df_filtrado.groupby('Nombre_Producto')['Total_Venta'].sum().nlargest(5).reset_index()
                    fig_top = px.bar(
                        top_productos, x='Total_Venta', y='Nombre_Producto', orientation='h',
                        title="Top 5 Productos (Estrellas)",
                        color_discrete_sequence=['#6EBA42']
                    )
                    fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_top, use_container_width=True)

                st.markdown("---")

                # --- FILA 4: LOCALIDAD/ESTRATO
                c7, c8 = st.columns(2)
                with c7:
                    # Agrupamos por Tienda y Nivel Socioeconómico sumando el Total de la Venta
                    ventas_tienda_estrato = df_filtrado.groupby(['Nombre_Tienda', 'Nivel_Socioeconomico'])['Total_Venta'].sum().reset_index()
                    
                    # Creamos el gráfico de barras apiladas (eliminando barmode='group' se apilan por defecto)
                    fig_apilado = px.bar(
                        ventas_tienda_estrato, 
                        x='Nombre_Tienda', 
                        y='Total_Venta', 
                        color='Nivel_Socioeconomico',
                        title="Análisis del Valor de Ventas por Tienda y Nivel Socioeconómico (Clientes que más compran)",
                        labels={'Nombre_Tienda': 'Sucursal / Tienda', 'Total_Venta': 'Suma Total de Ventas ($)', 'Nivel_Socioeconomico': 'Estrato Cliente'},
                        color_discrete_sequence=['#08529B', '#6EBA42', "#8935C0", "#D6BA3C"], # Manteniendo tus colores principales
                        text_auto='.2s' # Muestra las etiquetas resumidas de dinero en cada segmento (ej. 1.2M, 450k)
                    )
                    st.plotly_chart(fig_apilado, use_container_width=True)

                with c8:
                    # NUEVO: Tienda de mayor a menor volumen (Reemplaza el de bigotes)
                    ventas_tienda = df_filtrado.groupby('Nombre_Tienda')['Total_Venta'].sum().reset_index().sort_values('Total_Venta', ascending=False)
                    fig_tiendas = px.bar(
                        ventas_tienda, x='Nombre_Tienda', y='Total_Venta',
                        title="Volumen de Ventas por Tienda",
                        color_discrete_sequence=['#6EBA42'],
                        text_auto='.2s'
                    )
                    st.plotly_chart(fig_tiendas, use_container_width=True)
                
                # --- FILA 4: TOP DE PRODUCTOS (ANCHO COMPLETO - AL FINAL) ---
                c_full = st.container()
                with c_full:
                    # Incrementamos a los mejores 20 productos y los ordenamos de mayor a menor cantidad vendida
                    cant_prod = df_filtrado.groupby('Nombre_Producto')['Cantidad'].sum().reset_index().sort_values('Cantidad', ascending=True).tail(20)
                    
                    fig_cant = px.bar(
                        cant_prod, 
                        x='Cantidad', 
                        y='Nombre_Producto', 
                        orientation='h', # Gráfico horizontal para facilitar la lectura de los nombres largos de productos
                        title="Top 20: Productos Líderes en Unidades Vendidas (Mayor a Menor)",
                        labels={'Nombre_Producto': 'Producto', 'Cantidad': 'Unidades Totales Vendidas'},
                        color_discrete_sequence=['#08529B'], # Azul principal corporativo
                        text_auto=True
                    )
                    
                    # Le damos una altura personalizada (height=600) para que las 20 barras respiren y no se solapen
                    fig_cant.update_layout(height=600, margin=dict(l=150, r=20, t=50, b=50))
                    st.plotly_chart(fig_cant, use_container_width=True)

# MÓDULO 4: MODELADO Y SIMULACIÓN
    elif pagina == "🎲 Modelado y Simulación":
        st.title("🎲 Modelado Predictivo y Simulación de Escenarios")
        st.write("Análisis estocástico, proyecciones de Machine Learning y simulación de eventos discretos para la toma de decisiones operativas.")
        
        # Preparación de datos base para los modelos
        df_ventas = datos_limpios['Ventas'].copy()
        df_tiempo = datos_limpios['Tiempo'].copy()
        df_master = df_ventas.merge(df_tiempo[['ID_Tiempo', 'Fecha']], on='ID_Tiempo', how='inner')
        ventas_diarias = df_master.groupby('Fecha')['Total_Venta'].sum().reset_index()
        ventas_diarias['Dia_Semana'] = ventas_diarias['Fecha'].dt.dayofweek
        ventas_diarias['Mes'] = ventas_diarias['Fecha'].dt.month
        
        # Pestañas del módulo
        tab_pred, tab_mc, tab_des = st.tabs([
            "📈 Predicción (Random Forest)", 
            "🎲 Montecarlo (Riesgo Financiero)", 
            "⏱️ Eventos Discretos (Operación en Cajas)"
        ])
        
        # ---------------------------------------------------------
        # PESTAÑA 1: MODELO PREDICTIVO (Árboles de Decisión)
        # ---------------------------------------------------------
        with tab_pred:
            st.subheader("Predicción de Ventas con Random Forest Regressor")
            st.write("Utilizamos el histórico de ventas para predecir el comportamiento futuro basado en estacionalidad.")
            
            # Preparar Machine Learning
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error
            
            X = ventas_diarias[['Dia_Semana', 'Mes']]
            y = ventas_diarias['Total_Venta']
            
            if len(ventas_diarias) > 10: # Asegurar que hay suficientes datos
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                modelo_rf.fit(X_train, y_train)
                predicciones = modelo_rf.predict(X_test)
                mae = mean_absolute_error(y_test, predicciones)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.info(f"**Error Medio Absoluto (MAE):**\n\n${mae:,.0f}")
                    st.write("Este valor indica la desviación promedio (en pesos) de nuestras predicciones frente a la realidad.")
                
                with col2:
                    df_resultados = pd.DataFrame({'Real': y_test.values, 'Predicción': predicciones})
                    df_resultados = df_resultados.reset_index(drop=True).head(30) # Mostrar 30 días para claridad
                    
                    fig_rf = px.line(
                        df_resultados, 
                        title="Comparativa: Ventas Reales vs Predicción del Modelo (Muestra de 30 días)",
                        labels={'value': 'Total Ventas ($)', 'index': 'Días de Prueba'},
                        color_discrete_sequence=['#08529B', '#6EBA42']
                    )
                    st.plotly_chart(fig_rf, use_container_width=True)
            else:
                st.warning("No hay suficientes datos históricos para entrenar el modelo predictivo.")

        # ---------------------------------------------------------
        # PESTAÑA 2: SIMULACIÓN MONTECARLO
        # ---------------------------------------------------------
        with tab_mc:
            st.subheader("Simulación Montecarlo: Proyección de Ingresos Mensuales")
            st.write("Generamos **1000 iteraciones** estocásticas para evaluar el riesgo financiero en 3 escenarios macroeconómicos.")
            
            media_historica = ventas_diarias['Total_Venta'].mean() * 30 # Aproximación mensual
            std_historica = ventas_diarias['Total_Venta'].std() * np.sqrt(30)
            
            # Definir escenarios
            escenarios = {
                "Base (Tendencia Actual)": {'media': media_historica, 'std': std_historica},
                "Optimista (Crecimiento +20%)": {'media': media_historica * 1.2, 'std': std_historica * 0.9},
                "Pesimista (Recesión -20%)": {'media': media_historica * 0.8, 'std': std_historica * 1.1}
            }
            
            resultados_mc = {}
            metricas_mc = []
            meta_minima = media_historica * 0.85 # Umbral de riesgo
            
            for nombre, params in escenarios.items():
                # Simulación de 1000 meses posibles usando distribución normal
                simulacion = np.random.normal(params['media'], params['std'], 1000)
                resultados_mc[nombre] = simulacion
                
                metricas_mc.append({
                    "Escenario": nombre,
                    "Ingreso Promedio Esperado": f"${np.mean(simulacion):,.0f}",
                    "Pico Máximo Proyectado": f"${np.max(simulacion):,.0f}",
                    "Percentil 90 (Mejor de los casos)": f"${np.percentile(simulacion, 90):,.0f}",
                    "Probabilidad de Riesgo (Bajo Meta)": f"{(np.sum(simulacion < meta_minima) / 1000) * 100:.1f}%"
                })
            
            df_metricas = pd.DataFrame(metricas_mc)
            st.dataframe(df_metricas, use_container_width=True)
            
            # Graficar histogramas superpuestos
            df_plot_mc = pd.DataFrame(resultados_mc)
            fig_mc = px.histogram(
                df_plot_mc, 
                barmode='overlay',
                title="Distribución de Probabilidad de Ingresos (1000 Simulaciones)",
                labels={'value': 'Ingresos Mensuales ($)', 'variable': 'Escenario'},
                color_discrete_sequence=['#08529B', '#6EBA42', '#E53935'],
                opacity=0.7
            )
            fig_mc.add_vline(x=meta_minima, line_dash="dash", line_color="red", annotation_text="Línea de Riesgo")
            st.plotly_chart(fig_mc, use_container_width=True)

        # ---------------------------------------------------------
        # PESTAÑA 3: SIMULACIÓN DE EVENTOS DISCRETOS (DES)
        # ---------------------------------------------------------
        with tab_mc: # Nota técnica: Se reutiliza el entorno estético para la subsección o tab real
            pass
            
        with tab_des:
            st.subheader("Simulación de Eventos Discretos: Congestión en Cajas")
            st.write("Modelamos el comportamiento de las filas utilizando un proceso estocástico basado en teoría de colas.")
            
            col_a, col_b, col_c = st.columns(3)
            tasa_llegada = col_a.slider("Clientes por hora (Demanda)", 10, 100, 50)
            tasa_servicio = col_b.slider("Capacidad de atención por hora", 10, 100, 55)
            minutos_simulacion = col_c.number_input("Minutos a simular", min_value=60, value=480, step=60) # 8 horas por defecto
            
            if st.button("🚀 Ejecutar Simulación de Operación"):
                with st.spinner("Simulando eventos discretos..."):
                    # Lógica estocástica de colas (Emulación de SimPy)
                    tiempo_actual = 0
                    fila = 0
                    max_fila = 0
                    tiempos_espera = []
                    registro_evolucion = []
                    
                    # Convertir tasas a minutos
                    lambda_arr = tasa_llegada / 60.0
                    mu_srv = tasa_servicio / 60.0
                    
                    while tiempo_actual < minutos_simulacion:
                        # Tiempo hasta la próxima llegada (Exponencial)
                        t_llegada = np.random.exponential(1 / lambda_arr)
                        
                        # Si hay fila, se atiende a alguien
                        if fila > 0:
                            t_servicio = np.random.exponential(1 / mu_srv)
                            if t_servicio < t_llegada:
                                tiempo_actual += t_servicio
                                fila -= 1
                                # Se guarda el tiempo restante
                                t_llegada -= t_servicio
                            else:
                                tiempo_actual += t_llegada
                                fila += 1
                        else:
                            tiempo_actual += t_llegada
                            fila += 1
                            
                        # Actualizar métricas
                        max_fila = max(max_fila, fila)
                        # Tiempo estimado de espera para el que acaba de llegar (fórmula empírica simplificada)
                        espera_estimada = fila * (1 / mu_srv) if fila > 0 else 0
                        tiempos_espera.append(espera_estimada)
                        
                        registro_evolucion.append({'Minuto': tiempo_actual, 'Clientes_en_Fila': fila})
                
                # Resultados de la simulación
                df_evolucion = pd.DataFrame(registro_evolucion)
                espera_promedio = np.mean(tiempos_espera) if tiempos_espera else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Pico de Congestión (Max. Fila)", f"{max_fila} clientes")
                c2.metric("Tiempo de Espera Promedio", f"{espera_promedio:.1f} min")
                
                # Evaluación de capacidad
                rho = tasa_llegada / tasa_servicio
                if rho >= 1:
                    c3.error(f"Sistema Colapsado (Saturación: {rho*100:.1f}%)")
                elif rho > 0.8:
                    c3.warning(f"Riesgo Alto (Saturación: {rho*100:.1f}%)")
                else:
                    c3.success(f"Operación Estable (Saturación: {rho*100:.1f}%)")
                
                # Gráfico de evolución temporal
                fig_des = px.area(
                    df_evolucion, 
                    x='Minuto', 
                    y='Clientes_en_Fila',
                    title="Evolución de la Fila durante la Jornada",
                    color_discrete_sequence=['#E53935']
                )
                st.plotly_chart(fig_des, use_container_width=True)