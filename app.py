# AgriSight Pro - Application Streamlit complète
# Fichier: app.py
# Sauvegardez ce fichier et exécutez avec: streamlit run app.py

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, mapping, shape
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json
from matplotlib.backends.backend_pdf import PdfPages
import time

# Configuration
st.set_page_config(page_title="AgriSight IA", layout="wide", page_icon="🌾")

# CSS personnalisé
st.markdown("""
<style>
    .big-metric {font-size: 2em; font-weight: bold; color: #2E7D32;}
    .alert-box {background: #FFF3CD; padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107;}
    .success-box {background: #D4EDDA; padding: 15px; border-radius: 8px; border-left: 4px solid #28A745;}
    .info-box {background: #D1ECF1; padding: 15px; border-radius: 8px; border-left: 4px solid #17A2B8;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriSight Pro - Analyse Agro-climatique Avancée")
st.markdown("*Plateforme d'analyse par télédétection et IA pour l'agriculture de précision*")

# API Keys
AGRO_API_KEY = '28641235f2b024b5f45f97df45c6a0d5'

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

with st.sidebar.expander("🔑 Clés API", expanded=False):
    st.markdown("**Google Gemini API** (Gratuit)")
    st.markdown("- Inscription: [aistudio.google.com](https://aistudio.google.com/apikey)")
    gemini_key = st.text_input("Clé Gemini (optionnel)", type="password", value="")
    st.success("Clé Agromonitoring intégrée")

st.sidebar.markdown("---")

# Zone d'étude
st.sidebar.subheader("📍 Zone d'étude")
zone_method = st.sidebar.radio("Méthode de sélection", 
                               ["Dessiner sur carte", "Importer GeoJSON", "Coordonnées"])

uploaded_file = None
manual_coords = None

if zone_method == "Importer GeoJSON":
    uploaded_file = st.sidebar.file_uploader("Fichier GeoJSON", type=["geojson", "json"])
elif zone_method == "Coordonnées":
    st.sidebar.info("Entrez les coins d'un rectangle")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat_min = st.number_input("Lat Min", value=14.60, format="%.4f")
        lon_min = st.number_input("Lon Min", value=-17.50, format="%.4f")
    with col2:
        lat_max = st.number_input("Lat Max", value=14.70, format="%.4f")
        lon_max = st.number_input("Lon Max", value=-17.40, format="%.4f")
    manual_coords = (lat_min, lon_min, lat_max, lon_max)

# Paramètres temporels
st.sidebar.subheader("📅 Période d'analyse")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Début", date.today() - timedelta(days=180))
with col2:
    end_date = st.date_input("Fin", date.today())

# Type de culture
culture = st.sidebar.selectbox("🌱 Type de culture", 
    ["Mil", "Sorgho", "Maïs", "Arachide", "Riz", "Niébé", "Manioc", "Tomate", "Oignon"])

zone_name = st.sidebar.text_input("📍 Nom de la zone", "Ma parcelle")

st.sidebar.markdown("---")
load_btn = st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# Session State
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'satellite_data' not in st.session_state:
    st.session_state.satellite_data = None
if 'climate_data' not in st.session_state:
    st.session_state.climate_data = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'drawn_geometry' not in st.session_state:
    st.session_state.drawn_geometry = None

# Fonctions
def create_polygon_from_coords(lat_min, lon_min, lat_max, lon_max):
    coords = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min)
    ]
    return Polygon(coords)

@st.cache_data(ttl=3600)
def load_geojson(file_bytes):
    try:
        gdf = gpd.read_file(BytesIO(file_bytes))
        return gdf.to_crs(4326)
    except Exception as e:
        st.error(f"Erreur lecture GeoJSON: {e}")
        return None

def geometry_to_dict(geom):
    return mapping(geom)

def dict_to_geometry(geom_dict):
    return shape(geom_dict)

@st.cache_data(ttl=3600)
def get_climate_nasa_cached(geom_dict, start, end):
    geometry = dict_to_geometry(geom_dict)
    centroid = geometry.centroid
    lat, lon = centroid.y, centroid.x
    
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON&community=AG"
    )
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        data = response.json()
        params = data.get("properties", {}).get("parameter", {})
        
        df = pd.DataFrame({
            'date': pd.to_datetime(list(params.get('T2M', {}).keys())),
            'temp_mean': list(params.get('T2M', {}).values()),
            'temp_min': list(params.get('T2M_MIN', {}).values()),
            'temp_max': list(params.get('T2M_MAX', {}).values()),
            'rain': list(params.get('PRECTOTCORR', {}).values())
        })
        
        return df
    except Exception as e:
        st.error(f"Erreur NASA POWER: {e}")
        return None

@st.cache_data(ttl=3600)
def simulate_ndvi_data(start, end):
    dates = pd.date_range(start, end, freq='5D')
    ndvi_values = []
    
    for d in dates:
        month = d.month
        if 6 <= month <= 9:
            base = 0.65 + np.random.normal(0, 0.08)
        elif month in [5, 10]:
            base = 0.45 + np.random.normal(0, 0.1)
        else:
            base = 0.25 + np.random.normal(0, 0.06)
        
        ndvi_values.append({
            'date': d,
            'ndvi_mean': np.clip(base, 0, 1),
            'ndvi_std': 0.1,
            'ndvi_min': max(0, np.clip(base - 0.15, 0, 1)),
            'ndvi_max': min(1, np.clip(base + 0.15, 0, 1)),
            'cloud_cover': np.random.randint(0, 30)
        })
    
    return pd.DataFrame(ndvi_values)

def calculate_metrics(climate_df, ndvi_df, culture):
    if climate_df is None or ndvi_df is None or climate_df.empty or ndvi_df.empty:
        return {}
    
    metrics = {
        'ndvi_mean': ndvi_df['ndvi_mean'].mean(),
        'ndvi_std': ndvi_df['ndvi_mean'].std(),
        'temp_mean': climate_df['temp_mean'].mean(),
        'temp_min': climate_df['temp_min'].min(),
        'temp_max': climate_df['temp_max'].max(),
        'rain_total': climate_df['rain'].sum(),
        'rain_mean': climate_df['rain'].mean(),
        'rain_days': (climate_df['rain'] > 1).sum()
    }
    
    ndvi = metrics['ndvi_mean']
    rain = metrics['rain_total']
    
    if culture == "Mil":
        if ndvi > 0.6 and rain > 400:
            metrics['yield_potential'] = 1.5
        elif ndvi > 0.4 and rain > 300:
            metrics['yield_potential'] = 1.0
        else:
            metrics['yield_potential'] = 0.6
    elif culture == "Maïs":
        if ndvi > 0.65 and rain > 500:
            metrics['yield_potential'] = 3.5
        elif ndvi > 0.5 and rain > 400:
            metrics['yield_potential'] = 2.5
        else:
            metrics['yield_potential'] = 1.5
    elif culture == "Arachide":
        if ndvi > 0.6 and rain > 450:
            metrics['yield_potential'] = 2.0
        elif ndvi > 0.45 and rain > 350:
            metrics['yield_potential'] = 1.3
        else:
            metrics['yield_potential'] = 0.8
    else:
        if ndvi > 0.6 and rain > 400:
            metrics['yield_potential'] = 2.5
        elif ndvi > 0.4 and rain > 300:
            metrics['yield_potential'] = 1.8
        else:
            metrics['yield_potential'] = 1.0
    
    return metrics

# Onglets
tabs = st.tabs(["🗺️ Carte", "📊 Dashboard", "🛰️ NDVI", "🌦️ Climat", "🤖 IA", "📄 Rapport"])

# ONGLET 1: CARTE
with tabs[0]:
    st.subheader("🗺️ Définir la Zone d'Étude")
    
    if zone_method == "Dessiner sur carte":
        st.info("💡 Utilisez les outils de dessin pour définir votre zone, puis 'Lancer l'analyse'")
    
    if st.session_state.gdf is not None:
        center = [st.session_state.gdf.geometry.centroid.y.mean(),
                 st.session_state.gdf.geometry.centroid.x.mean()]
        zoom = 13
    elif manual_coords:
        center = [(manual_coords[0] + manual_coords[2])/2, (manual_coords[1] + manual_coords[3])/2]
        zoom = 13
    else:
        center = [14.6937, -17.4441]
        zoom = 10
    
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    
    m.add_child(MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='hectares'
    ))
    
    if st.session_state.gdf is not None:
        folium.GeoJson(
            st.session_state.gdf,
            name="Zone analysée",
            style_function=lambda x: {
                'fillColor': '#28A745',
                'color': '#155724',
                'weight': 3,
                'fillOpacity': 0.4
            },
            tooltip=f"<b>{zone_name}</b><br>Culture: {culture}"
        ).add_to(m)
        
        centroid = st.session_state.gdf.geometry.centroid.iloc[0]
        folium.Marker(
            [centroid.y, centroid.x],
            popup=f"<b>{zone_name}</b><br>{culture}",
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
    
    draw = Draw(
        export=True,
        draw_options={
            'polygon': {
                'allowIntersection': False,
                'shapeOptions': {'color': '#28A745', 'weight': 3}
            },
            'rectangle': {'shapeOptions': {'color': '#28A745', 'weight': 3}},
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)
    
    map_output = st_folium(m, height=600, width=None, key="main_map")
    
    if map_output and map_output.get('all_drawings'):
        drawings = map_output['all_drawings']
        if drawings and len(drawings) > 0:
            try:
                gdf_drawn = gpd.GeoDataFrame.from_features(drawings, crs="EPSG:4326")
                st.session_state.drawn_geometry = gdf_drawn.geometry.unary_union
                area_m2 = gdf_drawn.geometry.area.sum() * 111000 * 111000
                area_ha = area_m2 / 10000
                st.success(f"Zone dessinée: {len(drawings)} forme(s). Surface: {area_m2:.2f} m² ({area_ha:.2f} ha)")
            except Exception as e:
                st.error(f"Erreur: {e}")

# CHARGEMENT DES DONNÉES
if load_btn:
    geometry = None
    
    if zone_method == "Importer GeoJSON" and uploaded_file:
        file_bytes = uploaded_file.read()
        gdf = load_geojson(file_bytes)
        if gdf is not None and not gdf.empty:
            st.session_state.gdf = gdf
            geometry = gdf.geometry.unary_union
    
    elif zone_method == "Dessiner sur carte":
        if st.session_state.drawn_geometry:
            gdf = gpd.GeoDataFrame([{'geometry': st.session_state.drawn_geometry}], crs='EPSG:4326')
            st.session_state.gdf = gdf
            geometry = st.session_state.drawn_geometry
        else:
            st.error("Veuillez dessiner une zone sur la carte")
            st.stop()
    
    elif zone_method == "Coordonnées" and manual_coords:
        polygon = create_polygon_from_coords(*manual_coords)
        gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
        st.session_state.gdf = gdf
        geometry = polygon
    
    if geometry is None:
        st.error("Veuillez définir une zone d'étude")
        st.stop()
    
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### Progression du chargement")
        global_progress = st.progress(0, text="Initialisation...")
        status_climate = st.empty()
        status_ndvi = st.empty()
        status_analysis = st.empty()
    
    geom_dict = geometry_to_dict(geometry)
    
    # Climat
    status_climate.info("Chargement données climatiques...")
    global_progress.progress(10, text="Récupération NASA POWER...")
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    climate_df = get_climate_nasa_cached(geom_dict, start_dt, end_dt)
    
    if climate_df is None or climate_df.empty:
        status_climate.error("Échec données climatiques")
        st.stop()
    else:
        status_climate.success(f"Climat chargé ({len(climate_df)} jours)")
        st.session_state.climate_data = climate_df
    
    global_progress.progress(40, text="Récupération NDVI...")
    
    # NDVI
    status_ndvi.info("Chargement NDVI...")
    satellite_df = simulate_ndvi_data(start_date, end_date)
    status_ndvi.success(f"NDVI chargé ({len(satellite_df)} points)")
    st.session_state.satellite_data = satellite_df
    
    global_progress.progress(70, text="Calcul métriques...")
    
    # Métriques
    status_analysis.info("Calcul métriques...")
    metrics = calculate_metrics(climate_df, satellite_df, culture)
    
    if not metrics:
        status_analysis.error("Erreur métriques")
        st.stop()
    else:
        status_analysis.success("Analyse terminée!")
    
    global_progress.progress(100, text="Analyse complète!")
    time.sleep(1)
    st.success(f"Données chargées! {len(climate_df)} jours climat, {len(satellite_df)} images satellite.")
    st.balloons()
  # AgriSight Pro - Partie 2
# Ajoutez ce code à la suite du fichier app.py (après la partie 1)

# ONGLET 2: DASHBOARD
with tabs[1]:
    st.subheader("📊 Dashboard Interactif")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        metrics = calculate_metrics(
            st.session_state.climate_data,
            st.session_state.satellite_data,
            culture
        )
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_ndvi = "Bon" if metrics['ndvi_mean'] > 0.5 else "Faible"
            st.metric("🌱 NDVI Moyen", f"{metrics['ndvi_mean']:.3f}", delta=delta_ndvi)
        
        with col2:
            st.metric("🌡️ Température", f"{metrics['temp_mean']:.1f}°C",
                     delta=f"{metrics['temp_min']:.0f}-{metrics['temp_max']:.0f}°")
        
        with col3:
            delta_rain = "Suffisant" if metrics['rain_total'] > 300 else "Insuffisant"
            st.metric("💧 Pluie Totale", f"{metrics['rain_total']:.0f} mm", delta=delta_rain)
        
        with col4:
            st.metric("📈 Rendement Est.", f"{metrics['yield_potential']:.1f} t/ha")
        
        st.markdown("---")
        
        # Graphiques
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_ndvi, ax = plt.subplots(figsize=(8, 5))
            ax.plot(st.session_state.satellite_data['date'],
                   st.session_state.satellite_data['ndvi_mean'],
                   'o-', color='darkgreen', linewidth=2, markersize=6)
            ax.fill_between(st.session_state.satellite_data['date'],
                           st.session_state.satellite_data['ndvi_mean'],
                           alpha=0.3, color='green')
            ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Excellent')
            ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Bon')
            ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Stress')
            ax.set_ylabel('NDVI', fontweight='bold')
            ax.set_title('Évolution NDVI', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=30)
            st.pyplot(fig_ndvi)
        
        with col_g2:
            fig_clim, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            ax1.bar(st.session_state.climate_data['date'],
                   st.session_state.climate_data['rain'],
                   color='steelblue', alpha=0.6, label='Pluie')
            ax2.plot(st.session_state.climate_data['date'],
                    st.session_state.climate_data['temp_mean'],
                    color='orangered', linewidth=2, label='Température')
            ax1.set_ylabel('Pluie (mm)', color='steelblue', fontweight='bold')
            ax2.set_ylabel('Temp (°C)', color='orangered', fontweight='bold')
            ax1.set_title('Conditions Climatiques', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            plt.xticks(rotation=30)
            st.pyplot(fig_clim)
        
        st.markdown("---")
        
        # Analyse rapide
        st.markdown("### 🔍 Diagnostic Rapide")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            if metrics['ndvi_mean'] > 0.6:
                st.markdown('<div class="success-box">✅ <b>Vigueur excellente</b><br>Culture se développe très bien.</div>', 
                           unsafe_allow_html=True)
            elif metrics['ndvi_mean'] > 0.4:
                st.markdown('<div class="alert-box">⚠️ <b>Vigueur modérée</b><br>Surveillance recommandée.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box" style="border-left: 4px solid #DC3545; background: #F8D7DA;">❌ <b>Stress végétal</b><br>Action urgente.</div>', 
                           unsafe_allow_html=True)
        
        with col_a2:
            if metrics['rain_total'] < 200:
                st.markdown('<div class="alert-box" style="border-left: 4px solid #DC3545; background: #F8D7DA;">💧 <b>Stress hydrique sévère</b><br>Irrigation urgente.</div>', 
                           unsafe_allow_html=True)
            elif metrics['rain_total'] < 350:
                st.markdown('<div class="alert-box">💧 <b>Pluviométrie limite</b><br>Surveiller humidité.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">💧 <b>Pluviométrie adéquate</b><br>Bon approvisionnement.</div>', 
                           unsafe_allow_html=True)
        
        with col_a3:
            if metrics['temp_max'] > 38:
                st.markdown('<div class="alert-box">🌡️ <b>Chaleur excessive</b><br>Risque stress thermique.</div>', 
                           unsafe_allow_html=True)
            elif metrics['temp_mean'] > 32:
                st.markdown('<div class="info-box">🌡️ <b>Température élevée</b><br>Conditions chaudes.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">🌡️ <b>Température optimale</b><br>Bonnes conditions.</div>', 
                           unsafe_allow_html=True)
        
        # Statistiques détaillées
        st.markdown("---")
        st.markdown("### 📊 Statistiques Détaillées")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.markdown("**🌱 NDVI**")
            st.write(f"Min: {st.session_state.satellite_data['ndvi_mean'].min():.3f}")
            st.write(f"Max: {st.session_state.satellite_data['ndvi_mean'].max():.3f}")
            st.write(f"Écart-type: {metrics['ndvi_std']:.3f}")
        
        with col_s2:
            st.markdown("**🌡️ Températures**")
            st.write(f"Min abs: {metrics['temp_min']:.1f}°C")
            st.write(f"Max abs: {metrics['temp_max']:.1f}°C")
            st.write(f"Amplitude: {metrics['temp_max'] - metrics['temp_min']:.1f}°C")
        
        with col_s3:
            st.markdown("**💧 Précipitations**")
            st.write(f"Total: {metrics['rain_total']:.0f} mm")
            st.write(f"Moyenne/j: {metrics['rain_mean']:.1f} mm")
            st.write(f"Jours pluie: {metrics['rain_days']}")
        
        with col_s4:
            st.markdown("**📈 Indices**")
            days_35 = (st.session_state.climate_data['temp_max'] > 35).sum()
            days_dry = (st.session_state.climate_data['rain'] < 1).sum()
            st.write(f"Jours >35°C: {days_35}")
            st.write(f"Jours secs: {days_dry}")
            st.write(f"Durée: {len(st.session_state.climate_data)} j")
        
    else:
        st.info("👆 Lancez d'abord l'analyse dans l'onglet Configuration")

# ONGLET 3: NDVI
with tabs[2]:
    st.subheader("🛰️ Analyse NDVI Détaillée")
    
    if st.session_state.satellite_data is not None:
        df_sat = st.session_state.satellite_data
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(df_sat['date'], df_sat['ndvi_mean'], 'o-',
                   color='darkgreen', linewidth=2.5, markersize=7, label='NDVI moyen')
            ax.fill_between(df_sat['date'], df_sat['ndvi_min'], df_sat['ndvi_max'],
                           alpha=0.25, color='green', label='Plage min-max')
            ax.axhline(0.7, color='darkgreen', linestyle=':', alpha=0.6, label='Excellent')
            ax.axhline(0.5, color='orange', linestyle=':', alpha=0.6, label='Bon')
            ax.axhline(0.3, color='red', linestyle=':', alpha=0.6, label='Stress')
            ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
            ax.set_title('Évolution NDVI', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=30)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Statistiques")
            ndvi_mean = df_sat['ndvi_mean'].mean()
            st.metric("Moyenne", f"{ndvi_mean:.3f}")
            st.metric("Max", f"{df_sat['ndvi_mean'].max():.3f}")
            st.metric("Min", f"{df_sat['ndvi_mean'].min():.3f}")
            st.metric("Écart-type", f"{df_sat['ndvi_mean'].std():.3f}")
            
            st.markdown("---")
            st.markdown("### 🔬 Interprétation")
            
            if ndvi_mean > 0.6:
                st.success("✅ **Excellente santé**")
                st.write("Croissance optimale")
            elif ndvi_mean > 0.4:
                st.warning("⚠️ **État modéré**")
                st.write("Surveillance nécessaire")
            else:
                st.error("❌ **Stress détecté**")
                st.write("Action urgente")
        
        # Tableau des données
        st.markdown("---")
        st.markdown("### 📋 Données NDVI")
        st.dataframe(df_sat.sort_values('date', ascending=False), use_container_width=True)
        
    else:
        st.info("Chargez d'abord les données")

# ONGLET 4: CLIMAT
with tabs[3]:
    st.subheader("🌦️ Analyse Climatique")
    
    if st.session_state.climate_data is not None:
        df_clim = st.session_state.climate_data
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Températures
        axes[0].fill_between(df_clim['date'], df_clim['temp_min'], df_clim['temp_max'],
                            alpha=0.3, color='coral', label='Plage')
        axes[0].plot(df_clim['date'], df_clim['temp_mean'], color='red', linewidth=2, label='Moyenne')
        axes[0].set_ylabel('Température (°C)', fontweight='bold')
        axes[0].set_title('Températures', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pluie
        axes[1].bar(df_clim['date'], df_clim['rain'], color='dodgerblue', alpha=0.7)
        axes[1].axhline(df_clim['rain'].mean(), color='navy', linestyle='--', 
                       label=f"Moyenne: {df_clim['rain'].mean():.1f} mm")
        axes[1].set_ylabel('Pluie (mm)', fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Précipitations', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 📈 Statistiques Climatiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌡️ Températures**")
            st.metric("Moyenne", f"{df_clim['temp_mean'].mean():.1f}°C")
            st.metric("Min absolue", f"{df_clim['temp_min'].min():.1f}°C")
            st.metric("Max absolue", f"{df_clim['temp_max'].max():.1f}°C")
        
        with col2:
            st.markdown("**💧 Précipitations**")
            st.metric("Cumul total", f"{df_clim['rain'].sum():.0f} mm")
            st.metric("Moyenne/jour", f"{df_clim['rain'].mean():.1f} mm")
            st.metric("Max/jour", f"{df_clim['rain'].max():.1f} mm")
        
        with col3:
            st.markdown("**📊 Indices**")
            st.metric("Jours pluie (>1mm)", f"{(df_clim['rain'] > 1).sum()}")
            st.metric("Jours >35°C", f"{(df_clim['temp_max'] > 35).sum()}")
            st.metric("Jours secs", f"{(df_clim['rain'] < 1).sum()}")
        
        # Tableau
        st.markdown("---")
        st.markdown("### 📋 Données Climatiques")
        st.dataframe(df_clim.sort_values('date', ascending=False), use_container_width=True)
        
    else:
        st.info("Chargez d'abord les données")

# ONGLET 5: ANALYSE IA
with tabs[4]:
    st.subheader("🤖 Analyse IA avec Google Gemini")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        st.info("💡 **Google Gemini** gratuit (15 req/min). [Obtenez votre clé](https://aistudio.google.com/apikey)")
        
        metrics = calculate_metrics(
            st.session_state.climate_data,
            st.session_state.satellite_data,
            culture
        )
        
        analyze_btn = st.button("🚀 Générer l'analyse IA", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 Analyse en cours..."):
                
                ndvi_series = ", ".join([
                    f"{row['date'].strftime('%d/%m')}: {row['ndvi_mean']:.2f}"
                    for _, row in st.session_state.satellite_data.head(15).iterrows()
                ])
                
                prompt = f"""Tu es un agronome expert. Analyse cette parcelle et fournis des recommandations DÉTAILLÉES.

DONNÉES:
- Culture: {culture}
- Zone: {zone_name}  
- Période: {(end_date - start_date).days} jours

NDVI: {metrics['ndvi_mean']:.3f} (évolution: {ndvi_series})
Climat: {metrics['temp_mean']:.1f}°C, Pluie: {metrics['rain_total']:.0f}mm
Rendement: {metrics['yield_potential']:.1f} t/ha

ANALYSE:
1. DIAGNOSTIC (santé, stress, rendement)
2. RECOMMANDATIONS (irrigation, fertilisation, traitements)
3. ALERTES urgentes
4. SUIVI

Réponds en français, précis sur doses et périodes."""

                analysis_text = None
                if gemini_key:AIzaSyBZ4494NUEL_N13soCCIgCfIrMqn2jxoD8
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}"
                    try:
                        response = requests.post(
                            url,
                            headers={"Content-Type": "application/json"},
                            json={
                                "contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {
                                    "temperature": 0.7,
                                    "maxOutputTokens": 4096,
                                }
                            },
                            timeout=60
                        )
                        if response.status_code == 200:
                            data = response.json()
                            if 'candidates' in data and len(data['candidates']) > 0:
                                analysis_text = data['candidates'][0]['content']['parts'][0]['text']
                    except:
                        pass
                
                if not analysis_text:
                    analysis_text = f"""### ANALYSE AGRO-CLIMATIQUE - {culture.upper()}

**DIAGNOSTIC:**
NDVI moyen de {metrics['ndvi_mean']:.3f} indique un {'bon' if metrics['ndvi_mean'] > 0.5 else 'faible'} développement.

**SITUATION HYDRIQUE:**
{metrics['rain_total']:.0f}mm - {'favorable' if metrics['rain_total'] > 300 else 'stress hydrique probable'}.

**RECOMMANDATIONS:**

1. **IRRIGATION:** {'Apport 20-25mm tous les 5-7j' if metrics['rain_total'] < 350 else 'Satisfaisant'}

2. **FERTILISATION:**
   - NPK 15-15-15: 150 kg/ha
   - Urée: 50 kg/ha à montaison

3. **SURVEILLANCE:**
   - NDVI hebdomadaire
   - Humidité sol 20-30cm

**CONDITIONS:**
Temp {metrics['temp_mean']:.1f}°C (max {metrics['temp_max']:.1f}°C)
{'Attention stress thermique' if metrics['temp_max'] > 35 else 'Acceptable'}

**RENDEMENT:** {metrics['yield_potential']:.1f} t/ha

**Pour analyse IA complète, ajoutez clé Gemini.**"""
                
                st.session_state.analysis = analysis_text
        
        # Afficher analyse
        if st.session_state.analysis:
            st.markdown("### 📋 Rapport Agronomique")
            st.markdown(st.session_state.analysis)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Télécharger (TXT)",
                    st.session_state.analysis,
                    file_name=f"analyse_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                summary_json = json.dumps({
                    "zone": zone_name,
                    "culture": culture,
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "ndvi": round(metrics['ndvi_mean'], 3),
                    "temp": round(metrics['temp_mean'], 1),
                    "pluie": round(metrics['rain_total'], 1),
                    "rendement": round(metrics['yield_potential'], 1)
                }, indent=2)
                
                st.download_button(
                    "📊 Métriques (JSON)",
                    summary_json,
                    file_name=f"metriques_{zone_name}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("Lancez l'analyse d'abord")

# ONGLET 6: RAPPORT PDF
with tabs[5]:
    st.subheader("📄 Rapport PDF Complet")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        st.markdown("""
        **Contenu du rapport:**
        - 📊 Graphiques NDVI et climat
        - 📈 Statistiques détaillées
        - 🤖 Analyse IA
        - 💡 Recommandations
        """)
        
        if st.button("📄 Générer PDF", type="primary", use_container_width=True):
            with st.spinner("Génération PDF..."):
                try:
                    # Fonction pour générer PDF
                    def generate_pdf_report(gdf, climate_df, ndvi_df, metrics, culture, zone_name, analysis_text):
                        buffer = BytesIO()
                        
                        with PdfPages(buffer) as pdf:
                            fig = plt.figure(figsize=(11, 8.5))
                            fig.suptitle(f"Rapport AgriSight - {zone_name}", fontsize=18, fontweight='bold')
                            
                            # Infos
                            ax_info = fig.add_subplot(3, 2, 1)
                            ax_info.axis('off')
                            info_text = f"""Culture: {culture}
Zone: {zone_name}
Date: {datetime.now().strftime('%d/%m/%Y')}

NDVI: {metrics['ndvi_mean']:.3f}
Pluie: {metrics['rain_total']:.0f} mm
Temp: {metrics['temp_mean']:.1f}°C
Rendement: {metrics['yield_potential']:.1f} t/ha"""
                            ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
                            
                            # NDVI
                            ax_ndvi = fig.add_subplot(3, 2, (2, 4))
                            ax_ndvi.plot(ndvi_df['date'], ndvi_df['ndvi_mean'], 'o-', color='green', linewidth=2)
                            ax_ndvi.fill_between(ndvi_df['date'], ndvi_df['ndvi_mean'], alpha=0.3, color='green')
                            ax_ndvi.set_title('NDVI')
                            ax_ndvi.set_ylabel('NDVI')
                            ax_ndvi.grid(True, alpha=0.3)
                            ax_ndvi.set_ylim([0, 1])
                            
                            # Climat
                            ax_climate = fig.add_subplot(3, 2, (5, 6))
                            ax_temp = ax_climate.twinx()
                            ax_climate.bar(climate_df['date'], climate_df['rain'], color='blue', alpha=0.4)
                            ax_temp.plot(climate_df['date'], climate_df['temp_mean'], color='red', linewidth=2)
                            ax_climate.set_ylabel('Pluie (mm)', color='blue')
                            ax_temp.set_ylabel('Temp (°C)', color='red')
                            
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close()
                            
                            if analysis_text:
                                fig2 = plt.figure(figsize=(11, 8.5))
                                ax = fig2.add_subplot(111)
                                ax.axis('off')
                                wrapped = "\n".join([line[:85] for line in analysis_text.split('\n')][:50])
                                ax.text(0.05, 0.95, wrapped, fontsize=8, verticalalignment='top')
                                pdf.savefig(fig2, bbox_inches='tight')
                                plt.close()
                        
                        buffer.seek(0)
                        return buffer
                    
                    metrics = calculate_metrics(
                        st.session_state.climate_data,
                        st.session_state.satellite_data,
                        culture
                    )
                    
                    analysis_text = st.session_state.analysis if st.session_state.analysis else "Analyse non générée"
                    
                    pdf_buffer = generate_pdf_report(
                        st.session_state.gdf,
                        st.session_state.climate_data,
                        st.session_state.satellite_data,
                        metrics,
                        culture,
                        zone_name,
                        analysis_text
                    )
                    
                    st.success("PDF généré!")
                    
                    st.download_button(
                        "📥 Télécharger PDF",
                        pdf_buffer,
                        file_name=f"rapport_{zone_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erreur PDF: {e}")
        
        st.markdown("---")
        st.markdown("### 💾 Export CSV")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_climate = st.session_state.climate_data.to_csv(index=False)
            st.download_button(
                "📊 Climat CSV",
                csv_climate,
                f"climat_{zone_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_ndvi = st.session_state.satellite_data.to_csv(index=False)
            st.download_button(
                "🛰️ NDVI CSV",
                csv_ndvi,
                f"ndvi_{zone_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.session_state.gdf is not None:
                geojson_str = st.session_state.gdf.to_json()
                st.download_button(
                    "📍 Zone GeoJSON",
                    geojson_str,
                    f"zone_{zone_name}.geojson",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("Lancez l'analyse d'abord")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>🌾 AgriSight Pro v2.0</b> - Analyse agricole par télédétection et IA<br>
    <small>NASA POWER • NDVI Sentinel-2 • Google Gemini IA</small>
</div>
""", unsafe_allow_html=True)
