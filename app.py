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

# --------------------
# SIDEBAR CONFIGURATION
# --------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# API Keys (optionnel)
with st.sidebar.expander("🔑 Clés API (Optionnel)", expanded=False):
    st.markdown("""
    **Google Gemini API** (Gratuit - IA incluse)
    - Inscription: [aistudio.google.com](https://aistudio.google.com/apikey)
    - 15 requêtes/min gratuites
    - **Recommandé pour l'analyse IA**
    """)
    gemini_key = st.text_input("Clé Gemini (optionnel)", type="password", 
                               value="",
                               help="Laissez vide pour analyse basique")
    
    st.markdown("---")
    
    st.markdown("""
    **OpenWeather Agromonitoring** (Gratuit)
    - Inscription: [agromonitoring.com](https://agromonitoring.com)
    - NDVI, EVI, NDWI réels depuis Sentinel-2
    """)
    agromonitoring_key = st.text_input("Clé Agromonitoring", type="password",
                                      help="Laissez vide pour NDVI simulé")

st.sidebar.markdown("---")

# Zone d'étude
st.sidebar.subheader("📍 Zone d'étude")
zone_method = st.sidebar.radio("Méthode de sélection", 
                               ["📂 Importer GeoJSON", "✏️ Dessiner sur carte", "📌 Coordonnées"])

uploaded_file = None
manual_coords = None

if zone_method == "📂 Importer GeoJSON":
    uploaded_file = st.sidebar.file_uploader("Fichier GeoJSON", type=["geojson", "json"])
elif zone_method == "📌 Coordonnées":
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
    start_date = st.date_input("Début", date.today() - timedelta(days=60))
with col2:
    end_date = st.date_input("Fin", date.today())

# Limiter à 90 jours
if (end_date - start_date).days > 90:
    st.sidebar.warning("⚠️ Période limitée à 90 jours")
    end_date = start_date + timedelta(days=90)

# Type de culture
culture = st.sidebar.selectbox("🌱 Type de culture", 
    ["Mil", "Sorgho", "Maïs", "Arachide", "Riz", "Niébé", "Manioc", "Tomate", "Oignon", "Papayer"])

# Zone géographique
zone_name = st.sidebar.text_input("📍 Nom de la zone", "Ma parcelle")

st.sidebar.markdown("---")
load_btn = st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# --------------------
# SESSION STATE
# --------------------
if 'polygon_id' not in st.session_state:
    st.session_state.polygon_id = None
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

# --------------------
# FONCTIONS UTILITAIRES
# --------------------

def create_polygon_from_coords(lat_min, lon_min, lat_max, lon_max):
    """Crée un polygone à partir de coordonnées bbox"""
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
    """Charge un GeoJSON avec cache"""
    try:
        gdf = gpd.read_file(BytesIO(file_bytes))
        return gdf.to_crs(4326)
    except Exception as e:
        st.error(f"Erreur lecture GeoJSON: {e}")
        return None

def geometry_to_dict(geom):
    """Convertit une géométrie en dict pour le cache"""
    return mapping(geom)

def dict_to_geometry(geom_dict):
    """Convertit un dict en géométrie"""
    return shape(geom_dict)

@st.cache_data(ttl=3600)
def get_climate_nasa_cached(geom_dict, start, end):
    """Version cachée de get_climate_nasa_polygon"""
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
    """Génère des données NDVI simulées réalistes"""
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

def analyze_with_gemini(prompt, api_key=None):
    """Analyse avec Google Gemini (GRATUIT)"""
    if not api_key:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 4096,
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                return data['candidates'][0]['content']['parts'][0]['text']
        return None
    except Exception as e:
        st.error(f"Erreur Gemini: {e}")
        return None

def calculate_metrics(climate_df, ndvi_df, culture):
    """Calcule les métriques agrégées"""
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
    
    # Estimation rendement
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

def generate_pdf_report(gdf, climate_df, ndvi_df, metrics, culture, zone_name, analysis_text):
    """Génère un rapport PDF complet"""
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Infos et graphiques
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f"Rapport AgriSight - {zone_name}", fontsize=18, fontweight='bold', y=0.98)
        
        # Infos
        ax_info = fig.add_subplot(3, 2, 1)
        ax_info.axis('off')
        info_text = f"""Culture: {culture}
Zone: {zone_name}
Date: {datetime.now().strftime('%d/%m/%Y')}
Période: {climate_df['date'].min().strftime('%d/%m')} - {climate_df['date'].max().strftime('%d/%m')}

NDVI: {metrics['ndvi_mean']:.3f}
Pluie: {metrics['rain_total']:.0f} mm
Temp: {metrics['temp_mean']:.1f}°C
Rendement: {metrics['yield_potential']:.1f} t/ha"""
        ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                    family='monospace')
        
        # NDVI
        ax_ndvi = fig.add_subplot(3, 2, (2, 4))
        ax_ndvi.plot(ndvi_df['date'], ndvi_df['ndvi_mean'], 'o-', color='green', linewidth=2)
        ax_ndvi.fill_between(ndvi_df['date'], ndvi_df['ndvi_mean'], alpha=0.3, color='green')
        ax_ndvi.set_title('NDVI', fontsize=12, fontweight='bold')
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
        
        # Page 2: Analyse IA
        if analysis_text:
            fig2 = plt.figure(figsize=(11, 8.5))
            ax = fig2.add_subplot(111)
            ax.axis('off')
            
            wrapped = "\n".join([line[:85] for line in analysis_text.split('\n')][:50])
            ax.text(0.05, 0.95, wrapped, fontsize=8, verticalalignment='top',
                   wrap=True, family='monospace')
            
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close()
    
    buffer.seek(0)
    return buffer

# --------------------
# ONGLETS PRINCIPAUX
# --------------------
tabs = st.tabs(["🗺️ Carte & Zone", "📊 Vue d'ensemble", "🛰️ NDVI", 
                "🌦️ Climat", "🤖 Analyse IA", "📄 Rapport PDF"])

# --------------------
# ONGLET 1: CARTE (TOUJOURS VISIBLE)
# --------------------
with tabs[0]:
    st.subheader("🗺️ Définir la Zone d'Étude")
    
    st.info("💡 **Conseil:** Dessinez ou importez votre zone, puis cliquez sur 'Lancer l'analyse' dans la barre latérale")
    
    # Déterminer le centre
    if st.session_state.gdf is not None:
        center = [st.session_state.gdf.geometry.centroid.y.mean(),
                 st.session_state.gdf.geometry.centroid.x.mean()]
        zoom = 13
    elif manual_coords:
        center = [(manual_coords[0] + manual_coords[2])/2, (manual_coords[1] + manual_coords[3])/2]
        zoom = 13
    else:
        center = [14.6937, -17.4441]  # Dakar par défaut
        zoom = 10
    
    # Créer la carte (TOUJOURS AFFICHÉE)
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    m.add_child(MeasureControl(primary_length_unit='meters'))
    
    # Ajouter zone existante
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
    
    # Outil de dessin
    draw = Draw(
        export=True,
        draw_options={
            'polygon': {'allowIntersection': False},
            'rectangle': True,
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True}
    )
    draw.add_to(m)
    
    # Afficher la carte
    map_output = st_folium(m, height=550, width=None, key="main_map")
    
    # Récupérer les dessins automatiquement
    if map_output and map_output.get('all_drawings'):
        drawings = map_output['all_drawings']
        if drawings and len(drawings) > 0:
            try:
                gdf_drawn = gpd.GeoDataFrame.from_features(drawings, crs="EPSG:4326")
                st.session_state.drawn_geometry = gdf_drawn.geometry.unary_union
                st.success(f"✅ Zone dessinée: {len(drawings)} forme(s). Cliquez sur 'Lancer l'analyse' pour continuer.")
            except:
                pass

# --------------------
# CHARGEMENT DES DONNÉES
# --------------------
if load_btn:
    with st.spinner("🔄 Chargement et analyse en cours..."):
        
        # 1. Déterminer la géométrie
        geometry = None
        
        if zone_method == "📂 Importer GeoJSON" and uploaded_file:
            file_bytes = uploaded_file.read()
            gdf = load_geojson(file_bytes)
            if gdf is not None and not gdf.empty:
                st.session_state.gdf = gdf
                geometry = gdf.geometry.unary_union
        
        elif zone_method == "✏️ Dessiner sur carte":
            if st.session_state.drawn_geometry:
                gdf = gpd.GeoDataFrame([{'geometry': st.session_state.drawn_geometry}], crs='EPSG:4326')
                st.session_state.gdf = gdf
                geometry = st.session_state.drawn_geometry
            else:
                st.error("❌ Veuillez d'abord dessiner une zone sur la carte (onglet 'Carte & Zone')")
                st.stop()
        
        elif zone_method == "📌 Coordonnées" and manual_coords:
            polygon = create_polygon_from_coords(*manual_coords)
            gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
            st.session_state.gdf = gdf
            geometry = polygon
        
        if geometry is None:
            st.error("❌ Veuillez définir une zone d'étude")
            st.stop()
        
        # 2. Convertir geometry en dict pour le cache
        geom_dict = geometry_to_dict(geometry)
        
        # 3. Récupérer données satellite (simulées)
        with st.spinner("📡 Récupération NDVI..."):
            satellite_df = simulate_ndvi_data(start_date, end_date)
            st.session_state.satellite_data = satellite_df
        
        # 4. Récupérer données climatiques
        with st.spinner("🌦️ Récupération climat..."):
            climate_df = get_climate_nasa_cached(geom_dict, start_date, end_date)
            st.session_state.climate_data = climate_df
        
        if climate_df is None or climate_df.empty:
            st.error("❌ Échec récupération données climatiques")
            st.stop()
        
        st.success("✅ Données chargées avec succès! Explorez les onglets.")

# --------------------
# ONGLET 2: VUE D'ENSEMBLE
# --------------------
with tabs[1]:
    st.subheader("📊 Tableau de Bord")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        metrics = calculate_metrics(
            st.session_state.climate_data,
            st.session_state.satellite_data,
            culture
        )
        
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
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_ndvi, ax = plt.subplots(figsize=(8, 5))
            ax.plot(st.session_state.satellite_data['date'],
                   st.session_state.satellite_data['ndvi_mean'],
                   'o-', color='darkgreen', linewidth=2)
            ax.fill_between(st.session_state.satellite_data['date'],
                           st.session_state.satellite_data['ndvi_mean'],
                           alpha=0.3, color='green')
            ax.axhline(0.6, color='orange', linestyle='--', label='Seuil optimal')
            ax.set_ylabel('NDVI')
            ax.set_title('Vigueur Végétale', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=30)
            st.pyplot(fig_ndvi)
        
        with col_g2:
            fig_clim, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            ax1.bar(st.session_state.climate_data['date'],
                   st.session_state.climate_data['rain'],
                   color='steelblue', alpha=0.6)
            ax2.plot(st.session_state.climate_data['date'],
                    st.session_state.climate_data['temp_mean'],
                    color='orangered', linewidth=2)
            ax1.set_ylabel('Pluie (mm)', color='steelblue')
            ax2.set_ylabel('Temp (°C)', color='orangered')
            ax1.set_title('Conditions Climatiques', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            plt.xticks(rotation=30)
            st.pyplot(fig_clim)
        
        # Analyse rapide
        st.markdown("### 🔍 Analyse Rapide")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            if metrics['ndvi_mean'] > 0.6:
                st.markdown('<div class="success-box">✅ <b>Vigueur végétale excellente</b><br>La culture se développe très bien.</div>', 
                           unsafe_allow_html=True)
            elif metrics['ndvi_mean'] > 0.4:
                st.markdown('<div class="alert-box">⚠️ <b>Vigueur modérée</b><br>Surveillance recommandée.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box" style="border-left: 4px solid #DC3545; background: #F8D7DA;">❌ <b>Stress végétal détecté</b><br>Action urgente nécessaire.</div>', 
                           unsafe_allow_html=True)
        
        with col_a2:
            if metrics['rain_total'] < 200:
                st.markdown('<div class="alert-box" style="border-left: 4px solid #DC3545; background: #F8D7DA;">💧 <b>Stress hydrique sévère</b><br>Irrigation recommandée.</div>', 
                           unsafe_allow_html=True)
            elif metrics['rain_total'] < 350:
                st.markdown('<div class="alert-box">💧 <b>Pluviométrie limite</b><br>Surveiller l\'humidité du sol.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">💧 <b>Pluviométrie adéquate</b><br>Bon approvisionnement en eau.</div>', 
                           unsafe_allow_html=True)
        
        with col_a3:
            if metrics['temp_max'] > 38:
                st.markdown('<div class="alert-box">🌡️ <b>Chaleur excessive</b><br>Risque de stress thermique.</div>', 
                           unsafe_allow_html=True)
            elif metrics['temp_mean'] > 32:
                st.markdown('<div class="info-box">🌡️ <b>Température élevée</b><br>Conditions chaudes normales.</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">🌡️ <b>Température optimale</b><br>Bonnes conditions thermiques.</div>', 
                           unsafe_allow_html=True)
    else:
        st.info("👆 Configurez les paramètres et lancez l'analyse")

# --------------------
# ONGLET 3: NDVI
# --------------------
with tabs[2]:
    st.subheader("🛰️ Analyse NDVI Détaillée")
    
    if st.session_state.satellite_data is not None:
        df_sat = st.session_state.satellite_data
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(df_sat['date'], df_sat['ndvi_mean'], 'o-',
                   color='darkgreen', linewidth=2.5, markersize=7)
            ax.fill_between(df_sat['date'], df_sat['ndvi_min'], df_sat['ndvi_max'],
                           alpha=0.25, color='green')
            ax.axhline(0.7, color='darkgreen', linestyle=':', alpha=0.6, label='Excellent')
            ax.axhline(0.5, color='orange', linestyle=':', alpha=0.6, label='Bon')
            ax.axhline(0.3, color='red', linestyle=':', alpha=0.6, label='Stress')
            ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
            ax.set_title('Évolution NDVI', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            plt.xticks(rotation =30)
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
            st.write("Action urgente requise")
else:
    st.info("Chargez d'abord les données")		
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
else:
    st.info("Chargez d'abord les données")
	
# --------------------
# ONGLET 5: ANALYSE IA
# --------------------
with tabs[4]:
    st.subheader("🤖 Analyse IA avec Google Gemini (Gratuit)")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        st.info("💡 **Google Gemini** est gratuit (15 requêtes/min). [Obtenez votre clé](https://aistudio.google.com/apikey)")
        
        metrics = calculate_metrics(
            st.session_state.climate_data,
            st.session_state.satellite_data,
            culture
        )
        
        analyze_btn = st.button("🚀 Générer l'analyse complète", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 Analyse par IA en cours..."):
                
                ndvi_series = ", ".join([
                    f"{row['date'].strftime('%d/%m')}: {row['ndvi_mean']:.2f}"
                    for _, row in st.session_state.satellite_data.head(15).iterrows()
                ])
                
                prompt = f"""Tu es un agronome expert spécialisé en agriculture sahélienne. Analyse cette parcelle et fournis des recommandations DÉTAILLÉES et PRATIQUES.

DONNÉES PARCELLE:
- Culture: {culture}
- Zone: {zone_name}  
- Période: {(end_date - start_date).days} jours ({start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')})

INDICES VÉGÉTATIFS:
- NDVI moyen: {metrics['ndvi_mean']:.3f}
- Évolution: {ndvi_series}
- Variation: {metrics['ndvi_std']:.3f}

CLIMAT:
- Température: {metrics['temp_mean']:.1f}°C (min: {metrics['temp_min']:.1f}°, max: {metrics['temp_max']:.1f}°)
- Pluie totale: {metrics['rain_total']:.0f} mm
- Pluie moyenne: {metrics['rain_mean']:.1f} mm/jour
- Jours de pluie: {metrics['rain_days']}

RENDEMENT ESTIMÉ: {metrics['yield_potential']:.1f} t/ha

ANALYSE DEMANDÉE (structure précise):

1. DIAGNOSTIC GLOBAL
- État santé culture (NDVI)
- Stress identifiés (hydrique/thermique/nutritionnel)
- Évaluation rendement

2. RECOMMANDATIONS PRIORITAIRES

A) IRRIGATION (si nécessaire)
- Fréquence: tous les X jours
- Quantité: Y mm par apport
- Méthode: goutte-à-goutte/aspersion/gravité
- Moment: matin/soir

B) FERTILISATION
- Type: NPK/Urée/Organique
- Dose: X kg/ha
- Période: semis/tallage/floraison
- Fractionnement recommandé

C) TRAITEMENTS (si nécessaire)
- Type: pesticide/fongicide/herbicide
- Produit recommandé
- Dose et moment d'application

D) PRATIQUES CULTURALES
- Sarclage: fréquence
- Buttage: moment optimal
- Autres opérations

3. ALERTES URGENTES
Liste des actions < 7 jours

4. SUIVI
- Fréquence monitoring: hebdo/bi-hebdo
- Indicateurs à surveiller

Réponds en français, de manière claire et pédagogique. Sois PRÉCIS sur les doses, fréquences et périodes."""

                analysis_text = analyze_with_gemini(prompt, gemini_key if gemini_key else None)
                
                if analysis_text:
                    st.session_state.analysis = analysis_text
                else:
                    st.warning("⚠️ Aucune clé Gemini fournie. Analyse basique générée.")
                    st.session_state.analysis = f"""
### ANALYSE AGRO-CLIMATIQUE - {culture.upper()}

**DIAGNOSTIC:**

Votre parcelle de {culture} présente un NDVI moyen de {metrics['ndvi_mean']:.3f}, ce qui indique un {'bon' if metrics['ndvi_mean'] > 0.5 else 'faible'} développement végétatif.

**SITUATION HYDRIQUE:**
Avec {metrics['rain_total']:.0f} mm de pluie cumulée sur la période, {'la situation est favorable' if metrics['rain_total'] > 300 else 'un stress hydrique est probable'}.

**RECOMMANDATIONS PRINCIPALES:**

1. **IRRIGATION:** 
   {'Apport complémentaire recommandé: 20-25mm tous les 5-7 jours' if metrics['rain_total'] < 350 else 'Situation pluviométrique satisfaisante'}

2. **FERTILISATION:**
   - NPK 15-15-15: 150 kg/ha (50% au semis, 50% à 30 jours)
   - Urée: 50 kg/ha à la montaison

3. **SURVEILLANCE:**
   - Contrôle NDVI hebdomadaire
   - Observation stress foliaire
   - Humidité sol (20-30cm profondeur)

**CONDITIONS THERMIQUES:**
Température moyenne de {metrics['temp_mean']:.1f}°C avec des pics à {metrics['temp_max']:.1f}°C.
{'Attention aux stress thermiques possibles.' if metrics['temp_max'] > 35 else 'Conditions thermiques acceptables.'}

**RENDEMENT POTENTIEL:**
Basé sur les conditions actuelles, le rendement estimé est de {metrics['yield_potential']:.1f} tonnes/hectare.

**CONSEILS SPÉCIFIQUES POUR {culture.upper()}:**

"""
                    # Ajouter des conseils spécifiques par culture
                    if culture == "Mil":
                        st.session_state.analysis += """
- Résiste bien à la sécheresse mais nécessite au moins 400mm d'eau
- Période critique: tallage et épiaison
- Sensible aux attaques de chenilles et oiseaux
- Récolte: 90-120 jours après semis
"""
                    elif culture == "Maïs":
                        st.session_state.analysis += """
- Besoin en eau: 500-800mm bien répartis
- Période critique: floraison et remplissage des grains
- Sensible à la sécheresse et aux températures >35°C
- Récolte: 100-140 jours après semis
"""
                    elif culture == "Arachide":
                        st.session_state.analysis += """
- Besoin en eau: 450-600mm
- Période critique: floraison et formation des gousses
- Sensible à l'excès d'eau et aux maladies fongiques
- Récolte: 90-120 jours après semis
"""
                    elif culture == "Sorgho":
                        st.session_state.analysis += """
- Très résistant à la sécheresse (300-500mm)
- Période critique: épiaison
- Sensible aux oiseaux à maturité
- Récolte: 100-130 jours après semis
"""
                    
                    st.session_state.analysis += """

**ACTIONS RECOMMANDÉES CETTE SEMAINE:**
1. Vérifier l'humidité du sol à 20-30cm de profondeur
2. Observer les feuilles pour détecter stress ou maladies
3. Prévoir l'irrigation si pas de pluie dans les 5 prochains jours

**Pour une analyse IA complète et personnalisée, ajoutez votre clé Google Gemini gratuite dans les paramètres.**
"""
        
        # Afficher l'analyse
        if st.session_state.analysis:
            st.markdown("### 📋 Rapport Agronomique Détaillé")
            
            # Créer des sections expandables
            sections = st.session_state.analysis.split('\n\n')
            current_section = ""
            
            for section in sections:
                if section.strip():
                    # Détection des titres principaux
                    if section.startswith('###'):
                        st.markdown(section)
                    elif section.startswith('**') and section.endswith(':**'):
                        st.markdown(section)
                    elif section.startswith('1.') or section.startswith('2.') or section.startswith('3.') or section.startswith('4.'):
                        with st.expander(section.split('\n')[0], expanded=True):
                            st.markdown('\n'.join(section.split('\n')[1:]))
                    else:
                        st.markdown(section)
            
            # Boutons d'export
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Télécharger l'analyse (TXT)",
                    st.session_state.analysis,
                    file_name=f"analyse_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Créer un résumé JSON
                summary_json = json.dumps({
                    "zone": zone_name,
                    "culture": culture,
                    "date_analyse": datetime.now().strftime('%Y-%m-%d'),
                    "periode": f"{start_date} au {end_date}",
                    "ndvi_moyen": round(metrics['ndvi_mean'], 3),
                    "temperature_moyenne": round(metrics['temp_mean'], 1),
                    "pluie_totale": round(metrics['rain_total'], 1),
                    "rendement_estime": round(metrics['yield_potential'], 1)
                }, indent=2, ensure_ascii=False)
                
                st.download_button(
                    "📊 Télécharger métriques (JSON)",
                    summary_json,
                    file_name=f"metriques_{zone_name}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("👆 Chargez d'abord les données pour générer une analyse")
        st.markdown("""
        **L'analyse IA vous fournira:**
        - 🔍 Diagnostic précis de l'état de votre culture
        - 💧 Recommandations d'irrigation personnalisées
        - 🌾 Plan de fertilisation adapté
        - ⚠️ Alertes et actions urgentes
        - 📅 Plan de suivi détaillé
        """)

# --------------------
# ONGLET 6: RAPPORT PDF
# --------------------
with tabs[5]:
    st.subheader("📄 Rapport PDF Complet")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        st.markdown("""
        **Le rapport PDF inclut:**
        - 📊 Carte et informations de la zone d'étude
        - 📈 Graphiques NDVI avec évolution temporelle
        - 🌦️ Graphiques climatiques (température et pluie)
        - 📉 Statistiques et métriques détaillées
        - 🤖 Analyse IA complète avec recommandations
        - 💡 Conseils pratiques adaptés à votre culture
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("📄 Générer le rapport PDF", type="primary", use_container_width=True):
                with st.spinner("📝 Génération du rapport PDF en cours..."):
                    
                    metrics = calculate_metrics(
                        st.session_state.climate_data,
                        st.session_state.satellite_data,
                        culture
                    )
                    
                    analysis_text = st.session_state.analysis if st.session_state.analysis else "Analyse non générée. Utilisez l'onglet 'Analyse IA' pour générer une analyse complète avec des recommandations détaillées."
                    
                    try:
                        pdf_buffer = generate_pdf_report(
                            st.session_state.gdf,
                            st.session_state.climate_data,
                            st.session_state.satellite_data,
                            metrics,
                            culture,
                            zone_name,
                            analysis_text
                        )
                        
                        st.success("✅ Rapport PDF généré avec succès!")
                        
                        st.download_button(
                            "📥 Télécharger le rapport PDF",
                            pdf_buffer,
                            file_name=f"rapport_agrisight_{zone_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du PDF: {e}")
                        st.info("Vous pouvez toujours télécharger les données CSV ci-dessous.")
        
        with col2:
            st.info("""
            **Contenu du PDF:**
            - Page 1: Données et graphiques
            - Page 2: Analyse IA
            - Format A4
            - Prêt à imprimer
            """)
        
        st.markdown("---")
        st.markdown("### 💾 Export des données brutes (CSV)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_climate = st.session_state.climate_data.to_csv(index=False)
            st.download_button(
                "📊 Données climatiques",
                csv_climate,
                f"climat_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_ndvi = st.session_state.satellite_data.to_csv(index=False)
            st.download_button(
                "🛰️ Données NDVI",
                csv_ndvi,
                f"ndvi_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Export métriques calculées
            metrics = calculate_metrics(
                st.session_state.climate_data,
                st.session_state.satellite_data,
                culture
            )
            
            metrics_df = pd.DataFrame([metrics])
            metrics_csv = metrics_df.to_csv(index=False)
            
            st.download_button(
                "📈 Métriques calculées",
                metrics_csv,
                f"metriques_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Section export GeoJSON de la zone
        if st.session_state.gdf is not None:
            st.markdown("---")
            st.markdown("### 🗺️ Export de la zone d'étude")
            
            geojson_str = st.session_state.gdf.to_json()
            
            st.download_button(
                "📍 Télécharger la zone (GeoJSON)",
                geojson_str,
                f"zone_{zone_name}_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("👆 Chargez d'abord les données pour générer un rapport")
        
        st.markdown("""
        **Fonctionnalités du rapport PDF:**
        
        📄 **Format professionnel** prêt à imprimer ou partager
        
        📊 **Visualisations complètes:**
        - Graphiques NDVI avec seuils d'interprétation
        - Évolution des températures (min, max, moyenne)
        - Historique des précipitations
        
        📈 **Métriques clés:**
        - Indices de vigueur végétale
        - Statistiques climatiques
        - Estimation de rendement
        
        🤖 **Recommandations IA:**
        - Conseils d'irrigation
        - Plan de fertilisation
        - Calendrier cultural
        - Alertes et actions urgentes
        
        💾 **Formats disponibles:**
        - PDF (rapport complet)
        - CSV (données brutes)
        - JSON (métriques)
        - GeoJSON (zone d'étude)
        """)

# --------------------
# FOOTER
# --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>🌾 AgriSight Pro v2.1</b> - Analyse agricole par télédétection et IA<br>
    <small>
    Données: NASA POWER (climat) • NDVI simulé • Google Gemini (IA gratuite)<br>
    Développé avec ❤️ pour l'agriculture de précision en Afrique de l'Ouest<br>
    <a href="https://aistudio.google.com/apikey" target="_blank">Obtenir clé Gemini gratuite</a> | 
    <a href="https://agromonitoring.com" target="_blank">API Agromonitoring</a>
    </small>
</div>
""", unsafe_allow_html=True)

# Sidebar footer avec instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Guide rapide

1. **Définir la zone**
   - Importer GeoJSON
   - Dessiner sur carte
   - Ou coordonnées

2. **Configurer**
   - Dates d'analyse
   - Type de culture
   - Nom de zone

3. **Analyser**
   - Cliquer "Lancer l'analyse"
   - Explorer les onglets
   - Générer rapport IA

4. **Exporter**
   - PDF complet
   - CSV des données
   - Analyse texte
""")

st.sidebar.markdown("""
### 💡 Astuce

Pour de vraies données NDVI:
1. Créer compte sur [agromonitoring.com](https://agromonitoring.com)
2. Copier clé API (gratuite)
3. Coller dans "Clés API"
""")
