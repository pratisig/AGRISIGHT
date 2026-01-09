import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, mapping
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json
from matplotlib.backends.backend_pdf import PdfPages
import base64

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

# API Keys (stockées en session state pour sécurité)
with st.sidebar.expander("🔑 Clés API (Optionnel)", expanded=False):
    st.markdown("""
    **OpenWeather Agromonitoring** (Gratuit - 1000 appels/jour)
    - Inscription: [agromonitoring.com](https://agromonitoring.com)
    - NDVI, EVI, NDWI, Images satellite réelles
    """)
    agromonitoring_key = st.text_input("Clé Agromonitoring", type="password", 
                                      help="Laissez vide pour mode démo simulé")
    
    st.markdown("""
    **Ollama (Local - Gratuit)** 
    - Installation: [ollama.com](https://ollama.com)
    - Modèles: llama3, mistral, gemma
    - Fonctionne hors ligne
    """)
    use_ollama = st.checkbox("Utiliser Ollama (IA locale)", value=False)
    ollama_url = st.text_input("URL Ollama", value="http://localhost:11434", 
                               help="URL de votre serveur Ollama local")
    ollama_model = st.selectbox("Modèle Ollama", 
                                ["llama3.2", "mistral", "gemma2:2b", "phi3"],
                                help="Modèle à utiliser pour l'analyse")

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
    st.sidebar.info("Entrez les coins d'un rectangle (min/max)")
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

def register_polygon_agro(geometry, api_key):
    """Enregistre un polygone sur Agromonitoring API"""
    if not api_key:
        return None
    
    try:
        coords = list(mapping(geometry)['coordinates'][0])
        
        url = f"http://api.agromonitoring.com/agro/1.0/polygons?appid={api_key}"
        payload = {
            "name": "parcelle_temp",
            "geo_json": {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 201:
            return response.json()['id']
        else:
            st.warning(f"Erreur enregistrement polygone: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur API Agromonitoring: {e}")
        return None

@st.cache_data(ttl=3600)
def get_satellite_imagery_agro(polygon_id, api_key, start, end):
    """Récupère les données satellite via Agromonitoring"""
    if not polygon_id or not api_key:
        return None
    
    try:
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end, datetime.max.time()).timestamp())
        
        url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history?polyid={polygon_id}&start={start_ts}&end={end_ts}&appid={api_key}"
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            
            if not data:
                return None
            
            # Parser les données
            results = []
            for item in data:
                dt = datetime.fromtimestamp(item['dt'])
                
                # Récupérer les stats NDVI si disponibles
                stats_url = item.get('data', {}).get('std')
                if stats_url:
                    stats_response = requests.get(f"{stats_url}?appid={api_key}", timeout=10)
                    if stats_response.status_code == 200:
                        stats = stats_response.json()
                        results.append({
                            'date': dt,
                            'ndvi_mean': stats.get('mean', np.nan),
                            'ndvi_std': stats.get('std', np.nan),
                            'ndvi_min': stats.get('min', np.nan),
                            'ndvi_max': stats.get('max', np.nan),
                            'cloud_cover': item.get('cl', 0)
                        })
            
            return pd.DataFrame(results) if results else None
        else:
            st.warning(f"Pas de données satellite disponibles (code {response.status_code})")
            return None
    except Exception as e:
        st.error(f"Erreur récupération satellite: {e}")
        return None

@st.cache_data(ttl=3600)
def get_climate_nasa_polygon(geometry, start, end):
    """Récupère les données climatiques NASA POWER pour un polygone (centroïde)"""
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

def simulate_ndvi_data(start, end):
    """Génère des données NDVI simulées réalistes (fallback)"""
    dates = pd.date_range(start, end, freq='5D')
    ndvi_values = []
    
    for d in dates:
        month = d.month
        # Simuler selon saison
        if 6 <= month <= 9:  # Saison pluies
            base = 0.65 + np.random.normal(0, 0.08)
        elif month in [5, 10]:  # Transition
            base = 0.45 + np.random.normal(0, 0.1)
        else:  # Saison sèche
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

def analyze_with_ollama(prompt, url, model):
    """Analyse avec Ollama (IA locale)"""
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return None
    except Exception as e:
        st.error(f"Erreur Ollama: {e}. Vérifiez que Ollama est démarré.")
        return None

def calculate_metrics(climate_df, ndvi_df, culture):
    """Calcule les métriques agrégées et le rendement estimé"""
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
    
    # Estimation rendement selon culture et conditions
    ndvi = metrics['ndvi_mean']
    rain = metrics['rain_total']
    
    # Modèles simplifiés de rendement par culture
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
    else:  # Défaut
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
        # Page 1: Carte et infos générales
        fig = plt.figure(figsize=(11, 8.5))
        
        # Titre
        fig.suptitle(f"Rapport d'Analyse Agricole - {zone_name}", 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Informations générales
        ax_info = fig.add_subplot(3, 2, 1)
        ax_info.axis('off')
        info_text = f"""
        Culture: {culture}
        Zone: {zone_name}
        Date d'analyse: {datetime.now().strftime('%d/%m/%Y')}
        Période: {climate_df['date'].min().strftime('%d/%m/%Y')} - {climate_df['date'].max().strftime('%d/%m/%Y')}
        
        Superficie: {gdf.geometry.area.sum():.2f} ha
        """
        ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
        
        # Métriques clés
        ax_metrics = fig.add_subplot(3, 2, 2)
        ax_metrics.axis('off')
        metrics_text = f"""
        📊 INDICATEURS CLÉS
        
        NDVI moyen: {metrics['ndvi_mean']:.3f}
        Température moy: {metrics['temp_mean']:.1f}°C
        Pluviométrie totale: {metrics['rain_total']:.0f} mm
        Rendement estimé: {metrics['yield_potential']:.1f} t/ha
        """
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Graphique NDVI
        ax_ndvi = fig.add_subplot(3, 2, (3, 4))
        ax_ndvi.plot(ndvi_df['date'], ndvi_df['ndvi_mean'], 'o-', color='green', linewidth=2)
        ax_ndvi.fill_between(ndvi_df['date'], ndvi_df['ndvi_mean'], alpha=0.3, color='green')
        ax_ndvi.set_title('Évolution NDVI', fontsize=12, fontweight='bold')
        ax_ndvi.set_ylabel('NDVI')
        ax_ndvi.grid(True, alpha=0.3)
        ax_ndvi.set_ylim([0, 1])
        
        # Graphique Climat
        ax_climate = fig.add_subplot(3, 2, (5, 6))
        ax_temp = ax_climate.twinx()
        
        ax_climate.bar(climate_df['date'], climate_df['rain'], color='blue', alpha=0.4, label='Pluie')
        ax_temp.plot(climate_df['date'], climate_df['temp_mean'], color='red', linewidth=2, label='Temp')
        
        ax_climate.set_xlabel('Date')
        ax_climate.set_ylabel('Pluie (mm)', color='blue')
        ax_temp.set_ylabel('Température (°C)', color='red')
        ax_climate.legend(loc='upper left')
        ax_temp.legend(loc='upper right')
        ax_climate.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Analyse et recommandations
        if analysis_text:
            fig2 = plt.figure(figsize=(11, 8.5))
            ax_analysis = fig2.add_subplot(111)
            ax_analysis.axis('off')
            
            # Formatter le texte pour le PDF
            wrapped_text = "\n".join([line[:90] for line in analysis_text.split('\n')])
            ax_analysis.text(0.05, 0.95, wrapped_text, 
                           fontsize=9, verticalalignment='top',
                           wrap=True, family='monospace')
            
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close()
    
    buffer.seek(0)
    return buffer

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
        
        elif zone_method == "📌 Coordonnées" and manual_coords:
            polygon = create_polygon_from_coords(*manual_coords)
            gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
            st.session_state.gdf = gdf
            geometry = polygon
        
        elif zone_method == "✏️ Dessiner sur carte":
            st.info("Utilisez l'outil de dessin sur la carte ci-dessous, puis relancez l'analyse")
        
        if geometry is None:
            st.error("❌ Veuillez définir une zone d'étude")
            st.stop()
        
        # 2. Enregistrer le polygone (si API key fournie)
        if agromonitoring_key:
            polygon_id = register_polygon_agro(geometry, agromonitoring_key)
            st.session_state.polygon_id = polygon_id
        
        # 3. Récupérer données satellite
        satellite_df = None
        
        if agromonitoring_key and st.session_state.polygon_id:
            with st.spinner("📡 Récupération images satellite..."):
                satellite_df = get_satellite_imagery_agro(
                    st.session_state.polygon_id, 
                    agromonitoring_key, 
                    start_date, 
                    end_date
                )
        
        # Fallback: données simulées si pas d'API
        if satellite_df is None or satellite_df.empty:
            st.warning("⚠️ Utilisation de données NDVI simulées (API non configurée)")
            satellite_df = simulate_ndvi_data(start_date, end_date)
        
        st.session_state.satellite_data = satellite_df
        
        # 4. Récupérer données climatiques
        with st.spinner("🌦️ Récupération données climatiques..."):
            climate_df = get_climate_nasa_polygon(geometry, start_date, end_date)
        
        st.session_state.climate_data = climate_df
        
        if climate_df is None or climate_df.empty:
            st.error("❌ Échec récupération données climatiques")
            st.stop()
        
        st.success("✅ Données chargées avec succès!")

# --------------------
# ONGLETS PRINCIPAUX
# --------------------

tabs = st.tabs(["📊 Vue d'ensemble", "🗺️ Carte Interactive", "🛰️ NDVI", 
                "🌦️ Climat", "🤖 Analyse IA", "📄 Rapport PDF"])

# --------------------
# ONGLET 1: VUE D'ENSEMBLE
# --------------------
with tabs[0]:
    st.subheader("📊 Tableau de Bord Synthétique")
    
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
            st.metric("🌡️ Température Moy.", f"{metrics['temp_mean']:.1f}°C",
                     delta=f"{metrics['temp_min']:.0f}° - {metrics['temp_max']:.0f}°")
        
        with col3:
            delta_rain = "Suffisant" if metrics['rain_total'] > 300 else "Insuffisant"
            st.metric("💧 Pluie Totale", f"{metrics['rain_total']:.0f} mm", delta=delta_rain)
        
        with col4:
            st.metric("📈 Rendement Estimé", f"{metrics['yield_potential']:.1f} t/ha")
        
        st.markdown("---")
        
        # Graphiques combinés
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            # NDVI Evolution
            fig_ndvi, ax = plt.subplots(figsize=(8, 5))
            ax.plot(st.session_state.satellite_data['date'], 
                   st.session_state.satellite_data['ndvi_mean'],
                   'o-', color='darkgreen', linewidth=2, markersize=6, label='NDVI')
            ax.fill_between(st.session_state.satellite_data['date'],
                           st.session_state.satellite_data['ndvi_min'],
                           st.session_state.satellite_data['ndvi_max'],
                           alpha=0.2, color='green', label='Plage NDVI')
            ax.axhline(0.6, color='orange', linestyle='--', alpha=0.7, label='Seuil optimal')
            ax.axhline(0.3, color='red', linestyle='--', alpha=0.7, label='Seuil stress')
            ax.set_ylabel('NDVI', fontsize=11)
            ax.set_title('Évolution de la Vigueur Végétale', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=30)
            st.pyplot(fig_ndvi)
        
        with col_graph2:
            # Climat
            fig_clim, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            
            ax1.bar(st.session_state.climate_data['date'],
                   st.session_state.climate_data['rain'],
                   color='steelblue', alpha=0.6, label='Pluie (mm)')
            ax2.plot(st.session_state.climate_data['date'],
                    st.session_state.climate_data['temp_mean'],
                    color='orangered', linewidth=2.5, label='Température (°C)')
            
            ax1.set_ylabel('Précipitations (mm)', color='steelblue', fontsize=11)
            ax2.set_ylabel('Température (°C)', color='orangered', fontsize=11)
            ax1.set_title('Conditions Climatiques', fontsize=13, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=9)
            ax2.legend(loc='upper right', fontsize=9)
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
        st.info("👆 Configurez les paramètres et cliquez sur 'Lancer l'analyse'")

# --------------------
# ONGLET 2: CARTE
# --------------------
with tabs[1]:
    st.subheader("🗺️ Carte Interactive de la Zone d'Étude")
    
    # Déterminer le centre
    if st.session_state.gdf is not None:
        center = [st.session_state.gdf.geometry.centroid.y.mean(),
                 st.session_state.gdf.geometry.centroid.x.mean()]
    else:
        center = [14.6937, -17.4441]
    
    # Créer la carte
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
    
    # Ajouter contrôles
    m.add_child(MeasureControl(primary_length_unit='meters'))
    
    # Ajouter la zone d'étude
    if st.session_state.gdf is not None:
        folium.GeoJson(
            st.session_state.gdf,
            name="Zone d'analyse",
            style_function=lambda x: {
                'fillColor': '#28A745',
                'color': '#155724',
                'weight': 3,
                'fillOpacity': 0.4
            },
            tooltip=f"<b>{zone_name}</b><br>Culture: {culture}"
        ).add_to(m)
        
        # Ajouter marqueur au centroïde
        centroid = st.session_state.gdf.geometry.centroid.iloc[0]
        folium.Marker(
            [centroid.y, centroid.x],
            popup=f"<b>{zone_name}</b><br>Culture: {culture}<br>Surface: {st.session_state.gdf.geometry.area.sum():.2f} ha",
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
    
    # Outil de dessin
    draw = Draw(
        export=True,
        draw_options={
            'polygon': True,
            'rectangle': True,
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    draw.add_to(m)
    
    # Afficher la carte
    map_output = st_folium(m, height=550, width=None)
    
    # Récupérer les dessins
    if map_output and map_output.get('all_drawings'):
        st.info(f"✏️ {len(map_output['all_drawings'])} forme(s) dessinée(s). Relancez l'analyse pour les utiliser.")

# --------------------
# ONGLET 3: NDVI
# --------------------
with tabs[2]:
    st.subheader("🛰️ Analyse NDVI Détaillée")
    
    if st.session_state.satellite_data is not None:
        df_sat = st.session_state.satellite_data
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            # Graphique NDVI avancé
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            
            # Evolution NDVI
            ax1.plot(df_sat['date'], df_sat['ndvi_mean'], 'o-', 
                    color='darkgreen', linewidth=2.5, markersize=7, label='NDVI moyen')
            ax1.fill_between(df_sat['date'], df_sat['ndvi_min'], df_sat['ndvi_max'],
                            alpha=0.25, color='green', label='Plage min-max')
            
            # Seuils
            ax1.axhline(0.7, color='darkgreen', linestyle=':', alpha=0.6, label='Excellent (>0.7)')
            ax1.axhline(0.5, color='orange', linestyle=':', alpha=0.6, label='Bon (0.5-0.7)')
            ax1.axhline(0.3, color='red', linestyle=':', alpha=0.6, label='Stress (<0.3)')
            
            ax1.set_ylabel('NDVI', fontsize=12, fontweight='bold')
            ax1.set_title('Évolution Temporelle du NDVI', fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim([0, 1])
            
            # Couverture nuageuse
            ax2.bar(df_sat['date'], df_sat['cloud_cover'], 
                   color='gray', alpha=0.5, label='Couverture nuageuse (%)')
            ax2.set_ylabel('Nuages (%)', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Statistiques NDVI")
            
            stats_ndvi = {
                "Moyenne": df_sat['ndvi_mean'].mean(),
                "Médiane": df_sat['ndvi_mean'].median(),
                "Écart-type": df_sat['ndvi_mean'].std(),
                "Minimum": df_sat['ndvi_mean'].min(),
                "Maximum": df_sat['ndvi_mean'].max(),
                "Tendance": "↗️ Croissance" if df_sat['ndvi_mean'].iloc[-1] > df_sat['ndvi_mean'].iloc[0] else "↘️ Décroissance"
            }
            
            for key, val in stats_ndvi.items():
                if isinstance(val, str):
                    st.metric(key, val)
                else:
                    st.metric(key, f"{val:.3f}")
            
            st.markdown("---")
            st.markdown("### 🔬 Interprétation")
            
            ndvi_mean = df_sat['ndvi_mean'].mean()
            
            if ndvi_mean > 0.7:
                st.success("🌟 **Excellent état végétatif**")
                st.write("Croissance optimale, culture en très bonne santé.")
            elif ndvi_mean > 0.5:
                st.info("✅ **Bon développement**")
                st.write("Culture en bonne santé avec potentiel d'amélioration.")
            elif ndvi_mean > 0.3:
                st.warning("⚠️ **État modéré**")
                st.write("Croissance ralentie, surveillance nécessaire.")
            else:
                st.error("❌ **Stress végétal sévère**")
                st.write("Action immédiate requise: irrigation, fertilisation.")
            
            st.markdown("---")
            st.markdown("### 📅 Données Temporelles")
            st.dataframe(df_sat[['date', 'ndvi_mean', 'cloud_cover']].tail(10), 
                        use_container_width=True)
    else:
        st.info("Lancez d'abord l'analyse pour voir les données NDVI")

# --------------------
# ONGLET 4: CLIMAT
# --------------------
with tabs[3]:
    st.subheader("🌦️ Analyse Climatique Complète")
    
    if st.session_state.climate_data is not None:
        df_clim = st.session_state.climate_data
        
        # Graphiques climatiques
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Températures
        axes[0].fill_between(df_clim['date'], df_clim['temp_min'], df_clim['temp_max'],
                            alpha=0.3, color='coral', label='Plage température')
        axes[0].plot(df_clim['date'], df_clim['temp_mean'], 
                    color='red', linewidth=2.5, label='Température moyenne')
        axes[0].axhline(30, color='orange', linestyle='--', alpha=0.5, label='Seuil chaleur (30°C)')
        axes[0].set_ylabel('Température (°C)', fontsize=11, fontweight='bold')
        axes[0].set_title('Évolution des Températures', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Précipitations
        axes[1].bar(df_clim['date'], df_clim['rain'], color='dodgerblue', alpha=0.7)
        axes[1].axhline(df_clim['rain'].mean(), color='navy', linestyle='--', 
                       linewidth=2, label=f'Moyenne: {df_clim["rain"].mean():.1f} mm/jour')
        axes[1].set_ylabel('Précipitations (mm)', fontsize=11, fontweight='bold')
        axes[1].set_title('Précipitations Journalières', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Cumul pluie
        cumul_rain = df_clim['rain'].cumsum()
        axes[2].plot(df_clim['date'], cumul_rain, color='darkblue', linewidth=2.5)
        axes[2].fill_between(df_clim['date'], cumul_rain, alpha=0.2, color='blue')
        axes[2].set_ylabel('Cumul (mm)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Date', fontsize=11)
        axes[2].set_title('Cumul de Précipitations', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistiques climatiques
        st.markdown("### 📈 Statistiques Climatiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌡️ Températures**")
            st.write(f"• Moyenne: {df_clim['temp_mean'].mean():.1f}°C")
            st.write(f"• Min absolue: {df_clim['temp_min'].min():.1f}°C")
            st.write(f"• Max absolue: {df_clim['temp_max'].max():.1f}°C")
            st.write(f"• Amplitude: {df_clim['temp_max'].max() - df_clim['temp_min'].min():.1f}°C")
        
        with col2:
            st.markdown("**💧 Précipitations**")
            st.write(f"• Cumul total: {df_clim['rain'].sum():.0f} mm")
            st.write(f"• Moyenne/jour: {df_clim['rain'].mean():.1f} mm")
            st.write(f"• Max/jour: {df_clim['rain'].max():.1f} mm")
            st.write(f"• Jours pluie (>1mm): {(df_clim['rain'] > 1).sum()}")
        
        with col3:
            st.markdown("**📊 Indices**")
            
            # Indice de stress hydrique (simplifié)
            if df_clim['rain'].sum() < 200:
                stress_hydrique = "Sévère"
                color = "🔴"
            elif df_clim['rain'].sum() < 350:
                stress_hydrique = "Modéré"
                color = "🟠"
            else:
                stress_hydrique = "Faible"
                color = "🟢"
            
            st.write(f"{color} Stress hydrique: {stress_hydrique}")
            
            # Indice de stress thermique
            jours_chaleur = (df_clim['temp_max'] > 35).sum()
            st.write(f"🌡️ Jours >35°C: {jours_chaleur}")
            
            # Distribution pluie
            jours_sans_pluie = (df_clim['rain'] < 1).sum()
            st.write(f"☀️ Jours secs: {jours_sans_pluie}")
    else:
        st.info("Lancez d'abord l'analyse pour voir les données climatiques")

# --------------------
# ONGLET 5: ANALYSE IA
# --------------------
with tabs[4]:
    st.subheader("🤖 Analyse et Recommandations par Intelligence Artificielle")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        metrics = calculate_metrics(
            st.session_state.climate_data,
            st.session_state.satellite_data,
            culture
        )
        
        # Bouton d'analyse
        analyze_btn = st.button("🚀 Générer l'analyse IA complète", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 Analyse en cours avec l'IA..."):
                
                # Préparer les données
                ndvi_series = ", ".join([
                    f"{row['date'].strftime('%d/%m')}: {row['ndvi_mean']:.2f}"
                    for _, row in st.session_state.satellite_data.head(15).iterrows()
                ])
                
                prompt = f"""Tu es un agronome expert spécialisé dans l'agriculture sahélienne. 
Analyse les données de cette parcelle et fournis des recommandations détaillées et pratiques.

DONNÉES DE LA PARCELLE:
• Culture: {culture}
• Zone: {zone_name}
• Période: {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}
• Surface: {st.session_state.gdf.geometry.area.sum() if st.session_state.gdf is not None else 'N/A'} ha

INDICES VÉGÉTATIFS:
• NDVI moyen: {metrics['ndvi_mean']:.3f} (min: {st.session_state.satellite_data['ndvi_mean'].min():.2f}, max: {st.session_state.satellite_data['ndvi_mean'].max():.2f})
• Série temporelle (15 derniers points): {ndvi_series}
• Écart-type NDVI: {metrics['ndvi_std']:.3f}

DONNÉES CLIMATIQUES:
• Température moyenne: {metrics['temp_mean']:.1f}°C
• Plage température: {metrics['temp_min']:.1f}°C à {metrics['temp_max']:.1f}°C
• Pluviométrie totale: {metrics['rain_total']:.0f} mm
• Pluviométrie moyenne: {metrics['rain_mean']:.1f} mm/jour
• Nombre de jours de pluie (>1mm): {metrics['rain_days']}

RENDEMENT ESTIMÉ: {metrics['yield_potential']:.1f} tonnes/hectare

MISSION:
Fournis une analyse COMPLÈTE et DÉTAILLÉE structurée comme suit:

1. DIAGNOSTIC GÉNÉRAL
- État de santé de la culture (basé sur NDVI)
- Identification des stress (hydrique, thermique, nutritionnel)
- Évaluation du potentiel de rendement

2. ANALYSE DÉTAILLÉE PAR FACTEUR
- Vigueur végétative (NDVI): interprétation et tendance
- Conditions hydriques: adéquation besoins/apports
- Conditions thermiques: impact sur la culture
- Stress identifiés et leur impact

3. RECOMMANDATIONS PRATIQUES (par priorité)
Pour chaque recommandation, précise:
- Action concrète à mener
- Moment optimal (quand?)
- Dosage/quantité (combien?)
- Méthode d'application (comment?)
- Justification agronomique (pourquoi?)

Catégories:
a) IRRIGATION: fréquence, quantité, méthode
b) FERTILISATION: NPK, doses, périodes d'application
c) TRAITEMENTS: pesticides, fongicides si nécessaire
d) PRATIQUES CULTURALES: sarclage, buttage, etc.
e) SURVEILLANCE: indicateurs à suivre

4. ALERTES ET ACTIONS URGENTES
- Problèmes critiques détectés
- Actions immédiates requises (<7 jours)
- Risques à anticiper

5. PLAN DE SUIVI
- Fréquence de monitoring recommandée
- Indicateurs clés à surveiller
- Seuils d'alerte

6. PRÉVISIONS ET CONSEILS POUR LA SUITE
- Évolution attendue de la culture
- Préparation pour les prochaines étapes
- Conseils pour maximiser le rendement

Adapte tes conseils au contexte sahélien (climat aride, ressources limitées, techniques traditionnelles).
Sois précis, pratique et pédagogique. Utilise un langage compréhensible par un agriculteur.
"""
                
                # Appel IA
                analysis_text = None
                
                if use_ollama:
                    # Utiliser Ollama (local, gratuit)
                    analysis_text = analyze_with_ollama(prompt, ollama_url, ollama_model)
                else:
                    st.warning("⚠️ Ollama non activé. Activez-le dans les paramètres ou utilisez une API externe.")
                    analysis_text = """
### ANALYSE AUTOMATIQUE (Mode démo)

**DIAGNOSTIC:**
Basé sur les données collectées, votre culture montre des signes de développement modéré. 
Le NDVI moyen de {:.2f} indique une activité photosynthétique acceptable mais avec un potentiel d'amélioration.

**RECOMMANDATIONS PRIORITAIRES:**

1. **IRRIGATION** (Priorité HAUTE)
   - Apporter 25-30 mm d'eau tous les 5-7 jours
   - Privilégier l'irrigation en début de journée
   - Justification: Le cumul pluviométrique de {} mm est insuffisant

2. **FERTILISATION** (Priorité MOYENNE)
   - Apport NPK 15-15-15 à raison de 150 kg/ha
   - Fractionnement recommandé: 50% au semis, 50% à 30 jours
   - Compléter avec urée (50 kg/ha) à la floraison

3. **SURVEILLANCE** (Priorité HAUTE)
   - Contrôler le NDVI chaque semaine
   - Surveiller l'humidité du sol (profondeur 20-30 cm)
   - Observer les signes de stress foliaire

**Pour une analyse complète avec IA, activez Ollama dans les paramètres.**
                    """.format(metrics['ndvi_mean'], metrics['rain_total'])
                
                st.session_state.analysis = analysis_text
        
        # Afficher l'analyse
        if st.session_state.analysis:
            st.markdown("### 📋 Rapport d'Analyse Agronomique")
            
            # Créer des sections expandables
            sections = st.session_state.analysis.split('\n\n')
            
            for section in sections:
                if section.strip():
                    # Détecter si c'est un titre (commence par ###, ##, ou chiffre)
                    if section.startswith('###') or section.startswith('##') or section[0].isdigit():
                        st.markdown(section)
                    else:
                        st.write(section)
            
            # Bouton de téléchargement du rapport texte
            st.download_button(
                "📥 Télécharger l'analyse (TXT)",
                st.session_state.analysis,
                file_name=f"analyse_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    else:
        st.info("Chargez d'abord les données pour générer une analyse")

# --------------------
# ONGLET 6: RAPPORT PDF
# --------------------
with tabs[5]:
    st.subheader("📄 Génération de Rapport PDF Complet")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        st.markdown("""
        Le rapport PDF inclura:
        - 📊 Carte de la zone d'étude
        - 📈 Tous les graphiques (NDVI, climat, tendances)
        - 📉 Statistiques détaillées
        - 🤖 Analyse et recommandations IA (si générée)
        - 💡 Conseils agronomiques adaptés à votre culture
        """)
        
        if st.button("📄 Générer le rapport PDF", type="primary", use_container_width=True):
            with st.spinner("📝 Génération du rapport en cours..."):
                
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
                
                st.success("✅ Rapport PDF généré avec succès!")
                
                st.download_button(
                    "📥 Télécharger le rapport PDF",
                    pdf_buffer,
                    file_name=f"rapport_agrisight_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        st.markdown("---")
        st.markdown("### 💾 Export des données brutes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export données climatiques
            csv_climate = st.session_state.climate_data.to_csv(index=False)
            st.download_button(
                "📊 Télécharger données climatiques (CSV)",
                csv_climate,
                f"climat_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export données NDVI
            csv_ndvi = st.session_state.satellite_data.to_csv(index=False)
            st.download_button(
                "🛰️ Télécharger données NDVI (CSV)",
                csv_ndvi,
                f"ndvi_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info("Chargez d'abord les données pour générer un rapport")

# --------------------
# FOOTER
# --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>🌾 AgriSight Pro v2.0</b> - Plateforme d'analyse agricole par télédétection et IA<br>
    Données: NASA POWER (climat) • OpenWeather Agromonitoring (NDVI) • Ollama (IA locale)<br>
    💚 Développé pour l'agriculture de précision en Afrique de l'Ouest
</div>
""", unsafe_allow_html=True)
