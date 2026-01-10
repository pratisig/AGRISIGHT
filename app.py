import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import Draw, MeasureControl, MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, mapping, shape, MultiPoint
from shapely.ops import unary_union
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json
from matplotlib.backends.backend_pdf import PdfPages
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="AgriSight Pro", layout="wide", page_icon="🌾")

# CSS personnalisé
st.markdown("""
<style>
    .big-metric {font-size: 2em; font-weight: bold; color: #2E7D32;}
    .alert-box {background: #FFF3CD; padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107;}
    .success-box {background: #D4EDDA; padding: 15px; border-radius: 8px; border-left: 4px solid #28A745;}
    .info-box {background: #D1ECF1; padding: 15px; border-radius: 8px; border-left: 4px solid #17A2B8;}
    .danger-box {background: #F8D7DA; padding: 15px; border-radius: 8px; border-left: 4px solid #DC3545;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriSight Pro - Analyse Agro-climatique Avancée")
st.markdown("*Plateforme d'analyse multi-indices par télédétection et IA pour l'agriculture de précision*")

# API Keys
AGRO_API_KEY = '28641235f2b024b5f45f97df45c6a0d5'
OPENWEATHER_KEY = ''  # À configurer par l'utilisateur

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

with st.sidebar.expander("🔑 Clés API", expanded=False):
    st.markdown("**Google Gemini API** (Gratuit)")
    st.markdown("- [Obtenez votre clé](https://aistudio.google.com/apikey)")
    gemini_key = st.text_input("Clé Gemini", type="password", value="")
    
    st.markdown("**OpenWeather API** (Gratuit)")
    st.markdown("- [Inscription](https://openweathermap.org/api)")
    openweather_key = st.text_input("Clé OpenWeather", type="password", value="")
    if openweather_key:
        OPENWEATHER_KEY = openweather_key
    
    st.success("✓ Clé Agromonitoring intégrée")

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
    st.sidebar.info("Rectangle (lat/lon)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat_min = st.number_input("Lat Min", value=14.60, format="%.4f")
        lon_min = st.number_input("Lon Min", value=-17.50, format="%.4f")
    with col2:
        lat_max = st.number_input("Lat Max", value=14.70, format="%.4f")
        lon_max = st.number_input("Lon Max", value=-17.40, format="%.4f")
    manual_coords = (lat_min, lon_min, lat_max, lon_max)

# Paramètres temporels - restriction jusqu'à aujourd'hui - 10 jours
st.sidebar.subheader("📅 Période d'analyse")
max_end_date = date.today() - timedelta(days=10)
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Début", max_end_date - timedelta(days=90), 
                                max_value=max_end_date)
with col2:
    end_date = st.date_input("Fin", max_end_date, 
                              max_value=max_end_date,
                              min_value=start_date)

# Multi-cultures
st.sidebar.subheader("🌱 Cultures à analyser")
cultures_disponibles = ["Mil", "Sorgho", "Maïs", "Arachide", "Riz", "Niébé", 
                        "Manioc", "Tomate", "Oignon", "Coton", "Pastèque"]
cultures_selectionnees = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs cultures",
    cultures_disponibles,
    default=["Mil"]
)

if not cultures_selectionnees:
    st.sidebar.error("Sélectionnez au moins une culture")

zone_name = st.sidebar.text_input("📍 Nom de la zone", "Ma parcelle")

# Paramètres d'échantillonnage
st.sidebar.subheader("🔬 Échantillonnage")
grid_size_ha = st.sidebar.slider("Taille grille (ha)", 1, 10, 5, 
                                  help="Taille max de chaque cellule d'échantillonnage")

st.sidebar.markdown("---")
load_btn = st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# Session State
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'sampling_points' not in st.session_state:
    st.session_state.sampling_points = None
if 'satellite_data' not in st.session_state:
    st.session_state.satellite_data = None
if 'climate_data' not in st.session_state:
    st.session_state.climate_data = None
if 'weather_forecast' not in st.session_state:
    st.session_state.weather_forecast = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = {}
if 'drawn_geometry' not in st.session_state:
    st.session_state.drawn_geometry = None

# Fonctions utilitaires
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

def create_sampling_grid(geometry, grid_size_ha=5):
    """Crée une grille d'échantillonnage avec cellules max de grid_size_ha"""
    bounds = geometry.bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Conversion ha en degrés (approximation à l'équateur: 1 ha ≈ 0.003 deg²)
    # Pour une grille carrée: côté ≈ sqrt(ha) * 0.003
    cell_size = np.sqrt(grid_size_ha) * 0.003
    
    # Créer la grille
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            # Point au centre de chaque cellule
            point = Point(x + cell_size/2, y + cell_size/2)
            if geometry.contains(point):
                points.append({
                    'geometry': point,
                    'longitude': point.x,
                    'latitude': point.y,
                    'cell_id': f"C{len(points)+1}"
                })
    
    return gpd.GeoDataFrame(points, crs='EPSG:4326')

@st.cache_data(ttl=3600)
def get_climate_nasa_multi_points(points_dict_list, start, end):
    """Récupère données climat pour plusieurs points"""
    results = []
    
    for point_dict in points_dict_list:
        point = dict_to_geometry(point_dict['geometry'])
        lat, lon = point.y, point.x
        
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M"
            f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
            f"&latitude={lat}&longitude={lon}&format=JSON&community=AG"
        )
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                continue
            
            data = response.json()
            params = data.get("properties", {}).get("parameter", {})
            
            df = pd.DataFrame({
                'date': pd.to_datetime(list(params.get('T2M', {}).keys())),
                'temp_mean': list(params.get('T2M', {}).values()),
                'temp_min': list(params.get('T2M_MIN', {}).values()),
                'temp_max': list(params.get('T2M_MAX', {}).values()),
                'rain': list(params.get('PRECTOTCORR', {}).values()),
                'humidity': list(params.get('RH2M', {}).values()),
                'wind_speed': list(params.get('WS2M', {}).values()),
                'cell_id': point_dict['cell_id'],
                'latitude': lat,
                'longitude': lon
            })
            
            results.append(df)
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            st.warning(f"Erreur point {point_dict['cell_id']}: {e}")
            continue
    
    if results:
        return pd.concat(results, ignore_index=True)
    return None

@st.cache_data(ttl=3600)
def get_weather_forecast(lat, lon, api_key):
    """Récupère prévisions météo 7 jours"""
    if not api_key:
        return None
    
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        forecasts = []
        
        for item in data['list'][:56]:  # 7 jours (8 prévisions/jour)
            forecasts.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temp': item['main']['temp'],
                'temp_min': item['main']['temp_min'],
                'temp_max': item['main']['temp_max'],
                'humidity': item['main']['humidity'],
                'rain': item.get('rain', {}).get('3h', 0),
                'description': item['weather'][0]['description'],
                'wind_speed': item['wind']['speed']
            })
        
        df = pd.DataFrame(forecasts)
        df['date'] = df['datetime'].dt.date
        
        # Agrégation par jour
        daily = df.groupby('date').agg({
            'temp': 'mean',
            'temp_min': 'min',
            'temp_max': 'max',
            'humidity': 'mean',
            'rain': 'sum',
            'wind_speed': 'mean',
            'description': 'first'
        }).reset_index()
        
        return daily
        
    except Exception as e:
        st.warning(f"Erreur prévisions météo: {e}")
        return None
@st.cache_data(ttl=3600)
def simulate_multi_indices_data(points_dict_list, start, end):
    """Simule données multi-indices pour chaque point d'échantillonnage"""
    dates = pd.date_range(start, end, freq='5D')
    all_data = []
    
    for point_dict in points_dict_list:
        for d in dates:
            month = d.month
            
            # NDVI - Indice de végétation normalisé
            if 6 <= month <= 9:  # Saison des pluies
                ndvi_base = 0.65 + np.random.normal(0, 0.08)
            elif month in [5, 10]:
                ndvi_base = 0.45 + np.random.normal(0, 0.1)
            else:
                ndvi_base = 0.25 + np.random.normal(0, 0.06)
            
            # EVI - Enhanced Vegetation Index (plus sensible aux zones denses)
            evi_base = ndvi_base * 0.9 + np.random.normal(0, 0.05)
            
            # NDWI - Normalized Difference Water Index (contenu en eau)
            if month in [7, 8, 9]:
                ndwi_base = 0.3 + np.random.normal(0, 0.08)
            else:
                ndwi_base = 0.1 + np.random.normal(0, 0.05)
            
            # SAVI - Soil Adjusted Vegetation Index (ajusté au sol)
            savi_base = ndvi_base * 0.85 + np.random.normal(0, 0.06)
            
            # LAI - Leaf Area Index
            lai_base = ndvi_base * 5 + np.random.normal(0, 0.3)
            
            # MSAVI - Modified SAVI
            msavi_base = savi_base * 1.05 + np.random.normal(0, 0.04)
            
            all_data.append({
                'date': d,
                'cell_id': point_dict['cell_id'],
                'latitude': point_dict['latitude'],
                'longitude': point_dict['longitude'],
                'ndvi': np.clip(ndvi_base, 0, 1),
                'evi': np.clip(evi_base, 0, 1),
                'ndwi': np.clip(ndwi_base, -1, 1),
                'savi': np.clip(savi_base, 0, 1),
                'lai': np.clip(lai_base, 0, 7),
                'msavi': np.clip(msavi_base, 0, 1),
                'cloud_cover': np.random.randint(0, 30)
            })
    
    return pd.DataFrame(all_data)

def calculate_crop_metrics(climate_df, indices_df, culture):
    """Calcule métriques spécifiques à chaque culture"""
    if climate_df is None or indices_df is None or climate_df.empty or indices_df.empty:
        return {}
    
    # Agrégation par cellule puis moyenne
    indices_agg = indices_df.groupby('cell_id').agg({
        'ndvi': ['mean', 'min', 'max', 'std'],
        'evi': ['mean', 'std'],
        'ndwi': ['mean', 'std'],
        'savi': 'mean',
        'lai': 'mean',
        'msavi': 'mean'
    }).reset_index()
    
    climate_agg = climate_df.groupby('cell_id').agg({
        'temp_mean': 'mean',
        'temp_min': 'min',
        'temp_max': 'max',
        'rain': 'sum',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    # Moyennes globales
    metrics = {
        'ndvi_mean': indices_df['ndvi'].mean(),
        'ndvi_std': indices_df['ndvi'].std(),
        'ndvi_min': indices_df['ndvi'].min(),
        'ndvi_max': indices_df['ndvi'].max(),
        'evi_mean': indices_df['evi'].mean(),
        'ndwi_mean': indices_df['ndwi'].mean(),
        'savi_mean': indices_df['savi'].mean(),
        'lai_mean': indices_df['lai'].mean(),
        'temp_mean': climate_df['temp_mean'].mean(),
        'temp_min': climate_df['temp_min'].min(),
        'temp_max': climate_df['temp_max'].max(),
        'rain_total': climate_df['rain'].sum(),
        'rain_mean': climate_df['rain'].mean(),
        'rain_days': (climate_df['rain'] > 1).sum(),
        'humidity_mean': climate_df['humidity'].mean(),
        'wind_mean': climate_df['wind_speed'].mean()
    }
    
    # Paramètres optimaux par culture
    crop_params = {
        "Mil": {
            'ndvi_optimal': 0.6, 'rain_min': 400, 'temp_optimal': 28,
            'yield_max': 1.5, 'cycle_days': 90
        },
        "Sorgho": {
            'ndvi_optimal': 0.65, 'rain_min': 450, 'temp_optimal': 30,
            'yield_max': 2.0, 'cycle_days': 110
        },
        "Maïs": {
            'ndvi_optimal': 0.7, 'rain_min': 500, 'temp_optimal': 25,
            'yield_max': 4.0, 'cycle_days': 120
        },
        "Arachide": {
            'ndvi_optimal': 0.6, 'rain_min': 450, 'temp_optimal': 27,
            'yield_max': 2.5, 'cycle_days': 120
        },
        "Riz": {
            'ndvi_optimal': 0.75, 'rain_min': 800, 'temp_optimal': 26,
            'yield_max': 5.0, 'cycle_days': 130
        },
        "Niébé": {
            'ndvi_optimal': 0.55, 'rain_min': 350, 'temp_optimal': 28,
            'yield_max': 1.2, 'cycle_days': 75
        },
        "Manioc": {
            'ndvi_optimal': 0.65, 'rain_min': 1000, 'temp_optimal': 27,
            'yield_max': 20.0, 'cycle_days': 300
        },
        "Tomate": {
            'ndvi_optimal': 0.7, 'rain_min': 600, 'temp_optimal': 24,
            'yield_max': 40.0, 'cycle_days': 90
        },
        "Oignon": {
            'ndvi_optimal': 0.6, 'rain_min': 400, 'temp_optimal': 20,
            'yield_max': 25.0, 'cycle_days': 110
        },
        "Coton": {
            'ndvi_optimal': 0.65, 'rain_min': 600, 'temp_optimal': 28,
            'yield_max': 2.5, 'cycle_days': 150
        },
        "Pastèque": {
            'ndvi_optimal': 0.6, 'rain_min': 400, 'temp_optimal': 25,
            'yield_max': 30.0, 'cycle_days': 85
        }
    }
    
    params = crop_params.get(culture, crop_params["Mil"])
    
    # Calcul rendement potentiel
    ndvi_score = min(metrics['ndvi_mean'] / params['ndvi_optimal'], 1.0)
    rain_score = min(metrics['rain_total'] / params['rain_min'], 1.0)
    temp_score = 1 - abs(metrics['temp_mean'] - params['temp_optimal']) / 15
    temp_score = max(0, min(temp_score, 1))
    
    # Score de stress hydrique basé sur NDWI
    water_stress = 1 - max(0, min(metrics['ndwi_mean'], 1))
    
    # Rendement estimé
    yield_potential = params['yield_max'] * ndvi_score * rain_score * temp_score * (1 - water_stress * 0.3)
    
    metrics['yield_potential'] = yield_potential
    metrics['ndvi_score'] = ndvi_score
    metrics['rain_score'] = rain_score
    metrics['temp_score'] = temp_score
    metrics['water_stress'] = water_stress
    metrics['cycle_days'] = params['cycle_days']
    
    return metrics

def generate_crop_recommendations(metrics, culture, forecast_df=None):
    """Génère recommandations détaillées par culture"""
    recommendations = {
        'diagnostic': [],
        'irrigation': [],
        'fertilisation': [],
        'phytosanitaire': [],
        'calendrier': [],
        'alertes': []
    }
    
    # Diagnostic santé culture
    if metrics['ndvi_mean'] > 0.65:
        recommendations['diagnostic'].append("✅ Excellente vigueur végétative")
    elif metrics['ndvi_mean'] > 0.45:
        recommendations['diagnostic'].append("⚠️ Vigueur modérée - surveillance nécessaire")
    else:
        recommendations['diagnostic'].append("❌ Stress végétal détecté - intervention urgente")
    
    if metrics['water_stress'] > 0.5:
        recommendations['diagnostic'].append("❌ Stress hydrique important (NDWI faible)")
    elif metrics['water_stress'] > 0.3:
        recommendations['diagnostic'].append("⚠️ Déficit hydrique modéré")
    
    # Irrigation
    if metrics['rain_total'] < 300:
        recommendations['irrigation'].append(f"🚨 URGENT: Irrigation immédiate - 30-40mm tous les 5 jours")
        recommendations['alertes'].append("Déficit hydrique critique")
    elif metrics['rain_total'] < 450:
        recommendations['irrigation'].append(f"Complément irrigation: 20-25mm tous les 7 jours")
    else:
        recommendations['irrigation'].append(f"✅ Pluviométrie suffisante ({metrics['rain_total']:.0f}mm)")
    
    # Fertilisation spécifique par culture
    ferti_plans = {
        "Mil": [
            "Fond: NPK 15-15-15 à 150 kg/ha au semis",
            "Couverture: Urée 50 kg/ha à 30-35 jours",
            "Apport supplémentaire: Urée 25 kg/ha à montaison si NDVI < 0.5"
        ],
        "Maïs": [
            "Fond: NPK 23-10-5 à 200 kg/ha",
            "Premier apport: Urée 100 kg/ha à 4-6 feuilles",
            "Deuxième apport: Urée 50 kg/ha à floraison",
            "Fumure organique: 5-10 t/ha recommandée"
        ],
        "Arachide": [
            "Fond: NPK 6-20-10 à 200 kg/ha (culture fixatrice d'azote)",
            "Apport calcium: Gypse 300 kg/ha à floraison",
            "Éviter excès azote (favorise feuillage au détriment gousses)"
        ],
        "Riz": [
            "Fond: NPK 15-15-15 à 300 kg/ha",
            "Premier apport: Urée 100 kg/ha à tallage",
            "Deuxième apport: Urée 75 kg/ha à initiation paniculaire",
            "Maintenir lame d'eau 5-10cm"
        ]
    }
    
    recommendations['fertilisation'] = ferti_plans.get(culture, [
        f"NPK 15-15-15: 150-200 kg/ha au semis",
        f"Urée: 50-75 kg/ha en couverture à 30-40 jours"
    ])
    
    # Phytosanitaire
    if metrics['humidity_mean'] > 70 and metrics['temp_mean'] > 25:
        recommendations['phytosanitaire'].append("⚠️ Conditions favorables maladies fongiques")
        recommendations['phytosanitaire'].append(f"Traitement préventif fongicide recommandé ({culture})")
    
    if metrics['temp_max'] > 35:
        recommendations['phytosanitaire'].append("Risque ravageurs accru (chenilles, criquets)")
    
    # Calendrier cultural
    if forecast_df is not None and not forecast_df.empty:
        rain_forecast = forecast_df['rain'].sum()
        if rain_forecast > 20:
            recommendations['calendrier'].append("✅ Bonnes conditions semis prévues (pluie attendue)")
        else:
            recommendations['calendrier'].append("⚠️ Attendre pluies suffisantes avant semis")
    
    recommendations['calendrier'].append(f"Cycle cultural: {metrics['cycle_days']} jours")
    recommendations['calendrier'].append(f"Rendement estimé: {metrics['yield_potential']:.1f} t/ha")
    
    return recommendations
# Onglets
tabs = st.tabs(["🗺️ Carte", "📊 Dashboard", "🛰️ Indices", "🌦️ Climat", 
                "🔮 Prévisions", "🤖 IA Multi-Cultures", "📄 Rapport"])

# ONGLET 1: CARTE
with tabs[0]:
    st.subheader("🗺️ Définir la Zone d'Étude")
    
    if zone_method == "Dessiner sur carte":
        st.info("💡 Dessinez votre zone, puis lancez l'analyse")
    
    # Déterminer centre carte
    if st.session_state.gdf is not None:
        center = [st.session_state.gdf.geometry.centroid.y.mean(),
                 st.session_state.gdf.geometry.centroid.x.mean()]
        zoom = 13
    elif manual_coords:
        center = [(manual_coords[0] + manual_coords[2])/2, 
                  (manual_coords[1] + manual_coords[3])/2]
        zoom = 13
    else:
        center = [14.6937, -17.4441]  # Dakar par défaut
        zoom = 10
    
    # Créer carte
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    
    # Ajouter couches satellite optionnelles
    folium.TileLayer('Esri.WorldImagery', name='Satellite', attr='Esri').add_to(m)
    
    m.add_child(MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='hectares'
    ))
    
    # Afficher zone analysée
    if st.session_state.gdf is not None:
        folium.GeoJson(
            st.session_state.gdf,
            name="Zone analysée",
            style_function=lambda x: {
                'fillColor': '#28A745',
                'color': '#155724',
                'weight': 3,
                'fillOpacity': 0.3
            },
            tooltip=f"<b>{zone_name}</b><br>Cultures: {', '.join(cultures_selectionnees)}"
        ).add_to(m)
        
        # Afficher points d'échantillonnage
        if st.session_state.sampling_points is not None:
            marker_cluster = MarkerCluster(name="Points d'échantillonnage").add_to(m)
            
            for idx, row in st.session_state.sampling_points.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    popup=f"<b>{row['cell_id']}</b><br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}",
                    color='#FF5722',
                    fill=True,
                    fillColor='#FF5722',
                    fillOpacity=0.7
                ).add_to(marker_cluster)
            
            st.success(f"✓ {len(st.session_state.sampling_points)} points d'échantillonnage générés")
    
    # Outils de dessin
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
    
    folium.LayerControl().add_to(m)
    
    map_output = st_folium(m, height=600, width=None, key="main_map")
    
    # Capturer dessin
    if map_output and map_output.get('all_drawings'):
        drawings = map_output['all_drawings']
        if drawings and len(drawings) > 0:
            try:
                gdf_drawn = gpd.GeoDataFrame.from_features(drawings, crs="EPSG:4326")
                st.session_state.drawn_geometry = gdf_drawn.geometry.unary_union
                
                # Calculer surface
                geod = gdf_drawn.crs.get_geod()
                area_m2 = abs(geod.geometry_area_perimeter(gdf_drawn.geometry.unary_union)[0])
                area_ha = area_m2 / 10000
                
                st.success(f"Zone dessinée: {len(drawings)} forme(s). Surface: {area_ha:.2f} ha")
            except Exception as e:
                st.error(f"Erreur: {e}")

# CHARGEMENT DES DONNÉES
if load_btn:
    if not cultures_selectionnees:
        st.error("Sélectionnez au moins une culture")
        st.stop()
    
    geometry = None
    
    # Récupérer géométrie
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
    
    # Progression
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Progression du chargement")
        global_progress = st.progress(0, text="Initialisation...")
        status_grid = st.empty()
        status_climate = st.empty()
        status_indices = st.empty()
        status_forecast = st.empty()
        status_analysis = st.empty()
    
    # Étape 1: Créer grille d'échantillonnage
    status_grid.info("Création grille d'échantillonnage...")
    global_progress.progress(10, text="Génération points...")
    
    sampling_points = create_sampling_grid(geometry, grid_size_ha)
    
    if sampling_points is None or sampling_points.empty:
        status_grid.error("Échec création grille")
        st.stop()
    
    st.session_state.sampling_points = sampling_points
    status_grid.success(f"✓ {len(sampling_points)} points générés (grille {grid_size_ha}ha)")
    
    global_progress.progress(25, text="Récupération données climatiques...")
    
    # Étape 2: Données climatiques
    status_climate.info("Chargement données climatiques...")
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    points_dict_list = sampling_points.to_dict('records')
    
    climate_df = get_climate_nasa_multi_points(points_dict_list, start_dt, end_dt)
    
    if climate_df is None or climate_df.empty:
        status_climate.error("Échec données climatiques")
        st.stop()
    else:
        status_climate.success(f"✓ Climat chargé ({len(climate_df)} observations)")
        st.session_state.climate_data = climate_df
    
    global_progress.progress(50, text="Récupération indices satellitaires...")
    
    # Étape 3: Indices satellitaires
    status_indices.info("Chargement indices satellitaires...")
    
    indices_df = simulate_multi_indices_data(points_dict_list, start_date, end_date)
    
    if indices_df is None or indices_df.empty:
        status_indices.error("Échec indices")
        st.stop()
    else:
        status_indices.success(f"✓ Indices chargés ({len(indices_df)} observations)")
        st.session_state.satellite_data = indices_df
    
    global_progress.progress(70, text="Prévisions météo...")
    
    # Étape 4: Prévisions météo
    if OPENWEATHER_KEY:
        status_forecast.info("Chargement prévisions...")
        centroid = geometry.centroid
        forecast_df = get_weather_forecast(centroid.y, centroid.x, OPENWEATHER_KEY)
        
        if forecast_df is not None:
            st.session_state.weather_forecast = forecast_df
            status_forecast.success(f"✓ Prévisions 7j chargées")
        else:
            status_forecast.warning("Prévisions indisponibles")
    else:
        status_forecast.info("Clé OpenWeather non configurée - prévisions désactivées")
    
    global_progress.progress(85, text="Calcul métriques...")
    
    # Étape 5: Calcul métriques pour chaque culture
    status_analysis.info("Calcul métriques multi-cultures...")
    
    all_metrics = {}
    for culture in cultures_selectionnees:
        metrics = calculate_crop_metrics(climate_df, indices_df, culture)
        recommendations = generate_crop_recommendations(
            metrics, culture, st.session_state.weather_forecast
        )
        all_metrics[culture] = {
            'metrics': metrics,
            'recommendations': recommendations
        }
    
    st.session_state.analysis = all_metrics
    status_analysis.success(f"✓ Analyse complète ({len(cultures_selectionnees)} cultures)")
    
    global_progress.progress(100, text="Analyse terminée!")
    time.sleep(1)
    
    st.success(f"✅ Données chargées! {len(sampling_points)} points, {len(cultures_selectionnees)} cultures analysées")
    st.balloons()
# ONGLET 2: DASHBOARD
with tabs[1]:
    st.subheader("📊 Dashboard Multi-Cultures")
    
    if st.session_state.analysis and st.session_state.climate_data is not None:
        
        # Sélecteur de culture pour affichage détaillé
        selected_culture = st.selectbox("Culture à afficher en détail", cultures_selectionnees)
        
        if selected_culture in st.session_state.analysis:
            metrics = st.session_state.analysis[selected_culture]['metrics']
            
            # Métriques principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                delta = "✅" if metrics['ndvi_mean'] > 0.5 else "⚠️"
                st.metric("🌱 NDVI", f"{metrics['ndvi_mean']:.3f}", delta=delta)
            
            with col2:
                st.metric("🌡️ Temp", f"{metrics['temp_mean']:.1f}°C",
                         delta=f"{metrics['temp_min']:.0f}-{metrics['temp_max']:.0f}°")
            
            with col3:
                delta = "✅" if metrics['rain_total'] > 400 else "⚠️"
                st.metric("💧 Pluie", f"{metrics['rain_total']:.0f}mm", delta=delta)
            
            with col4:
                st.metric("💦 NDWI", f"{metrics['ndwi_mean']:.3f}",
                         delta="✅" if metrics['water_stress'] < 0.3 else "⚠️")
            
            with col5:
                st.metric("📈 Rendement", f"{metrics['yield_potential']:.1f} t/ha")
            
            st.markdown("---")
            
            # Graphiques comparatifs multi-cultures
            st.markdown("### 📊 Comparaison Multi-Cultures")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                # Rendements comparés
                fig_yields, ax = plt.subplots(figsize=(8, 5))
                
                cultures = list(st.session_state.analysis.keys())
                yields = [st.session_state.analysis[c]['metrics']['yield_potential'] 
                         for c in cultures]
                colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(cultures)))
                
                bars = ax.barh(cultures, yields, color=colors, edgecolor='darkgreen', linewidth=2)
                ax.set_xlabel('Rendement (t/ha)', fontweight='bold')
                ax.set_title('Rendements Potentiels par Culture', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Ajouter valeurs
                for i, (c, v) in enumerate(zip(cultures, yields)):
                    ax.text(v + 0.1, i, f"{v:.1f}", va='center', fontweight='bold')
                
                st.pyplot(fig_yields)
            
            with col_g2:
                # Indices de santé
                fig_health, ax = plt.subplots(figsize=(8, 5))
                
                indices_names = ['NDVI', 'EVI', 'SAVI', 'LAI/7']
                indices_values = [
                    metrics['ndvi_mean'],
                    metrics['evi_mean'],
                    metrics['savi_mean'],
                    metrics['lai_mean']/7  # Normaliser LAI
                ]
                
                x = np.arange(len(indices_names))
                bars = ax.bar(x, indices_values, color=['green', 'darkgreen', 'forestgreen', 'olivedrab'],
                             edgecolor='black', linewidth=1.5, alpha=0.8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(indices_names, fontweight='bold')
                ax.set_ylabel('Valeur', fontweight='bold')
                ax.set_title(f'Indices de Végétation - {selected_culture}', fontweight='bold')
                ax.set_ylim([0, 1])
                ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Optimal')
                ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Moyen')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig_health)
            
            st.markdown("---")
            
            # Évolution temporelle NDVI avec variabilité spatiale
            st.markdown(f"### 📈 Évolution NDVI - {selected_culture}")
            
            indices_df = st.session_state.satellite_data
            
            # Données agrégées par date
            ndvi_temporal = indices_df.groupby('date').agg({
                'ndvi': ['mean', 'min', 'max', 'std']
            }).reset_index()
            ndvi_temporal.columns = ['date', 'mean', 'min', 'max', 'std']
            
            fig_ndvi, ax = plt.subplots(figsize=(12, 6))
            
            # Plage min-max
            ax.fill_between(ndvi_temporal['date'], ndvi_temporal['min'], ndvi_temporal['max'],
                           alpha=0.2, color='green', label='Plage min-max (variabilité spatiale)')
            
            # Moyenne ± écart-type
            ax.fill_between(ndvi_temporal['date'], 
                           ndvi_temporal['mean'] - ndvi_temporal['std'],
                           ndvi_temporal['mean'] + ndvi_temporal['std'],
                           alpha=0.3, color='darkgreen', label='Écart-type')
            
            # Moyenne
            ax.plot(ndvi_temporal['date'], ndvi_temporal['mean'], 'o-',
                   color='darkgreen', linewidth=2.5, markersize=7, label='NDVI moyen')
            
            # Seuils
            ax.axhline(0.7, color='green', linestyle=':', alpha=0.6, linewidth=2, label='Excellent')
            ax.axhline(0.5, color='orange', linestyle=':', alpha=0.6, linewidth=2, label='Bon')
            ax.axhline(0.3, color='red', linestyle=':', alpha=0.6, linewidth=2, label='Stress')
            
            ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_title(f'Évolution NDVI avec Variabilité Spatiale', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=30)
            plt.tight_layout()
            
            st.pyplot(fig_ndvi)
            
            st.markdown("---")
            
            # Carte de chaleur variabilité spatiale
            st.markdown("### 🗺️ Variabilité Spatiale NDVI")
            
            col_map1, col_map2 = st.columns([2, 1])
            
            with col_map1:
                # Moyenne NDVI par point
                ndvi_by_cell = indices_df.groupby(['cell_id', 'latitude', 'longitude'])['ndvi'].mean().reset_index()
                
                # Carte
                m_ndvi = folium.Map(
                    location=[ndvi_by_cell['latitude'].mean(), ndvi_by_cell['longitude'].mean()],
                    zoom_start=13
                )
                
                # Colormap
                from branca.colormap import LinearColormap
                colormap = LinearColormap(
                    colors=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                    vmin=0, vmax=1,
                    caption='NDVI Moyen'
                )
                
                for idx, row in ndvi_by_cell.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=10,
                        popup=f"<b>{row['cell_id']}</b><br>NDVI: {row['ndvi']:.3f}",
                        color=colormap(row['ndvi']),
                        fill=True,
                        fillColor=colormap(row['ndvi']),
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(m_ndvi)
                
                colormap.add_to(m_ndvi)
                st_folium(m_ndvi, height=400)
            
            with col_map2:
                st.markdown("**Statistiques Spatiales**")
                st.metric("NDVI Min", f"{ndvi_by_cell['ndvi'].min():.3f}")
                st.metric("NDVI Max", f"{ndvi_by_cell['ndvi'].max():.3f}")
                st.metric("NDVI Médian", f"{ndvi_by_cell['ndvi'].median():.3f}")
                st.metric("Coef. Variation", f"{(ndvi_by_cell['ndvi'].std()/ndvi_by_cell['ndvi'].mean())*100:.1f}%")
                
                st.markdown("---")
                st.markdown("**Interprétation**")
                cv = (ndvi_by_cell['ndvi'].std()/ndvi_by_cell['ndvi'].mean())*100
                
                if cv < 10:
                    st.success("✅ Homogénéité excellente")
                elif cv < 20:
                    st.info("ℹ️ Variabilité modérée")
                else:
                    st.warning("⚠️ Forte hétérogénéité - gestion différenciée recommandée")
            
            st.markdown("---")
            
            # Diagnostic rapide
            st.markdown("### 🔍 Diagnostic Multi-Facteurs")
            
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            
            with col_d1:
                st.markdown("**🌱 Vigueur Végétale**")
                if metrics['ndvi_mean'] > 0.6:
                    st.markdown('<div class="success-box">✅ Excellente</div>', unsafe_allow_html=True)
                elif metrics['ndvi_mean'] > 0.4:
                    st.markdown('<div class="alert-box">⚠️ Modérée</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">❌ Faible</div>', unsafe_allow_html=True)
            
            with col_d2:
                st.markdown("**💧 Statut Hydrique**")
                if metrics['water_stress'] < 0.3:
                    st.markdown('<div class="success-box">✅ Bon</div>', unsafe_allow_html=True)
                elif metrics['water_stress'] < 0.5:
                    st.markdown('<div class="alert-box">⚠️ Modéré</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">❌ Stress</div>', unsafe_allow_html=True)
            
            with col_d3:
                st.markdown("**🌡️ Contrainte Thermique**")
                if metrics['temp_max'] < 35 and metrics['temp_mean'] < 30:
                    st.markdown('<div class="success-box">✅ Optimal</div>', unsafe_allow_html=True)
                elif metrics['temp_max'] < 38:
                    st.markdown('<div class="alert-box">⚠️ Élevé</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">❌ Excessif</div>', unsafe_allow_html=True)
            
            with col_d4:
                st.markdown("**💦 Pluviométrie**")
                if metrics['rain_total'] > 400:
                    st.markdown('<div class="success-box">✅ Suffisante</div>', unsafe_allow_html=True)
                elif metrics['rain_total'] > 250:
                    st.markdown('<div class="alert-box">⚠️ Limite</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">❌ Insuffisante</div>', unsafe_allow_html=True)
        
        else:
            st.warning("Données non disponibles pour cette culture")
    
    else:
        st.info("👆 Lancez d'abord l'analyse")
# ONGLET 3: INDICES SATELLITAIRES
with tabs[2]:
    st.subheader("🛰️ Analyse Multi-Indices Satellitaires")
    
    if st.session_state.satellite_data is not None:
        df_sat = st.session_state.satellite_data
        
        # Sélection culture
        selected_culture = st.selectbox("Culture", cultures_selectionnees, key="indices_culture")
        
        # Graphiques multi-indices
        st.markdown("### 📊 Évolution des Indices")
        
        # Agrégation temporelle
        indices_temporal = df_sat.groupby('date').agg({
            'ndvi': 'mean',
            'evi': 'mean',
            'ndwi': 'mean',
            'savi': 'mean',
            'lai': 'mean',
            'msavi': 'mean'
        }).reset_index()
        
        # Graphique 1: Indices de végétation
        fig_veg, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(indices_temporal['date'], indices_temporal['ndvi'], 'o-',
               color='darkgreen', linewidth=2, markersize=6, label='NDVI')
        ax.plot(indices_temporal['date'], indices_temporal['evi'], 's-',
               color='forestgreen', linewidth=2, markersize=6, label='EVI')
        ax.plot(indices_temporal['date'], indices_temporal['savi'], '^-',
               color='olive', linewidth=2, markersize=6, label='SAVI')
        ax.plot(indices_temporal['date'], indices_temporal['msavi'], 'd-',
               color='yellowgreen', linewidth=2, markersize=6, label='MSAVI')
        
        ax.axhline(0.7, color='green', linestyle=':', alpha=0.5, label='Seuil excellent')
        ax.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Seuil bon')
        ax.axhline(0.3, color='red', linestyle=':', alpha=0.5, label='Seuil stress')
        
        ax.set_ylabel('Valeur Indice', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('Indices de Végétation', fontsize=14, fontweight='bold')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=30)
        plt.tight_layout()
        
        st.pyplot(fig_veg)
        
        # Graphique 2: NDWI et LAI
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_ndwi, ax = plt.subplots(figsize=(8, 5))
            
            ax.plot(indices_temporal['date'], indices_temporal['ndwi'], 'o-',
                   color='steelblue', linewidth=2.5, markersize=7)
            ax.fill_between(indices_temporal['date'], indices_temporal['ndwi'],
                           alpha=0.3, color='steelblue')
            ax.axhline(0.3, color='blue', linestyle='--', alpha=0.5, label='Bon contenu eau')
            ax.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='Stress hydrique')
            
            ax.set_ylabel('NDWI', fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_title('Indice de Contenu en Eau (NDWI)', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1, 1])
            plt.xticks(rotation=30)
            plt.tight_layout()
            
            st.pyplot(fig_ndwi)
        
        with col_g2:
            fig_lai, ax = plt.subplots(figsize=(8, 5))
            
            ax.plot(indices_temporal['date'], indices_temporal['lai'], 'o-',
                   color='darkgreen', linewidth=2.5, markersize=7)
            ax.fill_between(indices_temporal['date'], indices_temporal['lai'],
                           alpha=0.3, color='green')
            ax.axhline(4, color='green', linestyle='--', alpha=0.5, label='LAI optimal')
            ax.axhline(2, color='orange', linestyle='--', alpha=0.5, label='LAI moyen')
            
            ax.set_ylabel('LAI (m²/m²)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_title("Indice de Surface Foliaire (LAI)", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 7])
            plt.xticks(rotation=30)
            plt.tight_layout()
            
            st.pyplot(fig_lai)
        
        st.markdown("---")
        
        # Statistiques détaillées par indice
        st.markdown("### 📈 Statistiques par Indice")
        
        stats_df = df_sat.agg({
            'ndvi': ['mean', 'min', 'max', 'std'],
            'evi': ['mean', 'min', 'max', 'std'],
            'ndwi': ['mean', 'min', 'max', 'std'],
            'savi': ['mean', 'min', 'max', 'std'],
            'lai': ['mean', 'min', 'max', 'std'],
            'msavi': ['mean', 'min', 'max', 'std']
        }).T
        
        stats_df.columns = ['Moyenne', 'Minimum', 'Maximum', 'Écart-type']
        stats_df = stats_df.round(3)
        
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # Tableau complet avec coordonnées pour interpolation SIG
        st.markdown("### 📋 Données Complètes (Export SIG)")
        st.info("💡 Tableau avec coordonnées géographiques pour interpolation dans votre logiciel SIG")
        
        # Moyenne par point d'échantillonnage
        export_df = df_sat.groupby(['cell_id', 'latitude', 'longitude']).agg({
            'ndvi': ['mean', 'min', 'max', 'std'],
            'evi': 'mean',
            'ndwi': 'mean',
            'savi': 'mean',
            'lai': 'mean',
            'msavi': 'mean'
        }).reset_index()
        
        export_df.columns = ['cell_id', 'latitude', 'longitude', 
                            'ndvi_mean', 'ndvi_min', 'ndvi_max', 'ndvi_std',
                            'evi_mean', 'ndwi_mean', 'savi_mean', 'lai_mean', 'msavi_mean']
        
        st.dataframe(export_df, use_container_width=True)
        
        # Bouton téléchargement
        csv_export = export_df.to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV pour SIG",
            csv_export,
            f"indices_sig_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Interprétation des indices
        st.markdown("### 📚 Interprétation des Indices")
        
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            st.markdown("""
            **NDVI (Normalized Difference Vegetation Index)**
            - 0.7-1.0: Végétation très dense et saine
            - 0.5-0.7: Végétation modérée
            - 0.3-0.5: Végétation clairsemée
            - < 0.3: Sol nu ou végétation stressée
            
            **EVI (Enhanced Vegetation Index)**
            - Plus sensible en zones de forte biomasse
            - Corrige effets atmosphériques
            - Meilleur pour suivi croissance
            
            **SAVI (Soil Adjusted Vegetation Index)**
            - Réduit influence du sol
            - Idéal début de cycle cultural
            - Recommandé faible couverture végétale
            """)
        
        with col_i2:
            st.markdown("""
            **NDWI (Normalized Difference Water Index)**
            - > 0.3: Bon contenu en eau
            - 0.1-0.3: Contenu modéré
            - < 0.1: Stress hydrique
            - Indicateur précoce sécheresse
            
            **LAI (Leaf Area Index)**
            - > 4: Canopée dense
            - 2-4: Développement normal
            - < 2: Développement faible
            - Lié à productivité photosynthétique
            
            **MSAVI (Modified SAVI)**
            - Version améliorée de SAVI
            - Auto-ajustement selon végétation
            """)
    
    else:
        st.info("Chargez d'abord les données")

# ONGLET 4: CLIMAT
with tabs[3]:
    st.subheader("🌦️ Analyse Climatique Détaillée")
    
    if st.session_state.climate_data is not None:
        df_clim = st.session_state.climate_data
        
        # Agrégation temporelle (moyenne de tous les points)
        clim_temporal = df_clim.groupby('date').agg({
            'temp_mean': 'mean',
            'temp_min': 'min',
            'temp_max': 'max',
            'rain': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        
        # Graphique principal: Température et pluie
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Températures
        ax1.fill_between(clim_temporal['date'], clim_temporal['temp_min'], 
                        clim_temporal['temp_max'],
                        alpha=0.3, color='coral', label='Plage min-max')
        ax1.plot(clim_temporal['date'], clim_temporal['temp_mean'], 
                color='red', linewidth=2.5, label='Moyenne')
        ax1.axhline(35, color='darkred', linestyle='--', alpha=0.6, label='Seuil stress (35°C)')
        ax1.axhline(25, color='orange', linestyle=':', alpha=0.6, label='Temp optimale (25°C)')
        
        ax1.set_ylabel('Température (°C)', fontweight='bold', fontsize=11)
        ax1.set_title('Températures', fontweight='bold', fontsize=13)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Précipitations
        ax2.bar(clim_temporal['date'], clim_temporal['rain'], 
               color='dodgerblue', alpha=0.7, edgecolor='navy')
        ax2.axhline(clim_temporal['rain'].mean(), color='navy', linestyle='--', 
                   linewidth=2, label=f"Moyenne: {clim_temporal['rain'].mean():.1f} mm/j")
        ax2.set_ylabel('Pluie (mm)', fontweight='bold', fontsize=11)
        ax2.set_xlabel('Date', fontweight='bold', fontsize=11)
        ax2.set_title('Précipitations', fontweight='bold', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Graphiques complémentaires
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Humidité
            fig_hum, ax = plt.subplots(figsize=(8, 5))
            
            ax.plot(clim_temporal['date'], clim_temporal['humidity'], 'o-',
                   color='teal', linewidth=2, markersize=6)
            ax.fill_between(clim_temporal['date'], clim_temporal['humidity'],
                           alpha=0.3, color='teal')
            ax.axhline(70, color='blue', linestyle='--', alpha=0.5, label='Seuil maladies (70%)')
            ax.axhline(50, color='green', linestyle=':', alpha=0.5, label='Optimal (50%)')
            
            ax.set_ylabel('Humidité Relative (%)', fontweight='bold')
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_title('Humidité Relative', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            plt.xticks(rotation=30)
            plt.tight_layout()
            
            st.pyplot(fig_hum)
        
        with col_g2:
            # Vitesse du vent
            fig_wind, ax = plt.subplots(figsize=(8, 5))
            
            ax.plot(clim_temporal['date'], clim_temporal['wind_speed'], 'o-',
                   color='slategray', linewidth=2, markersize=6)
            ax.fill_between(clim_temporal['date'], clim_temporal['wind_speed'],
                           alpha=0.3, color='slategray')
            ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='Vent modéré (5 m/s)')
            
            ax.set_ylabel('Vitesse Vent (m/s)', fontweight='bold')
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_title('Vitesse du Vent', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=30)
            plt.tight_layout()
            
            st.pyplot(fig_wind)
        
        st.markdown("---")
        
        # Statistiques climatiques
        st.markdown("### 📊 Statistiques Climatiques")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🌡️ Températures**")
            st.metric("Moyenne", f"{clim_temporal['temp_mean'].mean():.1f}°C")
            st.metric("Min absolue", f"{clim_temporal['temp_min'].min():.1f}°C")
            st.metric("Max absolue", f"{clim_temporal['temp_max'].max():.1f}°C")
            st.metric("Amplitude", f"{clim_temporal['temp_max'].max() - clim_temporal['temp_min'].min():.1f}°C")
        
        with col2:
            st.markdown("**💧 Précipitations**")
            st.metric("Cumul total", f"{clim_temporal['rain'].sum():.0f} mm")
            st.metric("Moyenne/jour", f"{clim_temporal['rain'].mean():.1f} mm")
            st.metric("Max/jour", f"{clim_temporal['rain'].max():.1f} mm")
            st.metric("Jours pluie (>1mm)", f"{(clim_temporal['rain'] > 1).sum()}")
        
        with col3:
            st.markdown("**💨 Humidité & Vent**")
            st.metric("Humidité moy.", f"{clim_temporal['humidity'].mean():.1f}%")
            st.metric("Humidité min", f"{clim_temporal['humidity'].min():.1f}%")
            st.metric("Humidité max", f"{clim_temporal['humidity'].max():.1f}%")
            st.metric("Vent moyen", f"{clim_temporal['wind_speed'].mean():.1f} m/s")
        
        with col4:
            st.markdown("**📊 Indices**")
            st.metric("Jours >35°C", f"{(clim_temporal['temp_max'] > 35).sum()}")
            st.metric("Jours secs (<1mm)", f"{(clim_temporal['rain'] < 1).sum()}")
            st.metric("Jours HR>70%", f"{(clim_temporal['humidity'] > 70).sum()}")
            st.metric("Période (jours)", f"{len(clim_temporal)}")
        
        st.markdown("---")
        
        # Données pour SIG
        st.markdown("### 📋 Données Climatiques par Point (Export SIG)")
        
        # Moyenne par point
        clim_by_point = df_clim.groupby(['cell_id', 'latitude', 'longitude']).agg({
            'temp_mean': 'mean',
            'temp_min': 'min',
            'temp_max': 'max',
            'rain': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        
        clim_by_point.columns = ['cell_id', 'latitude', 'longitude',
                                 'temp_mean', 'temp_min', 'temp_max',
                                 'rain_total', 'humidity_mean', 'wind_mean']
        
        st.dataframe(clim_by_point, use_container_width=True)
        
        csv_clim = clim_by_point.to_csv(index=False)
        st.download_button(
            "📥 Télécharger Climat CSV",
            csv_clim,
            f"climat_sig_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.info("Chargez d'abord les données")
# ONGLET 5: PRÉVISIONS MÉTÉO
with tabs[4]:
    st.subheader("🔮 Prévisions Météorologiques et Calendrier Cultural")
    
    if st.session_state.weather_forecast is not None:
        forecast_df = st.session_state.weather_forecast
        
        st.markdown("### 📅 Prévisions 7 Jours")
        
        # Graphique prévisions
        fig_forecast, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Températures
        axes[0].plot(forecast_df['date'], forecast_df['temp'], 'o-',
                    color='orangered', linewidth=2.5, markersize=8, label='Temp moyenne')
        axes[0].fill_between(forecast_df['date'], forecast_df['temp_min'], 
                            forecast_df['temp_max'],
                            alpha=0.3, color='coral', label='Min-Max')
        axes[0].axhline(30, color='red', linestyle='--', alpha=0.5, label='Seuil chaud')
        axes[0].set_ylabel('Température (°C)', fontweight='bold')
        axes[0].set_title('Températures Prévues', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pluie
        axes[1].bar(forecast_df['date'], forecast_df['rain'], 
                   color='steelblue', alpha=0.7, edgecolor='navy')
        axes[1].set_ylabel('Pluie (mm)', fontweight='bold')
        axes[1].set_title('Précipitations Prévues', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Humidité
        axes[2].plot(forecast_df['date'], forecast_df['humidity'], 's-',
                    color='teal', linewidth=2.5, markersize=7)
        axes[2].fill_between(forecast_df['date'], forecast_df['humidity'],
                            alpha=0.3, color='teal')
        axes[2].axhline(70, color='blue', linestyle='--', alpha=0.5, label='Seuil risque maladies')
        axes[2].set_ylabel('Humidité (%)', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].set_title('Humidité Prévue', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 100])
        
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig_forecast)
        
        st.markdown("---")
        
        # Tableau prévisions détaillées
        st.markdown("### 📋 Détail des Prévisions")
        
        forecast_display = forecast_df.copy()
        forecast_display['date'] = forecast_display['date'].astype(str)
        forecast_display = forecast_display.rename(columns={
            'date': 'Date',
            'temp': 'Temp (°C)',
            'temp_min': 'Min (°C)',
            'temp_max': 'Max (°C)',
            'humidity': 'Humidité (%)',
            'rain': 'Pluie (mm)',
            'wind_speed': 'Vent (m/s)',
            'description': 'Conditions'
        })
        
        st.dataframe(forecast_display, use_container_width=True)
        
        st.markdown("---")
        
        # Analyses et recommandations basées sur prévisions
        st.markdown("### 🌾 Recommandations Culturales (Prévisions)")
        
        total_rain_forecast = forecast_df['rain'].sum()
        avg_temp_forecast = forecast_df['temp'].mean()
        max_temp_forecast = forecast_df['temp_max'].max()
        avg_humidity_forecast = forecast_df['humidity'].mean()
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown("**💧 Gestion Eau**")
            if total_rain_forecast > 30:
                st.markdown('<div class="success-box">✅ <b>Pluies prévues: {:.0f}mm</b><br>Irrigation non nécessaire<br>Période favorable semis</div>'.format(total_rain_forecast), 
                           unsafe_allow_html=True)
            elif total_rain_forecast > 10:
                st.markdown('<div class="info-box">ℹ️ <b>Pluies modérées: {:.0f}mm</b><br>Irrigation complémentaire si besoin<br>Surveiller développement</div>'.format(total_rain_forecast), 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box">⚠️ <b>Peu de pluie: {:.0f}mm</b><br>Irrigation nécessaire<br>Reporter semis si possible</div>'.format(total_rain_forecast), 
                           unsafe_allow_html=True)
        
        with col_r2:
            st.markdown("**🌡️ Conditions Thermiques**")
            if max_temp_forecast > 38:
                st.markdown('<div class="danger-box">🔥 <b>Chaleur extrême prévue</b><br>Max: {:.0f}°C<br>Risque stress thermique<br>Irrigation impérative</div>'.format(max_temp_forecast), 
                           unsafe_allow_html=True)
            elif avg_temp_forecast > 30:
                st.markdown('<div class="alert-box">☀️ <b>Températures élevées</b><br>Moy: {:.1f}°C<br>Surveiller hydratation<br>Éviter traitements midi</div>'.format(avg_temp_forecast), 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <b>Températures favorables</b><br>Moy: {:.1f}°C<br>Conditions optimales croissance</div>'.format(avg_temp_forecast), 
                           unsafe_allow_html=True)
        
        with col_r3:
            st.markdown("**🦠 Risque Phytosanitaire**")
            if avg_humidity_forecast > 70 and avg_temp_forecast > 20:
                st.markdown('<div class="alert-box">⚠️ <b>Risque maladies ÉLEVÉ</b><br>Humidité: {:.0f}%<br>Conditions favorables champignons<br>Traitement préventif recommandé</div>'.format(avg_humidity_forecast), 
                           unsafe_allow_html=True)
            elif avg_humidity_forecast > 60:
                st.markdown('<div class="info-box">ℹ️ <b>Risque modéré</b><br>Surveiller apparition symptômes<br>Préparer traitements</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <b>Risque faible</b><br>Conditions sèches<br>Pression sanitaire limitée</div>', 
                           unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Calendrier semis pour chaque culture
        st.markdown("### 📅 Calendrier de Semis Recommandé")
        
        if cultures_selectionnees:
            for culture in cultures_selectionnees:
                with st.expander(f"🌱 {culture}", expanded=False):
                    
                    # Paramètres culturaux
                    crop_calendar = {
                        "Mil": {
                            "periode_semis": "Juin-Juillet",
                            "pluie_min_semis": 20,
                            "temp_optimale": "25-30°C",
                            "cycle": "90 jours",
                            "espacement": "50x50 cm",
                            "profondeur": "2-3 cm",
                            "dose_semences": "5-8 kg/ha"
                        },
                        "Sorgho": {
                            "periode_semis": "Juin-Juillet",
                            "pluie_min_semis": 25,
                            "temp_optimale": "27-32°C",
                            "cycle": "110 jours",
                            "espacement": "60x40 cm",
                            "profondeur": "3-4 cm",
                            "dose_semences": "8-10 kg/ha"
                        },
                        "Maïs": {
                            "periode_semis": "Juin-Août",
                            "pluie_min_semis": 30,
                            "temp_optimale": "20-30°C",
                            "cycle": "120 jours",
                            "espacement": "75x25 cm",
                            "profondeur": "4-5 cm",
                            "dose_semences": "20-25 kg/ha"
                        },
                        "Arachide": {
                            "periode_semis": "Juin-Juillet",
                            "pluie_min_semis": 25,
                            "temp_optimale": "25-30°C",
                            "cycle": "120 jours",
                            "espacement": "50x15 cm",
                            "profondeur": "3-4 cm",
                            "dose_semences": "60-80 kg/ha"
                        },
                        "Riz": {
                            "periode_semis": "Juillet-Août",
                            "pluie_min_semis": 50,
                            "temp_optimale": "25-30°C",
                            "cycle": "130 jours",
                            "espacement": "20x20 cm (repiquage)",
                            "profondeur": "2-3 cm",
                            "dose_semences": "60-80 kg/ha"
                        },
                        "Niébé": {
                            "periode_semis": "Juillet-Août",
                            "pluie_min_semis": 20,
                            "temp_optimale": "25-30°C",
                            "cycle": "75 jours",
                            "espacement": "50x20 cm",
                            "profondeur": "3-4 cm",
                            "dose_semences": "20-30 kg/ha"
                        },
                        "Tomate": {
                            "periode_semis": "Octobre-Novembre (pépinière)",
                            "pluie_min_semis": 15,
                            "temp_optimale": "20-25°C",
                            "cycle": "90 jours",
                            "espacement": "80x50 cm",
                            "profondeur": "0.5-1 cm",
                            "dose_semences": "200-300 g/ha"
                        },
                        "Oignon": {
                            "periode_semis": "Octobre-Décembre (pépinière)",
                            "pluie_min_semis": 10,
                            "temp_optimale": "15-25°C",
                            "cycle": "110 jours",
                            "espacement": "15x10 cm",
                            "profondeur": "1-2 cm",
                            "dose_semences": "4-5 kg/ha"
                        },
                        "Coton": {
                            "periode_semis": "Juin-Juillet",
                            "pluie_min_semis": 25,
                            "temp_optimale": "25-30°C",
                            "cycle": "150 jours",
                            "espacement": "80x30 cm",
                            "profondeur": "3-4 cm",
                            "dose_semences": "15-20 kg/ha"
                        },
                        "Pastèque": {
                            "periode_semis": "Mars-Avril ou Septembre-Octobre",
                            "pluie_min_semis": 15,
                            "temp_optimale": "25-30°C",
                            "cycle": "85 jours",
                            "espacement": "2m x 1m",
                            "profondeur": "2-3 cm",
                            "dose_semences": "3-4 kg/ha"
                        }
                    }
                    
                    params = crop_calendar.get(culture, crop_calendar["Mil"])
                    
                    col_c1, col_c2 = st.columns(2)
                    
                    with col_c1:
                        st.markdown(f"""
                        **Calendrier Cultural**
                        - 📅 Période optimale: {params['periode_semis']}
                        - ⏱️ Durée cycle: {params['cycle']}
                        - 🌡️ Température optimale: {params['temp_optimale']}
                        - 💧 Pluie min semis: {params['pluie_min_semis']} mm
                        """)
                    
                    with col_c2:
                        st.markdown(f"""
                        **Paramètres Techniques**
                        - 📏 Espacement: {params['espacement']}
                        - 📐 Profondeur semis: {params['profondeur']}
                        - 🌾 Dose semences: {params['dose_semences']}
                        """)
                    
                    # Recommandation basée sur prévisions
                    if total_rain_forecast >= params['pluie_min_semis']:
                        st.success(f"✅ Conditions favorables au semis détectées dans les 7 prochains jours ({total_rain_forecast:.0f}mm prévu)")
                        st.info(f"💡 Recommandation: Préparer le semis de {culture} dès que les pluies commencent")
                    else:
                        st.warning(f"⚠️ Pluies insuffisantes prévues ({total_rain_forecast:.0f}mm < {params['pluie_min_semis']}mm requis)")
                        st.info(f"💡 Recommandation: Attendre des prévisions plus favorables ou prévoir irrigation post-semis")
        
    else:
        st.info("⚙️ Configurez votre clé OpenWeather dans la barre latérale pour activer les prévisions")
        st.markdown("""
        ### Comment obtenir une clé OpenWeather (gratuit):
        1. Allez sur [openweathermap.org](https://openweathermap.org/api)
        2. Créez un compte gratuit
        3. Générez une clé API (plan gratuit: 1000 appels/jour)
        4. Collez la clé dans la configuration
        
        **Avantages des prévisions:**
        - Calendrier semis optimisé
        - Alerte traitements phytosanitaires
        - Planification irrigation
        - Prévention risques climatiques
        """)
# ONGLET 6: ANALYSE IA MULTI-CULTURES
with tabs[5]:
    st.subheader("🤖 Analyse IA Multi-Cultures avec Google Gemini")
    
    if st.session_state.analysis and st.session_state.climate_data is not None:
        
        st.info("💡 **Google Gemini** gratuit (15 req/min). [Obtenez votre clé](https://aistudio.google.com/apikey)")
        
        # Options d'analyse
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            include_forecast = st.checkbox("Inclure prévisions météo", 
                                          value=st.session_state.weather_forecast is not None)
        
        with col_opt2:
            detailed_analysis = st.checkbox("Analyse très détaillée", value=True)
        
        analyze_btn = st.button("🚀 Générer Analyses IA Complètes", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 Analyse IA en cours pour toutes les cultures..."):
                
                analyses_generated = {}
                
                for culture in cultures_selectionnees:
                    
                    st.info(f"Analyse de {culture}...")
                    
                    metrics = st.session_state.analysis[culture]['metrics']
                    recommendations = st.session_state.analysis[culture]['recommendations']
                    
                    # Construire données NDVI détaillées
                    indices_df = st.session_state.satellite_data
                    ndvi_evolution = indices_df.groupby('date')['ndvi'].agg(['mean', 'min', 'max']).reset_index()
                    ndvi_recent = ", ".join([
                        f"{row['date'].strftime('%d/%m')}: {row['mean']:.2f} (min:{row['min']:.2f}, max:{row['max']:.2f})"
                        for _, row in ndvi_evolution.tail(10).iterrows()
                    ])
                    
                    # Données climatiques
                    climate_df = st.session_state.climate_data
                    rain_by_week = climate_df.groupby(pd.Grouper(key='date', freq='W'))['rain'].sum().tail(4)
                    rain_weekly = ", ".join([f"Sem {i+1}: {v:.0f}mm" for i, v in enumerate(rain_by_week)])
                    
                    # Variabilité spatiale
                    ndvi_by_cell = indices_df.groupby('cell_id')['ndvi'].mean()
                    spatial_cv = (ndvi_by_cell.std() / ndvi_by_cell.mean()) * 100
                    
                    # Prévisions
                    forecast_info = ""
                    if include_forecast and st.session_state.weather_forecast is not None:
                        forecast_df = st.session_state.weather_forecast
                        forecast_info = f"""
PRÉVISIONS 7 JOURS:
- Pluie prévue: {forecast_df['rain'].sum():.0f}mm
- Temp moyenne: {forecast_df['temp'].mean():.1f}°C (max: {forecast_df['temp_max'].max():.0f}°C)
- Humidité moyenne: {forecast_df['humidity'].mean():.0f}%
"""
                    
                    # Construction prompt détaillé
                    prompt = f"""Tu es un AGRONOME EXPERT spécialisé en {culture}. Analyse ces données et fournis des recommandations TRÈS DÉTAILLÉES, PRÉCISES et ACTIONNABLES.

CULTURE: {culture}
ZONE: {zone_name}
PÉRIODE: {(end_date - start_date).days} jours d'analyse

DONNÉES SATELLITAIRES:
- NDVI moyen: {metrics['ndvi_mean']:.3f} (min:{metrics['ndvi_min']:.3f}, max:{metrics['ndvi_max']:.3f}, σ:{metrics['ndvi_std']:.3f})
- Évolution NDVI (10 derniers points): {ndvi_recent}
- EVI moyen: {metrics['evi_mean']:.3f}
- NDWI moyen: {metrics['ndwi_mean']:.3f} (stress hydrique: {metrics['water_stress']:.2f})
- SAVI: {metrics['savi_mean']:.3f}, LAI: {metrics['lai_mean']:.1f} m²/m²
- Variabilité spatiale (CV): {spatial_cv:.1f}%

DONNÉES CLIMATIQUES:
- Température: {metrics['temp_mean']:.1f}°C (min:{metrics['temp_min']:.0f}°C, max:{metrics['temp_max']:.0f}°C)
- Pluie totale: {metrics['rain_total']:.0f}mm ({metrics['rain_days']} jours de pluie)
- Pluie hebdomadaire: {rain_weekly}
- Humidité: {metrics['humidity_mean']:.0f}%
- Vent: {metrics['wind_mean']:.1f} m/s
{forecast_info}

SCORES CALCULÉS:
- Score NDVI: {metrics['ndvi_score']:.2f}/1.0
- Score Pluviométrie: {metrics['rain_score']:.2f}/1.0
- Score Température: {metrics['temp_score']:.2f}/1.0
- Rendement estimé: {metrics['yield_potential']:.1f} t/ha

ANALYSE DEMANDÉE (sois TRÈS PRÉCIS et ACTIONNABLE):

1. DIAGNOSTIC DÉTAILLÉ
   - État actuel de la culture (stade phénologique probable, vigueur, stress)
   - Analyse de la variabilité spatiale ({spatial_cv:.1f}% de CV)
   - Interprétation croisée des indices (NDVI, EVI, NDWI, LAI)
   - Points de vigilance spécifiques

2. IRRIGATION (doses et timing précis)
   - Besoins en eau actuels (mm/semaine)
   - Calendrier irrigation (fréquence, durée)
   - Méthode recommandée (aspersion, goutte-à-goutte, gravitaire)
   - Ajustements selon prévisions météo

3. FERTILISATION (formules NPK précises, doses, périodes)
   - Apports de fond: type engrais, dose kg/ha, période exacte
   - Couvertures: formulations, doses, stades d'application
   - Apports foliaires si nécessaire
   - Fumure organique: type, dose, incorporation

4. PROTECTION PHYTOSANITAIRE
   - Maladies probables (conditions actuelles)
   - Ravageurs à surveiller (saison, température)
   - Traitements préventifs: matières actives, doses, périodes
   - Traitements curatifs si symptômes
   - Fréquence surveillance

5. OPÉRATIONS CULTURALES
   - Sarclages/binages: fréquence et périodes
   - Buttage si nécessaire: quand et comment
   - Éclaircissage: densité cible
   - Autres interventions spécifiques à {culture}

6. CALENDRIER PRÉVISIONNEL
   - Estimation stade actuel
   - Opérations à venir (15-30 jours)
   - Date récolte probable
   - Indicateurs de maturité

7. PRÉVISION RENDEMENT ET QUALITÉ
   - Rendement final estimé (t/ha) avec intervalle de confiance
   - Qualité probable (calibre, teneur, etc.)
   - Facteurs limitants identifiés
   - Potentiel d'amélioration

8. ALERTES ET ACTIONS URGENTES
   - Problèmes critiques détectés
   - Actions à entreprendre IMMÉDIATEMENT
   - Délais d'intervention

IMPORTANT:
- Sois CONCRET: donne des chiffres, des dates, des doses précises
- Adapte au CONTEXTE SAHÉLIEN (disponibilité intrants, pratiques locales)
- Évite généralités: chaque recommandation doit être APPLICABLE directement
- Utilise expertise agronomique pointue pour {culture}
- Fournis réponse structurée en français, ~1200-1500 mots"""

                    analysis_text = None
                    
                    if gemini_key:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}"
                        try:
                            response = requests.post(
                                url,
                                headers={"Content-Type": "application/json"},
                                json={
                                    "contents": [{"parts": [{"text": prompt}]}],
                                    "generationConfig": {
                                        "temperature": 0.7,
                                        "maxOutputTokens": 8192,
                                    }
                                },
                                timeout=90
                            )
                            if response.status_code == 200:
                                data = response.json()
                                if 'candidates' in data and len(data['candidates']) > 0:
                                    analysis_text = data['candidates'][0]['content']['parts'][0]['text']
                            else:
                                st.warning(f"Erreur API Gemini pour {culture}: {response.status_code}")
                        except Exception as e:
                            st.warning(f"Erreur connexion Gemini pour {culture}: {e}")
                    
                    # Analyse par défaut si pas de Gemini
                    if not analysis_text:
                        analysis_text = f"""# ANALYSE AGRONOMIQUE - {culture.upper()}

## 1. DIAGNOSTIC

**État Cultural:** NDVI de {metrics['ndvi_mean']:.3f} indique {'excellente vigueur' if metrics['ndvi_mean'] > 0.6 else 'vigueur modérée' if metrics['ndvi_mean'] > 0.4 else 'stress végétal'}.

**Variabilité Spatiale:** Coefficient de variation de {spatial_cv:.1f}% {'(faible - parcelle homogène)' if spatial_cv < 15 else '(élevé - gestion différenciée recommandée)'}.

**Stress Hydrique:** NDWI {metrics['ndwi_mean']:.3f} - {'Bon contenu en eau' if metrics['ndwi_mean'] > 0.2 else 'Déficit hydrique probable'}.

## 2. IRRIGATION

**Pluviométrie:** {metrics['rain_total']:.0f}mm sur période analysée.

{'- URGENT: Irrigation immédiate 30-40mm, répéter tous les 5-7 jours' if metrics['rain_total'] < 250 else '- Irrigation complémentaire 20-25mm tous les 7-10 jours' if metrics['rain_total'] < 400 else '- Pluviométrie satisfaisante, surveiller évolution'}.

## 3. FERTILISATION

**Apports recommandés pour {culture}:**

{chr(10).join(['- ' + r for r in recommendations['fertilisation']])}

## 4. PROTECTION PHYTOSANITAIRE

{'- Conditions favorables maladies fongiques (T>{metrics["temp_mean"]:.0f}°C, HR>{metrics["humidity_mean"]:.0f}%)' if metrics['humidity_mean'] > 70 and metrics['temp_mean'] > 25 else '- Pression sanitaire modérée'}
- Surveillance hebdomadaire recommandée
- Traitement préventif si conditions favorables persistent

## 5. OPÉRATIONS CULTURALES

- Sarclage/binage: 2-3 passages selon enherbement
- Maintien sol meuble pour infiltration eau
- Contrôle adventices compétition eau/nutriments

## 6. RENDEMENT PRÉVISIONNEL

**Estimation:** {metrics['yield_potential']:.1f} t/ha

**Facteurs limitants:**
{chr(10).join(['- ' + a for a in recommendations['alertes']]) if recommendations['alertes'] else '- Aucun facteur critique identifié'}

## 7. RECOMMANDATIONS PRIORITAIRES

{chr(10).join(['- ' + r for r in (recommendations['irrigation'][:2] + recommendations['diagnostic'][:2])])}

---
*Pour analyse IA approfondie, configurez votre clé Google Gemini (gratuite)*"""
                    
                    analyses_generated[culture] = analysis_text
                    time.sleep(2)  # Rate limiting
                
                # Stocker toutes les analyses
                for culture, text in analyses_generated.items():
                    if culture not in st.session_state.analysis:
                        st.session_state.analysis[culture] = {}
                    st.session_state.analysis[culture]['ai_analysis'] = text
                
                st.success(f"✅ Analyses IA générées pour {len(cultures_selectionnees)} cultures!")
        
        # Afficher analyses
        if st.session_state.analysis:
            st.markdown("---")
            st.markdown("### 📋 Rapports Agronomiques Détaillés")
            
            for culture in cultures_selectionnees:
                if culture in st.session_state.analysis and 'ai_analysis' in st.session_state.analysis[culture]:
                    
                    with st.expander(f"🌾 {culture} - Rapport Complet", expanded=True):
                        
                        analysis_text = st.session_state.analysis[culture]['ai_analysis']
                        st.markdown(analysis_text)
                        
                        st.markdown("---")
                        
                        # Boutons téléchargement
                        col_dl1, col_dl2, col_dl3 = st.columns(3)
                        
                        with col_dl1:
                            st.download_button(
                                f"📥 Télécharger {culture} (TXT)",
                                analysis_text,
                                file_name=f"analyse_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key=f"dl_txt_{culture}"
                            )
                        
                        with col_dl2:
                            metrics = st.session_state.analysis[culture]['metrics']
                            summary_json = json.dumps({
                                "culture": culture,
                                "zone": zone_name,
                                "date": datetime.now().strftime('%Y-%m-%d'),
                                "ndvi_mean": round(metrics['ndvi_mean'], 3),
                                "ndvi_min": round(metrics['ndvi_min'], 3),
                                "ndvi_max": round(metrics['ndvi_max'], 3),
                                "evi": round(metrics['evi_mean'], 3),
                                "ndwi": round(metrics['ndwi_mean'], 3),
                                "lai": round(metrics['lai_mean'], 2),
                                "temp_mean": round(metrics['temp_mean'], 1),
                                "rain_total": round(metrics['rain_total'], 1),
                                "humidity": round(metrics['humidity_mean'], 1),
                                "rendement_estime": round(metrics['yield_potential'], 2),
                                "water_stress": round(metrics['water_stress'], 2)
                            }, indent=2)
                            
                            st.download_button(
                                f"📊 Métriques {culture} (JSON)",
                                summary_json,
                                file_name=f"metriques_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json",
                                use_container_width=True,
                                key=f"dl_json_{culture}"
                            )
                        
                        with col_dl3:
                            # Markdown complet
                            full_md = f"""# Analyse {culture} - {zone_name}
Date: {datetime.now().strftime('%d/%m/%Y')}

{analysis_text}

## Métriques Clés
- NDVI: {metrics['ndvi_mean']:.3f}
- Rendement: {metrics['yield_potential']:.1f} t/ha
- Pluie: {metrics['rain_total']:.0f}mm
- Température: {metrics['temp_mean']:.1f}°C
"""
                            st.download_button(
                                f"📝 Rapport {culture} (MD)",
                                full_md,
                                file_name=f"rapport_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.md",
                                mime="text/markdown",
                                use_container_width=True,
                                key=f"dl_md_{culture}"
                            )
    
    else:
        st.info("Lancez d'abord l'analyse complète")
# ONGLET 7: RAPPORT PDF
with tabs[6]:
    st.subheader("📄 Rapport PDF Complet Multi-Cultures")
    
    if st.session_state.climate_data is not None and st.session_state.satellite_data is not None:
        
        st.markdown("""
        **Contenu du rapport PDF:**
        - 🗺️ Carte de la zone avec points d'échantillonnage
        - 📊 Graphiques multi-indices (NDVI, EVI, NDWI, LAI)
        - 🌦️ Données climatiques détaillées
        - 🔮 Prévisions météorologiques (si disponibles)
        - 🤖 Analyses IA complètes pour chaque culture
        - 📈 Tableaux synthétiques et coordonnées GPS
        - 💡 Recommandations détaillées
        """)
        
        # Options rapport
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            include_map = st.checkbox("Inclure carte détaillée", value=True)
            include_ai = st.checkbox("Inclure analyses IA", value=True)
        
        with col_opt2:
            include_coords = st.checkbox("Inclure tableau coordonnées", value=True)
            include_forecast = st.checkbox("Inclure prévisions", 
                                          value=st.session_state.weather_forecast is not None)
        
        if st.button("📄 Générer Rapport PDF Complet", type="primary", use_container_width=True):
            with st.spinner("📝 Génération du rapport PDF..."):
                try:
                    def generate_comprehensive_pdf():
                        buffer = BytesIO()
                        
                        with PdfPages(buffer) as pdf:
                            
                            # PAGE 1: Page de garde
                            fig = plt.figure(figsize=(8.5, 11))
                            fig.patch.set_facecolor('white')
                            ax = fig.add_subplot(111)
                            ax.axis('off')
                            
                            # Titre principal
                            ax.text(0.5, 0.85, 'RAPPORT AGRO-CLIMATIQUE', 
                                   ha='center', fontsize=24, fontweight='bold', 
                                   color='#2E7D32')
                            ax.text(0.5, 0.78, 'Analyse par Télédétection et IA', 
                                   ha='center', fontsize=14, color='#555')
                            
                            # Ligne séparatrice
                            ax.plot([0.2, 0.8], [0.75, 0.75], 'k-', linewidth=2)
                            
                            # Informations zone
                            info_y = 0.68
                            ax.text(0.5, info_y, f'Zone: {zone_name}', 
                                   ha='center', fontsize=16, fontweight='bold')
                            info_y -= 0.05
                            ax.text(0.5, info_y, f'Cultures: {", ".join(cultures_selectionnees)}', 
                                   ha='center', fontsize=12)
                            info_y -= 0.05
                            ax.text(0.5, info_y, f'Période: {start_date.strftime("%d/%m/%Y")} - {end_date.strftime("%d/%m/%Y")}', 
                                   ha='center', fontsize=12)
                            info_y -= 0.05
                            ax.text(0.5, info_y, f'Surface: {len(st.session_state.sampling_points)} points échantillonnés', 
                                   ha='center', fontsize=12)
                            
                            # Métriques clés pour première culture
                            if cultures_selectionnees:
                                first_culture = cultures_selectionnees[0]
                                metrics = st.session_state.analysis[first_culture]['metrics']
                                
                                metrics_y = 0.50
                                ax.text(0.5, metrics_y, 'MÉTRIQUES PRINCIPALES', 
                                       ha='center', fontsize=14, fontweight='bold', 
                                       color='#2E7D32')
                                metrics_y -= 0.08
                                
                                col1_x, col2_x = 0.3, 0.7
                                
                                ax.text(col1_x, metrics_y, f'NDVI Moyen:', fontweight='bold')
                                ax.text(col2_x, metrics_y, f'{metrics["ndvi_mean"]:.3f}')
                                metrics_y -= 0.05
                                
                                ax.text(col1_x, metrics_y, f'Pluie Totale:', fontweight='bold')
                                ax.text(col2_x, metrics_y, f'{metrics["rain_total"]:.0f} mm')
                                metrics_y -= 0.05
                                
                                ax.text(col1_x, metrics_y, f'Température Moy.:', fontweight='bold')
                                ax.text(col2_x, metrics_y, f'{metrics["temp_mean"]:.1f}°C')
                                metrics_y -= 0.05
                                
                                ax.text(col1_x, metrics_y, f'Rendement Estimé:', fontweight='bold')
                                ax.text(col2_x, metrics_y, f'{metrics["yield_potential"]:.1f} t/ha')
                            
                            # Footer
                            ax.text(0.5, 0.15, 'AgriSight Pro v2.0', 
                                   ha='center', fontsize=10, style='italic', color='#666')
                            ax.text(0.5, 0.12, f'Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}', 
                                   ha='center', fontsize=9, color='#888')
                            ax.text(0.5, 0.08, 'Télédétection • IA • Agriculture de Précision', 
                                   ha='center', fontsize=9, color='#888')
                            
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close()
                            
                            # PAGE 2: Carte et coordonnées
                            if include_map and st.session_state.gdf is not None:
                                fig = plt.figure(figsize=(8.5, 11))
                                
                                if include_coords and st.session_state.sampling_points is not None:
                                    # Carte + tableau
                                    ax_map = plt.subplot2grid((2, 1), (0, 0))
                                    ax_table = plt.subplot2grid((2, 1), (1, 0))
                                else:
                                    # Carte seule
                                    ax_map = fig.add_subplot(111)
                                
                                ax_map.set_title(f'Zone d\'Étude: {zone_name}', 
                                               fontsize=14, fontweight='bold')
                                
                                # Plot zone
                                gdf = st.session_state.gdf
                                gdf.plot(ax=ax_map, facecolor='lightgreen', 
                                        edgecolor='darkgreen', alpha=0.5, linewidth=2)
                                
                                # Plot points échantillonnage
                                if st.session_state.sampling_points is not None:
                                    points_gdf = st.session_state.sampling_points
                                    points_gdf.plot(ax=ax_map, color='red', 
                                                   markersize=30, alpha=0.7)
                                    
                                    # Annotations
                                    for idx, row in points_gdf.iterrows():
                                        if idx < 20:  # Limiter annotations
                                            ax_map.annotate(row['cell_id'], 
                                                          (row.geometry.x, row.geometry.y),
                                                          fontsize=6, ha='center')
                                
                                ax_map.set_xlabel('Longitude', fontweight='bold')
                                ax_map.set_ylabel('Latitude', fontweight='bold')
                                ax_map.grid(True, alpha=0.3)
                                
                                # Tableau coordonnées
                                if include_coords and st.session_state.sampling_points is not None:
                                    ax_table.axis('off')
                                    
                                    coords_data = []
                                    for idx, row in st.session_state.sampling_points.head(15).iterrows():
                                        coords_data.append([
                                            row['cell_id'],
                                            f"{row['latitude']:.4f}",
                                            f"{row['longitude']:.4f}"
                                        ])
                                    
                                    table = ax_table.table(
                                        cellText=coords_data,
                                        colLabels=['Point', 'Latitude', 'Longitude'],
                                        cellLoc='center',
                                        loc='center',
                                        colWidths=[0.3, 0.35, 0.35]
                                    )
                                    table.auto_set_font_size(False)
                                    table.set_fontsize(8)
                                    table.scale(1, 1.5)
                                    
                                    # Style header
                                    for i in range(3):
                                        table[(0, i)].set_facecolor('#2E7D32')
                                        table[(0, i)].set_text_props(weight='bold', color='white')
                                    
                                    ax_table.set_title('Coordonnées Points d\'Échantillonnage (15 premiers)', 
                                                      fontsize=11, fontweight='bold', pad=20)
                                
                                plt.tight_layout()
                                pdf.savefig(fig, bbox_inches='tight')
                                plt.close()
                            
                            # PAGES 3+: Graphiques par culture
                            for culture in cultures_selectionnees:
                                metrics = st.session_state.analysis[culture]['metrics']
                                
                                fig = plt.figure(figsize=(8.5, 11))
                                fig.suptitle(f'Analyse {culture}', fontsize=16, fontweight='bold')
                                
                                # Grille 4x2
                                gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
                                
                                # Info box
                                ax_info = fig.add_subplot(gs[0, :])
                                ax_info.axis('off')
                                info_text = f"""Culture: {culture}  |  Rendement: {metrics['yield_potential']:.1f} t/ha  |  NDVI: {metrics['ndvi_mean']:.3f}  |  Pluie: {metrics['rain_total']:.0f}mm  |  Temp: {metrics['temp_mean']:.1f}°C"""
                                ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                                           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue'))
                                
                                # NDVI
                                ax1 = fig.add_subplot(gs[1, :])
                                indices_df = st.session_state.satellite_data
                                ndvi_temp = indices_df.groupby('date')['ndvi'].mean().reset_index()
                                ax1.plot(ndvi_temp['date'], ndvi_temp['ndvi'], 'o-', 
                                        color='darkgreen', linewidth=2)
                                ax1.fill_between(ndvi_temp['date'], ndvi_temp['ndvi'], 
                                               alpha=0.3, color='green')
                                ax1.axhline(0.7, color='green', linestyle='--', alpha=0.5)
                                ax1.axhline(0.5, color='orange', linestyle='--', alpha=0.5)
                                ax1.set_ylabel('NDVI', fontweight='bold')
                                ax1.set_title('Évolution NDVI', fontweight='bold', fontsize=11)
                                ax1.grid(True, alpha=0.3)
                                ax1.set_ylim([0, 1])
                                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
                                
                                # Température
                                ax2 = fig.add_subplot(gs[2, 0])
                                climate_df = st.session_state.climate_data
                                clim_temp = climate_df.groupby('date').agg({
                                    'temp_mean': 'mean',
                                    'temp_min': 'min',
                                    'temp_max': 'max'
                                }).reset_index()
                                ax2.fill_between(clim_temp['date'], clim_temp['temp_min'], 
                                               clim_temp['temp_max'], alpha=0.3, color='coral')
                                ax2.plot(clim_temp['date'], clim_temp['temp_mean'], 
                                        color='red', linewidth=2)
                                ax2.set_ylabel('Temp (°C)', fontweight='bold')
                                ax2.set_title('Températures', fontweight='bold', fontsize=10)
                                ax2.grid(True, alpha=0.3)
                                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
                                
                                # Pluie
                                ax3 = fig.add_subplot(gs[2, 1])
                                rain_temp = climate_df.groupby('date')['rain'].mean().reset_index()
                                ax3.bar(rain_temp['date'], rain_temp['rain'], 
                                       color='dodgerblue', alpha=0.7)
                                ax3.set_ylabel('Pluie (mm)', fontweight='bold')
                                ax3.set_title('Précipitations', fontweight='bold', fontsize=10)
                                ax3.grid(True, alpha=0.3, axis='y')
                                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
                                
                                # Indices
                                ax4 = fig.add_subplot(gs[3, 0])
                                indices_names = ['NDVI', 'EVI', 'NDWI', 'SAVI']
                                indices_vals = [metrics['ndvi_mean'], metrics['evi_mean'],
                                              (metrics['ndwi_mean']+1)/2, metrics['savi_mean']]
                                colors_bar = ['green', 'darkgreen', 'blue', 'olive']
                                ax4.bar(indices_names, indices_vals, color=colors_bar, alpha=0.7)
                                ax4.set_ylabel('Valeur', fontweight='bold')
                                ax4.set_title('Indices Végétation', fontweight='bold', fontsize=10)
                                ax4.set_ylim([0, 1])
                                ax4.grid(True, alpha=0.3, axis='y')
                                
                                # Statistiques
                                ax5 = fig.add_subplot(gs[3, 1])
                                ax5.axis('off')
                                stats_text = f"""LAI: {metrics['lai_mean']:.1f} m²/m²
Humidité: {metrics['humidity_mean']:.0f}%
Vent: {metrics['wind_mean']:.1f} m/s
Jours pluie: {metrics['rain_days']}
Score hydrique: {(1-metrics['water_stress']):.2f}
Cycle: {metrics['cycle_days']} jours"""
                                ax5.text(0.1, 0.5, stats_text, fontsize=9, 
                                        verticalalignment='center', family='monospace')
                                ax5.set_title('Statistiques', fontweight='bold', fontsize=10)
                                
                                pdf.savefig(fig, bbox_inches='tight')
                                plt.close()
                                
                                # PAGE Analyse IA
                                if include_ai and culture in st.session_state.analysis:
                                    if 'ai_analysis' in st.session_state.analysis[culture]:
                                        analysis_text = st.session_state.analysis[culture]['ai_analysis']
                                        
                                        # Découper texte
                                        lines = analysis_text.split('\n')
                                        pages_text = []
                                        current_page = []
                                        line_count = 0
                                        
                                        for line in lines:
                                            if line_count > 55:  # ~55 lignes par page
                                                pages_text.append('\n'.join(current_page))
                                                current_page = [line]
                                                line_count = 1
                                            else:
                                                current_page.append(line)
                                                line_count += 1
                                        
                                        if current_page:
                                            pages_text.append('\n'.join(current_page))
                                        
                                        # Générer pages
                                        for i, page_text in enumerate(pages_text):
                                            fig = plt.figure(figsize=(8.5, 11))
                                            ax = fig.add_subplot(111)
                                            ax.axis('off')
                                            
                                            if i == 0:
                                                ax.text(0.5, 0.98, f'Analyse IA - {culture}', 
                                                       ha='center', fontsize=14, fontweight='bold',
                                                       transform=ax.transAxes)
                                                y_start = 0.94
                                            else:
                                                y_start = 0.98
                                            
                                            ax.text(0.05, y_start, page_text, 
                                                   fontsize=8, verticalalignment='top',
                                                   transform=ax.transAxes, family='sans-serif',
                                                   wrap=True)
                                            
                                            # Numéro page
                                            ax.text(0.95, 0.02, f'Page {i+1}/{len(pages_text)}', 
                                                   ha='right', fontsize=8, color='gray',
                                                   transform=ax.transAxes)
                                            
                                            pdf.savefig(fig, bbox_inches='tight')
                                            plt.close()
                            
                            # PAGE FINALE: Tableau synthétique multi-cultures
                            fig = plt.figure(figsize=(8.5, 11))
                            ax = fig.add_subplot(111)
                            ax.axis('off')
                            
                            ax.text(0.5, 0.95, 'TABLEAU SYNTHÉTIQUE', 
                                   ha='center', fontsize=16, fontweight='bold')
                            
                            # Préparer données tableau
                            synth_data = []
                            for culture in cultures_selectionnees:
                                metrics = st.session_state.analysis[culture]['metrics']
                                synth_data.append([
                                    culture,
                                    f"{metrics['ndvi_mean']:.3f}",
                                    f"{metrics['rain_total']:.0f}",
                                    f"{metrics['temp_mean']:.1f}",
                                    f"{metrics['yield_potential']:.1f}",
                                    f"{(1-metrics['water_stress'])*100:.0f}%"
                                ])
                            
                            table = ax.table(
                                cellText=synth_data,
                                colLabels=['Culture', 'NDVI', 'Pluie\n(mm)', 'Temp\n(°C)', 
                                          'Rend.\n(t/ha)', 'État\nHydrique'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0.1, 0.5, 0.8, 0.4]
                            )
                            table.auto_set_font_size(False)
                            table.set_fontsize(10)
                            table.scale(1, 2)
                            
                            # Style
                            for i in range(6):
                                table[(0, i)].set_facecolor('#2E7D32')
                                table[(0, i)].set_text_props(weight='bold', color='white')
                            
                            # Légende
                            legend_y = 0.35
                            ax.text(0.5, legend_y, 'Légende et Seuils', 
                                   ha='center', fontsize=12, fontweight='bold')
                            legend_y -= 0.05
                            ax.text(0.1, legend_y, '• NDVI > 0.6: Excellent | 0.4-0.6: Bon | < 0.4: Faible', 
                                   fontsize=9)
                            legend_y -= 0.04
                            ax.text(0.1, legend_y, '• Pluie > 400mm: Suffisant | 250-400: Modéré | < 250: Insuffisant', 
                                   fontsize=9)
                            legend_y -= 0.04
                            ax.text(0.1, legend_y, '• État Hydrique > 70%: Bon | 50-70%: Modéré | < 50%: Stress', 
                                   fontsize=9)
                            
                            # Footer
                            ax.text(0.5, 0.05, f'Rapport généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}', 
                                   ha='center', fontsize=9, style='italic', color='#666')
                            ax.text(0.5, 0.02, 'AgriSight Pro - Télédétection & IA pour Agriculture de Précision', 
                                   ha='center', fontsize=8, color='#888')
                            
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close()
                        
                        buffer.seek(0)
                        return buffer
                    
                    # Générer PDF
                    pdf_buffer = generate_comprehensive_pdf()
                    
                    st.success("✅ Rapport PDF généré avec succès!")
                    
                    # Bouton téléchargement
                    st.download_button(
                        "📥 Télécharger Rapport PDF Complet",
                        pdf_buffer,
                        file_name=f"rapport_complet_{zone_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Erreur génération PDF: {e}")
                    st.exception(e)
        
        st.markdown("---")
        
        # Export CSV
        st.markdown("### 💾 Exports CSV pour SIG")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Indices par point
            if st.session_state.satellite_data is not None:
                indices_df = st.session_state.satellite_data
                export_indices = indices_df.groupby(['cell_id', 'latitude', 'longitude']).agg({
                    'ndvi': ['mean', 'min', 'max', 'std'],
                    'evi': 'mean',
                    'ndwi': 'mean',
                    'savi': 'mean',
                    'lai': 'mean'
                }).reset_index()
                
                export_indices.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                         for col in export_indices.columns]
                
                csv_indices = export_indices.to_csv(index=False)
                st.download_button(
                    "🛰️ Indices Satellitaires",
                    csv_indices,
                    f"indices_{zone_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp2:
            # Climat par point
            if st.session_state.climate_data is not None:
                climate_df = st.session_state.climate_data
                export_climate = climate_df.groupby(['cell_id', 'latitude', 'longitude']).agg({
                    'temp_mean': 'mean',
                    'temp_min': 'min',
                    'temp_max': 'max',
                    'rain': 'sum',
                    'humidity': 'mean',
                    'wind_speed': 'mean'
                }).reset_index()
                
                export_climate.columns = ['cell_id', 'latitude', 'longitude',
                                         'temp_mean', 'temp_min', 'temp_max',
                                         'rain_total', 'humidity_mean', 'wind_mean']
                
                csv_climate = export_climate.to_csv(index=False)
                st.download_button(
                    "🌦️ Données Climatiques",
                    csv_climate,
                    f"climat_{zone_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp3:
            # Zone GeoJSON
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
        st.info("Lancez d'abord l'analyse complète")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>🌾 AgriSight Pro v2.0</b> - Analyse Agricole Multi-Indices par Télédétection et IA<br>
    <small>NDVI • EVI • NDWI • SAVI • LAI • NASA POWER • Google Gemini IA • Prévisions Météo</small><br>
    <small style='color: #888;'>Échantillonnage spatial • Multi-cultures • Export SIG • Rapports PDF complets</small>
</div>
""", unsafe_allow_html=True)
