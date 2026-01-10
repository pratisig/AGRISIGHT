# =========================
# IMPORTS
# =========================
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

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="AgriSight IA",
    layout="wide",
    page_icon="🌾"
)

# =========================
# CSS
# =========================
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

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

with st.sidebar.expander("🔑 Clés API (Optionnel)", expanded=False):
    st.markdown("""
    **Google Gemini API** (Gratuit)
    - 15 requêtes/min
    """)
    gemini_key = st.text_input("Clé Gemini (optionnel)", type="password")

    st.markdown("---")
    st.markdown("**Agromonitoring NDVI**")
    agromonitoring_key = st.text_input("Clé Agromonitoring", type="password")

st.sidebar.subheader("📍 Zone d'étude")
zone_method = st.sidebar.radio(
    "Méthode de sélection",
    ["📂 Importer GeoJSON", "✏️ Dessiner sur carte", "📌 Coordonnées"]
)

uploaded_file = None
manual_coords = None

if zone_method == "📂 Importer GeoJSON":
    uploaded_file = st.sidebar.file_uploader(
        "Fichier GeoJSON",
        type=["geojson", "json"]
    )

elif zone_method == "📌 Coordonnées":
    st.sidebar.info("Coins d'un rectangle")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        lat_min = st.number_input("Lat Min", value=14.60)
        lon_min = st.number_input("Lon Min", value=-17.50)
    with c2:
        lat_max = st.number_input("Lat Max", value=14.70)
        lon_max = st.number_input("Lon Max", value=-17.40)
    manual_coords = (lat_min, lon_min, lat_max, lon_max)

st.sidebar.subheader("📅 Période")
c1, c2 = st.sidebar.columns(2)
with c1:
    start_date = st.date_input("Début", date.today() - timedelta(days=60))
with c2:
    end_date = st.date_input("Fin", date.today())

culture = st.sidebar.selectbox(
    "🌱 Culture",
    ["Mil", "Sorgho", "Maïs", "Arachide", "Riz", "Niébé", "Manioc", "Tomate", "Oignon", "Papayer"]
)

zone_name = st.sidebar.text_input("📍 Nom de la zone", "Ma parcelle")

st.sidebar.markdown("---")
load_btn = st.sidebar.button(
    "🚀 Lancer l'analyse",
    type="primary",
    use_container_width=True
)

# =========================
# SESSION STATE
# =========================
for k in [
    "gdf", "satellite_data", "climate_data",
    "analysis", "drawn_geometry"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# =========================
# FONCTIONS (INCHANGÉES)
# =========================
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
    gdf = gpd.read_file(BytesIO(file_bytes))
    return gdf.to_crs(4326)

def geometry_to_dict(geom):
    return mapping(geom)

def dict_to_geometry(d):
    return shape(d)

@st.cache_data(ttl=3600)
def get_climate_nasa_cached(geom_dict, start, end):
    geom = dict_to_geometry(geom_dict)
    c = geom.centroid
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}"
        f"&latitude={c.y}&longitude={c.x}&format=JSON&community=AG"
    )
    r = requests.get(url, timeout=30)
    data = r.json()["properties"]["parameter"]

    return pd.DataFrame({
        "date": pd.to_datetime(list(data["T2M"].keys())),
        "temp_mean": list(data["T2M"].values()),
        "temp_min": list(data["T2M_MIN"].values()),
        "temp_max": list(data["T2M_MAX"].values()),
        "rain": list(data["PRECTOTCORR"].values())
    })

@st.cache_data(ttl=3600)
def simulate_ndvi_data(start, end):
    dates = pd.date_range(start, end, freq="5D")
    rows = []
    for d in dates:
        base = 0.6 + np.random.normal(0, 0.1)
        rows.append({
            "date": d,
            "ndvi_mean": np.clip(base, 0, 1),
            "ndvi_min": np.clip(base - 0.15, 0, 1),
            "ndvi_max": np.clip(base + 0.15, 0, 1)
        })
    return pd.DataFrame(rows)

# =========================
# ONGLET NDVI (CORRIGÉ)
# =========================
tabs = st.tabs([
    "🗺️ Carte & Zone",
    "📊 Vue d'ensemble",
    "🛰️ NDVI",
    "🌦️ Climat",
    "🤖 Analyse IA",
    "📄 Rapport PDF"
])

with tabs[2]:
    st.subheader("🛰️ Analyse NDVI Détaillée")

    if st.session_state.satellite_data is not None:
        df_sat = st.session_state.satellite_data

        col1, col2 = st.columns([2.5, 1.5])

        with col1:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(
                df_sat["date"],
                df_sat["ndvi_mean"],
                "o-",
                color="darkgreen",
                linewidth=2.5
            )
            ax.fill_between(
                df_sat["date"],
                df_sat["ndvi_min"],
                df_sat["ndvi_max"],
                alpha=0.25,
                color="green"
            )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.markdown("### 📊 Statistiques")
            st.metric("Moyenne", f"{df_sat['ndvi_mean'].mean():.3f}")
            st.metric("Max", f"{df_sat['ndvi_mean'].max():.3f}")
            st.metric("Min", f"{df_sat['ndvi_mean'].min():.3f}")
            st.metric("Écart-type", f"{df_sat['ndvi_mean'].std():.3f}")

    else:
        st.info("Chargez d'abord les données")

# =========================
# ONGLET CLIMAT (CORRIGÉ)
# =========================
with tabs[3]:
    st.subheader("🌦️ Analyse Climatique")

    if st.session_state.climate_data is not None:
        df = st.session_state.climate_data

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df["date"], df["rain"], alpha=0.6)
        ax.set_ylabel("Pluie (mm)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    else:
        st.info("Chargez d'abord les données")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666">
<b>AgriSight Pro</b> – Version stable (formatage corrigé uniquement)
</div>
""", unsafe_allow_html=True)
