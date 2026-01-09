# ============================================================
# APPLICATION AGRO-CLIMATIQUE & VEGETATION (STREAMLIT CLOUD READY)
# 100% OPEN & FREE – Compatible Streamlit.io
# Auteur : ChatGPT
# ============================================================

# --------------------
# REQUIREMENTS (requirements.txt)
# --------------------
# streamlit
# pandas
# numpy
# requests
# folium
# streamlit-folium
# shapely
# pyogrio
# pystac-client
# planetary-computer
# xarray
# stackstac
# matplotlib

# --------------------
# IMPORTS
# --------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping
from datetime import date
import matplotlib.pyplot as plt

import pyogrio
import shapely.geometry as geom

from pystac_client import Client
import planetary_computer as pc
import stackstac

# --------------------
# CONFIG STREAMLIT
# --------------------
st.set_page_config(page_title="Analyse Agro-Climatique & NDVI", layout="wide")
st.title("🌾 Application d’analyse agro-climatique et végétative")

# ============================================================
# 1. CHARGEMENT ZONE D'ÉTUDE (LIGHT & CLOUD SAFE)
# ============================================================
st.sidebar.header("1️⃣ Zone d’étude")
uploaded_file = st.sidebar.file_uploader(
    "Importer un GeoJSON ou SHP (zip)", type=["geojson", "zip"]
)

@st.cache_data
def load_vector(file):
    # Lecture GeoJSON sans geopandas (Streamlit Cloud safe)
    if file.name.endswith(".geojson"):
        import json
        data = json.load(file)
        features = data.get("features", [])
        geometries = [geom.shape(f["geometry"]) for f in features]
        gdf = pd.DataFrame({"geometry": geometries})
        gdf["geometry"] = gdf["geometry"].apply(lambda g: g)
        gdf = gdf.set_geometry("geometry", inplace=False)
        return gdf

    # Lecture SHP zip (nécessite pyogrio + geopandas)
    if file.name.endswith(".zip"):
        import geopandas as gpd
        gdf = gpd.read_file(file)
        gdf = gdf.to_crs(4326)
        return gdf

    raise ValueError("Format non supporté")

if not uploaded_file:
    st.info("Veuillez charger une zone d’étude pour commencer")
    st.stop()

gdf = load_vector(uploaded_file)
st.success("Zone chargée avec succès")

geometry = gdf.geometry.unary_union
centroid = geometry.centroid
lat, lon = centroid.y, centroid.x

# ============================================================
# 2. PARAMÈTRES D’ANALYSE
# ============================================================
st.sidebar.header("2️⃣ Paramètres d’analyse")
start_date = st.sidebar.date_input("Date de début", date(2023, 6, 1))
end_date = st.sidebar.date_input("Date de fin", date(2023, 10, 31))

culture = st.sidebar.selectbox(
    "Type de culture",
    ["Mil", "Sorgho", "Maïs", "Arachide", "Papayer"],
)

# ============================================================
# 3. DONNÉES CLIMATIQUES – NASA POWER
# ============================================================
st.subheader("🌦️ Données climatiques (NASA POWER)")

@st.cache_data
def get_nasa_power(lat, lon, start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOT,RH2M,WS2M"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON"
    )
    r = requests.get(url)
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df

climate_df = get_nasa_power(lat, lon, start_date, end_date)
st.dataframe(climate_df.head())

rain_total = climate_df["PRECTOT"].sum()
temp_mean = climate_df["T2M"].mean()

st.metric("🌧️ Pluie cumulée (mm)", round(rain_total, 1))
st.metric("🌡️ Température moyenne (°C)", round(temp_mean, 1))

# ============================================================
# 4. NDVI SENTINEL-2 (SAFE POUR STREAMLIT CLOUD)
# ============================================================
st.subheader("🛰️ NDVI moyen (Sentinel-2)")

@st.cache_data
def compute_ndvi_mean(geometry, start, end):
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=mapping(geometry),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 20}},
    )

    items = list(search.get_items())
    if not items:
        return None

    stack = stackstac.stack(
        items,
        assets=["B04", "B08"],
        bounds=geometry.bounds,
        resolution=100,
        chunksize=2048,
    )

    red = stack.sel(band="B04").mean("time")
    nir = stack.sel(band="B08").mean("time")

    ndvi = (nir - red) / (nir + red)
    return float(ndvi.mean().values)

ndvi_mean = compute_ndvi_mean(geometry, start_date, end_date)

if ndvi_mean is not None:
    st.metric("🌿 NDVI moyen", round(ndvi_mean, 3))
    st.metric("🌱 Indice biomasse (proxy)", round(ndvi_mean * 100, 1))
else:
    st.warning("Aucune image Sentinel-2 disponible pour cette période")

# ============================================================
# 5. CARTE INTERACTIVE
# ============================================================
st.subheader("🗺️ Carte interactive")

m = folium.Map(location=[lat, lon], zoom_start=10)
folium.GeoJson(gdf, name="Zone d’étude").add_to(m)
folium.LayerControl().add_to(m)

st_folium(m, height=500)

# ============================================================
# 6. INTERPRÉTATION AGRONOMIQUE (IA RULE-BASED FREE)
# ============================================================
st.subheader("🤖 Interprétation agronomique")

CROP_RULES = {
    "Mil": {"rain": (300, 800), "temp": (25, 35)},
    "Sorgho": {"rain": (400, 900), "temp": (24, 34)},
    "Maïs": {"rain": (500, 1200), "temp": (20, 30)},
    "Arachide": {"rain": (400, 1000), "temp": (22, 32)},
    "Papayer": {"rain": (800, 2000), "temp": (22, 30)},
}

rules = CROP_RULES[culture]

diagnostic = []

if rain_total < rules["rain"][0]:
    diagnostic.append("🌧️ Pluviométrie insuffisante (stress hydrique)")
elif rain_total > rules["rain"][1]:
    diagnostic.append("🌧️ Excès de pluie (risque maladies)")
else:
    diagnostic.append("✅ Pluviométrie favorable")

if not (rules["temp"][0] <= temp_mean <= rules["temp"][1]):
    diagnostic.append("🌡️ Température hors plage optimale")
else:
    diagnostic.append("✅ Température adaptée")

if ndvi_mean is not None and ndvi_mean < 0.4:
    diagnostic.append("🌱 Vigueur végétative faible")
elif ndvi_mean is not None:
    diagnostic.append("🌿 Bonne vigueur végétative")

st.markdown(f"### 🌾 Diagnostic – **{culture}**")
for d in diagnostic:
    st.write("-", d)

# ============================================================
# 7. EXPORT DES RÉSULTATS
# ============================================================
st.subheader("📤 Export des résultats")

export_df = climate_df.copy()
export_df["NDVI_mean"] = ndvi_mean
export_df["Culture"] = culture

csv = export_df.to_csv().encode("utf-8")
st.download_button(
    "📥 Télécharger les résultats (CSV)",
    csv,
    "resultats_agro_climat.csv",
    "text/csv",
)

st.success("✅ Analyse terminée – Application 100% OPEN & STREAMLIT CLOUD READY")
