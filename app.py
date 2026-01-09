# ============================================================
# AGRISIGHT – APPLICATION AGRO-CLIMATIQUE & NDVI (VERSION COMPLETE)
# NDVI Sentinel-2 + Climat NASA POWER + Diagnostic cultures
# Version STABLE – Streamlit Cloud compatible (FREE & OPEN DATA)
# ============================================================

# --------------------
# DEPENDANCES (requirements.txt)
# --------------------
# streamlit
# geopandas
# pandas
# numpy
# shapely
# folium
# streamlit-folium
# requests
# pystac-client
# planetary-computer
# rasterio
# matplotlib

# --------------------
# IMPORTS
# --------------------
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, mapping
from datetime import date
import random
import matplotlib.pyplot as plt

from pystac_client import Client
import planetary_computer as pc
import rasterio

# --------------------
# CONFIG STREAMLIT
# --------------------
st.set_page_config(page_title="AgriSight – Agro-climat & NDVI", layout="wide")
st.title("🌾 AgriSight – Analyse agro-climatique & végétative")
st.caption("Données open : NASA POWER & Sentinel-2 | Analyse spatiale agrégée")

# ============================================================
# SIDEBAR – PARAMETRES
# ============================================================
st.sidebar.header("📂 Zone & paramètres")

uploaded_file = st.sidebar.file_uploader(
    "Importer une zone (GeoJSON ou SHP zippé)",
    type=["geojson", "zip"]
)

start_date = st.sidebar.date_input("Date de début", date(2023, 6, 1))
end_date = st.sidebar.date_input("Date de fin", date(2023, 10, 31))

culture = st.sidebar.selectbox(
    "Type de culture",
    ["Mil", "Sorgho", "Maïs", "Arachide", "Papayer"]
)

# ============================================================
# CHARGEMENT DE LA ZONE (ROBUSTE)
# ============================================================

def load_zone(file):
    if file is None:
        return None
    gdf = gpd.read_file(file)
    return gdf.to_crs(4326)

try:
    gdf = load_zone(uploaded_file)
except Exception:
    gdf = None

# ============================================================
# CARTE INTERACTIVE (TOUJOURS AFFICHEE)
# ============================================================
st.subheader("🗺️ Carte interactive")

m = folium.Map(location=[14, -14], zoom_start=6, tiles="OpenStreetMap")

if gdf is not None:
    folium.GeoJson(gdf, name="Zone d'étude").add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=500)

# ============================================================
# ECHANTILLONNAGE SPATIAL – POINTS INTERNES
# ============================================================

def sample_points_in_polygon(geometry, n_points=5):
    minx, miny, maxx, maxy = geometry.bounds
    points = []
    attempts = 0
    while len(points) < n_points and attempts < 1000:
        p = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )
        if geometry.contains(p):
            points.append(p)
        attempts += 1
    return points

# ============================================================
# NASA POWER – EXTRACTION PAR POINT
# ============================================================

def get_nasa_power_point(lat, lon, start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOT"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON"
    )
    r = requests.get(url)
    data = r.json()

    if "properties" not in data:
        return pd.DataFrame()

    params = data["properties"].get("parameter", {})
    if not params:
        return pd.DataFrame()

    df = pd.DataFrame(params)
    df.index = pd.to_datetime(df.index)
    df["lat"] = lat
    df["lon"] = lon
    return df

# ============================================================
# CLIMAT – AGRÉGATION SPATIALE & TEMPORELLE
# ============================================================
st.subheader("🌦️ Données climatiques agrégées")

climate_points_df = pd.DataFrame()
climate_agg_df = pd.DataFrame()

if gdf is not None:
    point_records = []

    for geom in gdf.geometry:
        points = sample_points_in_polygon(geom, n_points=5)
        for p in points:
            df_p = get_nasa_power_point(p.y, p.x, start_date, end_date)
            if not df_p.empty:
                df_p = df_p.reset_index().rename(columns={"index": "date"})
                point_records.append(df_p)

    if point_records:
        climate_points_df = pd.concat(point_records, ignore_index=True)

        climate_agg_df = (
            climate_points_df
            .groupby("date")
            .agg({
                "T2M": ["mean", "min", "max"],
                "PRECTOT": ["mean", "min", "max"],
            })
        )

        st.markdown("**Aperçu – données par points (avec coordonnées)**")
        st.dataframe(climate_points_df.head())

        st.markdown("**Aperçu – données climatiques agrégées (zone)**")
        st.dataframe(climate_agg_df.head())
    else:
        st.info("Aucune donnée climatique récupérée")
else:
    st.info("Chargez une zone pour lancer l'analyse climatique")

# ============================================================
# NDVI – SENTINEL-2 (MOYENNE ZONALE)
# ============================================================
st.subheader("🛰️ NDVI Sentinel-2 (moyenne zonale)")

ndvi_mean = None

if gdf is not None:
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=mapping(gdf.geometry.unary_union),
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}},
    )

    items = list(search.get_items())

    if items:
        item = items[0]
        with rasterio.open(item.assets["B04"].href) as red, \
             rasterio.open(item.assets["B08"].href) as nir:

            red_arr = red.read(1).astype(float)
            nir_arr = nir.read(1).astype(float)

            ndvi = (nir_arr - red_arr) / (nir_arr + red_arr + 1e-6)
            ndvi_mean = np.nanmean(ndvi)

            st.metric("NDVI moyen", round(ndvi_mean, 3))

            fig, ax = plt.subplots()
            im = ax.imshow(ndvi, cmap="RdYlGn")
            plt.colorbar(im, ax=ax, label="NDVI")
            st.pyplot(fig)
    else:
        st.info("Aucune image Sentinel-2 disponible sur la période")
else:
    st.info("Chargez une zone pour calculer le NDVI")

# ============================================================
# DIAGNOSTIC AGRONOMIQUE – SYSTEME EXPERT
# ============================================================
st.subheader("🤖 Diagnostic agronomique")

CROP_RULES = {
    "Mil": {"rain": (300, 800), "temp": (25, 35), "ndvi": 0.35},
    "Sorgho": {"rain": (400, 900), "temp": (24, 34), "ndvi": 0.40},
    "Maïs": {"rain": (500, 1200), "temp": (20, 30), "ndvi": 0.45},
    "Arachide": {"rain": (400, 1000), "temp": (22, 32), "ndvi": 0.40},
    "Papayer": {"rain": (800, 2000), "temp": (22, 30), "ndvi": 0.50},
}

if not climate_agg_df.empty:
    rain_total = climate_agg_df[('PRECTOT', 'mean')].sum()
    temp_mean = climate_agg_df[('T2M', 'mean')].mean()

    rules = CROP_RULES[culture]
    diagnosis = []

    if rain_total < rules['rain'][0]:
        diagnosis.append("🌧️ Pluviométrie insuffisante – stress hydrique")
    elif rain_total > rules['rain'][1]:
        diagnosis.append("🌧️ Excès de pluie – risque maladies")
    else:
        diagnosis.append("✅ Pluviométrie favorable")

    if not (rules['temp'][0] <= temp_mean <= rules['temp'][1]):
        diagnosis.append("🌡️ Température moyenne hors plage optimale")
    else:
        diagnosis.append("✅ Température adaptée à la culture")

    if ndvi_mean is not None:
        if ndvi_mean < rules['ndvi']:
            diagnosis.append("🌱 Vigueur végétative faible (NDVI bas)")
        else:
            diagnosis.append("🌿 Bonne vigueur végétative")

    st.markdown(f"### Diagnostic pour la culture : **{culture}**")
    for d in diagnosis:
        st.write("-", d)
else:
    st.info("Diagnostic disponible après calcul climatique")

# ============================================================
# EXPORT DES RESULTATS
# ============================================================
st.subheader("📤 Export des résultats")

if not climate_points_df.empty:
    export_df = climate_points_df.copy()
    export_df["Culture"] = culture
    if ndvi_mean is not None:
        export_df["NDVI_mean_zone"] = ndvi_mean

    csv = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Télécharger les données climatiques (CSV)",
        csv,
        file_name="agrisight_climat_points.csv",
        mime="text/csv",
    )

st.success("Analyse terminée – Application agro-climatique opérationnelle")
