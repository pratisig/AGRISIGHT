# Agrisight – Application Streamlit Agro-climatique & Végétative
# VERSION FINALE CORRIGÉE – STREAMLIT CLOUD READY

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import requests
from datetime import date

# =============================
# CONFIG STREAMLIT
# =============================
st.set_page_config(page_title="Agrisight", layout="wide")

st.title("🌱 Agrisight – Analyse Agro-climatique et Végétative")

# =============================
# FONCTIONS UTILITAIRES
# =============================

@st.cache_data(show_spinner=False)
def load_vector(file):
    try:
        gdf = gpd.read_file(file)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_nasa_power_point(lat, lon, start_date, end_date):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOT",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        if "properties" not in data:
            return pd.DataFrame()
        params_data = data["properties"].get("parameter", {})
        if not params_data:
            return pd.DataFrame()
        df = pd.DataFrame(params_data)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def sample_points_in_polygon(geom, n=5):
    minx, miny, maxx, maxy = geom.bounds
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 20:
        p = Point(
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        )
        if geom.contains(p):
            points.append(p)
        attempts += 1
    return points


def aggregate_climate_over_geometry(geometry, start_date, end_date):
    all_df = []
    points = sample_points_in_polygon(geometry, n=5)
    for p in points:
        df = get_nasa_power_point(p.y, p.x, start_date, end_date)
        if not df.empty:
            all_df.append(df)
    if not all_df:
        return pd.DataFrame()
    concat = pd.concat(all_df)
    agg = concat.groupby(concat.index).agg([
        "mean", "min", "max"
    ])
    return agg

# =============================
# SIDEBAR
# =============================

st.sidebar.header("📂 Données")
uploaded_file = st.sidebar.file_uploader("Charger un GeoJSON", type=["geojson", "json"])

start_date = st.sidebar.date_input("Date début", date(2020, 1, 1))
end_date = st.sidebar.date_input("Date fin", date.today())

# =============================
# CHARGEMENT DONNÉES
# =============================

gdf = None
if uploaded_file:
    gdf = load_vector(uploaded_file)

# =============================
# ONGLET
# =============================

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Cartographie",
    "🌦️ Climat",
    "📊 Indices",
    "📤 Export"
])

# =============================
# TAB 1 – CARTE
# =============================

with tab1:
    st.subheader("Carte interactive")
    m = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="OpenStreetMap")

    if gdf is not None:
        folium.GeoJson(gdf, name="Zones").add_to(m)

    st_folium(m, height=500, width="100%")

# =============================
# TAB 2 – CLIMAT
# =============================

with tab2:
    st.subheader("Données climatiques agrégées")
    if gdf is None:
        st.info("Veuillez charger un GeoJSON pour activer cette section")
    else:
        results = []
        for idx, row in gdf.iterrows():
            agg = aggregate_climate_over_geometry(row.geometry, start_date, end_date)
            if not agg.empty:
                agg["zone_id"] = idx
                results.append(agg)
        if results:
            climate_df = pd.concat(results)
            st.dataframe(climate_df)
        else:
            st.warning("Aucune donnée climatique disponible pour la période")

# =============================
# TAB 3 – INDICES
# =============================

with tab3:
    st.subheader("Indices agro-climatiques")
    st.info("Les indices (GDD, SPI, anomalies) seront calculés à partir des données climatiques agrégées")

# =============================
# TAB 4 – EXPORT
# =============================

with tab4:
    st.subheader("Export")
    st.info("Export CSV / PDF à implémenter")
