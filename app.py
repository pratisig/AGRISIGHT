# ============================================================
# AGRISIGHT – DASHBOARD AVANCEE
# Version Streamlit interactive avec onglets : Carte, NDVI, Climat, Rendement, PDF
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, mapping
from datetime import date, timedelta
import rasterio
import matplotlib.pyplot as plt
from fpdf import FPDF
import random
from pystac_client import Client
import planetary_computer as pc

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="AgriSight Dashboard", layout="wide")
st.title("🌾 AgriSight – Dashboard interactif avancé")

# --------------------
# SIDEBAR
# --------------------
st.sidebar.header("📂 Paramètres")
uploaded_file = st.sidebar.file_uploader("Importer une zone (GeoJSON ou SHP zip)", type=["geojson", "zip"])
start_date = st.sidebar.date_input("Date de début", date(2023,6,1))
end_date = st.sidebar.date_input("Date de fin", date(2023,10,31))
culture = st.sidebar.selectbox("Type de culture", ["Mil", "Sorgho", "Maïs", "Arachide", "Papayer"])

# --------------------
# CHARGEMENT ZONE
# --------------------
def load_zone(file):
    if file is None: return None
    gdf = gpd.read_file(file)
    return gdf.to_crs(4326)

try:
    gdf = load_zone(uploaded_file)
except:
    gdf = None

# --------------------
# ONGLET INTERACTIF
# --------------------
tabs = st.tabs(["Carte","NDVI","Climat","Rendement","PDF"])

# --------------------
# Onglet Carte
# --------------------
with tabs[0]:
    st.subheader("🗺️ Carte interactive")
    m = folium.Map(location=[14,-14], zoom_start=6)
    if gdf is not None:
        folium.GeoJson(gdf, name="Zone").add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, height=500)

# --------------------
# Fonction échantillonnage points
# --------------------
def sample_points(geometry, n=5):
    minx, miny, maxx, maxy = geometry.bounds
    pts = []
    while len(pts)<n:
        p = Point(random.uniform(minx,maxx), random.uniform(miny,maxy))
        if geometry.contains(p): pts.append(p)
    return pts

# --------------------
# NASA POWER
# --------------------
def nasa_power(lat, lon, start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOT"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON"
    )
    r = requests.get(url)
    data = r.json()
    if "properties" not in data: return pd.DataFrame()
    params = data["properties"].get("parameter", {})
    if not params: return pd.DataFrame()
    df = pd.DataFrame(params)
    df.index = pd.to_datetime(df.index)
    df['lat'] = lat
    df['lon'] = lon
    return df

# --------------------
# Onglet Climat
# --------------------
with tabs[2]:
    st.subheader("🌦️ Données climatiques")
    climate_points = []
    if gdf is not None:
        for geom in gdf.geometry:
            for p in sample_points(geom,5):
                df_p = nasa_power(p.y, p.x, start_date, end_date)
                if not df_p.empty: climate_points.append(df_p.reset_index().rename(columns={'index':'date'}))
        if climate_points:
            climate_points_df = pd.concat(climate_points, ignore_index=True)
            climate_daily = climate_points_df.groupby('date').agg({'T2M':['mean','min','max'],'PRECTOT':['mean','min','max']})
            st.line_chart(climate_daily['T2M']['mean'], use_container_width=True)
        else:
            st.info("Pas de données climatiques disponibles")

# --------------------
# Onglet NDVI
# --------------------
with tabs[1]:
    st.subheader("🛰️ NDVI par date")
    ndvi_timeseries = pd.DataFrame()
    if gdf is not None:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        search = catalog.search(collections=["sentinel-2-l2a"], intersects=mapping(gdf.geometry.unary_union), datetime=f"{start_date}/{end_date}", query={"eo:cloud_cover":{"lt":20}})
        items = list(search.get_items())
        if items:
            ndvi_list = []
            for item in items:
                with rasterio.open(item.assets['B04'].href) as red, rasterio.open(item.assets['B08'].href) as nir:
                    ndvi = (nir.read(1)-red.read(1))/(nir.read(1)+red.read(1)+1e-6)
                    ndvi_list.append({'date':pd.to_datetime(item.datetime),'ndvi_mean':np.nanmean(ndvi)})
            ndvi_timeseries = pd.DataFrame(ndvi_list).sort_values('date')
            st.line_chart(ndvi_timeseries.set_index('date')['ndvi_mean'])
        else:
            st.info("Pas d'image Sentinel-2 disponible")

# --------------------
# Onglet Rendement
# --------------------
with tabs[3]:
    st.subheader("🌾 Rendement potentiel")
    if not climate_daily.empty and not ndvi_timeseries.empty:
        ndvi_mean = ndvi_timeseries['ndvi_mean'].mean()
        rain_total = climate_daily['PRECTOT']['mean'].sum()
        temp_mean = climate_daily['T2M']['mean'].mean()
        water_factor = min(1,rain_total/600)
        temp_factor = max(0,1-abs(temp_mean-28)/20)
        ndvi_factor = min(1,ndvi_mean/0.6)
        YIELD_REF = {"Mil":2.5,"Sorgho":3.0,"Maïs":6.0,"Arachide":3.5,"Papayer":40.0}
        yield_potential = YIELD_REF[culture]*water_factor*temp_factor*ndvi_factor
        st.metric("Rendement potentiel estimé", f"{round(yield_potential,2)} t/ha")
        st.line_chart(pd.DataFrame({'Rendement potentiel':[yield_potential]*len(ndvi_timeseries)}, index=ndvi_timeseries['date']))
    else:
        st.info("Rendement disponible après extraction climat + NDVI")

# --------------------
# Onglet PDF
# --------------------
with tabs[4]:
    st.subheader("📄 Rapport PDF")
    if st.button("Générer PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial","B",16)
        pdf.cell(0,10,"AgriSight – Rapport avancé",0,1,'C')
        pdf.ln(5)
        pdf.set_font("Arial","",12)
        pdf.cell(0,10,f"Culture : {culture}",0,1)
        pdf.cell(0,10,f"Période : {start_date} à {end_date}",0,1)
        if not climate_daily.empty: pdf.cell(0,10,f"Pluviométrie totale : {rain_total:.1f} mm",0,1)
        if not ndvi_timeseries.empty: pdf.cell(0,10,f"NDVI moyen : {ndvi_mean:.2f}",0,1)
        if 'yield_potential' in locals(): pdf.cell(0,10,f"Rendement potentiel : {round(yield_potential,2)} t/ha",0,1)
        pdf_file = "agrisight_dashboard_rapport.pdf"
        pdf.output(pdf_file)
        with open(pdf_file,'rb') as f: st.download_button("📥 Télécharger PDF", f, pdf_file,"application/pdf")

st.success("✅ Dashboard complet avec onglets NDVI, Climat, Rendement et PDF")
