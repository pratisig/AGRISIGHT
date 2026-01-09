# ============================================================
# AGRISIGHT – DASHBOARD INTERACTIF AVANCEE
# Chargement sur demande, dessin parcelle, diagnostic IA, PDF/CSV
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import Point, mapping
from datetime import date
import rasterio
import matplotlib.pyplot as plt
from fpdf import FPDF
import random

# Optional: import OpenAI if you have API key
import openai

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="AgriSight IA", layout="wide")
st.title("🌾 AgriSight – Analyse Agro-climatique et Végétative")

# --------------------
# SIDEBAR
# --------------------
st.sidebar.header("📂 Paramètres")
uploaded_file = st.sidebar.file_uploader("Importer une zone (GeoJSON ou ZIP)", type=["geojson", "zip"])
start_date = st.sidebar.date_input("Date de début", date(2023,6,1))
end_date = st.sidebar.date_input("Date de fin", date(2023,10,31))
culture = st.sidebar.selectbox("Type de culture", ["Mil", "Sorgho", "Maïs", "Arachide", "Papayer"])

# Bouton pour charger les données
load_data = st.sidebar.button("📥 Charger les données")

# --------------------
# INITIALISATION VARIABLES
# --------------------
gdf = None
climate_daily = pd.DataFrame()
ndvi_timeseries = pd.DataFrame()
ndvi_mean, rain_total, temp_mean, yield_potential = [np.nan]*4

# --------------------
# Fonction chargement zone
# --------------------
def load_zone(file):
    if file is None: return None
    try:
        gdf = gpd.read_file(file)
        return gdf.to_crs(4326)
    except:
        return None

if uploaded_file:
    gdf = load_zone(uploaded_file)

# --------------------
# Onglets
# --------------------
tabs = st.tabs(["Carte","NDVI","Climat","Rendement","Conseils IA","PDF"])

# --------------------
# Onglet Carte + dessin
# --------------------
with tabs[0]:
    st.subheader("🗺️ Carte interactive")
    if gdf is not None:
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    else:
        center = [14, -14]  # Default center

    m = folium.Map(location=center, zoom_start=6)

    if gdf is not None:
        folium.GeoJson(gdf, name="Zone importée").add_to(m)

    draw = Draw(export=True)
    draw.add_to(m)

    drawn = st_folium(m, height=500, width=900)

    gdf_drawn = None
    if drawn and 'all_drawings' in drawn and drawn['all_drawings']:
        features = drawn['all_drawings']
        gdf_drawn = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        st.success(f"Zone dessinée avec {len(gdf_drawn)} polygones")

# --------------------
# Fonction échantillonnage points
# --------------------
def sample_points(geometry, n=5):
    minx, miny, maxx, maxy = geometry.bounds
    pts = []
    while len(pts) < n:
        p = Point(random.uniform(minx,maxx), random.uniform(miny,maxy))
        if geometry.contains(p): pts.append(p)
    return pts

# --------------------
# Fonction NASA POWER
# --------------------
def nasa_power(lat, lon, start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOT"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON"
    )
    try:
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
    except:
        return pd.DataFrame()

# --------------------
# Extraction données sur bouton
# --------------------
if load_data and (gdf is not None or gdf_drawn is not None):
    st.success("🔄 Extraction des données en cours...")

    zone_to_use = gdf_drawn if gdf_drawn is not None else gdf

    # Climat
    climate_points = []
    for geom in zone_to_use.geometry:
        for p in sample_points(geom,5):
            df_p = nasa_power(p.y, p.x, start_date, end_date)
            if not df_p.empty: climate_points.append(df_p.reset_index().rename(columns={'index':'date'}))

    if climate_points:
        climate_points_df = pd.concat(climate_points, ignore_index=True)
        climate_daily = climate_points_df.groupby('date').agg({'T2M':['mean','min','max'],'PRECTOT':['mean','min','max']})

    # NDVI (simplifié: moyenne NDVI sur les images Sentinel-2)
    try:
        from pystac_client import Client
        import planetary_computer as pc
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        from rasterio import open as rio_open

        search = catalog.search(collections=["sentinel-2-l2a"], intersects=mapping(zone_to_use.geometry.unary_union), datetime=f"{start_date}/{end_date}", query={"eo:cloud_cover":{"lt":20}})
        items = list(search.get_items())
        ndvi_list = []
        for item in items:
            with rio_open(item.assets['B04'].href) as red, rio_open(item.assets['B08'].href) as nir:
                ndvi = (nir.read(1)-red.read(1))/(nir.read(1)+red.read(1)+1e-6)
                ndvi_list.append({'date':pd.to_datetime(item.datetime),'ndvi_mean':np.nanmean(ndvi)})
        if ndvi_list:
            ndvi_timeseries = pd.DataFrame(ndvi_list).sort_values('date')
    except:
        st.warning("Impossible de récupérer NDVI Sentinel-2")

    st.success("✅ Données extraites")

# --------------------
# Onglet Climat
# --------------------
with tabs[2]:
    st.subheader("🌦️ Climat")
    if not climate_daily.empty:
        st.line_chart(climate_daily['T2M']['mean'])
    else:
        st.info("Données climatiques non encore chargées")

# --------------------
# Onglet NDVI
# --------------------
with tabs[1]:
    st.subheader("🛰️ NDVI")
    if not ndvi_timeseries.empty:
        st.line_chart(ndvi_timeseries.set_index('date')['ndvi_mean'])
    else:
        st.info("NDVI non encore chargé")

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
    else:
        st.info("Rendement disponible après extraction climat + NDVI")

# --------------------
# Onglet Conseils IA
# --------------------
with tabs[4]:
    st.subheader("🤖 Conseils IA")
    if not climate_daily.empty and not ndvi_timeseries.empty:
        try:
            prompt = f"Analyse pour la culture {culture}. NDVI moyen: {ndvi_mean:.2f}, Pluie totale: {rain_total:.1f} mm, Temp moyenne: {temp_mean:.1f}°C. Donne des conseils sur période de semis, amendement, stress hydrique, rotation, et autres conseils agronomiques."
            # Remplacez par votre clé API OpenAI dans votre environnement
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}]
            )
            advice = response.choices[0].message.content
            st.write(advice)
        except:
            st.info("Impossible de générer conseil IA (API clé manquante ou problème réseau)")
    else:
        st.info("Les données doivent être chargées pour obtenir les conseils IA")

# --------------------
# Onglet PDF
# --------------------
with tabs[5]:
    st.subheader("📄 Rapport PDF")
    if st.button("Générer PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial","B",16)
        pdf.cell(0,10,"AgriSight – Rapport avancé",0,1,'C')
        pdf.set_font("Arial","",12)
        pdf.ln(5)
        pdf.cell(0,10,f"Culture : {culture}",0,1)
        pdf.cell(0,10,f"Période : {start_date} à {end_date}",0,1)
        if not climate_daily.empty: pdf.cell(0,10,f"Pluviométrie totale : {rain_total:.1f} mm",0,1)
        if not ndvi_timeseries.empty: pdf.cell(0,10,f"NDVI moyen : {ndvi_mean:.2f}",0,1)
        if 'yield_potential' in locals(): pdf.cell(0,10,f"Rendement potentiel : {round(yield_potential,2)} t/ha",0,1)
        pdf_file = "agrisight_dashboard_rapport.pdf"
        pdf.output(pdf_file)
        with open(pdf_file,'rb') as f: st.download_button("📥 Télécharger PDF", f, pdf_file,"application/pdf")

st.success("✅ Dashboard prêt et interactif")
