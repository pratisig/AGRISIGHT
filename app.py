import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
from shapely.geometry import Point, mapping
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json

# Configuration
st.set_page_config(page_title="AgriSight IA", layout="wide")

# Style CSS personnalisé
st.markdown("""
<style>
    .stAlert {border-radius: 10px;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriSight – Analyse Agro-climatique et Végétative Optimisée")

# --------------------
# SIDEBAR
# --------------------
st.sidebar.header("📂 Paramètres")

# Option 1: Upload fichier
uploaded_file = st.sidebar.file_uploader("Importer une zone (GeoJSON)", type=["geojson"])

# Option 2: Coordonnées manuelles (OPTIMISATION)
st.sidebar.subheader("OU entrer les coordonnées")
manual_lat = st.sidebar.number_input("Latitude", value=14.6937, format="%.4f")
manual_lon = st.sidebar.number_input("Longitude", value=-17.4441, format="%.4f")
use_manual = st.sidebar.checkbox("Utiliser les coordonnées manuelles")

# Dates (OPTIMISATION: limiter la période)
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Date début", date.today() - timedelta(days=90))
with col2:
    end_date = st.date_input("Date fin", date.today())

# Vérifier que la période n'excède pas 120 jours (OPTIMISATION)
date_diff = (end_date - start_date).days
if date_diff > 120:
    st.sidebar.warning("⚠️ Période limitée à 120 jours pour éviter les timeouts")
    end_date = start_date + timedelta(days=120)

culture = st.sidebar.selectbox("Type de culture", 
    ["Mil", "Sorgho", "Maïs", "Arachide", "Papayer", "Riz", "Niébé", "Manioc"])

# Bouton de chargement
load_data = st.sidebar.button("📥 Charger les données", type="primary")

# --------------------
# SESSION STATE
# --------------------
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'climate_data' not in st.session_state:
    st.session_state.climate_data = None
if 'ndvi_data' not in st.session_state:
    st.session_state.ndvi_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --------------------
# FONCTIONS OPTIMISÉES
# --------------------

@st.cache_data(ttl=3600)
def load_zone(file_bytes):
    """Charge un fichier GeoJSON avec cache"""
    try:
        gdf = gpd.read_file(BytesIO(file_bytes))
        return gdf.to_crs(4326)
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None

@st.cache_data(ttl=3600)
def get_climate_data_optimized(lat, lon, start, end):
    """
    Récupère les données climatiques NASA POWER
    OPTIMISATION: Un seul point, pas d'échantillonnage multiple
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON&community=AG"
    )
    
    try:
        with st.spinner("🌍 Récupération données climatiques..."):
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None
            
            data = r.json()
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
def get_ndvi_mock(lat, lon, start, end):
    """
    OPTIMISATION: Génère des données NDVI simulées basées sur la saison
    Pour production réelle: utiliser Google Earth Engine ou API Sentinel Hub
    """
    dates = pd.date_range(start, end, freq='10D')
    ndvi_values = []
    
    for d in dates:
        # Simuler NDVI basé sur le mois (saison des pluies)
        month = d.month
        if 6 <= month <= 9:  # Saison des pluies
            base_ndvi = 0.6 + np.random.normal(0, 0.1)
        elif month in [5, 10]:  # Transition
            base_ndvi = 0.4 + np.random.normal(0, 0.1)
        else:  # Saison sèche
            base_ndvi = 0.2 + np.random.normal(0, 0.05)
        
        ndvi_values.append(np.clip(base_ndvi, 0, 1))
    
    return pd.DataFrame({'date': dates, 'ndvi': ndvi_values})

def calculate_metrics(climate_df, ndvi_df):
    """Calcule les métriques agrégées"""
    if climate_df is None or ndvi_df is None:
        return {}
    
    metrics = {
        'ndvi_mean': ndvi_df['ndvi'].mean() if not ndvi_df.empty else 0,
        'temp_mean': climate_df['temp_mean'].mean() if not climate_df.empty else 0,
        'temp_min': climate_df['temp_min'].min() if not climate_df.empty else 0,
        'temp_max': climate_df['temp_max'].max() if not climate_df.empty else 0,
        'rain_total': climate_df['rain'].sum() if not climate_df.empty else 0,
        'rain_mean': climate_df['rain'].mean() if not climate_df.empty else 0
    }
    
    # Estimation rendement basique (à améliorer)
    if metrics['ndvi_mean'] > 0.6 and metrics['rain_total'] > 400:
        metrics['yield_potential'] = 2.5
    elif metrics['ndvi_mean'] > 0.4 and metrics['rain_total'] > 300:
        metrics['yield_potential'] = 1.8
    else:
        metrics['yield_potential'] = 1.0
    
    return metrics

# --------------------
# CHARGEMENT DES DONNÉES
# --------------------
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.session_state.gdf = load_zone(file_bytes)

if load_data:
    with st.spinner("⏳ Chargement en cours..."):
        # Déterminer les coordonnées
        if use_manual:
            lat, lon = manual_lat, manual_lon
        elif st.session_state.gdf is not None:
            centroid = st.session_state.gdf.geometry.centroid.iloc[0]
            lat, lon = centroid.y, centroid.x
        else:
            st.error("Veuillez importer une zone ou activer les coordonnées manuelles")
            st.stop()
        
        # Charger climat (1 seul point)
        climate_df = get_climate_data_optimized(lat, lon, start_date, end_date)
        st.session_state.climate_data = climate_df
        
        # Charger NDVI (simulé pour éviter timeout)
        ndvi_df = get_ndvi_mock(lat, lon, start_date, end_date)
        st.session_state.ndvi_data = ndvi_df
        
        if climate_df is not None:
            st.success("✅ Données chargées avec succès!")
        else:
            st.error("❌ Échec du chargement")

# --------------------
# ONGLETS
# --------------------
tabs = st.tabs(["📊 Vue d'ensemble", "🗺️ Carte", "🛰️ NDVI", "🌦️ Climat", "🤖 Analyse IA", "📄 Rapport"])

# --------------------
# ONGLET VUE D'ENSEMBLE
# --------------------
with tabs[0]:
    st.subheader("📊 Tableau de bord")
    
    if st.session_state.climate_data is not None and st.session_state.ndvi_data is not None:
        metrics = calculate_metrics(st.session_state.climate_data, st.session_state.ndvi_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌱 NDVI Moyen", f"{metrics['ndvi_mean']:.2f}", 
                     delta="Santé végétale")
        with col2:
            st.metric("🌡️ Température Moy.", f"{metrics['temp_mean']:.1f}°C",
                     delta=f"{metrics['temp_min']:.1f} - {metrics['temp_max']:.1f}°C")
        with col3:
            st.metric("💧 Pluie Totale", f"{metrics['rain_total']:.0f} mm",
                     delta=f"{metrics['rain_mean']:.1f} mm/jour")
        with col4:
            st.metric("📈 Rendement Estimé", f"{metrics['yield_potential']:.1f} t/ha",
                     delta="Potentiel")
        
        # Graphique combiné
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        # NDVI
        ax1.plot(st.session_state.ndvi_data['date'], 
                st.session_state.ndvi_data['ndvi'], 
                color='green', linewidth=2, marker='o')
        ax1.fill_between(st.session_state.ndvi_data['date'], 
                         st.session_state.ndvi_data['ndvi'], 
                         alpha=0.3, color='green')
        ax1.set_ylabel('NDVI', fontsize=12)
        ax1.set_title('Évolution de la vigueur végétale (NDVI)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Climat
        ax2_temp = ax2.twinx()
        ax2.bar(st.session_state.climate_data['date'], 
               st.session_state.climate_data['rain'], 
               color='blue', alpha=0.4, label='Pluie (mm)')
        ax2_temp.plot(st.session_state.climate_data['date'], 
                     st.session_state.climate_data['temp_mean'], 
                     color='red', linewidth=2, label='Température (°C)')
        
        ax2.set_ylabel('Pluie (mm)', fontsize=12, color='blue')
        ax2_temp.set_ylabel('Température (°C)', fontsize=12, color='red')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_temp.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper left')
        ax2_temp.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("👆 Cliquez sur 'Charger les données' dans la barre latérale pour commencer")

# --------------------
# ONGLET CARTE
# --------------------
with tabs[1]:
    st.subheader("🗺️ Carte Interactive")
    
    if use_manual:
        center = [manual_lat, manual_lon]
    elif st.session_state.gdf is not None:
        center = [st.session_state.gdf.geometry.centroid.y.mean(), 
                 st.session_state.gdf.geometry.centroid.x.mean()]
    else:
        center = [14.6937, -17.4441]  # Dakar par défaut
    
    m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")
    m.add_child(MeasureControl())
    
    # Ajouter zone importée
    if st.session_state.gdf is not None:
        folium.GeoJson(
            st.session_state.gdf,
            style_function=lambda x: {
                'fillColor': '#228B22',
                'color': '#006400',
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip="Zone d'analyse"
        ).add_to(m)
    
    # Ajouter marqueur point manuel
    if use_manual:
        folium.Marker(
            [manual_lat, manual_lon],
            popup=f"Point d'analyse<br>Lat: {manual_lat}<br>Lon: {manual_lon}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Outil de dessin
    draw = Draw(export=True)
    draw.add_to(m)
    
    map_data = st_folium(m, height=500, width=None)

# --------------------
# ONGLET NDVI
# --------------------
with tabs[2]:
    st.subheader("🛰️ Analyse NDVI (Indice de Végétation)")
    
    if st.session_state.ndvi_data is not None:
        df = st.session_state.ndvi_data
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['date'], df['ndvi'], color='darkgreen', linewidth=2, marker='o')
            ax.fill_between(df['date'], df['ndvi'], alpha=0.3, color='green')
            ax.axhline(y=0.6, color='orange', linestyle='--', label='Seuil optimal')
            ax.axhline(y=0.3, color='red', linestyle='--', label='Seuil stress')
            ax.set_ylabel('NDVI', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title('Évolution temporelle du NDVI', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Interprétation")
            ndvi_mean = df['ndvi'].mean()
            
            if ndvi_mean > 0.6:
                st.success("✅ **Excellente vigueur végétale**")
                st.write("La culture se développe très bien.")
            elif ndvi_mean > 0.4:
                st.warning("⚠️ **Vigueur modérée**")
                st.write("La culture se développe correctement mais peut être améliorée.")
            else:
                st.error("❌ **Stress végétal détecté**")
                st.write("La culture montre des signes de stress.")
            
            st.markdown("### Statistiques")
            st.write(f"**NDVI moyen:** {ndvi_mean:.3f}")
            st.write(f"**NDVI max:** {df['ndvi'].max():.3f}")
            st.write(f"**NDVI min:** {df['ndvi'].min():.3f}")
            st.write(f"**Écart-type:** {df['ndvi'].std():.3f}")
    else:
        st.info("Chargez d'abord les données")

# --------------------
# ONGLET CLIMAT
# --------------------
with tabs[3]:
    st.subheader("🌦️ Analyse Climatique")
    
    if st.session_state.climate_data is not None:
        df = st.session_state.climate_data
        
        # Graphique température
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.fill_between(df['date'], df['temp_min'], df['temp_max'], 
                         color='lightcoral', alpha=0.3, label='Plage de température')
        ax1.plot(df['date'], df['temp_mean'], color='red', linewidth=2, label='Température moyenne')
        ax1.set_ylabel('Température (°C)', fontsize=12)
        ax1.set_title('Évolution des températures', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # Graphique précipitations
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.bar(df['date'], df['rain'], color='blue', alpha=0.6)
        ax2.set_ylabel('Précipitation (mm)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Précipitations journalières', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig2)
        
        # Statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temp. Moyenne", f"{df['temp_mean'].mean():.1f}°C")
            st.metric("🔥 Temp. Max", f"{df['temp_max'].max():.1f}°C")
        with col2:
            st.metric("💧 Pluie Totale", f"{df['rain'].sum():.0f} mm")
            st.metric("☔ Pluie Max/jour", f"{df['rain'].max():.1f} mm")
        with col3:
            st.metric("📅 Jours de pluie", f"{(df['rain'] > 1).sum()} jours")
            st.metric("💦 Pluie Moyenne", f"{df['rain'].mean():.1f} mm/jour")
    else:
        st.info("Chargez d'abord les données")

# --------------------
# ONGLET ANALYSE IA
# --------------------
with tabs[4]:
    st.subheader("🤖 Analyse et Recommandations par IA")
    
    if st.session_state.climate_data is not None and st.session_state.ndvi_data is not None:
        metrics = calculate_metrics(st.session_state.climate_data, st.session_state.ndvi_data)
        
        if st.button("🚀 Générer l'analyse IA", type="primary"):
            with st.spinner("🤖 Claude analyse vos données..."):
                # Préparer les données pour l'IA
                ndvi_series = ", ".join([f"{row['date'].strftime('%Y-%m-%d')}: {row['ndvi']:.2f}" 
                                        for _, row in st.session_state.ndvi_data.iterrows()])
                
                prompt = f"""Tu es un expert en agronomie, climatologie et télédétection. Analyse cette parcelle agricole et fournis un diagnostic détaillé.

INFORMATIONS DE LA PARCELLE :
- Culture : {culture}
- Période d'analyse : {start_date} à {end_date}
- NDVI moyen : {metrics['ndvi_mean']:.3f}
- Série temporelle NDVI : {ndvi_series}
- Température moyenne : {metrics['temp_mean']:.1f} °C
- Température min-max : {metrics['temp_min']:.1f} - {metrics['temp_max']:.1f} °C
- Pluviométrie totale : {metrics['rain_total']:.0f} mm
- Pluviométrie moyenne : {metrics['rain_mean']:.1f} mm/jour
- Rendement potentiel estimé : {metrics['yield_potential']:.1f} t/ha
- Zone géographique : Sénégal (contexte sahélien)

Fournis une analyse structurée au format JSON avec cette structure :
{{
  "diagnostic": {{
    "vigueur": "Description de la vigueur végétative basée sur le NDVI",
    "stress": "Identification des stress climatiques (hydrique, température, etc.)",
    "etat_general": "État général de la culture et perspectives"
  }},
  "recommandations": [
    {{
      "categorie": "Catégorie (Semis, Irrigation, Fertilisation, etc.)",
      "conseil": "Conseil pratique et actionnable",
      "explication": "Explication scientifique simple du pourquoi",
      "priorite": "haute/moyenne/basse"
    }}
  ],
  "alertes": [
    {{
      "type": "Type d'alerte",
      "message": "Description de l'alerte",
      "action": "Action immédiate recommandée"
    }}
  ],
  "suivi": {{
    "frequence": "Fréquence de monitoring recommandée",
    "indicateurs": ["Liste des indicateurs clés à surveiller"]
  }}
}}

Adapte les conseils au contexte sahélien et rends-les compréhensibles pour un agriculteur. Réponds UNIQUEMENT avec le JSON."""

                try:
                    # Appel API Anthropic
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 4000,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        text_content = "".join([item['text'] for item in data['content'] if item['type'] == 'text'])
                        clean_text = text_content.strip().replace('```json', '').replace('```', '')
                        analysis = json.loads(clean_text)
                        st.session_state.analysis_result = analysis
                    else:
                        st.error(f"Erreur API: {response.status_code}")
                        analysis = None
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {e}")
                    analysis = None
        
        # Afficher les résultats
        if st.session_state.analysis_result:
            analysis = st.session_state.analysis_result
            
            # Diagnostic
            st.markdown("### 📋 Diagnostic")
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                st.info(f"**Vigueur végétative:**\n\n{analysis['diagnostic']['vigueur']}")
            with diag_col2:
                st.warning(f"**Stress identifiés:**\n\n{analysis['diagnostic']['stress']}")
            
            st.success(f"**État général:**\n\n{analysis['diagnostic']['etat_general']}")
            
            # Alertes
            if analysis.get('alertes'):
                st.markdown("### ⚠️ Alertes")
                for alerte in analysis['alertes']:
                    with st.expander(f"🚨 {alerte['type']}", expanded=True):
                        st.write(alerte['message'])
                        st.markdown(f"**Action:** {alerte['action']}")
            
            # Recommandations
            st.markdown("### 💡 Recommandations")
            for rec in analysis['recommandations']:
                priority_color = {
                    'haute': '🔴',
                    'moyenne': '🟠',
                    'basse': '🟢'
                }.get(rec['priorite'], '⚪')
                
                with st.expander(f"{priority_color} {rec['categorie']} - Priorité {rec['priorite']}", 
                               expanded=(rec['priorite'] == 'haute')):
                    st.markdown(f"**Conseil:** {rec['conseil']}")
                    st.markdown(f"**Pourquoi:** {rec['explication']}")
            
            # Plan de suivi
            st.markdown("### 📅 Plan de Suivi")
            st.info(f"**Fréquence:** {analysis['suivi']['frequence']}")
            st.write("**Indicateurs à surveiller:**")
            for ind in analysis['suivi']['indicateurs']:
                st.write(f"• {ind}")
    else:
        st.info("Chargez d'abord les données pour générer une analyse")

# --------------------
# ONGLET RAPPORT
# --------------------
with tabs[5]:
    st.subheader("📄 Génération de Rapport")
    
    if st.session_state.climate_data is not None:
        st.write("Fonctionnalité de génération PDF en cours de développement")
        
        if st.button("📥 Télécharger les données CSV"):
            # Export CSV des données
            csv_climate = st.session_state.climate_data.to_csv(index=False)
            st.download_button(
                "Télécharger données climatiques",
                csv_climate,
                "climate_data.csv",
                "text/csv"
            )
            
            if st.session_state.ndvi_data is not None:
                csv_ndvi = st.session_state.ndvi_data.to_csv(index=False)
                st.download_button(
                    "Télécharger données NDVI",
                    csv_ndvi,
                    "ndvi_data.csv",
                    "text/csv"
                )
    else:
        st.info("Chargez d'abord les données")

# Footer
st.markdown("---")
st.markdown("🌍 **AgriSight** - Analyse agricole intelligente par télédétection et IA | Optimisé pour Streamlit Cloud")
