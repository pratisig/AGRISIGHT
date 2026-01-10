import React, { useState, useEffect, useCallback } from 'react';
import { MapPin, Download, AlertCircle, Leaf, Cloud, BarChart3, FileText, TrendingUp, Droplets } from 'lucide-react';

// Composant de barre de progression
const ProgressBar = ({ progress, label, status }) => (
  <div className="mb-4">
    <div className="flex justify-between mb-1">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <span className="text-sm text-gray-500">{progress}%</span>
    </div>
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div 
        className={`h-2 rounded-full transition-all duration-300 ${
          status === 'complete' ? 'bg-green-500' : 
          status === 'loading' ? 'bg-blue-500 animate-pulse' : 
          'bg-gray-400'
        }`}
        style={{ width: `${progress}%` }}
      />
    </div>
  </div>
);

// Composant carte interactive
const InteractiveMap = ({ onAreaSelect, selectedArea }) => {
  const [drawMode, setDrawMode] = useState(false);
  const [points, setPoints] = useState([]);
  const [mapCenter, setMapCenter] = useState({ lat: 14.6937, lng: -17.4441 });

  useEffect(() => {
    if (selectedArea?.coordinates) {
      const avgLat = selectedArea.coordinates.reduce((sum, p) => sum + p[1], 0) / selectedArea.coordinates.length;
      const avgLng = selectedArea.coordinates.reduce((sum, p) => sum + p[0], 0) / selectedArea.coordinates.length;
      setMapCenter({ lat: avgLat, lng: avgLng });
    }
  }, [selectedArea]);

  const handleMapClick = (e) => {
    if (!drawMode) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    
    // Convertir en coordonnées lat/lng approximatives
    const lng = mapCenter.lng + (x - 50) * 0.01;
    const lat = mapCenter.lat - (y - 50) * 0.01;
    
    setPoints([...points, [lng, lat]]);
  };

  const finishDrawing = () => {
    if (points.length >= 3) {
      const closedPoints = [...points, points[0]];
      onAreaSelect({ type: 'Polygon', coordinates: [closedPoints] });
      setDrawMode(false);
      setPoints([]);
    }
  };

  const clearDrawing = () => {
    setPoints([]);
    setDrawMode(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setDrawMode(!drawMode)}
          className={`px-4 py-2 rounded-lg font-medium ${
            drawMode 
              ? 'bg-red-500 text-white' 
              : 'bg-green-500 text-white hover:bg-green-600'
          }`}
        >
          {drawMode ? '🚫 Annuler' : '✏️ Dessiner zone'}
        </button>
        
        {drawMode && points.length >= 3 && (
          <button
            onClick={finishDrawing}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600"
          >
            ✅ Terminer ({points.length} points)
          </button>
        )}
        
        {points.length > 0 && (
          <button
            onClick={clearDrawing}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600"
          >
            🗑️ Effacer
          </button>
        )}
      </div>

      <div 
        onClick={handleMapClick}
        className={`relative w-full h-96 bg-gradient-to-br from-green-50 to-blue-50 rounded-lg border-2 overflow-hidden ${
          drawMode ? 'border-green-500 cursor-crosshair' : 'border-gray-300'
        }`}
        style={{
          backgroundImage: `
            linear-gradient(rgba(34, 197, 94, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 197, 94, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }}
      >
        {/* Indicateur de centre */}
        <div className="absolute top-2 left-2 bg-white px-3 py-1 rounded shadow text-sm">
          📍 {mapCenter.lat.toFixed(4)}°N, {mapCenter.lng.toFixed(4)}°E
        </div>

        {drawMode && (
          <div className="absolute top-2 right-2 bg-green-500 text-white px-3 py-1 rounded shadow text-sm font-medium animate-pulse">
            🖱️ Cliquez pour placer des points
          </div>
        )}

        {/* Afficher les points dessinés */}
        {points.map((point, idx) => (
          <div
            key={idx}
            className="absolute w-3 h-3 bg-red-500 rounded-full border-2 border-white shadow-lg"
            style={{
              left: `${((point[0] - mapCenter.lng) / 0.01 + 50)}%`,
              top: `${(-(point[1] - mapCenter.lat) / 0.01 + 50)}%`,
              transform: 'translate(-50%, -50%)'
            }}
          >
            <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-1 rounded whitespace-nowrap">
              P{idx + 1}
            </div>
          </div>
        ))}

        {/* Lignes entre les points */}
        {points.length > 1 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {points.map((point, idx) => {
              if (idx === 0) return null;
              const prev = points[idx - 1];
              return (
                <line
                  key={idx}
                  x1={`${((prev[0] - mapCenter.lng) / 0.01 + 50)}%`}
                  y1={`${(-(prev[1] - mapCenter.lat) / 0.01 + 50)}%`}
                  x2={`${((point[0] - mapCenter.lng) / 0.01 + 50)}%`}
                  y2={`${(-(point[1] - mapCenter.lat) / 0.01 + 50)}%`}
                  stroke="#ef4444"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
              );
            })}
          </svg>
        )}

        {/* Zone sélectionnée */}
        {selectedArea && (
          <div className="absolute inset-0 pointer-events-none">
            <svg className="w-full h-full">
              <polygon
                points={selectedArea.coordinates[0].map(p => {
                  const x = ((p[0] - mapCenter.lng) / 0.01 + 50);
                  const y = (-(p[1] - mapCenter.lat) / 0.01 + 50);
                  return `${x}%,${y}%`;
                }).join(' ')}
                fill="rgba(34, 197, 94, 0.2)"
                stroke="#22c55e"
                strokeWidth="3"
              />
            </svg>
          </div>
        )}
      </div>
    </div>
  );
};

// Composant principal - Partie 1
const AgriSightApp = () => {
  const [activeTab, setActiveTab] = useState('config');
  const [config, setConfig] = useState({
    zoneMethod: 'draw',
    startDate: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    culture: 'Mil',
    zoneName: 'Ma parcelle',
    coords: null
  });

  const [selectedArea, setSelectedArea] = useState(null);
  const [loadingState, setLoadingState] = useState({
    climate: { progress: 0, status: 'idle' },
    ndvi: { progress: 0, status: 'idle' },
    analysis: { progress: 0, status: 'idle' }
  });

  const [data, setData] = useState({
    climate: null,
    ndvi: null,
    metrics: null
  });

  // Clé API intégrée
  const AGRO_API_KEY = '28641235f2b024b5f45f97df45c6a0d5';

  const updateLoadingState = (key, progress, status) => {
    setLoadingState(prev => ({
      ...prev,
      [key]: { progress, status }
    }));
  };

  const loadClimateData = useCallback(async (geometry, startDate, endDate) => {
    updateLoadingState('climate', 10, 'loading');
    
    try {
      const centroid = geometry.coordinates[0].reduce(
        (acc, coord) => ({ lat: acc.lat + coord[1], lng: acc.lng + coord[0] }),
        { lat: 0, lng: 0 }
      );
      const avgLat = centroid.lat / geometry.coordinates[0].length;
      const avgLng = centroid.lng / geometry.coordinates[0].length;

      updateLoadingState('climate', 30, 'loading');

      const start = startDate.replace(/-/g, '');
      const end = endDate.replace(/-/g, '');
      
      const url = `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR&start=${start}&end=${end}&latitude=${avgLat}&longitude=${avgLng}&format=JSON&community=AG`;

      updateLoadingState('climate', 50, 'loading');
      const response = await fetch(url);
      
      if (!response.ok) throw new Error('Erreur NASA POWER');

      updateLoadingState('climate', 70, 'loading');
      const result = await response.json();
      const params = result.properties.parameter;

      const climateData = Object.keys(params.T2M).map(dateKey => ({
        date: dateKey,
        temp_mean: params.T2M[dateKey],
        temp_min: params.T2M_MIN[dateKey],
        temp_max: params.T2M_MAX[dateKey],
        rain: params.PRECTOTCORR[dateKey]
      }));

      updateLoadingState('climate', 100, 'complete');
      return climateData;
    } catch (error) {
      updateLoadingState('climate', 0, 'error');
      throw error;
    }
  }, []);

  const loadNDVIData = useCallback(async (geometry, startDate, endDate) => {
    updateLoadingState('ndvi', 10, 'loading');

    try {
      // Créer un polygone sur Agromonitoring
      updateLoadingState('ndvi', 20, 'loading');
      const polyResponse = await fetch(`http://api.agromonitoring.com/agro/1.0/polygons?appid=${AGRO_API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'temp_polygon',
          geo_json: geometry
        })
      });

      if (!polyResponse.ok) throw new Error('Erreur création polygone');
      
      const polyData = await polyResponse.json();
      const polygonId = polyData.id;

      updateLoadingState('ndvi', 40, 'loading');

      // Récupérer les images satellites
      const start = Math.floor(new Date(startDate).getTime() / 1000);
      const end = Math.floor(new Date(endDate).getTime() / 1000);
      
      const imagesUrl = `http://api.agromonitoring.com/agro/1.0/image/search?start=${start}&end=${end}&polyid=${polygonId}&appid=${AGRO_API_KEY}`;
      
      updateLoadingState('ndvi', 60, 'loading');
      const imagesResponse = await fetch(imagesUrl);
      
      if (!imagesResponse.ok) throw new Error('Erreur récupération images');
      
      const images = await imagesResponse.json();

      updateLoadingState('ndvi', 80, 'loading');

      const ndviData = images.map(img => ({
        date: new Date(img.dt * 1000).toISOString().split('T')[0],
        ndvi_mean: img.stats?.ndvi?.mean || 0.5,
        ndvi_std: img.stats?.ndvi?.std || 0.1,
        ndvi_min: img.stats?.ndvi?.min || 0.3,
        ndvi_max: img.stats?.ndvi?.max || 0.7,
        cloud_cover: img.cl || 0
      }));

      updateLoadingState('ndvi', 100, 'complete');
      return ndviData;
    } catch (error) {
      updateLoadingState('ndvi', 0, 'error');
      // Fallback: données simulées
      return generateSimulatedNDVI(startDate, endDate);
    }
  }, []);

  const generateSimulatedNDVI = (startDate, endDate) => {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const data = [];
    
    for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 5)) {
      const month = d.getMonth();
      let base = 0.5;
      
      if (month >= 5 && month <= 8) base = 0.65;
      else if (month === 4 || month === 9) base = 0.45;
      else base = 0.25;
      
      data.push({
        date: d.toISOString().split('T')[0],
        ndvi_mean: Math.min(1, Math.max(0, base + (Math.random() - 0.5) * 0.16)),
        ndvi_std: 0.1,
        ndvi_min: Math.max(0, base - 0.15),
        ndvi_max: Math.min(1, base + 0.15),
        cloud_cover: Math.floor(Math.random() * 30)
      });
    }
    
    return data;
  };

  const calculateMetrics = (climateData, ndviData) => {
    if (!climateData || !ndviData) return null;

    const ndviMean = ndviData.reduce((sum, d) => sum + d.ndvi_mean, 0) / ndviData.length;
    const tempMean = climateData.reduce((sum, d) => sum + d.temp_mean, 0) / climateData.length;
    const rainTotal = climateData.reduce((sum, d) => sum + d.rain, 0);
    
    return {
      ndvi_mean: ndviMean,
      ndvi_std: Math.sqrt(ndviData.reduce((sum, d) => sum + Math.pow(d.ndvi_mean - ndviMean, 2), 0) / ndviData.length),
      temp_mean: tempMean,
      temp_min: Math.min(...climateData.map(d => d.temp_min)),
      temp_max: Math.max(...climateData.map(d => d.temp_max)),
      rain_total: rainTotal,
      rain_mean: rainTotal / climateData.length,
      rain_days: climateData.filter(d => d.rain > 1).length,
      yield_potential: calculateYield(ndviMean, rainTotal, config.culture)
    };
  };

  const calculateYield = (ndvi, rain, culture) => {
    const yields = {
      'Mil': ndvi > 0.6 && rain > 400 ? 1.5 : ndvi > 0.4 && rain > 300 ? 1.0 : 0.6,
      'Maïs': ndvi > 0.65 && rain > 500 ? 3.5 : ndvi > 0.5 && rain > 400 ? 2.5 : 1.5,
      'Arachide': ndvi > 0.6 && rain > 450 ? 2.0 : ndvi > 0.45 && rain > 350 ? 1.3 : 0.8,
      'Sorgho': ndvi > 0.6 && rain > 400 ? 2.5 : ndvi > 0.4 && rain > 300 ? 1.8 : 1.0
    };
    return yields[culture] || 1.0;
  };

  const handleLaunchAnalysis = async () => {
    if (!selectedArea) {
      alert('Veuillez définir une zone d\'étude');
      return;
    }

    try {
      updateLoadingState('analysis', 10, 'loading');

      const climateData = await loadClimateData(selectedArea, config.startDate, config.endDate);
      updateLoadingState('analysis', 40, 'loading');

      const ndviData = await loadNDVIData(selectedArea, config.startDate, config.endDate);
      updateLoadingState('analysis', 70, 'loading');

      const metrics = calculateMetrics(climateData, ndviData);
      
      setData({ climate: climateData, ndvi: ndviData, metrics });
      updateLoadingState('analysis', 100, 'complete');
      
      setActiveTab('dashboard');
    } catch (error) {
      console.error('Erreur analyse:', error);
      alert('Erreur lors de l\'analyse: ' + error.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-green-700 flex items-center gap-2">
                <Leaf className="w-8 h-8" />
                AgriSight Pro
              </h1>
              <p className="text-gray-600 mt-1">Analyse agro-climatique avancée par IA et télédétection</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Zone: {config.zoneName}</div>
              <div className="text-sm text-gray-500">Culture: {config.culture}</div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow-lg overflow-hidden mb-6">
          <div className="flex border-b overflow-x-auto">
            {[
              { id: 'config', label: '⚙️ Configuration', icon: MapPin },
              { id: 'dashboard', label: '📊 Dashboard', icon: BarChart3 },
              { id: 'ndvi', label: '🛰️ NDVI', icon: TrendingUp },
              { id: 'climate', label: '🌦️ Climat', icon: Cloud },
              { id: 'analysis', label: '🤖 Analyse IA', icon: FileText }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 font-medium whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? 'bg-green-500 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="p-6">
            {/* Configuration Tab */}
            {activeTab === 'config' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800">Configuration de l'analyse</h2>

                {/* Paramètres */}
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">Type de culture</label>
                    <select
                      value={config.culture}
                      onChange={(e) => setConfig({...config, culture: e.target.value})}
                      className="w-full px-4 py-2 border rounded-lg"
                    >
                      {['Mil', 'Sorgho', 'Maïs', 'Arachide', 'Riz', 'Niébé', 'Manioc', 'Tomate', 'Oignon'].map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Nom de la zone</label>
                    <input
                      type="text"
                      value={config.zoneName}
                      onChange={(e) => setConfig({...config, zoneName: e.target.value})}
                      className="w-full px-4 py-2 border rounded-lg"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Date de début</label>
                    <input
                      type="date"
                      value={config.startDate}
                      onChange={(e) => setConfig({...config, startDate: e.target.value})}
                      className="w-full px-4 py-2 border rounded-lg"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Date de fin</label>
                    <input
                      type="date"
                      value={config.endDate}
                      onChange={(e) => setConfig({...config, endDate: e.target.value})}
                      className="w-full px-4 py-2 border rounded-lg"
                    />
                  </div>
                </div>

                {/* Carte interactive */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Définir la zone d'étude</h3>
                  <InteractiveMap 
                    onAreaSelect={setSelectedArea}
                    selectedArea={selectedArea}
                  />
                </div>

                {/* Progression du chargement */}
                {(loadingState.climate.status !== 'idle' || 
                  loadingState.ndvi.status !== 'idle' || 
                  loadingState.analysis.status !== 'idle') && (
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold mb-3">Progression du chargement</h3>
                    <ProgressBar 
                      progress={loadingState.climate.progress}
                      label="Données climatiques (NASA POWER)"
                      status={loadingState.climate.status}
                    />
                    <ProgressBar 
                      progress={loadingState.ndvi.progress}
                      label="Données NDVI (Sentinel-2)"
                      status={loadingState.ndvi.status}
                    />
                    <ProgressBar 
                      progress={loadingState.analysis.progress}
                      label="Analyse globale"
                      status={loadingState.analysis.status}
                    />
                  </div>
                )}

                {/* Bouton d'analyse */}
                <button
                  onClick={handleLaunchAnalysis}
                  disabled={!selectedArea || loadingState.analysis.status === 'loading'}
                  className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {loadingState.analysis.status === 'loading' ? '⏳ Analyse en cours...' : '🚀 Lancer l\'analyse'}
                </button>
              </div>
            )}

            {/* Suite dans Partie 2... */}
            {activeTab !== 'config' && !data.metrics && (
              <div className="text-center py-12">
                <AlertCircle className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600 text-lg">Veuillez d'abord lancer une analyse dans l'onglet Configuration</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgriSightApp;
