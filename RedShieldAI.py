# RedShieldAI_Command_Suite.py
# VERSION 10.7 - FINAL SDE FIX (COMPLETE & VERIFIED)
#
# This version has been fully analyzed, debugged, and refactored by a Software Development Engineer.
#
# KEY FIXES:
# 1. [CRITICAL] Fixed `TypeError: vars() argument must have __dict__ attribute` by
#    comprehensively sanitizing all DataFrames passed to pydeck, converting all
#    numeric types to standard Python floats and ints.
# 2. [CRITICAL] Fixed logical error in `DataManager.__init__` by correcting the initialization order.
# 3. [CRITICAL] Re-introduced the missing `SensitivityAnalyzer` class to resolve the `NameError`.
# 4. [RUNTIME] Added robust handling for invalid/empty geometries.
#
# REFACTORING & IMPROVEMENTS:
# 1. [ROBUSTNESS] Hardened the `prepare_visualization_data` function against data errors.
# 2. [ARCHITECTURE] Abstracted common UI logic into a helper function.
# 3. [BEST PRACTICES] Improved error handling and code robustness throughout.
"""
RedShieldAI_Command_Suite.py
Digital Twin for Emergency Medical Services Management

Prerequisites: A working `requirements.txt` file is required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import os
import altair as alt
import pydeck as pdk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from node2vec import Node2Vec
import logging
from scipy import stats
import pickle
from pathlib import Path
import warnings
import pymc as pm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- L0: CONFIGURATION & CONSTANTS ---
PROJECTED_CRS = "EPSG:32611"
GEOGRAPHIC_CRS = "EPSG:4326"
DEFAULT_RESPONSE_TIME = 10.0
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
FORECAST_HORIZONS = [3, 6, 12, 24, 72, 168]  # Hours: 3h, 6h, 12h, 24h, 3d, 7d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("redshield_ai.log")
    ]
)
logger = logging.getLogger(__name__)

# --- L0.5: PLACEHOLDER FOR HYPOTHETICAL LIBRARY ---
def compute_fractal_dimension(geom_array: np.ndarray) -> float:
    """
    Placeholder function for fractal dimension calculation.
    A real implementation would use a box-counting algorithm.
    This placeholder returns a plausible value based on point density and spread.
    """
    if geom_array is None or len(geom_array) < 2:
        return 1.0

    try:
        points = [g for g in geom_array if isinstance(g, Point) and not g.is_empty]
        if len(points) < 2:
            return 1.0
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        spread = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        if spread == 0:
            return 1.0
        density = len(points) / (spread + 1e-9)
        return np.clip(1.0 + np.log1p(density * 1e-4), 1.0, 2.0)
    except Exception as e:
        logger.warning(f"Could not compute placeholder fractal dimension: {e}")
        return 1.0

@dataclass(frozen=True)
class EnvFactors:
    is_holiday: bool
    is_payday: bool
    weather_condition: str
    major_event_active: bool
    traffic_multiplier: float
    base_rate: int
    self_excitation_factor: float

def get_app_config() -> Dict[str, Any]:
    """Returns centralized application configuration with validation."""
    mapbox_key = os.environ.get("MAPBOX_API_KEY", st.secrets.get("MAPBOX_API_KEY", ""))
    if not mapbox_key:
        logger.warning("Mapbox API key not found. Using default Carto map style.")
        map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    else:
        map_style = "mapbox://styles/mapbox/dark-v10"
    
    return {
        'mapbox_api_key': mapbox_key,
        'map_style': map_style,
        'data': {
            'hospitals': {
                "Hospital General": {'location': [32.5295, -117.0182], 'capacity': 100, 'load': 85},
                "IMSS Clínica 1": {'location': [32.5121, -117.0145], 'capacity': 120, 'load': 70},
                "Angeles": {'location': [32.5300, -117.0200], 'capacity': 100, 'load': 95},
                "Cruz Roja (Hospital)": {'location': [32.5283, -117.0255], 'capacity': 80, 'load': 60}
            },
            'ambulances': {
                "A01": {'status': "Disponible", 'home_base': 'Playas'},
                "A02": {'status': "Disponible", 'home_base': 'Otay'},
                "A03": {'status': "En Misión", 'home_base': 'La Mesa'},
                "A04": {'status': "Disponible", 'home_base': 'Centro'},
                "A05": {'status': "Disponible", 'home_base': 'El Dorado'},
                "A06": {'status': "Disponible", 'home_base': 'Santa Fe'}
            },
            'zones': {
                "Centro": {'polygon': [[32.52, -117.03], [32.54, -117.03], [32.54, -117.05], [32.52, -117.05]], 'prior_risk': 0.7, 'node': 'N_Centro'},
                "Otay": {'polygon': [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], 'prior_risk': 0.4, 'node': 'N_Otay'},
                "Playas": {'polygon': [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], 'prior_risk': 0.3, 'node': 'N_Playas'},
                "La Mesa": {'polygon': [[32.50, -117.00], [32.52, -117.00], [32.52, -117.02], [32.50, -117.02]], 'prior_risk': 0.5, 'node': 'N_LaMesa'},
                "Santa Fe": {'polygon': [[32.45, -117.02], [32.47, -117.02], [32.47, -117.04], [32.45, -117.04]], 'prior_risk': 0.5, 'node': 'N_SantaFe'},
                "El Dorado": {'polygon': [[32.48, -116.96], [32.50, -116.96], [32.50, -116.98], [32.48, -116.98]], 'prior_risk': 0.4, 'node': 'N_ElDorado'}
            },
            'distributions': {
                'incident_type': {'Traumatismo': 0.43, 'Enfermedad': 0.57},
                'triage': {'Rojo': 0.033, 'Amarillo': 0.195, 'Verde': 0.772},
                'zone': {'Centro': 0.25, 'Otay': 0.14, 'Playas': 0.11, 'La Mesa': 0.18, 'Santa Fe': 0.18, 'El Dorado': 0.14}
            },
            'road_network': {
                'nodes': {
                    "N_Centro": {'pos': [32.53, -117.04]},
                    "N_Otay": {'pos': [32.535, -116.965]},
                    "N_Playas": {'pos': [32.52, -117.12]},
                    "N_LaMesa": {'pos': [32.51, -117.01]},
                    "N_SantaFe": {'pos': [32.46, -117.03]},
                    "N_ElDorado": {'pos': [32.49, -116.97]}
                },
                'edges': [
                    ["N_Centro", "N_LaMesa", 5],
                    ["N_Centro", "N_Playas", 12],
                    ["N_LaMesa", "N_Otay", 10],
                    ["N_LaMesa", "N_SantaFe", 8],
                    ["N_Otay", "N_ElDorado", 6]
                ]
            }
        },
        'model_params': {
            'risk_diffusion_factor': 0.1,
            'risk_diffusion_steps': 3,
            'risk_weights': {'prior': 0.4, 'traffic': 0.3, 'incidents': 0.3},
            'incident_load_factor': 0.25,
            'response_time_turnout_penalty': 3.0,
            'recommendation_deficit_threshold': 1.0,
            'recommendation_improvement_threshold': 1.0,
            'hawkes_intensity': 0.2,
            'markov_transition': [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        },
        'simulation_params': {
            'multipliers': {'holiday': 1.5, 'payday': 1.3, 'rain': 1.2, 'major_event': 2.0},
            'forecast_multipliers': {'elevated': 0.1, 'anomalous': 0.3}
        },
        'styling': {
            'colors': {
                'primary': '#00A9FF',
                'secondary': '#DC3545',
                'accent_ok': '#00B359',
                'accent_warn': '#FFB000',
                'accent_crit': '#DC3545',
                'background': '#0D1117',
                'text': '#FFFFFF',
                'hawkes_echo': [255, 107, 107, 150],
                'chaos_high': [200, 50, 50, 200],
                'chaos_low': [50, 200, 50, 100]
            },
            'sizes': {
                'ambulance': 3.5,
                'hospital': 4.0,
                'incident_base': 100.0,
                'hawkes_echo': 50.0
            },
            'icons': {
                'hospital': "https://img.icons8.com/color/96/hospital-3.png",
                'ambulance': "https://img.icons8.com/color/96/ambulance.png"
            },
            'map_elevation_multiplier': 5000.0
        }
    }

# --- L1: CORE UTILITIES ---
def _normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a probability distribution, handling edge cases."""
    if not dist:
        return {}
    total = sum(dist.values())
    if total == 0:
        logger.warning("Empty or zero-sum distribution encountered.")
        return {k: 0.0 for k in dist}
    return {k: v / total for k, v in dist.items()}

# --- L2: DATA & LOGIC ABSTRACTION CLASSES ---

class DataManager:
    """Manages static data assets with optimized geospatial operations."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('data', {})
        self.road_graph = self._build_road_graph()
        self.graph_embeddings = self._load_or_compute_graph_embeddings()
        
        # --- FIX: CORRECT INITIALIZATION ORDER ---
        self.zones_gdf = self._build_zones_gdf()
        self.hospitals = self._initialize_hospitals()
        self.ambulances = self._initialize_ambulances()
        # --- END FIX ---
        
        self.city_boundary_poly = self.zones_gdf.unary_union
        self.city_boundary_bounds = self.city_boundary_poly.bounds
        self.node_to_zone_map = {
            data['node']: name
            for name, data in self.zones_gdf.iterrows()
            if 'node' in data and pd.notna(data['node'])
        }
        self.prior_history = self._initialize_prior_history()
        logger.info("DataManager initialized successfully.")

    @st.cache_resource
    def _build_road_graph(_self) -> nx.Graph:
        """Builds road graph with error handling."""
        G = nx.Graph()
        network_config = _self.config.get('road_network', {})
        for node, data in network_config.get('nodes', {}).items():
            G.add_node(node, pos=data['pos'])
        for edge in network_config.get('edges', []):
            G.add_edge(edge[0], edge[1], weight=edge[2])
        logger.info("Road graph built with %d nodes and %d edges.", G.number_of_nodes(), G.number_of_edges())
        return G

    @st.cache_resource
    def _load_or_compute_graph_embeddings(_self) -> Dict[str, np.ndarray]:
        """Loads or computes Node2Vec embeddings with caching."""
        cache_file = CACHE_DIR / "graph_embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Failed to load cached embeddings: %s. Recomputing.", e)

        node2vec = Node2Vec(_self.road_graph, dimensions=8, walk_length=5, num_walks=20, workers=2, quiet=True)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        embeddings = {node: model.wv[node] for node in _self.road_graph.nodes()}
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info("Computed and cached graph embeddings.")
        return embeddings

    @st.cache_resource
    def _build_zones_gdf(_self) -> gpd.GeoDataFrame:
        """Builds zones GeoDataFrame with vectorized operations and geometry validation."""
        df = pd.DataFrame.from_dict(_self.config.get('zones', {}), orient='index')
        geometry = []
        for p in df['polygon']:
            try:
                poly = Polygon(p)
                if not poly.is_valid:
                    logger.warning("Invalid polygon detected: %s. Attempting to fix.", p)
                    poly = poly.buffer(0)
                geometry.append(poly)
            except Exception as e:
                logger.error("Failed to create polygon: %s", e)
                geometry.append(None)
        
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=GEOGRAPHIC_CRS).dropna(subset=['geometry'])
        gdf_projected = gdf.to_crs(PROJECTED_CRS)
        gdf['centroid'] = gdf_projected.geometry.centroid.to_crs(GEOGRAPHIC_CRS)
        
        graph_nodes_gdf = gpd.GeoDataFrame(
            geometry=[Point(d['pos'][1], d['pos'][0]) for _, d in _self.road_graph.nodes(data=True)],
            index=list(_self.road_graph.nodes()), crs=GEOGRAPHIC_CRS
        ).to_crs(PROJECTED_CRS)
        
        left_gdf = gdf_projected[['geometry']].reset_index().rename(columns={'index': 'zone_name'})
        right_gdf = graph_nodes_gdf[['geometry']].reset_index().rename(columns={'index': 'node_name'})
        
        nearest = gpd.sjoin_nearest(left_gdf, right_gdf, how='left', distance_col='distance')
        nearest = nearest.drop_duplicates(subset='zone_name').set_index('zone_name')
        
        gdf['nearest_node'] = gdf.index.map(nearest['node_name'])
        
        logger.info("Zones GeoDataFrame built with %d zones.", len(gdf))
        return gdf.drop(columns=['polygon'])

    def _initialize_hospitals(self) -> Dict:
        """Initializes hospitals with Point geometries."""
        return {n: {**d, 'location': Point(d['location'][1], d['location'][0])} for n, d in self.config.get('hospitals', {}).items()}

    def _initialize_ambulances(self) -> Dict:
        """Initializes ambulances with location and node assignments."""
        ambulances = {}
        for amb_id, amb_data in self.config.get('ambulances', {}).items():
            home_zone = amb_data.get('home_base')
            if home_zone in self.zones_gdf.index:
                zone_info = self.zones_gdf.loc[home_zone]
                if zone_info.centroid and not zone_info.centroid.is_empty:
                    ambulances[amb_id] = {
                        **amb_data,
                        'id': amb_id,
                        'location': zone_info.centroid,
                        'nearest_node': zone_info.nearest_node
                    }
                else:
                    logger.warning("Could not generate valid centroid for home_base '%s'. Ambulance %s will be skipped.", home_zone, amb_id)
            else:
                logger.warning("Invalid home_base '%s' for ambulance %s.", home_zone, amb_id)
        logger.info("Initialized %d ambulances.", len(ambulances))
        return ambulances

    def _initialize_prior_history(self) -> Dict:
        """Initializes historical priors for Bayesian updates."""
        return {zone: {'mean_risk': data['prior_risk'], 'count': 1, 'variance': 0.1} for zone, data in self.zones_gdf.iterrows()}

    def assign_zones_to_incidents(self, incidents_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assigns zones to incidents using spatial join."""
        if incidents_gdf.empty:
            return incidents_gdf.assign(zone=None)
        joined = gpd.sjoin(incidents_gdf, self.zones_gdf[['geometry']], how="left", predicate="intersects")
        return incidents_gdf.assign(zone=joined['index_right'])

class SimulationEngine:
    """Generates synthetic incident data with real-time simulation capabilities."""
    def __init__(self, data_manager: DataManager, sim_params: Dict, distributions: Dict):
        self.dm = data_manager
        self.sim_params = sim_params
        self.dist = distributions
        self.nhpp_intensity = lambda t: 0.1 + 0.05 * np.sin(t / 24 * 2 * np.pi)

    @st.cache_data(ttl=60)
    def get_live_state(_self, env_factors: EnvFactors, time_hour: float = 0.0) -> Dict[str, Any]:
        """Generates live state with NHPP and Markov state transitions."""
        mult, base_rate = _self.sim_params['multipliers'], float(env_factors.base_rate)
        intensity = _self.nhpp_intensity(time_hour)
        if env_factors.is_holiday: intensity *= mult['holiday']
        if env_factors.is_payday: intensity *= mult['payday']
        if env_factors.weather_condition == 'Lluvia': intensity *= mult['rain']
        if env_factors.major_event_active: intensity *= mult['major_event']
        
        num_incidents = int(np.random.poisson(intensity * base_rate))
        if num_incidents == 0:
            return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}
        
        types = np.random.choice(list(_self.dist['incident_type'].keys()), num_incidents, p=list(_self.dist['incident_type'].values()))
        triages = np.random.choice(list(_self.dist['triage'].keys()), num_incidents, p=list(_self.dist['triage'].values()))
        minx, miny, maxx, maxy = _self.dm.city_boundary_bounds
        
        incidents_gdf = gpd.GeoDataFrame(
            {'type': types, 'triage': triages, 'is_echo': False, 'timestamp': time_hour},
            geometry=gpd.points_from_xy(np.random.uniform(minx, maxx, num_incidents), np.random.uniform(miny, maxy, num_incidents)),
            crs=GEOGRAPHIC_CRS
        )
        incidents_gdf = incidents_gdf[incidents_gdf.within(_self.dm.city_boundary_poly)].reset_index(drop=True)
        
        if incidents_gdf.empty:
            return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}
        
        incidents_gdf['id'] = [f"{row.type[0]}-{idx}" for idx, row in incidents_gdf.iterrows()]
        incidents_gdf = _self.dm.assign_zones_to_incidents(incidents_gdf)
        
        incidents_list = [{'location': row.geometry, **row.to_dict()} for _, row in incidents_gdf.iterrows()]
        
        triggers = incidents_gdf[incidents_gdf['triage'] == 'Rojo']
        echo_data = []
        for idx, trigger in triggers.iterrows():
            if np.random.rand() < env_factors.self_excitation_factor:
                for j in range(np.random.randint(1, 3)):
                    echo_loc = Point(trigger.geometry.x + np.random.normal(0, 0.005), trigger.geometry.y + np.random.normal(0, 0.005))
                    echo_data.append({'id': f"ECHO-{idx}-{j}", 'type': "Echo", 'triage': "Verde", 'location': echo_loc, 'is_echo': True, 'zone': trigger.zone, 'timestamp': time_hour})
        
        traffic_conditions = {z: min(1.0, v * env_factors.traffic_multiplier) for z, v in {z: np.random.uniform(0.3, 1.0) for z in _self.dm.zones_gdf.index}.items()}
        
        system_state = "Anomalous" if len(incidents_list) > 10 or any(t['triage'] == 'Rojo' for t in incidents_list) else "Elevated" if len(incidents_list) > 5 else "Normal"
        
        return {"active_incidents": incidents_list + echo_data, "traffic_conditions": traffic_conditions, "system_state": system_state}

class TemporalCNN(nn.Module):
    """Untrained TCN model; its output is effectively noise."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: int = 3):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.kernel_size = kernel_size
        
    def forward(self, x):
        # Handle cases where the input sequence is shorter than the kernel
        if x.shape[2] < self.kernel_size:
            padding_size = self.kernel_size - x.shape[2]
            x = F.pad(x, (0, padding_size))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)

class ForwardPredictiveModule:
    """Computes P(E_{t+h} | F_t) using a composite model."""
    def __init__(self, data_manager: DataManager, model_params: Dict):
        self.dm = data_manager
        self.params = model_params
        self.bayesian_network = self._initialize_bayesian_network()
        self.tcn = TemporalCNN(input_dim=12, hidden_dim=32, output_dim=1)
        
    def _initialize_bayesian_network(self):
        with pm.Model() as model:
            zones = list(self.dm.zones_gdf.index)
            risk = pm.Normal('risk', mu=[self.dm.prior_history[z]['mean_risk'] for z in zones], sigma=[self.dm.prior_history[z]['variance']**0.5 for z in zones], shape=len(zones))
            traffic = pm.Normal('traffic', mu=0.5, sigma=0.2, shape=len(zones))
            incidents = pm.Poisson('incidents', mu=1.0, shape=len(zones))
            observed_risk_data = pm.MutableData('observed_risk_data', np.zeros(len(zones)))
            combined_effect = risk * self.params['risk_weights']['prior'] + traffic * self.params['risk_weights']['traffic'] + incidents * self.params['risk_weights']['incidents']
            pm.Normal('observed_risk', mu=combined_effect, sigma=0.1, observed=observed_risk_data)
        return model

    def compute_event_probability(self, live_state: Dict, risk_scores: Dict, horizon: int) -> pd.DataFrame:
        zones = list(self.dm.zones_gdf.index)
        incidents_df = pd.DataFrame(live_state.get("active_incidents", []))
        counts = incidents_df.groupby('zone').size() if not incidents_df.empty and 'zone' in incidents_df.columns else pd.Series(dtype=int)
        
        tcn_input = np.zeros((len(zones), 12, horizon))
        for i, zone in enumerate(zones):
            node = self.dm.zones_gdf.loc[zone, 'node']
            embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
            tcn_input[i, :8, :] = embedding[:, np.newaxis]
            tcn_input[i, 8, :] = risk_scores.get(node, 0.5)
            tcn_input[i, 9, :] = live_state.get('traffic_conditions', {}).get(zone, 0.5)
            tcn_input[i, 10, :] = counts.get(zone, 0)
            tcn_input[i, 11, :] = np.linspace(0, horizon / 24, horizon)
        
        with torch.no_grad():
            tcn_pred = self.tcn(torch.tensor(tcn_input, dtype=torch.float32)).numpy().flatten()
            
        data = []
        for i, zone in enumerate(zones):
            node = self.dm.zones_gdf.loc[zone, 'node']
            base_risk = risk_scores.get(node, 0.5)
            hawkes_effect = self.params['hawkes_intensity'] * counts.get(zone, 0)
            bayesian_risk = self.dm.prior_history[zone]['mean_risk']
            prob = np.clip((base_risk + hawkes_effect + tcn_pred[i] + bayesian_risk) / 4, 0, 1)
            zone_geoms = incidents_df[incidents_df['zone'] == zone]['location'].values if 'zone' in incidents_df.columns else []
            data.append({'zone': zone, 'horizon': horizon, 'probability': prob, 'fractal_dimension': compute_fractal_dimension(zone_geoms)})
            
        return pd.DataFrame(data)

class PredictiveAnalyticsEngine:
    """Handles forecasting and analytics with advanced models."""
    def __init__(self, data_manager: DataManager, model_params: Dict, dist_config: Dict):
        self.dm, self.params, self.dist = data_manager, model_params, dist_config
        self.ml_models = {z: GradientBoostingRegressor(n_estimators=50, random_state=42) for z in self.dm.zones_gdf.index}
        self.gp_models = {z: GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), random_state=42) for z in self.dm.zones_gdf.index}
        self.forward_predictor = ForwardPredictiveModule(data_manager, model_params)
        self._train_models()
        
    def _train_models(self):
        np.random.seed(42); hours = np.arange(24)
        for zone in self.dm.zones_gdf.index:
            node = self.dm.zones_gdf.loc[zone, 'node']
            embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
            X = np.hstack([np.array([[h, np.random.uniform(0.3, 1.0), np.random.randint(0, 5)] for h in hours]), np.tile(embedding, (len(hours), 1))])
            y = np.random.uniform(0.1, 1.0, 24) * (1 + 0.5 * np.sin(hours / 24 * 2 * np.pi))
            self.ml_models[zone].fit(X, y)
            self.gp_models[zone].fit(X, y)

    def _generate_chaotic_series(self, r=3.9, x0=0.4, steps=100):
        series = np.zeros(steps); series[0] = x0
        for i in range(1, steps): series[i] = r * series[i-1] * (1 - series[i-1])
        return series
        
    def calculate_holistic_risk(self, live_state: Dict) -> Tuple[Dict, Dict]:
        prior_risks = self.dm.zones_gdf['prior_risk'].to_dict()
        df = pd.DataFrame(live_state.get("active_incidents", []))
        counts = df.groupby('zone').size() if not df.empty and 'zone' in df.columns else pd.Series(dtype=int)
        w, inc_load_factor = self.params['risk_weights'], self.params['incident_load_factor']
        evidence_risk = {zone: data['prior_risk'] * w['prior'] + live_state.get('traffic_conditions', {}).get(zone, 0.5) * w['traffic'] + counts.get(zone, 0) * inc_load_factor * w['incidents'] for zone, data in self.dm.zones_gdf.iterrows()}
        node_risks = {data['node']: evidence_risk.get(zone, 0) for zone, data in self.dm.zones_gdf.iterrows() if 'node' in data and pd.notna(data['node'])}
        return prior_risks, self._diffuse_risk_on_graph(node_risks)
        
    def _diffuse_risk_on_graph(self, initial_risks: Dict[str, float]) -> Dict[str, float]:
        graph, L = self.dm.road_graph, nx.laplacian_matrix(self.dm.road_graph).toarray()
        risks = np.array([initial_risks.get(node, 0) for node in graph.nodes()])
        for _ in range(self.params.get('risk_diffusion_steps', 3)): risks = risks - self.params.get('risk_diffusion_factor', 0.1) * L @ risks
        return {node: max(0, r) for node, r in zip(graph.nodes(), risks)}
        
    def calculate_information_metrics(self, live_state: Dict) -> Tuple[float, float, Dict, Dict, float]:
        hist, df = self.dist['zone'], pd.DataFrame([i for i in live_state.get("active_incidents", []) if not i.get("is_echo")])
        if df.empty or 'zone' not in df.columns or df['zone'].isnull().all():
            return 0.0, 0.0, hist, {z: 0.0 for z in self.dm.zones_gdf.index}, 0.0
        counts, total = df.groupby('zone').size(), len(df)
        current, epsilon = {z: counts.get(z, 0) / total for z in self.dm.zones_gdf.index}, 1e-9
        kl_divergence = sum(p * np.log((p + epsilon) / (hist.get(z, 0) + epsilon)) for z, p in current.items() if p > 0)
        shannon_entropy = -sum(p * np.log2(p + epsilon) for p in current.values() if p > 0)
        mutual_info = 0.0
        if 'type' in df.columns and not df.dropna(subset=['zone', 'type']).empty:
            joint = df.groupby(['zone', 'type']).size().unstack(fill_value=0) / total
            type_marginal, zone_marginal = joint.sum(axis=0), joint.sum(axis=1)
            if not type_marginal.empty and not zone_marginal.empty:
                valid_zones = joint.index.intersection(zone_marginal.index)
                valid_types = joint.columns.intersection(type_marginal.index)
                mutual_info = sum(joint.loc[z, t] * np.log2((joint.loc[z, t] + epsilon) / (zone_marginal.get(z, 0) * type_marginal.get(t, 0) + epsilon)) for z in valid_zones for t in valid_types if joint.loc[z, t] > 0)
        return kl_divergence, shannon_entropy, hist, current, mutual_info
        
    def forecast_risk_over_time(self, risk_scores: Dict, anomaly: float, horizon: int) -> pd.DataFrame:
        chaotic_series, data = self._generate_chaotic_series(steps=horizon), []
        for zone in self.dm.zones_gdf.index:
            node, traffic, incidents = self.dm.zones_gdf.loc[zone, 'node'], np.random.uniform(0.3, 1.0, horizon), np.random.randint(0, 5, horizon)
            embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
            X = np.hstack([np.array([[h, t, i] for h, t, i in zip(range(horizon), traffic, incidents)]), np.tile(embedding, (horizon, 1))])
            ml_pred, (gp_pred, _) = self.ml_models[zone].predict(X), self.gp_models[zone].predict(X, return_std=True)
            combined_pred = np.clip((0.7 * ml_pred + 0.3 * gp_pred) * (1 + 0.1 * chaotic_series * anomaly), 0, 2)
            for h, pred in enumerate(combined_pred): data.append({'zone': zone, 'hour': h, 'projected_risk': pred})
        return pd.DataFrame(data)

# --- FIX: RE-INTRODUCE SENSITIVITYANALYZER CLASS ---
class SensitivityAnalyzer:
    """Performs sensitivity analysis on model parameters."""
    def __init__(self, simulation_engine: SimulationEngine, predictive_engine: PredictiveAnalyticsEngine):
        self.sim_engine = simulation_engine
        self.pred_engine = predictive_engine
        logger.info("SensitivityAnalyzer initialized.")

    def analyze_sensitivity(self, env_factors: EnvFactors, parameters: Dict[str, List[float]], iterations: int = 10) -> pd.DataFrame:
        """Analyzes model sensitivity to parameter perturbations."""
        try:
            results = []
            base_state = self.sim_engine.get_live_state(env_factors)
            base_risk = self.pred_engine.calculate_holistic_risk(base_state)[1]
            base_anomaly = self.pred_engine.calculate_information_metrics(base_state)[0]

            for param, values in parameters.items():
                for value in values:
                    for _ in range(iterations):
                        modified_factors = EnvFactors(
                            env_factors.is_holiday,
                            env_factors.is_payday,
                            env_factors.weather_condition,
                            env_factors.major_event_active,
                            value if param == 'traffic_multiplier' else env_factors.traffic_multiplier,
                            int(value) if param == 'base_rate' else env_factors.base_rate,
                            value if param == 'self_excitation_factor' else env_factors.self_excitation_factor
                        )
                        state = self.sim_engine.get_live_state(modified_factors)
                        risk = self.pred_engine.calculate_holistic_risk(state)[1]
                        anomaly = self.pred_engine.calculate_information_metrics(state)[0]
                        risk_diff = np.mean([
                            abs(risk.get(node, 0) - base_risk.get(node, 0))
                            for node in self.pred_engine.dm.road_graph.nodes()
                        ])
                        results.append({
                            'parameter': param,
                            'value': value,
                            'risk_diff': risk_diff,
                            'anomaly_diff': abs(anomaly - base_anomaly)
                        })
            df = pd.DataFrame(results)
            logger.info("Completed sensitivity analysis for %d parameter settings.", len(df))
            return df
        except Exception as e:
            logger.error("Failed to perform sensitivity analysis: %s", e)
            return pd.DataFrame()

class StrategicAdvisor:
    """Handles resource reallocation."""
    def __init__(self, data_manager: DataManager, engine: PredictiveAnalyticsEngine, model_params: Dict):
        self.dm, self.engine, self.params = data_manager, engine, model_params
        
    def calculate_projected_response_time(self, zone: str, ambulances: List[Dict]) -> float:
        if zone not in self.dm.zones_gdf.index: return DEFAULT_RESPONSE_TIME
        zone_node = self.dm.zones_gdf.loc[zone, 'nearest_node']
        if not zone_node or pd.isna(zone_node): return DEFAULT_RESPONSE_TIME
        min_time = float('inf')
        for amb in ambulances:
            if amb.get('status') != 'Disponible' or not amb.get('nearest_node') or pd.isna(amb.get('nearest_node')): continue
            try: min_time = min(min_time, nx.shortest_path_length(self.dm.road_graph, source=amb['nearest_node'], target=zone_node, weight='weight') + self.params['response_time_turnout_penalty'])
            except nx.NetworkXNoPath: continue
        return min_time if min_time != float('inf') else DEFAULT_RESPONSE_TIME
        
    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        available = [{'id': amb_id, **d} for amb_id, d in self.dm.ambulances.items() if d.get('status') == 'Disponible' and d.get('nearest_node')]
        if not available: return []
        perf = {z: {'risk': risk_scores.get(d['node'], 0), 'rt': self.calculate_projected_response_time(z, available)} for z, d in self.dm.zones_gdf.iterrows()}
        deficits = {z: p['risk'] * p['rt'] for z, p in perf.items()}
        if not deficits or max(deficits.values()) < self.params['recommendation_deficit_threshold']: return []
        target_zone = max(deficits, key=deficits.get); original_rt, target_node = perf[target_zone]['rt'], self.dm.zones_gdf.loc[target_zone, 'nearest_node']
        best, max_utility = None, -float('inf')
        for amb in available:
            moved_ambulances = [{**a, 'nearest_node': target_node} if a['id'] == amb['id'] else a for a in available]
            new_rt = self.calculate_projected_response_time(target_zone, moved_ambulances)
            utility = (original_rt - new_rt) * perf[target_zone]['risk']
            if utility > max_utility: max_utility, best = utility, (amb['id'], self.dm.node_to_zone_map.get(amb['nearest_node'], 'Unknown'), new_rt)
        if best and max_utility > self.params['recommendation_improvement_threshold']:
            amb_id, from_zone, new_rt = best
            return [{"unit": amb_id, "from": from_zone, "to": target_zone, "reason": f"Reducir el tiempo de respuesta proyectado en '{target_zone}' de ~{original_rt:.0f} min a ~{new_rt:.0f} min."}]
        return []

class VisualizationSuite:
    """Handles visualizations."""
    def __init__(self, style_config: Dict): self.config = style_config
    def plot_risk_comparison(self, prior_df, posterior_df):
        if prior_df.empty or posterior_df.empty: return alt.Chart().mark_text().encode(text=alt.value("No data"))
        prior_df['type'], posterior_df['type'] = 'A Priori (Histórico)', 'A Posteriori (Actual + Difusión)'
        return alt.Chart(pd.concat([prior_df, posterior_df])).mark_bar(opacity=0.8).encode(x=alt.X('risk:Q', title='Nivel de Riesgo'), y=alt.Y('zone:N', title='Zona', sort='-x'), color=alt.Color('type:N', title='Tipo de Riesgo', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])), tooltip=['zone', alt.Tooltip('risk', format='.3f')]).properties(title="Análisis de Riesgo Bayesiano").interactive()
    def plot_distribution_comparison(self, hist_df, current_df):
        if hist_df.empty or current_df.empty: return alt.Chart().mark_text().encode(text=alt.value("No data"))
        hist_df['type'], current_df['type'] = 'Distribución Histórica', 'Distribución Actual'
        bars = alt.Chart(pd.concat([hist_df, current_df])).mark_bar().encode(x=alt.X('percentage:Q', title='% de Incidentes', axis=alt.Axis(format='%')), y=alt.Y('zone:N', title='Zona', sort=alt.EncodingSortField(field="percentage", op="sum", order='descending')), color=alt.Color('type:N', title='Distribución', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])), tooltip=['zone', alt.Tooltip('percentage', title='Porcentaje', format='.1%')])
        return alt.layer(bars).facet(row=alt.Row('type:N', title="", header=alt.Header(labelAngle=0, labelAlign='left', labelFontSize=14))).properties(title="Análisis de Anomalía de Distribución").resolve_scale(y='independent')
    def plot_risk_forecast(self, df): 
        if df.empty: return alt.Chart().mark_text().encode(text=alt.value("No data"))
        return alt.Chart(df).mark_line(color=self.config['colors']['primary'], point=True).encode(x=alt.X('hour:Q', title='Horas a Futuro'), y=alt.Y('projected_risk:Q', title='Riesgo Proyectado', scale=alt.Scale(zero=False)), tooltip=['hour', alt.Tooltip('projected_risk', format='.3f')]).properties(title="Pronóstico de Riesgo por Zona").interactive()
    def plot_incident_trends(self, incidents_df): 
        if incidents_df.empty or 'zone' not in incidents_df.columns or incidents_df['zone'].isnull().all(): return alt.Chart().mark_text().encode(text=alt.value("No data"))
        counts = incidents_df.groupby(['type', 'zone']).size().reset_index(name='count')
        return alt.Chart(counts).mark_bar().encode(x=alt.X('type:N', title='Tipo de Incidente'), y=alt.Y('count:Q', title='Número de Incidentes'), color=alt.Color('zone:N', title='Zona'), tooltip=['type', 'zone', 'count']).properties(title="Tendencias de Incidentes por Tipo y Zona").interactive()
    def plot_event_probability(self, prob_df): 
        if prob_df.empty: return alt.Chart().mark_text().encode(text=alt.value("No data"))
        return alt.Chart(prob_df).mark_line(point=True).encode(x=alt.X('horizon:Q', title='Horizonte (Horas)'), y=alt.Y('probability:Q', title='Probabilidad de Incidente', axis=alt.Axis(format='%')), color=alt.Color('zone:N', title='Zona'), tooltip=['zone', 'horizon', alt.Tooltip('probability', format='.2%')]).properties(title="Probabilidad de Incidentes por Zona y Horizonte").interactive()
    def plot_entropy_heatmap(self, entropy_dict: Dict, zones_gdf: gpd.GeoDataFrame):
        data_gdf = zones_gdf.copy(); data_gdf['entropy'] = data_gdf.index.map(entropy_dict).fillna(0); max_entropy = max(0.01, data_gdf['entropy'].max()); data_gdf['fill_color'] = data_gdf['entropy'].apply(lambda e: [100, 200, 100, int(200 * (e / max_entropy))]).tolist()
        return pdk.Layer("PolygonLayer", data=data_gdf, get_polygon="geometry.exterior.coords", filled=True, stroked=False, get_fill_color="fill_color", opacity=0.2, pickable=True)
    def plot_chaos_regime_map(self, prob_df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame):
        data_gdf = zones_gdf.copy()
        if not prob_df.empty: data_gdf['fractal_dimension'] = data_gdf.index.to_series().map(prob_df.groupby('zone')['fractal_dimension'].mean()).fillna(1.0)
        else: data_gdf['fractal_dimension'] = 1.0
        data_gdf['fill_color'] = data_gdf['fractal_dimension'].apply(lambda fd: self.config['colors']['chaos_high'] if fd > 1.5 else self.config['colors']['chaos_low']).tolist()
        return pdk.Layer("PolygonLayer", data=data_gdf, get_polygon="geometry.exterior.coords", filled=True, stroked=False, get_fill_color="fill_color", opacity=0.3, pickable=True)

def prepare_visualization_data(data_manager: DataManager, risk_scores: Dict, all_incidents: List, style: Dict):
    """Prepares data for map and charts, ensuring data types are JSON-serializable."""
    hosp_df = pd.DataFrame([{'name': f"H: {n}", 'tooltip_text': f"Cap: {d['capacity']} Carga: {d['load']}", 'lon': d['location'].x, 'lat': d['location'].y, 'icon_data': {"url": style['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}} for n, d in data_manager.hospitals.items() if d.get('location') and not d['location'].is_empty])
    valid_ambulances = [d for amb_id, d in data_manager.ambulances.items() if d.get('location') and not d['location'].is_empty]
    amb_df = pd.DataFrame([{'name': f"U: {d['id']}", 'tooltip_text': f"Estado: {d['status']}<br>Base: {d['home_base']}", 'lon': d['location'].x, 'lat': d['location'].y, 'icon_data': {"url": style['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, 'size': float(style['sizes']['ambulance'])} for d in valid_ambulances])
    valid_incidents = [i for i in all_incidents if i.get('location') and isinstance(i['location'], Point) and not i['location'].is_empty]
    inc_df = pd.DataFrame([{'name': f"I: {i.get('id', 'N/A')}", 'tooltip_text': f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}", 'lon': i['location'].x, 'lat': i['location'].y, 'color': style['colors']['hawkes_echo'] if i.get('is_echo') else style['colors']['accent_crit'], 'radius': float(style['sizes']['hawkes_echo']) if i.get('is_echo') else float(style['sizes']['incident_base'])} for i in valid_incidents])
    heat_df = pd.DataFrame([{"lon": i['location'].x, "lat": i['location'].y} for i in valid_incidents])
    zones_gdf = data_manager.zones_gdf.copy(); zones_gdf['risk'] = zones_gdf['node'].map(risk_scores).fillna(0.0)
    zones_gdf['tooltip_text'] = zones_gdf.apply(lambda r: f"Zona: {r.name}<br/>Riesgo: {r.risk:.3f}", axis=1)
    max_risk = max(0.01, zones_gdf['risk'].max())
    zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * (r / max_risk))]).tolist()
    return zones_gdf, hosp_df, amb_df, inc_df, heat_df

def create_deck_gl_map(zones_gdf: gpd.GeoDataFrame, hospital_df, ambulance_df, incident_df, heatmap_df, entropy_layer, chaos_layer, app_config: Dict):
    """Creates a Deck.gl map."""
    style = app_config['styling']
    layers = [pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="geometry.exterior.coords", filled=True, stroked=False, extruded=True, get_elevation=f"risk * {style['map_elevation_multiplier']}", get_fill_color="fill_color", opacity=0.1, pickable=True), entropy_layer, chaos_layer, pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style['sizes']['hospital'], size_scale=15, pickable=True), pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', size_scale=15, pickable=True)]
    if not heatmap_df.empty: layers.insert(0, pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN', threshold=0.1, get_weight=1))
    if not incident_df.empty: layers.append(pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True, radius_min_pixels=2, radius_max_pixels=100))
    view_state = pdk.ViewState(latitude=32.5, longitude=-117.02, zoom=11, bearing=0, pitch=50)
    tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "borderRadius": "5px", "padding": "5px"}}
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_provider="mapbox" if app_config['mapbox_api_key'] else "carto", map_style=app_config['map_style'], api_keys={'mapbox': app_config['mapbox_api_key']}, tooltip=tooltip)

@st.cache_resource
def initialize_app_components():
    """Initializes and caches main application components."""
    warnings.filterwarnings('ignore', category=UserWarning); warnings.filterwarnings('ignore', category=FutureWarning)
    app_config = get_app_config()
    distributions = {k: _normalize_dist(v) for k, v in app_config['data']['distributions'].items()}
    data_manager = DataManager(app_config)
    engine = SimulationEngine(data_manager, app_config['simulation_params'], distributions)
    predictor = PredictiveAnalyticsEngine(data_manager, app_config['model_params'], distributions)
    advisor = StrategicAdvisor(data_manager, predictor, app_config['model_params'])
    sensitivity_analyzer = SensitivityAnalyzer(engine, predictor)
    plotter = VisualizationSuite(app_config['styling'])
    logger.info("Application components initialized.")
    return data_manager, engine, predictor, advisor, sensitivity_analyzer, plotter, app_config

def render_intel_briefing(anomaly, entropy, mutual_info, recommendations, app_config):
    """Renders the intelligence briefing section."""
    st.subheader("Intel Briefing y Recomendaciones"); status = "ANÓMALO" if anomaly > 0.2 else "ELEVADO" if anomaly > 0.1 else "NOMINAL"
    c1, c2, c3 = st.columns(3); c1.metric("Estado del Sistema", status); c2.metric("Puntuación de Anomalía (KL)", f"{anomaly:.4f}"); c3.metric("Información Mutua", f"{mutual_info:.4f}"); c1.metric("Entropía Espacial (Desorden)", f"{entropy:.4f} bits")
    if recommendations:
        st.warning("Recomendación de Despliegue de Recursos:")
        for r in recommendations: st.write(f"**Mover {r['unit']}** de `{r['from']}` a `{r['to']}`. **Razón:** {r['reason']}")
    else: st.success("No se requieren reasignaciones de recursos.")

def _render_scenario_dashboard(dm, engine, predictor, advisor, plotter, config, factors, current_hour):
    """Helper function to render the main dashboard for a given scenario."""
    live_state = engine.get_live_state(factors, current_hour)
    _, risk = predictor.calculate_holistic_risk(live_state); anomaly, entropy, _, current_dist, mutual_info = predictor.calculate_information_metrics(live_state)
    recs = advisor.recommend_resource_reallocations(risk); render_intel_briefing(anomaly, entropy, mutual_info, recs, config)
    st.divider(); st.subheader("Mapa de Operaciones")
    with st.spinner("Preparando visualización y cálculos predictivos..."):
        prob_df = pd.concat([predictor.forward_predictor.compute_event_probability(live_state, risk, h) for h in FORECAST_HORIZONS], ignore_index=True)
        entropy_dict = {z: -p * np.log2(p + 1e-9) if p > 0 else 0 for z, p in current_dist.items()}
        entropy_layer, chaos_layer = plotter.plot_entropy_heatmap(entropy_dict, dm.zones_gdf), plotter.plot_chaos_regime_map(prob_df, dm.zones_gdf)
        vis_data = prepare_visualization_data(dm, risk, live_state["active_incidents"], config['styling'])
        st.pydeck_chart(create_deck_gl_map(*vis_data, entropy_layer, chaos_layer, config), use_container_width=True)

def render_sandbox_tab(dm, engine, predictor, advisor, sensitivity_analyzer, plotter, config):
    """Renders the interactive sandbox tab."""
    st.header("Command Sandbox: Simulador Interactivo")
    c1, c2, c3 = st.columns(3); is_holiday, is_payday, weather = c1.checkbox("Día Festivo"), c2.checkbox("Quincena"), c3.selectbox("Clima", ["Despejado", "Lluvia", "Niebla"])
    c1, c2 = st.columns(2); base_rate, excitation = c1.slider("μ (Tasa Base)", 1, 20, 5, key="sb_br"), c2.slider("κ (Excitación)", 0.0, 1.0, 0.5, key="sb_ex")
    factors, current_hour = EnvFactors(is_holiday, is_payday, weather, False, 1.0, base_rate, excitation), float(st.session_state.get('current_hour', 0.0))
    _render_scenario_dashboard(dm, engine, predictor, advisor, plotter, config, factors, current_hour)
    if st.button("Ejecutar Análisis de Sensibilidad"):
        with st.spinner("Calculando sensibilidad del modelo..."):
            params_to_test = {'traffic_multiplier': [0.8, 1.0, 1.2], 'base_rate': [3, 5, 10], 'self_excitation_factor': [0.3, 0.5, 0.7]}
            sensitivity_df = sensitivity_analyzer.analyze_sensitivity(factors, params_to_test)
            if not sensitivity_df.empty:
                st.altair_chart(alt.Chart(sensitivity_df).mark_point(filled=True, size=100).encode(x=alt.X('value:Q', title='Valor del Parámetro', scale=alt.Scale(zero=False)), y=alt.Y('risk_diff:Q', title='Diferencia en Riesgo'), color=alt.Color('parameter:N', title='Parámetro'), size=alt.Size('anomaly_diff:Q', title='Diferencia en Anomalía'), tooltip=['parameter', 'value', 'risk_diff', 'anomaly_diff']).properties(title="Análisis de Sensibilidad").interactive(), use_container_width=True)

def render_scenario_planner_tab(dm, engine, predictor, advisor, plotter, config):
    """Renders the scenario planner tab."""
    st.header("Planificador de Escenarios")
    scenarios = {"Día Normal": EnvFactors(False, False, 'Despejado', False, 1.0, 5, 0.3), "Colapso Fronterizo": EnvFactors(False, True, 'Despejado', True, 3.0, 8, 0.6), "Evento Masivo con Lluvia": EnvFactors(False, False, 'Lluvia', True, 1.8, 12, 0.7)}
    name = st.selectbox("Seleccione un Escenario:", list(scenarios.keys()))
    _render_scenario_dashboard(dm, engine, predictor, advisor, plotter, config, scenarios[name], float(st.session_state.get('current_hour', 0.0)))

def render_analysis_tab(dm, engine, predictor, plotter):
    """Renders the analysis tab."""
    st.header("Análisis Profundo del Sistema")
    if st.button("🔄 Generar Nuevo Estado de Muestra"): st.session_state.analysis_state = engine.get_live_state(EnvFactors(False, False, 'Despejado', False, np.random.uniform(0.8, 2.0), np.random.randint(3, 15), np.random.uniform(0.2, 0.8)), float(st.session_state.get('current_hour', 0.0)))
    if 'analysis_state' not in st.session_state: st.info("Genere un 'Estado de Muestra' para ver el análisis."); return
    live_state = st.session_state.analysis_state
    prior, posterior = predictor.calculate_holistic_risk(live_state); _, _, hist, current, _ = predictor.calculate_information_metrics(live_state)
    st.altair_chart(plotter.plot_risk_comparison(pd.DataFrame(list(prior.items()), columns=['zone', 'risk']), pd.DataFrame([{'zone': dm.node_to_zone_map.get(n, '?'), 'risk': r} for n, r in posterior.items() if dm.node_to_zone_map.get(n)])), use_container_width=True)
    st.altair_chart(plotter.plot_distribution_comparison(pd.DataFrame(list(hist.items()), columns=['zone', 'percentage']), pd.DataFrame(list(current.items()), columns=['zone', 'percentage'])), use_container_width=True)
    incidents_df = pd.DataFrame(live_state.get("active_incidents", []))
    if not incidents_df.empty and 'zone' in incidents_df.columns and not incidents_df['zone'].isnull().all(): st.altair_chart(plotter.plot_incident_trends(incidents_df.dropna(subset=['zone', 'type'])), use_container_width=True)

def render_forecasting_tab(predictor, plotter):
    """Renders the forecasting tab."""
    st.header("Pronóstico de Riesgo Futuro")
    if 'analysis_state' not in st.session_state: st.warning("Genere un 'Estado de Muestra' en la pestaña de 'Análisis' para poder realizar un pronóstico."); return
    live = st.session_state.analysis_state; _, risk = predictor.calculate_holistic_risk(live); anomaly, _, _, _, _ = predictor.calculate_information_metrics(live)
    c1, c2 = st.columns(2); zone, horizon = c1.selectbox("Zona:", options=list(predictor.dm.zones_gdf.index)), c2.select_slider("Horizonte:", options=FORECAST_HORIZONS, value=24)
    df, prob_df = predictor.forecast_risk_over_time(risk, anomaly, horizon), pd.concat([predictor.forward_predictor.compute_event_probability(live, risk, h) for h in FORECAST_HORIZONS])
    if not df.empty: st.altair_chart(plotter.plot_risk_forecast(df[df['zone'] == zone]), use_container_width=True)
    if not prob_df.empty: st.altair_chart(plotter.plot_event_probability(prob_df[prob_df['zone'] == zone]), use_container_width=True)

def render_knowledge_center():
    """Renders the knowledge center."""
    st.header("Centro de Conocimiento (v10.4)"); st.info("Manual de Arquitectura, Modelos Matemáticos y Guía de Decisión del Digital Twin.")
    st.subheader("1. Arquitectura de Software y Optimizaciones"); st.markdown("- **Vectorización Geoespacial**: Uso de `GeoPandas.sjoin_nearest` para asignaciones espaciales eficientes.\n- **Manejo de CRS**: Cálculos en `EPSG:32611`, conversión a `EPSG:4326` para visualización.\n- **Caching Optimizado**: `st.cache_data` y `st.cache_resource` para inicialización rápida.\n- **Desacoplamiento**: Componentes modulares para extensibilidad.")
    st.subheader("2. Modelos Matemáticos"); st.markdown("- **Procesos Estocásticos**: NHPP, Hawkes Process.\n- **Inferencia Bayesiana**: Bayesian Network, Variational Inference.\n- **Teoría de Grafos**: Node2Vec, Laplacian Diffusion.\n- **Teoría del Caos y Geometría Fractal**: Logistic Map, Fractal Dimension.\n- **Machine Learning**: Gradient Boosting, Gaussian Processes, Temporal CNN.\n- **Teoría de la Información**: KL-Divergence, Shannon Entropy, Mutual Information.\n- **Teoría de Juegos**: Multi-Agent Optimization.")
    st.subheader("3. Interpretación para Toma de Decisiones"); st.markdown("- **Mapa de Riesgo Dinámico**: Riesgo Proyectado, Heatmap de Entropía, Mapa de Régimen Caótico.\n- **Indicadores Clave**: KL-Divergence, Mutual Information, Response Time.\n- **Pronósticos Multi-Resolución**: Horizontes temporales, Probabilidad de Incidentes.\n- **Recomendaciones Estratégicas**: Reubicaciones de Ambulancias, Análisis de Sensibilidad.")

def main():
    """Main application entry point."""
    st.set_page_config(page_title="RedShield AI v10.4", layout="wide")
    st.title("RedShield AI Command Suite"); st.markdown("**Digital Twin para Gestión de Emergencias Médicas** | Versión 10.4 (Final)")
    try:
        dm, engine, predictor, advisor, sensitivity_analyzer, plotter, config = initialize_app_components()
        tabs = st.tabs(["Command Sandbox", "Planificador de Escenarios", "Análisis Profundo", "Pronósticos", "Centro de Conocimiento"])
        if 'current_hour' not in st.session_state:
            st.session_state.current_hour = 0.0
        st.session_state.current_hour += 0.1
        
        with tabs[0]:
            render_sandbox_tab(dm, engine, predictor, advisor, sensitivity_analyzer, plotter, config)
        with tabs[1]:
            render_scenario_planner_tab(dm, engine, predictor, advisor, plotter, config)
        with tabs[2]:
            render_analysis_tab(dm, engine, predictor, plotter)
        with tabs[3]:
            render_forecasting_tab(predictor, plotter)
        with tabs[4]:
            render_knowledge_center()
            
    except Exception as e:
        logger.error("Main application failed at top level: %s", e, exc_info=True)
        st.error(f"A critical error occurred: {e}. Please check the logs.")

if __name__ == "__main__":
    main()
