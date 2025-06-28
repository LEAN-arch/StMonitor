```python
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
        return 1.0  # A single point is dimension 0, but 1.0 is a good baseline for non-chaotic distribution.

    try:
        points = [g for g in geom_array if isinstance(g, Point)]
        if len(points) < 2:
            return 1.0
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        # Calculate bounding box area
        spread = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        if spread == 0:
            return 1.0  # All points are coincident
        # Heuristic: relate fractal dimension to density. Higher density in a small area = higher complexity.
        density = len(points) / spread
        # Scale to a plausible range for 2D data (e.g., 1.0 to 2.0)
        return np.clip(1.0 + np.log1p(density * 1e-4), 1.0, 2.0)
    except Exception as e:
        logger.warning(f"Could not compute placeholder fractal dimension: {e}")
        return 1.0  # Return a default value on any error

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
        self.zones_gdf = self._build_zones_gdf()
        self.hospitals = self._initialize_hospitals()
        self.ambulances = self._initialize_ambulances()
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
        try:
            G = nx.Graph()
            network_config = _self.config.get('road_network', {})
            for node, data in network_config.get('nodes', {}).items():
                G.add_node(node, pos=data['pos'])
            for edge in network_config.get('edges', []):
                G.add_edge(edge[0], edge[1], weight=edge[2])
            logger.info("Road graph built with %d nodes and %d edges.", G.number_of_nodes(), G.number_of_edges())
            return G
        except Exception as e:
            logger.error("Failed to build road graph: %s", e)
            raise

    def _load_or_compute_graph_embeddings(self) -> Dict[str, np.ndarray]:
        """Loads or computes Node2Vec embeddings with caching."""
        cache_file = CACHE_DIR / "graph_embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info("Loaded cached graph embeddings.")
                return embeddings
            except Exception as e:
                logger.warning("Failed to load cached embeddings: %s. Recomputing.", e)

        try:
            node2vec = Node2Vec(self.road_graph, dimensions=8, walk_length=5, num_walks=20, workers=2, quiet=True)
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            embeddings = {node: model.wv[node] for node in self.road_graph.nodes()}
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info("Computed and cached graph embeddings.")
            return embeddings
        except Exception as e:
            logger.error("Failed to compute graph embeddings: %s", e)
            return {}

    @st.cache_resource
    def _build_zones_gdf(_self) -> gpd.GeoDataFrame:
        """Builds zones GeoDataFrame with vectorized operations and geometry validation."""
        try:
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

            # Rename index columns to avoid name collisions
            left_gdf = gdf_projected[['geometry']].reset_index().rename(columns={'index': 'zone_name'})
            right_gdf = graph_nodes_gdf[['geometry']].reset_index().rename(columns={'index': 'node_name'})
            
            nearest = gpd.sjoin_nearest(
                left_gdf,
                right_gdf,
                how='left', distance_col='distance'
            )
            
            nearest = nearest.drop_duplicates(subset='zone_name').set_index('zone_name')
            gdf['nearest_node'] = gdf.index.map(nearest['node_name'])
            
            logger.info("Zones GeoDataFrame built with %d zones.", len(gdf))
            return gdf.drop(columns=['polygon'])
        except Exception as e:
            logger.error("Failed to build zones GeoDataFrame: %s", e, exc_info=True)
            raise

    def _initialize_hospitals(self) -> Dict:
        """Initializes hospitals with Point geometries."""
        try:
            return {
                n: {**d, 'location': Point(d['location'][1], d['location'][0])}
                for n, d in self.config.get('hospitals', {}).items()
            }
        except Exception as e:
            logger.error("Failed to initialize hospitals: %s", e)
            return {}

    def _initialize_ambulances(self) -> Dict:
        """Initializes ambulances with location and node assignments."""
        ambulances = {}
        for amb_id, amb_data in self.config.get('ambulances', {}).items():
            home_zone = amb_data.get('home_base')
            try:
                if home_zone in self.zones_gdf.index:
                    zone_info = self.zones_gdf.loc[home_zone]
                    ambulances[amb_id] = {
                        **amb_data,
                        'location': zone_info.centroid,
                        'nearest_node': zone_info.nearest_node
                    }
                else:
                    logger.warning("Invalid home_base '%s' for ambulance %s.", home_zone, amb_id)
            except Exception as e:
                logger.error("Failed to initialize ambulance %s: %s", amb_id, e)
        logger.info("Initialized %d ambulances.", len(ambulances))
        return ambulances

    def _initialize_prior_history(self) -> Dict:
        """Initializes historical priors for Bayesian updates."""
        try:
            return {
                zone: {'mean_risk': data['prior_risk'], 'count': 1, 'variance': 0.1}
                for zone, data in self.zones_gdf.iterrows()
            }
        except Exception as e:
            logger.error("Failed to initialize prior history: %s", e)
            return {}

    def update_priors(self, live_state: Dict):
        """Updates zone priors using Bayesian updates with streaming data."""
        try:
            df = pd.DataFrame(live_state.get("active_incidents", []))
            if df.empty or 'zone' not in df.columns:
                logger.info("No new incidents for prior update.")
                return

            counts = df.groupby('zone').size()
            for zone in self.zones_gdf.index:
                new_count = counts.get(zone, 0)
                if new_count == 0:
                    continue
                prior = self.prior_history.get(zone, {'mean_risk': 0.5, 'count': 1, 'variance': 0.1})
                new_mean = (prior['mean_risk'] * prior['count'] + new_count * 0.1) / (prior['count'] + new_count)
                new_variance = max(0.01, prior['variance'] * prior['count'] / (prior['count'] + new_count))
                self.prior_history[zone] = {
                    'mean_risk': new_mean,
                    'count': prior['count'] + new_count,
                    'variance': new_variance
                }
                self.zones_gdf.loc[zone, 'prior_risk'] = new_mean
            logger.info("Updated priors for %d zones.", len(counts))
        except Exception as e:
            logger.error("Failed to update priors: %s", e)

    def assign_zones_to_incidents(self, incidents_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assigns zones to incidents using spatial join."""
        if incidents_gdf.empty:
            logger.warning("Empty incidents GeoDataFrame provided.")
            return incidents_gdf.assign(zone=None)
        try:
            joined = gpd.sjoin(incidents_gdf, self.zones_gdf[['geometry']], how="left", predicate="intersects")
            return incidents_gdf.assign(zone=joined['index_right'])
        except Exception as e:
            logger.error("Failed to assign zones to incidents: %s", e)
            return incidents_gdf.assign(zone=None)


```python
class SimulationEngine:
    """Generates synthetic incident data with real-time simulation capabilities."""
    def __init__(self, data_manager: DataManager, sim_params: Dict, distributions: Dict):
        self.dm = data_manager
        self.sim_params = sim_params
        self.dist = distributions
        self.nhpp_intensity = lambda t: 0.1 + 0.05 * np.sin(t / 24 * 2 * np.pi)  # Time-varying intensity
        logger.info("SimulationEngine initialized.")

    @st.cache_data(ttl=60)
    def get_live_state(_self, env_factors: EnvFactors, time_hour: float = 0.0) -> Dict[str, Any]:
        """Generates live state with NHPP and Markov state transitions."""
        try:
            mult, base_rate = _self.sim_params['multipliers'], float(env_factors.base_rate)
            intensity = _self.nhpp_intensity(time_hour)
            if env_factors.is_holiday:
                intensity *= mult['holiday']
            if env_factors.is_payday:
                intensity *= mult['payday']
            if env_factors.weather_condition == 'Lluvia':
                intensity *= mult['rain']
            if env_factors.major_event_active:
                intensity *= mult['major_event']

            num_incidents = int(np.random.poisson(intensity * base_rate))
            if num_incidents == 0:
                return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}

            types = np.random.choice(
                list(_self.dist['incident_type'].keys()),
                num_incidents,
                p=list(_self.dist['incident_type'].values())
            )
            triages = np.random.choice(
                list(_self.dist['triage'].keys()),
                num_incidents,
                p=list(_self.dist['triage'].values())
            )
            minx, miny, maxx, maxy = _self.dm.city_boundary_bounds

            incidents_gdf = gpd.GeoDataFrame(
                {'type': types, 'triage': triages, 'is_echo': False, 'timestamp': time_hour},
                geometry=gpd.points_from_xy(
                    np.random.uniform(minx, maxx, num_incidents),
                    np.random.uniform(miny, maxy, num_incidents)
                ),
                crs=GEOGRAPHIC_CRS
            )
            incidents_gdf = incidents_gdf[incidents_gdf.within(_self.dm.city_boundary_poly)].reset_index(drop=True)
            if incidents_gdf.empty:
                logger.info("No incidents within city boundary.")
                return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}

            incidents_gdf['id'] = [f"{row.type[0]}-{idx}" for idx, row in incidents_gdf.iterrows()]
            incidents_gdf = _self.dm.assign_zones_to_incidents(incidents_gdf)
            
            incidents_list = []
            for _, row in incidents_gdf.iterrows():
                record = row.to_dict()
                record['location'] = row.geometry
                incidents_list.append(record)

            triggers = incidents_gdf[incidents_gdf['triage'] == 'Rojo']
            echo_data = []
            for idx, trigger in triggers.iterrows():
                if np.random.rand() < env_factors.self_excitation_factor:
                    for j in range(np.random.randint(1, 3)):
                        echo_loc = Point(
                            trigger.geometry.x + np.random.normal(0, 0.005),
                            trigger.geometry.y + np.random.normal(0, 0.005)
                        )
                        echo_data.append({
                            'id': f"ECHO-{idx}-{j}",
                            'type': "Echo",
                            'triage': "Verde",
                            'location': echo_loc,
                            'is_echo': True,
                            'zone': trigger.zone,
                            'timestamp': time_hour
                        })
            
            traffic_conditions = {
                z: min(1.0, v * env_factors.traffic_multiplier)
                for z, v in {z: np.random.uniform(0.3, 1.0) for z in _self.dm.zones_gdf.index}.items()
            }

            if len(incidents_list) > 10 or any(t['triage'] == 'Rojo' for t in incidents_list):
                 system_state = "Anomalous"
            elif len(incidents_list) > 5:
                 system_state = "Elevated"
            else:
                 system_state = "Normal"

            logger.info("Generated live state with %d incidents, state: %s.", len(incidents_list) + len(echo_data), system_state)
            return {
                "active_incidents": incidents_list + echo_data,
                "traffic_conditions": traffic_conditions,
                "system_state": system_state
            }
        except Exception as e:
            logger.error("Failed to generate live state: %s", e, exc_info=True)
            return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}

class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling.
    SME Note: This model is initialized with random weights and is NOT trained.
    Its output is effectively noise. A real implementation requires a training pipeline.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: int = 3):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.kernel_size = kernel_size

    def forward(self, x):
        if x.shape[2] < self.kernel_size:
            logger.warning("Input horizon too short for TCN kernel size.")
            return torch.zeros(x.shape[0], 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        return x

class ForwardPredictiveModule:
    """Computes P(E_{t+h} | F_t) using Bayesian-Hawkes-Gaussian composition."""
    def __init__(self, data_manager: DataManager, model_params: Dict):
        self.dm = data_manager
        self.params = model_params
        self.bayesian_network = self._initialize_bayesian_network()
        self.tcn = TemporalCNN(input_dim=12, hidden_dim=32, output_dim=1)
        self.tcn_optimizer = torch.optim.Adam(self.tcn.parameters(), lr=0.001)
        logger.info("ForwardPredictiveModule initialized.")

    def _initialize_bayesian_network(self):
        """Initializes a Bayesian Network for structured inference."""
        try:
            with pm.Model() as model:
                zones = list(self.dm.zones_gdf.index)
                risk_prior_mean = [self.dm.prior_history[z]['mean_risk'] for z in zones]
                risk_prior_sigma = [self.dm.prior_history[z]['variance']**0.5 for z in zones]
                
                risk = pm.Normal('risk', mu=risk_prior_mean, sigma=risk_prior_sigma, shape=len(zones))
                traffic = pm.Normal('traffic', mu=0.5, sigma=0.2, shape=len(zones))
                incidents = pm.Poisson('incidents', mu=1.0, shape=len(zones))

                observed_risk_data = pm.MutableData('observed_risk_data', np.zeros(len(zones)))

                combined_effect = (risk * self.params['risk_weights']['prior'] +
                                   traffic * self.params['risk_weights']['traffic'] +
                                   incidents * self.params['risk_weights']['incidents'])
                
                pm.Normal('observed_risk', mu=combined_effect, sigma=0.1, observed=observed_risk_data)

            return model
        except Exception as e:
            logger.error("Failed to initialize Bayesian Network: %s", e)
            return None

    def update_bayesian_priors(self, live_state: Dict):
        """
        Updates Bayesian Network priors using variational inference for performance.
        """
        try:
            df = pd.DataFrame(live_state.get("active_incidents", []))
            if self.bayesian_network is None or df.empty or 'zone' not in df.columns:
                logger.info("Skipping Bayesian update (no model or no new incidents).")
                return

            counts = df.groupby('zone').size()
            traffic = live_state.get('traffic_conditions', {})
            with self.bayesian_network:
                observed_data = np.array([
                    counts.get(z, 0) * 0.1 + traffic.get(z, 0.5) * 0.3
                    for z in self.dm.zones_gdf.index
                ])
                pm.set_data({'observed_risk_data': observed_data})
                inference = pm.fit(n=10000, method='advi', progressbar=False)
                trace = inference.sample(500)
                for idx, zone in enumerate(self.dm.zones_gdf.index):
                    risk_samples = trace['risk'][:, idx]
                    self.dm.prior_history[zone]['mean_risk'] = float(risk_samples.mean())
                    self.dm.prior_history[zone]['variance'] = float(risk_samples.var())
            logger.info("Updated Bayesian priors with variational inference.")
        except Exception as e:
            logger.error("Failed to update Bayesian priors: %s", e)

    def compute_event_probability(self, live_state: Dict, risk_scores: Dict, horizon: int) -> pd.DataFrame:
        """Computes P(E_{t+h} | F_t) for each zone and horizon."""
        try:
            data = []
            zones = list(self.dm.zones_gdf.index)
            hawkes_intensity = self.params['hawkes_intensity']
            incidents_df = pd.DataFrame(live_state.get("active_incidents", []))
            counts = incidents_df.groupby('zone').size() if not incidents_df.empty else pd.Series(dtype=int)

            tcn_input = np.zeros((len(zones), 12, horizon))
            for i, zone in enumerate(zones):
                node = self.dm.zones_gdf.loc[zone, 'node']
                embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
                tcn_input[i, :8, :] = embedding[:, np.newaxis]
                tcn_input[i, 8, :] = risk_scores.get(node, 0.5)
                tcn_input[i, 9, :] = live_state.get('traffic_conditions', {}).get(zone, 0.5)
                tcn_input[i, 10, :] = counts.get(zone, 0)
                tcn_input[i, 11, :] = np.linspace(0, horizon / 24, horizon)

            tcn_input_tensor = torch.tensor(tcn_input, dtype=torch.float32)
            with torch.no_grad():
                if tcn_input_tensor.shape[2] < self.tcn.kernel_size:
                    logger.warning("Horizon too short for TCN kernel size. Skipping TCN.")
                    tcn_pred = np.zeros(len(zones))
                else:
                    tcn_pred = self.tcn(tcn_input_tensor).numpy().flatten()

            for i, zone in enumerate(zones):
                node = self.dm.zones_gdf.loc[zone, 'node']
                base_risk = risk_scores.get(node, 0.5)
                hawkes_effect = hawkes_intensity * counts.get(zone, 0)
                bayesian_risk = self.dm.prior_history[zone]['mean_risk']
                prob = np.clip((base_risk + hawkes_effect + tcn_pred[i] + bayesian_risk) / 4, 0, 1)
                
                zone_geoms = incidents_df[incidents_df['zone'] == zone]['location'].dropna().values
                data.append({
                    'zone': zone,
                    'horizon': horizon,
                    'probability': prob,
                    'fractal_dimension': compute_fractal_dimension(zone_geoms)
                })

            result = pd.DataFrame(data)
            logger.info("Computed event probabilities for horizon %d hours.", horizon)
            return result
        except Exception as e:
            logger.error("Failed to compute event probability: %s", e)
            return pd.DataFrame()

class PredictiveAnalyticsEngine:
    """Handles forecasting and analytics with advanced models."""
    def __init__(self, data_manager: DataManager, model_params: Dict, dist_config: Dict):
        self.dm = data_manager
        self.params = model_params
        self.dist = dist_config
        self.ml_models = {zone: GradientBoostingRegressor(n_estimators=50, random_state=42) for zone in self.dm.zones_gdf.index}
        self.gp_models = {
            zone: GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                random_state=42
            ) for zone in self.dm.zones_gdf.index
        }
        self.forward_predictor = ForwardPredictiveModule(data_manager, model_params)
        self._train_models()
        logger.info("PredictiveAnalyticsEngine initialized with %d ML/GP models.", len(self.ml_models))

    def _train_models(self):
        """Trains ML and GP models using synthetic historical data."""
        try:
            np.random.seed(42)
            hours = np.arange(24)
            for zone in self.dm.zones_gdf.index:
                node = self.dm.zones_gdf.loc[zone, 'node']
                embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
                X = np.hstack([
                    np.array([[h, np.random.uniform(0.3, 1.0), np.random.randint(0, 5)] for h in hours]),
                    np.tile(embedding, (len(hours), 1))
                ])
                y = np.random.uniform(0.1, 1.0, 24) * (1 + 0.5 * np.sin(hours / 24 * 2 * np.pi))
                self.ml_models[zone].fit(X, y)
                self.gp_models[zone].fit(X, y)
            logger.info("ML and GP models trained successfully.")
        except Exception as e:
            logger.error("Failed to train models: %s", e)

    def _generate_chaotic_series(self, r=3.9, x0=0.4, steps=100):
        """Generates logistic map chaotic series."""
        try:
            series = np.zeros(steps)
            series[0] = x0
            for i in range(1, steps):
                series[i] = r * series[i-1] * (1 - series[i-1])
            return series
        except Exception as e:
            logger.error("Failed to generate chaotic series: %s", e)
            return np.zeros(steps)

    def calculate_holistic_risk(self, live_state: Dict) -> Tuple[Dict, Dict]:
        """Calculates risk scores with graph Laplacian diffusion."""
        try:
            prior_risks = self.dm.zones_gdf['prior_risk'].to_dict()
            df = pd.DataFrame(live_state.get("active_incidents", []))
            counts = df.groupby('zone').size() if not df.empty and 'zone' in df.columns else pd.Series(dtype=int)

            w, inc_load_factor = self.params['risk_weights'], self.params['incident_load_factor']
            evidence_risk = {
                zone: data['prior_risk'] * w['prior'] +
                      live_state.get('traffic_conditions', {}).get(zone, 0.5) * w['traffic'] +
                      counts.get(zone, 0) * inc_load_factor * w['incidents']
                for zone, data in self.dm.zones_gdf.iterrows()
            }

            node_risks = {data['node']: evidence_risk.get(zone, 0) for zone, data in self.dm.zones_gdf.iterrows() if 'node' in data}
            diffused_risks = self._diffuse_risk_on_graph(node_risks)
            logger.info("Calculated holistic risk scores.")
            return prior_risks, diffused_risks
        except Exception as e:
            logger.error("Failed to calculate holistic risk: %s", e)
            return {}, {}

    def _diffuse_risk_on_graph(self, initial_risks: Dict[str, float]) -> Dict[str, float]:
        """Diffuses risk using graph Laplacian."""
        try:
            graph = self.dm.road_graph
            L = nx.laplacian_matrix(graph).toarray()
            risks = np.array([initial_risks.get(node, 0) for node in graph.nodes()])
            for _ in range(self.params.get('risk_diffusion_steps', 3)):
                risks = risks - self.params.get('risk_diffusion_factor', 0.1) * L @ risks
            diffused_risks = {node: max(0, r) for node, r in zip(graph.nodes(), risks)}
            logger.info("Diffused risk using graph Laplacian.")
            return diffused_risks
        except Exception as e:
            logger.error("Failed to diffuse risk on graph: %s", e)
            return initial_risks

    def calculate_information_metrics(self, live_state: Dict) -> Tuple[float, float, Dict, Dict, float, float, float]:
        """Calculates KL-divergence, Shannon entropy, mutual information, and chi-square test."""
        try:
            hist = self.dist['zone']
            df = pd.DataFrame([i for i in live_state.get("active_incidents", []) if not i.get("is_echo")])
            if df.empty or 'zone' not in df.columns or df['zone'].isnull().all():
                logger.warning("No valid incident data for information metrics.")
                return 0.0, 0.0, hist, {z: 0.0 for z in self.dm.zones_gdf.index}, 0.0, 0.0, 1.0

            counts, total = df.groupby('zone').size(), len(df)
            current = {z: counts.get(z, 0) / total for z in self.dm.zones_gdf.index}
            epsilon = 1e-9

            kl_divergence = sum(
                p * np.log((p + epsilon) / (hist.get(z, 0) + epsilon))
                for z, p in current.items() if p > 0
            )
            shannon_entropy = -sum(
                p * np.log2(p + epsilon)
                for p in current.values() if p > 0
            )

            if 'type' in df.columns:
                joint = df.groupby(['zone', 'type']).size().unstack(fill_value=0) / total
                type_marginal = joint.sum(axis=0)
                zone_marginal = joint.sum(axis=1)
                mutual_info = sum(
                    joint.loc[z, t] * np.log2(joint.loc[z, t] / (zone_marginal[z] * type_marginal[t] + epsilon))
                    for z in joint.index for t in joint.columns if joint.loc[z, t] > 0
                ) if joint.sum().sum() > 0 else 0.0
            else:
                mutual_info = 0.0

            observed = [counts.get(z, 0) for z in self.dm.zones_gdf.index]
            expected = [hist.get(z, 0) * total for z in self.dm.zones_gdf.index]
            if sum(observed) > 0 and sum(expected) > 0:
                chi2, p_value = stats.chisquare(observed, expected)
            else:
                chi2, p_value = 0.0, 1.0

            logger.info("Information metrics: KL=%.4f, Entropy=%.4f, MI=%.4f, Chi2=%.4f, p=%.4f",
                        kl_divergence, shannon_entropy, mutual_info, chi2, p_value)
            return kl_divergence, shannon_entropy, hist, current, mutual_info, chi2, p_value
        except Exception as e:
            logger.error("Failed to calculate information metrics: %s", e, exc_info=True)
            return 0.0, 0.0, hist, {z: 0.0 for z in self.dm.zones_gdf.index}, 0.0, 0.0, 1.0

    def forecast_risk_over_time(self, risk_scores: Dict, anomaly: float, horizon: int) -> pd.DataFrame:
        """Forecasts risk scores using ML, GP, and chaotic modulation."""
        try:
            chaotic_series = self._generate_chaotic_series(steps=horizon)
            data = []
            for zone in self.dm.zones_gdf.index:
                node = self.dm.zones_gdf.loc[zone, 'node']
                traffic = np.random.uniform(0.3, 1.0, horizon)
                incidents = np.random.randint(0, 5, horizon)
                embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
                X = np.hstack([
                    np.array([[h, t, i] for h, t, i in zip(range(horizon), traffic, incidents)]),
                    np.tile(embedding, (horizon, 1))
                ])
                
                ml_pred = self.ml_models[zone].predict(X)
                gp_pred, gp_std = self.gp_models[zone].predict(X, return_std=True)
                combined_pred = 0.7 * ml_pred + 0.3 * gp_pred
                combined_pred = np.clip(combined_pred * (1 + 0.1 * chaotic_series * anomaly), 0, 2)
                
                for h, pred in enumerate(combined_pred):
                    data.append({'zone': zone, 'hour': h, 'projected_risk': pred})
            
            df = pd.DataFrame(data)
            logger.info("Generated risk forecast for %d hours across %d zones.", horizon, len(self.dm.zones_gdf.index))
            return df
        except Exception as e:
            logger.error("Failed to forecast risk: %s", e)
            return pd.DataFrame()

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
    """Handles resource reallocation with multi-agent game theory."""
    def __init__(self, data_manager: DataManager, engine: PredictiveAnalyticsEngine, model_params: Dict):
        self.dm = data_manager
        self.engine = engine
        self.params = model_params
        logger.info("StrategicAdvisor initialized.")

    def calculate_projected_response_time(self, zone: str, ambulances: List[Dict]) -> float:
        """Calculates projected response time for a zone."""
        try:
            if zone not in self.dm.zones_gdf.index:
                logger.warning("Invalid zone: %s", zone)
                return DEFAULT_RESPONSE_TIME
            
            zone_node = self.dm.zones_gdf.loc[zone, 'nearest_node']
            if not zone_node or pd.isna(zone_node):
                logger.warning("No nearest node for zone: %s", zone)
                return DEFAULT_RESPONSE_TIME

            min_time = float('inf')
            for amb in ambulances:
                if amb.get('status') != 'Disponible' or not amb.get('nearest_node') or pd.isna(amb.get('nearest_node')):
                    continue
                try:
                    path_length = nx.shortest_path_length(
                        self.dm.road_graph,
                        source=amb['nearest_node'],
                        target=zone_node,
                        weight='weight'
                    )
                    min_time = min(min_time, path_length + self.params['response_time_turnout_penalty'])
                except nx.NetworkXNoPath:
                    continue
            return min_time if min_time != float('inf') else DEFAULT_RESPONSE_TIME
        except Exception as e:
            logger.error("Failed to calculate response time for zone %s: %s", zone, e)
            return DEFAULT_RESPONSE_TIME

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        """Recommends ambulance reallocations using multi-agent optimization."""
        try:
            available = [
                {'id': i, **d} for i, d in self.dm.ambulances.items()
                if d.get('status') == 'Disponible' and d.get('nearest_node')
            ]
            if not available:
                logger.info("No available ambulances for reallocation.")
                return []

            perf = {
                z: {
                    'risk': risk_scores.get(d['node'], 0),
                    'rt': self.calculate_projected_response_time(z, available)
                }
                for z, d in self.dm.zones_gdf.iterrows()
            }
            deficits = {z: p['risk'] * p['rt'] for z, p in perf.items()}
            if not deficits or max(deficits.values()) < self.params['recommendation_deficit_threshold']:
                logger.info("No reallocation needed: deficits below threshold.")
                return []

            target_zone = max(deficits, key=deficits.get)
            original_rt = perf[target_zone]['rt']
            target_node = self.dm.zones_gdf.loc[target_zone, 'nearest_node']

            best, max_utility = None, -float('inf')
            for amb in available:
                moved_ambulances = [
                    {**a, 'nearest_node': target_node} if a['id'] == amb['id'] else a
                    for a in available
                ]
                new_rt = self.calculate_projected_response_time(target_zone, moved_ambulances)
                utility = (original_rt - new_rt) * perf[target_zone]['risk']
                if utility > max_utility:
                    max_utility, best = utility, (amb['id'], self.dm.node_to_zone_map.get(amb['nearest_node'], 'Unknown'), new_rt)

            if best and max_utility > self.params['recommendation_improvement_threshold']:
                amb_id, from_zone, new_rt = best
                reason = f"Reducir el tiempo de respuesta proyectado en '{target_zone}' de ~{original_rt:.0f} min a ~{new_rt:.0f} min."
                logger.info("Reallocation recommended: %s from %s to %s.", amb_id, from_zone, target_zone)
                return [{"unit": amb_id, "from": from_zone, "to": target_zone, "reason": reason}]
            
            logger.info("No reallocation needed: utility below threshold.")
            return []
        except Exception as e:
            logger.error("Failed to recommend reallocations: %s", e)
            return []


```python
class VisualizationSuite:
    """Handles visualizations, including trends and chaos maps."""
    def __init__(self, style_config: Dict):
        self.config = style_config
        logger.info("VisualizationSuite initialized.")

    def plot_risk_comparison(self, prior_df, posterior_df):
        """Plots prior vs. posterior risk comparison."""
        if prior_df.empty or posterior_df.empty:
            logger.warning("Empty risk data for comparison plot.")
            return alt.Chart().mark_text().encode(text=alt.value("No data"))
        prior_df['type'], posterior_df['type'] = 'A Priori (Histórico)', 'A Posteriori (Actual + Difusión)'
        return alt.Chart(pd.concat([prior_df, posterior_df])).mark_bar(opacity=0.8).encode(
            x=alt.X('risk:Q', title='Nivel de Riesgo'),
            y=alt.Y('zone:N', title='Zona', sort='-x'),
            color=alt.Color('type:N', title='Tipo de Riesgo', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=['zone', alt.Tooltip('risk', format='.3f')]
        ).properties(title="Análisis de Riesgo Bayesiano").interactive()

    def plot_distribution_comparison(self, hist_df, current_df):
        """Plots historical vs. current distribution comparison."""
        if hist_df.empty or current_df.empty:
            logger.warning("Empty distribution data for comparison plot.")
            return alt.Chart().mark_text().encode(text=alt.value("No data"))
        hist_df['type'], current_df['type'] = 'Distribución Histórica', 'Distribución Actual'
        bars = alt.Chart(pd.concat([hist_df, current_df])).mark_bar().encode(
            x=alt.X('percentage:Q', title='% de Incidentes', axis=alt.Axis(format='%')),
            y=alt.Y('zone:N', title='Zona', sort=alt.EncodingSortField(field="percentage", op="sum", order='descending')),
            color=alt.Color('type:N', title='Distribución', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=['zone', alt.Tooltip('percentage', title='Porcentaje', format='.1%')]
        )
        return alt.layer(bars).facet(
            row=alt.Row('type:N', title="", header=alt.Header(labelAngle=0, labelAlign='left', labelFontSize=14))
        ).properties(title="Análisis de Anomalía de Distribución").resolve_scale(y='independent')

    def plot_risk_forecast(self, df):
        """Plots forecasted risk trends."""
        if df.empty:
            logger.warning("Empty forecast data for risk plot.")
            return alt.Chart().mark_text().encode(text=alt.value("No data"))
        return alt.Chart(df).mark_line(color=self.config['colors']['primary'], point=True).encode(
            x=alt.X('hour:Q', title='Horas a Futuro'),
            y=alt.Y('projected_risk:Q', title='Riesgo Proyectado', scale=alt.Scale(zero=False)),
            tooltip=['hour', alt.Tooltip('projected_risk', format='.3f')]
        ).properties(title="Pronóstico de Riesgo por Zona").interactive()

    def plot_incident_trends(self, incidents_df):
        """Plots temporal trends in incident counts by type and zone."""
        if incidents_df.empty or 'zone' not in incidents_df.columns or incidents_df['zone'].isnull().all():
            logger.warning("No valid incident data for trends.")
            return alt.Chart().mark_text().encode(text=alt.value("No data"))
        counts = incidents_df.groupby(['type', 'zone']).size().reset_index(name='count')
        return alt.Chart(counts).mark_bar().encode(
            x=alt.X('type:N', title='Tipo de Incidente'),
            y=alt.Y('count:Q', title='Número de Incidentes'),
            color=alt.Color('zone:N', title='Zona'),
            tooltip=['type', 'zone', 'count']
        ).properties(title="Tendencias de Incidentes por Tipo y Zona").interactive()

    def plot_event_probability(self, prob_df):
        """Plots incident probability over time by zone and horizon."""
        if prob_df.empty:
            logger.warning("Empty probability data for event plot.")
            return alt.Chart().mark_text().encode(text=alt.value("No data"))
        return alt.Chart(prob_df).mark_line(point=True).encode(
            x=alt.X('horizon:Q', title='Horizonte (Horas)'),
            y=alt.Y('probability:Q', title='Probabilidad de Incidente', axis=alt.Axis(format='%')),
            color=alt.Color('zone:N', title='Zona'),
            tooltip=['zone', 'horizon', alt.Tooltip('probability', format='.2%')]
        ).properties(title="Probabilidad de Incidentes por Zona y Horizonte").interactive()

    def plot_entropy_heatmap(self, entropy_dict: Dict, zones_gdf: gpd.GeoDataFrame):
        """Plots entropy as a heatmap over zones."""
        data_gdf = zones_gdf.copy()
        data_gdf['entropy'] = data_gdf.index.map(entropy_dict).fillna(0)
        max_entropy = max(0.01, data_gdf['entropy'].max())
        data_gdf['fill_color'] = data_gdf['entropy'].apply(
            lambda e: [100, 200, 100, int(200 * (e / max_entropy))]
        ).tolist()
        return pdk.Layer(
            "PolygonLayer",
            data=data_gdf,
            get_polygon="geometry.exterior.coords",
            filled=True,
            stroked=False,
            get_fill_color="fill_color",
            opacity=0.2,
            pickable=True
        )

    def plot_chaos_regime_map(self, prob_df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame):
        """Plots chaos regime map based on fractal dimensions."""
        data_gdf = zones_gdf.copy()
        if not prob_df.empty:
            fd_series = prob_df.groupby('zone')['fractal_dimension'].mean()
            data_gdf['fractal_dimension'] = data_gdf.index.to_series().map(fd_series).fillna(1.0)
        else:
            data_gdf['fractal_dimension'] = 1.0

        data_gdf['fill_color'] = data_gdf['fractal_dimension'].apply(
            lambda fd: self.config['colors']['chaos_high'] if fd > 1.5 else self.config['colors']['chaos_low']
        ).tolist()
        return pdk.Layer(
            "PolygonLayer",
            data=data_gdf,
            get_polygon="geometry.exterior.coords",
            filled=True,
            stroked=False,
            get_fill_color="fill_color",
            opacity=0.3,
            pickable=True
        )

def prepare_visualization_data(data_manager: DataManager, risk_scores: Dict, all_incidents: List, style: Dict):
    """Prepares data for map and charts."""
    try:
        hosp_df = pd.DataFrame([
            {
                "name": f"H: {n}",
                "tooltip_text": f"Cap: {d['capacity']} Carga: {d['load']}",
                "lon": d['location'].x,
                "lat": d['location'].y,
                "icon_data": {"url": style['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}
            } for n, d in data_manager.hospitals.items()
        ])
        amb_df = pd.DataFrame([
            {
                "name": f"U: {n}",
                "tooltip_text": f"Estado: {d['status']}<br>Base: {d['home_base']}",
                "lon": d['location'].x,
                "lat": d['location'].y,
                "icon_data": {"url": style['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128},
                "size": style['sizes']['ambulance']
            } for n, d in data_manager.ambulances.items()
        ])
        inc_df = pd.DataFrame([
            {
                "name": f"I: {i.get('id', 'N/A')}",
                "tooltip_text": f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}",
                "lon": i['location'].x,
                "lat": i['location'].y,
                "color": style['colors']['hawkes_echo'] if i.get('is_echo') else style['colors']['accent_crit'],
                "radius": style['sizes']['hawkes_echo'] if i.get('is_echo') else style['sizes']['incident_base']
            } for i in all_incidents if 'location' in i and isinstance(i['location'], Point)
        ])
        heat_df = pd.DataFrame([
            {"lon": i['location'].x, "lat": i['location'].y}
            for i in all_incidents if not i.get('is_echo') and 'location' in i and isinstance(i['location'], Point)
        ])
        zones_gdf = data_manager.zones_gdf.copy()
        zones_gdf['risk'] = zones_gdf['node'].map(risk_scores).fillna(0)
        zones_gdf['tooltip_text'] = zones_gdf.apply(lambda r: f"Zona: {r.name}<br/>Riesgo: {r.risk:.3f}", axis=1)
        max_risk = max(0.01, zones_gdf['risk'].max())
        zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * (r / max_risk))]).tolist()
        logger.info("Prepared visualization data.")
        return zones_gdf, hosp_df, amb_df, inc_df, heat_df
    except Exception as e:
        logger.error("Failed to prepare visualization data: %s", e)
        return data_manager.zones_gdf.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data
def create_deck_gl_map(zones_gdf: gpd.GeoDataFrame, hospital_df, ambulance_df, incident_df, heatmap_df, entropy_layer, chaos_layer, app_config: Dict):
    """Creates a Deck.gl map with entropy and chaos layers."""
    try:
        style = app_config['styling']
        layers = [
            pdk.Layer(
                "PolygonLayer",
                data=zones_gdf,
                get_polygon="geometry.exterior.coords",
                filled=True,
                stroked=False,
                extruded=True,
                get_elevation=f"risk * {style['map_elevation_multiplier']}",
                get_fill_color="fill_color",
                opacity=0.1,
                pickable=True
            ),
            entropy_layer,
            chaos_layer,
            pdk.Layer(
                "IconLayer",
                data=hospital_df,
                get_icon="icon_data",
                get_position='[lon, lat]',
                get_size=style['sizes']['hospital'],
                size_scale=15,
                pickable=True
            ),
            pdk.Layer(
                "IconLayer",
                data=ambulance_df,
                get_icon="icon_data",
                get_position='[lon, lat]',
                get_size='size',
                size_scale=15,
                pickable=True
            )
        ]
        if not heatmap_df.empty:
            layers.insert(0, pdk.Layer(
                "HeatmapLayer",
                data=heatmap_df,
                get_position='[lon, lat]',
                opacity=0.3,
                aggregation='MEAN',
            threshold=0.1,
                get_weight=1
            ))
        if not incident_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=incident_df,
                get_position='[lon, lat]',
                get_radius='radius',
                get_fill_color='color',
                radius_scale=1,
                pickable=True,
                radius_min_pixels=2,
                radius_max_pixels=100
            ))
        view_state = pdk.ViewState(latitude=32.5, longitude=-117.02, zoom=11, bearing=0, pitch=50)
        tooltip = {
            "html": "<b>{name}</b><br/>{tooltip_text}",
            "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "borderRadius": "5px", "padding": "5px"}
        }
        logger.info("Deck.gl map created with entropy and chaos layers.")
        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider="mapbox" if app_config['mapbox_api_key'] else "carto",
            map_style=app_config['map_style'],
            api_keys={'mapbox': app_config['mapbox_api_key']},
            tooltip=tooltip
        )
    except Exception as e:
        logger.error("Failed to create Deck.gl map: %s", e)
        return pdk.Deck(layers=[])

# --- L4: APPLICATION UI & EXECUTION ---

@st.cache_resource
def initialize_app_components():
    """Initializes and caches main application components."""
    try:
        warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
        warnings.filterwarnings('ignore', category=FutureWarning, module='pydeck')
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
    except Exception as e:
        logger.error("Failed to initialize app components: %s", e, exc_info=True)
        st.error(f"Fatal error during initialization: {e}")
        st.stop()

def render_intel_briefing(anomaly, entropy, mutual_info, chi2, p_value, recommendations, app_config):
    """Renders the intelligence briefing section."""
    st.subheader("Intel Briefing y Recomendaciones")
    if anomaly > 0.2:
        status = "ANÓMALO"
    elif anomaly > 0.1:
        status = "ELEVADO"
    else:
        status = "NOMINAL"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado del Sistema", status)
    c2.metric("Puntuación de Anomalía (KL)", f"{anomaly:.4f}")
    c3.metric("Información Mutua", f"{mutual_info:.4f}")
    c4.metric("Chi-Square p-valor", f"{p_value:.4f}")
    c1.metric("Entropía Espacial (Desorden)", f"{entropy:.4f} bits")

    if recommendations:
        st.warning("Recomendación de Despliegue de Recursos:")
        for r in recommendations:
            st.write(f"**Mover {r['unit']}** de `{r['from']}` a `{r['to']}`. **Razón:** {r['reason']}")
    else:
        st.success("No se requieren reasignaciones de recursos.")
    logger.info("Rendered intel briefing.")

def _render_scenario_dashboard(dm, engine, predictor, advisor, plotter, config, factors, current_hour):
    """Helper function to render the main dashboard for a given scenario."""
    live_state = engine.get_live_state(factors, current_hour)
    dm.update_priors(live_state)
    predictor.forward_predictor.update_bayesian_priors(live_state)
    prior_risk, risk = predictor.calculate_holistic_risk(live_state)
    anomaly, entropy, _, current_dist, mutual_info, chi2, p_value = predictor.calculate_information_metrics(live_state)
    recs = advisor.recommend_resource_reallocations(risk)

    render_intel_briefing(anomaly, entropy, mutual_info, chi2, p_value, recs, config)
    st.divider()
    st.subheader("Mapa de Operaciones")

    with st.spinner("Preparando visualización y cálculos predictivos..."):
        prob_df = pd.concat([
            predictor.forward_predictor.compute_event_probability(live_state, risk, h)
            for h in FORECAST_HORIZONS
        ], ignore_index=True)
        
        entropy_dict = {z: -p * np.log2(p + 1e-9) if p > 0 else 0 for z, p in current_dist.items()}
        entropy_layer = plotter.plot_entropy_heatmap(entropy_dict, dm.zones_gdf)
        chaos_layer = plotter.plot_chaos_regime_map(prob_df, dm.zones_gdf)
        vis_data = prepare_visualization_data(dm, risk, live_state["active_incidents"], config['styling'])
        st.pydeck_chart(create_deck_gl_map(*vis_data, entropy_layer, chaos_layer, config), use_container_width=True)

def render_sandbox_tab(dm, engine, predictor, advisor, sensitivity_analyzer, plotter, config):
    """Renders the interactive sandbox tab with sensitivity analysis."""
    st.header("Command Sandbox: Simulador Interactivo")
    c1, c2, c3 = st.columns(3)
    is_holiday = c1.checkbox("Día Festivo")
    is_payday = c2.checkbox("Quincena")
    weather = c3.selectbox("Clima", ["Despejado", "Lluvia", "Niebla"])
    c1, c2 = st.columns(2)
    base_rate = c1.slider("μ (Tasa Base)", 1, 20, 5, key="sb_br")
    excitation = c2.slider("κ (Excitación)", 0.0, 1.0, 0.5, key="sb_ex")

    factors = EnvFactors(is_holiday, is_payday, weather, False, 1.0, base_rate, excitation)
    current_hour = float(st.session_state.get('current_hour', 0.0))
    
    _render_scenario_dashboard(dm, engine, predictor, advisor, plotter, config, factors, current_hour)

    st.subheader("Análisis de Sensibilidad")
    if st.button("Ejecutar Análisis de Sensibilidad"):
        with st.spinner("Calculando sensibilidad del modelo..."):
            params_to_test = {
                'traffic_multiplier': [0.8, 1.0, 1.2],
                'base_rate': [3, 5, 10],
                'self_excitation_factor': [0.3, 0.5, 0.7]
            }
            sensitivity_df = sensitivity_analyzer.analyze_sensitivity(factors, params_to_test)
            if not sensitivity_df.empty:
                st.altair_chart(
                    alt.Chart(sensitivity_df).mark_point(filled=True, size=100).encode(
                        x=alt.X('value:Q', title='Valor del Parámetro', scale=alt.Scale(zero=False)),
                        y=alt.Y('risk_diff:Q', title='Diferencia en Riesgo'),
                        color=alt.Color('parameter:N', title='Parámetro'),
                        size=alt.Size('anomaly_diff:Q', title='Diferencia en Anomalía'),
                        tooltip=['parameter', 'value', 'risk_diff', 'anomaly_diff']
                    ).properties(title="Análisis de Sensibilidad").interactive(),
                    use_container_width=True
                )
            else:
                st.error("No se pudo completar el análisis de sensibilidad.")
    logger.info("Rendered sandbox tab.")

def render_scenario_planner_tab(dm, engine, predictor, advisor, plotter, config):
    """Renders the scenario planner tab."""
    st.header("Planificador de Escenarios")
    scenarios = {
        "Día Normal": EnvFactors(False, False, 'Despejado', False, 1.0, 5, 0.3),
        "Colapso Fronterizo": EnvFactors(False, True, 'Despejado', True, 3.0, 8, 0.6),
        "Evento Masivo con Lluvia": EnvFactors(False, False, 'Lluvia', True, 1.8, 12, 0.7)
    }
    name = st.selectbox("Seleccione un Escenario:", list(scenarios.keys()))
    current_hour = float(st.session_state.get('current_hour', 0.0))
    st.subheader(f"Simulación para: {name}")
    _render_scenario_dashboard(dm, engine, predictor, advisor, plotter, config, scenarios[name], current_hour)
    logger.info("Rendered scenario planner tab.")

def render_analysis_tab(dm, engine, predictor, plotter):
    """Renders the analysis tab with trends and probability visualizations."""
    st.header("Análisis Profundo del Sistema")
    if st.button("🔄 Generar Nuevo Estado de Muestra"):
        factors = EnvFactors(
            False, False, 'Despejado', False,
            np.random.uniform(0.8, 2.0),
            np.random.randint(3, 15),
            np.random.uniform(0.2, 0.8)
        )
        current_hour = float(st.session_state.get('current_hour', 0.0))
        st.session_state.analysis_state = engine.get_live_state(factors, current_hour)
    
    if 'analysis_state' not in st.session_state:
        st.info("Genere un 'Estado de Muestra' para ver el análisis.")
        return

    live_state = st.session_state.analysis_state
    prior, posterior = predictor.calculate_holistic_risk(live_state)
    anomaly, entropy, hist, current, mutual_info, chi2, p_value = predictor.calculate_information_metrics(live_state)

    prior_df = pd.DataFrame(list(prior.items()), columns=['zone', 'risk'])
    posterior_df = pd.DataFrame([
        {'zone': dm.node_to_zone_map.get(n, '?'), 'risk': r}
        for n, r in posterior.items() if dm.node_to_zone_map.get(n)
    ])
    st.altair_chart(plotter.plot_risk_comparison(prior_df, posterior_df), use_container_width=True)

    hist_df = pd.DataFrame(list(hist.items()), columns=['zone', 'percentage'])
    current_df = pd.DataFrame(list(current.items()), columns=['zone', 'percentage'])
    st.altair_chart(plotter.plot_distribution_comparison(hist_df, current_df), use_container_width=True)

    st.subheader("Tendencias de Incidentes")
    incidents_df = pd.DataFrame(live_state.get("active_incidents", []))
    if not incidents_df.empty and 'zone' in incidents_df.columns and not incidents_df['zone'].isnull().all():
        st.altair_chart(plotter.plot_incident_trends(incidents_df.dropna(subset=['zone', 'type'])), use_container_width=True)
    else:
        st.info("No hay incidentes para visualizar tendencias.")

    logger.info("Rendered analysis tab.")

def render_forecasting_tab(predictor, plotter):
    """Renders the forecasting tab with multi-resolution predictions."""
    st.header("Pronóstico de Riesgo Futuro")
    if 'analysis_state' not in st.session_state:
        st.warning("Genere un 'Estado de Muestra' en la pestaña de 'Análisis' para poder realizar un pronóstico.")
        return

    live = st.session_state.analysis_state
    _, risk = predictor.calculate_holistic_risk(live)
    anomaly, _, _, _, _, _, _ = predictor.calculate_information_metrics(live)
    c1, c2 = st.columns(2)
    zone = c1.selectbox("Zona:", options=list(predictor.dm.zones_gdf.index))
    horizon = c2.select_slider("Horizonte:", options=FORECAST_HORIZONS, value=24)

    df = predictor.forecast_risk_over_time(risk, anomaly, horizon)
    prob_df = pd.concat([
        predictor.forward_predictor.compute_event_probability(live, risk, h)
        for h in FORECAST_HORIZONS
    ])

    if not df.empty:
        st.altair_chart(plotter.plot_risk_forecast(df[df['zone'] == zone]), use_container_width=True)
    else:
        st.error("Error generando el pronóstico de riesgo.")

    if not prob_df.empty:
        st.altair_chart(plotter.plot_event_probability(prob_df[prob_df['zone'] == zone]), use_container_width=True)
    else:
        st.error("Error generando el pronóstico de probabilidad.")
    logger.info("Rendered forecasting tab.")

def render_knowledge_center():
    """Renders the knowledge center with documentation and insights."""
    try:
        st.header("Knowledge Center")
        st.markdown("""
        ### Documentación del Sistema RedShield AI
        **RedShield AI** es un sistema de gestión de servicios médicos de emergencia que utiliza un gemelo digital para modelar, simular y optimizar operaciones en tiempo real. A continuación, se describen los componentes principales y su funcionalidad:

        #### Componentes Principales
        - **DataManager**: Gestiona datos geoespaciales, incluyendo zonas, hospitales, ambulancias y redes viales. Utiliza GeoPandas para operaciones espaciales y NetworkX para modelado de grafos.
        - **SimulationEngine**: Genera datos sintéticos de incidentes utilizando un Proceso de Poisson No Homogéneo (NHPP) con factores ambientales (días festivos, clima, etc.).
        - **PredictiveAnalyticsEngine**: Combina modelos de machine learning (GradientBoosting, GaussianProcess) y redes bayesianas para pronosticar riesgos y calcular métricas de información (KL-divergencia, entropía, información mutua, chi-square).
        - **ForwardPredictiveModule**: Calcula probabilidades de eventos futuros utilizando una red bayesiana y un modelo TCN (no entrenado en esta versión).
        - **StrategicAdvisor**: Optimiza la reasignación de recursos (ambulancias) utilizando teoría de juegos y tiempos de respuesta proyectados.
        - **VisualizationSuite**: Genera visualizaciones interactivas, incluyendo mapas de calor, gráficos de tendencias y mapas de caos basados en dimensiones fractales (heurísticas).
        - **SensitivityAnalyzer**: Analiza la sensibilidad del sistema a cambios en parámetros clave (tráfico, tasa base, excitación).

        #### Métricas Clave
        - **KL-Divergence**: Mide la discrepancia entre la distribución histórica y actual de incidentes.
        - **Entropía Espacial**: Cuantifica el desorden en la distribución de incidentes.
        - **Información Mutua**: Evalúa la dependencia entre zonas y tipos de incidentes.
        - **Chi-Square p-valor**: Evalúa la significancia estadística de las diferencias entre distribuciones observadas y esperadas.
        - **Dimensión Fractal**: Heurística para evaluar la complejidad espacial de los incidentes (placeholder).

        #### Uso del Sistema
        - **Command Sandbox**: Permite simular escenarios personalizados ajustando parámetros como clima, tasa base y excitación.
        - **Planificador de Escenarios**: Ejecuta simulaciones predefinidas (e.g., Día Normal, Colapso Fronterizo).
        - **Análisis Profundo**: Visualiza tendencias de incidentes y compara riesgos a priori y a posteriori.
        - **Pronóstico de Riesgo**: Proyecta riesgos futuros para horizontes seleccionados (3h, 6h, 12h, 24h, 3d, 7d).

        #### Notas Técnicas
        - **Requisitos**: Ver `requirements.txt` para dependencias (Streamlit, GeoPandas, PyMC, PyTorch, etc.).
        - **Limitaciones Actuales**: El modelo TCN no está entrenado, y la dimensión fractal es una heurística. Se recomienda integrar datos reales y entrenar modelos para producción.
        - **Configuración**: Requiere una clave de Mapbox para mapas avanzados; de lo contrario, usa el estilo Carto por defecto.

        Para más información, contacte al equipo de desarrollo o consulte la documentación técnica completa.
        """)
        logger.info("Rendered knowledge center.")
    except Exception as e:
        logger.error("Failed to render knowledge center: %s", e)
        st.error("Error al renderizar el centro de conocimiento.")

def main():
    """Main application entry point."""
    try:
        st.set_page_config(page_title="RedShield AI: Command Suite", layout="wide")
        st.title("RedShield AI: Command Suite")
        st.markdown("**Gemelo Digital para la Gestión de Servicios Médicos de Emergencia**")
        
        if 'current_hour' not in st.session_state:
            st.session_state.current_hour = 0.0
        
        # Normalize current_hour to a 24-hour cycle
        st.session_state.current_hour = (st.session_state.current_hour + 0.1) % 24

        data_manager, engine, predictor, advisor, sensitivity_analyzer, plotter, app_config = initialize_app_components()
        
        tabs = st.tabs(["Sandbox", "Planificador de Escenarios", "Análisis", "Pronósticos", "Knowledge Center"])
        
        with tabs[0]:
            render_sandbox_tab(data_manager, engine, predictor, advisor, sensitivity_analyzer, plotter, app_config)
        with tabs[1]:
            render_scenario_planner_tab(data_manager, engine, predictor, advisor, plotter, app_config)
        with tabs[2]:
            render_analysis_tab(data_manager, engine, predictor, plotter)
        with tabs[3]:
            render_forecasting_tab(predictor, plotter)
        with tabs[4]:
            render_knowledge_center()

        logger.info("Main application executed successfully.")
    except Exception as e:
        logger.error("Main application failed: %s", e, exc_info=True)
        st.error(f"Error fatal en la ejecución de la aplicación: {e}")

if __name__ == "__main__":
    main()
