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
    handlers=[logging.StreamHandler(), logging.FileHandler("redshield_ai.log")]
)
logger = logging.getLogger(__name__)

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
    """Returns validated application configuration."""
    mapbox_key = os.environ.get("MAPBOX_API_KEY", st.secrets.get("MAPBOX_API_KEY", ""))
    map_style = "mapbox://styles/mapbox/dark-v10" if mapbox_key else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    if not mapbox_key:
        logger.warning("Mapbox API key not found. Using default Carto map style.")

    config = {
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
                'text': '#01FFFF',
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
    
    required_sections = ['data', 'model_params', 'simulation_params', 'styling']
    for section in required_sections:
        if section not in config or not config[section]:
            raise ValueError(f"Configuration section '{section}' is missing or empty.")
    return config

# --- L1: CORE UTILITIES ---
def _normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a probability distribution with edge case handling."""
    if not dist:
        return {}
    total = sum(v for v in dist.values() if isinstance(v, (int, float)) and v >= 0)
    if total <= 0:
        logger.warning("Invalid or zero-sum distribution encountered.")
        return {k: 0.0 for k in dist}
    return {k: v / total for k, v in dist.items() if isinstance(v, (int, float)) and v >= 0}

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
        self.city_boundary_bounds = self.city_boundary_poly.bounds if self.city_boundary_poly else (0, 0, 0, 0)
        self.node_to_zone_map = {data['node']: name for name, data in self.zones_gdf.iterrows() if 'node' in data and pd.notna(data['node'])}
        self.prior_history = self._initialize_prior_history()
        logger.info("DataManager initialized with %d zones, %d hospitals, %d ambulances.", len(self.zones_gdf), len(self.hospitals), len(self.ambulances))

    @st.cache_resource
    def _build_road_graph(_self) -> nx.Graph:
        """Builds road graph with error handling."""
        G = nx.Graph()
        network_config = _self.config.get('road_network', {})
        nodes = network_config.get('nodes', {})
        edges = network_config.get('edges', [])
        try:
            for node, data in nodes.items():
                if 'pos' in data and len(data['pos']) == 2:
                    G.add_node(node, pos=data['pos'])
            for edge in edges:
                if len(edge) == 3 and isinstance(edge[0], str) and isinstance(edge[1], str) and isinstance(edge[2], (int, float)) and edge[0] in G and edge[1] in G:
                    G.add_edge(edge[0], edge[1], weight=float(edge[2]))
                else:
                    logger.warning("Invalid edge format or nodes not found: %s", edge)
            logger.info("Road graph built with %d nodes and %d edges.", G.number_of_nodes(), G.number_of_edges())
        except Exception as e:
            logger.error("Failed to build road graph: %s", e)
            raise
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
        if not _self.road_graph.nodes:
            logger.warning("Road graph is empty. Returning empty embeddings.")
            return {}
        try:
            node2vec = Node2Vec(_self.road_graph, dimensions=8, walk_length=5, num_walks=20, workers=2, quiet=True)
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            embeddings = {node: model.wv[node] for node in _self.road_graph.nodes()}
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info("Computed and cached graph embeddings for %d nodes.", len(embeddings))
        except Exception as e:
            logger.error("Failed to compute graph embeddings: %s", e)
            raise
        return embeddings

    @st.cache_resource
    def _build_zones_gdf(_self) -> gpd.GeoDataFrame:
        """Builds zones GeoDataFrame with correct coordinate order and validation."""
        zones = _self.config.get('zones', {})
        if not zones:
            logger.error("No zones defined in config.")
            return gpd.GeoDataFrame()
        try:
            df = pd.DataFrame.from_dict(zones, orient='index')
            geometry = [Polygon([(lon, lat) for lat, lon in p]) if isinstance(p, list) and len(p) >= 3 else None for p in df['polygon']]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=GEOGRAPHIC_CRS).dropna(subset=['geometry'])
            gdf_projected = gdf.to_crs(PROJECTED_CRS)
            gdf['centroid'] = gdf_projected.geometry.centroid.to_crs(GEOGRAPHIC_CRS)
            graph_nodes_gdf = gpd.GeoDataFrame(
                geometry=[Point(d['pos'][1], d['pos'][0]) for _, d in _self.road_graph.nodes(data=True)],
                index=list(_self.road_graph.nodes()), crs=GEOGRAPHIC_CRS
            ).to_crs(PROJECTED_CRS)
            nearest = gpd.sjoin_nearest(gdf_projected[['geometry']], graph_nodes_gdf[['geometry']], how='left')
            gdf['nearest_node'] = nearest['index_right']
            logger.info("Zones GeoDataFrame built with %d valid zones.", len(gdf))
            return gdf.drop(columns=['polygon'])
        except Exception as e:
            logger.error("Failed to build zones GeoDataFrame: %s", e)
            raise

    def _initialize_hospitals(self) -> Dict:
        """Initializes hospitals with Point geometries."""
        hospitals = {}
        try:
            for name, data in self.config.get('hospitals', {}).items():
                if 'location' in data and len(data['location']) == 2:
                    hospitals[name] = {**data, 'location': Point(data['location'][1], data['location'][0])}
        except Exception as e:
            logger.error("Failed to initialize hospitals: %s", e)
            raise
        return hospitals

    def _initialize_ambulances(self) -> Dict:
        """Initializes ambulances with location and node assignments."""
        ambulances = {}
        try:
            for amb_id, amb_data in self.config.get('ambulances', {}).items():
                home_zone = amb_data.get('home_base')
                if home_zone in self.zones_gdf.index:
                    zone_info = self.zones_gdf.loc[home_zone]
                    if zone_info.geometry.is_valid and not zone_info.centroid.is_empty:
                        ambulances[amb_id] = {
                            **amb_data,
                            'id': amb_id,
                            'location': zone_info.centroid,
                            'nearest_node': zone_info.nearest_node
                        }
            logger.info("Initialized %d ambulances.", len(ambulances))
        except Exception as e:
            logger.error("Failed to initialize ambulances: %s", e)
            raise
        return ambulances

    def _initialize_prior_history(self) -> Dict:
        """Initializes historical priors for Bayesian updates."""
        try:
            return {zone: {'mean_risk': data['prior_risk'], 'count': 1, 'variance': 0.1} for zone, data in self.zones_gdf.iterrows()}
        except Exception as e:
            logger.error("Failed to initialize prior history: %s", e)
            raise

    def assign_zones_to_incidents(self, incidents_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assigns zones to incidents using spatial join."""
        if incidents_gdf.empty or self.zones_gdf.empty:
            return incidents_gdf.assign(zone=None)
        try:
            joined = gpd.sjoin(incidents_gdf, self.zones_gdf[['geometry']], how="left", predicate="intersects")
            return incidents_gdf.assign(zone=joined['index_right'])
        except Exception as e:
            logger.error("Failed to assign zones to incidents: %s", e)
            raise

class SimulationEngine:
    """Generates synthetic incident data with real-time simulation capabilities."""
    def __init__(self, data_manager: DataManager, sim_params: Dict, distributions: Dict):
        self.dm = data_manager
        self.sim_params = sim_params
        self.dist = distributions
        self.nhpp_intensity = lambda t: 0.1 + 0.05 * np.sin(t / 24 * 2 * np.pi)

    @st.cache_data(ttl=60)
    def get_live_state(_self, env_factors: EnvFactors, time_hour: float = 0.0) -> Dict[str, Any]:
        """Generates live state with NHPP and environmental factors."""
        try:
            mult = _self.sim_params.get('multipliers', {})
            intensity = _self.nhpp_intensity(time_hour) * float(env_factors.base_rate)
            if env_factors.is_holiday: intensity *= mult.get('holiday', 1.0)
            if env_factors.is_payday: intensity *= mult.get('payday', 1.0)
            if env_factors.weather_condition == 'Rain': intensity *= mult.get('rain', 1.0)
            if env_factors.major_event_active: intensity *= mult.get('major_event', 1.0)
            
            num_incidents = max(0, int(np.random.poisson(intensity)))
            if num_incidents == 0:
                return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}
            
            incidents_gdf = gpd.GeoDataFrame(
                {
                    'type': np.random.choice(list(_self.dist['incident_type'].keys()), num_incidents, p=list(_self.dist['incident_type'].values())),
                    'triage': np.random.choice(list(_self.dist['triage'].keys()), num_incidents, p=list(_self.dist['triage'].values())),
                    'is_echo': False,
                    'timestamp': time_hour
                },
                geometry=gpd.points_from_xy(
                    np.random.uniform(_self.dm.city_boundary_bounds[0], _self.dm.city_boundary_bounds[2], num_incidents),
                    np.random.uniform(_self.dm.city_boundary_bounds[1], _self.dm.city_boundary_bounds[3], num_incidents)
                ),
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
                        echo_data.append({
                            'id': f"ECHO-{idx}-{j}",
                            'type': "Echo",
                            'triage': "Verde",
                            'location': echo_loc,
                            'is_echo': True,
                            'zone': trigger.zone,
                            'timestamp': time_hour
                        })
            
            traffic_conditions = {z: min(1.0, env_factors.traffic_multiplier * np.random.uniform(0.3, 1.0)) for z in _self.dm.zones_gdf.index}
            if not traffic_conditions:
                logger.warning("No traffic conditions generated; zones may be missing.")
            system_state = "Anomalous" if len(incidents_list) > 10 or any(t['triage'] == 'Rojo' for t in incidents_list) else "Elevated" if len(incidents_list) > 5 else "Normal"
            return {"active_incidents": incidents_list + echo_data, "traffic_conditions": traffic_conditions, "system_state": system_state}
        except Exception as e:
            logger.error("Failed to generate live state: %s", e)
            raise

class ForwardPredictiveModule:
    """Computes event probabilities using a composite model."""
    def __init__(self, data_manager: DataManager, model_params: Dict):
        self.dm = data_manager
        self.params = model_params
        self.bayesian_network = self._initialize_bayesian_network()

    def _initialize_bayesian_network(self):
        """Initializes a Bayesian network for risk inference."""
        try:
            with pm.Model() as model:
                zones = list(self.dm.zones_gdf.index)
                risk = pm.Normal('risk', mu=[self.dm.prior_history[z]['mean_risk'] for z in zones], sigma=0.1, shape=len(zones))
                traffic = pm.Normal('traffic', mu=0.5, sigma=0.2, shape=len(zones))
                incidents = pm.Poisson('incidents', mu=1.0, shape=len(zones))
                observed_risk_data = pm.Data('observed_risk_data', np.zeros(len(zones)))
                combined_effect = risk * self.params['risk_weights']['prior'] + traffic * self.params['risk_weights']['traffic'] + incidents * self.params['risk_weights']['incidents']
                pm.Normal('observed_risk', mu=combined_effect, sigma=0.1, observed=observed_risk_data)
            return model
        except Exception as e:
            logger.error("Failed to initialize Bayesian network: %s", e)
            raise

    def compute_event_probability(self, live_state: Dict, risk_scores: Dict, horizon: int) -> pd.DataFrame:
        """Computes event probabilities without untrained TCN."""
        try:
            zones = list(self.dm.zones_gdf.index)
            incidents_df = pd.DataFrame(live_state.get("active_incidents", []))
            counts = incidents_df.groupby('zone').size() if not incidents_df.empty and 'zone' in incidents_df.columns else pd.Series(dtype=int)
            
            data = []
            for zone in zones:
                node = self.dm.zones_gdf.loc[zone, 'node']
                base_risk = risk_scores.get(node, 0.5)
                hawkes_effect = self.params['hawkes_intensity'] * counts.get(zone, 0)
                bayesian_risk = self.dm.prior_history[zone]['mean_risk']
                prob = np.clip((base_risk + hawkes_effect + bayesian_risk) / 3, 0, 1)
                data.append({'zone': zone, 'horizon': horizon, 'probability': prob})
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error("Failed to compute event probability: %s", e)
            raise

class PredictiveAnalyticsEngine:
    """Handles forecasting and analytics with machine learning models."""
    def __init__(self, data_manager: DataManager, model_params: Dict, dist_config: Dict):
        self.dm, self.params, self.dist = data_manager, model_params, dist_config
        self.ml_models = {z: GradientBoostingRegressor(n_estimators=50, random_state=42) for z in self.dm.zones_gdf.index}
        self.gp_models = {z: GaussianProcessRegressor(kernel=C(1.0) * RBF(10), random_state=42) for z in self.dm.zones_gdf.index}
        self.forward_predictor = ForwardPredictiveModule(data_manager, model_params)
        self._train_models()

    def _train_models(self):
        """Trains ML models with synthetic data."""
        try:
            np.random.seed(42)
            hours = np.arange(24)
            for zone in self.dm.zones_gdf.index:
                node = self.dm.zones_gdf.loc[zone, 'node']
                embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
                X = np.hstack([np.array([[h, np.random.uniform(0.3, 1.0), np.random.randint(0, 5)] for h in hours]), np.tile(embedding, (len(hours), 1))])
                y = np.random.uniform(0.1, 1.0, 24) * (1 + 0.5 * np.sin(hours / 24 * 2 * np.pi))
                self.ml_models[zone].fit(X, y)
                self.gp_models[zone].fit(X, y)
        except Exception as e:
            logger.error("Failed to train models: %s", e)
            raise

    def calculate_holistic_risk(self, live_state: Dict) -> Tuple[Dict, Dict]:
        """Computes and diffuses risk scores."""
        try:
            prior_risks = self.dm.zones_gdf['prior_risk'].to_dict()
            df = pd.DataFrame(live_state.get("active_incidents", []))
            counts = df.groupby('zone').size() if not df.empty and 'zone' in df.columns else pd.Series(dtype=int)
            w = self.params['risk_weights']
            evidence_risk = {
                zone: data['prior_risk'] * w['prior'] + live_state.get('traffic_conditions', {}).get(zone, 0.5) * w['traffic'] + 
                      counts.get(zone, 0) * self.params['incident_load_factor'] * w['incidents']
                for zone, data in self.dm.zones_gdf.iterrows()
            }
            node_risks = {data['node']: evidence_risk.get(zone, 0) for zone, data in self.dm.zones_gdf.iterrows() if 'node' in data}
            return prior_risks, self._diffuse_risk_on_graph(node_risks)
        except Exception as e:
            logger.error("Failed to calculate holistic risk: %s", e)
            raise

    def _diffuse_risk_on_graph(self, initial_risks: Dict[str, float]) -> Dict[str, float]:
        """Diffuses risk across the road graph."""
        try:
            graph = self.dm.road_graph
            L = nx.laplacian_matrix(graph).toarray()
            risks = np.array([initial_risks.get(node, 0) for node in graph.nodes()])
            for _ in range(self.params.get('risk_diffusion_steps', 3)):
                risks = risks - self.params.get('risk_diffusion_factor', 0.1) * L @ risks
            return {node: max(0, r) for node, r in zip(graph.nodes(), risks)}
        except Exception as e:
            logger.error("Failed to diffuse risk on graph: %s", e)
            raise

    def calculate_information_metrics(self, live_state: Dict) -> Tuple[float, float, Dict, Dict, float]:
        """Computes information-theoretic metrics."""
        try:
            hist = self.dist['zone']
            df = pd.DataFrame([i for i in live_state.get("active_incidents", []) if not i.get("is_echo")])
            if df.empty or 'zone' not in df.columns:
                return 0.0, 0.0, hist, {z: 0.0 for z in self.dm.zones_gdf.index}, 0.0
            counts = df.groupby('zone').size()
            total = len(df)
            current = {z: counts.get(z, 0) / total for z in self.dm.zones_gdf.index}
            epsilon = 1e-9
            kl_divergence = sum(p * np.log((p + epsilon) / (hist.get(z, 0) + epsilon)) for z, p in current.items() if p > 0)
            shannon_entropy = -sum(p * np.log2(p + epsilon) for p in current.values() if p > 0)
            mutual_info = 0.0
            if 'type' in df.columns:
                joint = df.groupby(['zone', 'type']).size().unstack(fill_value=0) / total
                type_marginal = joint.sum(axis=0)
                zone_marginal = joint.sum(axis=1)
                mutual_info = sum(
                    joint.loc[z, t] * np.log2((joint.loc[z, t] + epsilon) / (zone_marginal[z] * type_marginal[t] + epsilon))
                    for z in joint.index for t in joint.columns if joint.loc[z, t] > 0
                )
            return kl_divergence, shannon_entropy, hist, current, mutual_info
        except Exception as e:
            logger.error("Failed to calculate information metrics: %s", e)
            raise

    def forecast_risk_over_time(self, risk_scores: Dict, anomaly: float, horizon: int) -> pd.DataFrame:
        """Forecasts risk over a time horizon."""
        try:
            data = []
            for zone in self.dm.zones_gdf.index:
                node = self.dm.zones_gdf.loc[zone, 'node']
                embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
                X = np.hstack([np.array([[h, np.random.uniform(0.3, 1.0), np.random.randint(0, 5)] for h in range(horizon)]), 
                               np.tile(embedding, (horizon, 1))])
                ml_pred = self.ml_models[zone].predict(X)
                gp_pred, _ = self.gp_models[zone].predict(X, return_std=True)
                combined_pred = np.clip((0.7 * ml_pred + 0.3 * gp_pred) * (1 + 0.1 * anomaly), 0, 2)
                for h, pred in enumerate(combined_pred):
                    data.append({'zone': zone, 'hour': h, 'projected_risk': pred})
            return pd.DataFrame(data)
        except Exception as e:
            logger.error("Failed to forecast risk over time: %s", e)
            raise

class SensitivityAnalyzer:
    """Performs sensitivity analysis on model parameters."""
    def __init__(self, simulation_engine: SimulationEngine, predictive_engine: PredictiveAnalyticsEngine):
        self.sim_engine = simulation_engine
        self.pred_engine = predictive_engine
        logger.info("SensitivityAnalyzer initialized.")

    def analyze_sensitivity(self, env_factors: EnvFactors, parameters: Dict[str, List[float]], iterations: int = 10) -> pd.DataFrame:
        """Analyzes sensitivity to parameter changes."""
        try:
            results = []
            base_state = self.sim_engine.get_live_state(env_factors)
            base_risk = self.pred_engine.calculate_holistic_risk(base_state)[1]
            base_anomaly = self.pred_engine.calculate_information_metrics(base_state)[0]
            
            for param, values in parameters.items():
                for value in values:
                    for _ in range(iterations):
                        modified_factors = EnvFactors(
                            env_factors.is_holiday, env_factors.is_payday, env_factors.weather_condition,
                            env_factors.major_event_active,
                            value if param == 'traffic_multiplier' else env_factors.traffic_multiplier,
                            int(value) if param == 'base_rate' else env_factors.base_rate,
                            value if param == 'self_excitation_factor' else env_factors.self_excitation_factor
                        )
                        state = self.sim_engine.get_live_state(modified_factors)
                        risk = self.pred_engine.calculate_holistic_risk(state)[1]
                        anomaly = self.pred_engine.calculate_information_metrics(state)[0]
                        risk_diff = np.mean([abs(risk.get(node, 0) - base_risk.get(node, 0)) for node in self.pred_engine.dm.road_graph.nodes()])
                        results.append({
                            'parameter': param,
                            'value': value,
                            'risk_diff': risk_diff,
                            'anomaly_diff': abs(anomaly - base_anomaly)
                        })
            return pd.DataFrame(results)
        except Exception as e:
            logger.error("Failed to analyze sensitivity: %s", e)
            raise

class StrategicAdvisor:
    """Handles resource reallocation."""
    def __init__(self, data_manager: DataManager, engine: PredictiveAnalyticsEngine, model_params: Dict):
        self.dm, self.engine, self.params = data_manager, engine, model_params

    def calculate_projected_response_time(self, zone: str, ambulances: List[Dict]) -> float:
        """Calculates minimum response time to a zone."""
        try:
            if zone not in self.dm.zones_gdf.index or not ambulances:
                return DEFAULT_RESPONSE_TIME
            zone_node = self.dm.zones_gdf.loc[zone, 'nearest_node']
            min_time = float('inf')
            for amb in ambulances:
                if amb.get('status') == 'Disponible' and amb.get('nearest_node'):
                    try:
                        min_time = min(min_time, nx.shortest_path_length(self.dm.road_graph, amb['nearest_node'], zone_node, weight='weight') + 
                                      self.params['response_time_turnout_penalty'])
                    except nx.NetworkXNoPath:
                        continue
            return min_time if min_time != float('inf') else DEFAULT_RESPONSE_TIME
        except Exception as e:
            logger.error("Failed to calculate projected response time: %s", e)
            raise

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        """Recommends ambulance reallocations based on risk and response times."""
        try:
            available = [{'id': amb_id, **d} for amb_id, d in self.dm.ambulances.items() if d.get('status') == 'Disponible']
            if not available:
                return []
            perf = {z: {'risk': risk_scores.get(d['node'], 0), 'rt': self.calculate_projected_response_time(z, available)} 
                    for z, d in self.dm.zones_gdf.iterrows()}
            deficits = {z: p['risk'] * p['rt'] for z, p in perf.items()}
            if not deficits or max(deficits.values()) < self.params['recommendation_deficit_threshold']:
                return []
            target_zone = max(deficits, key=deficits.get)
            original_rt = perf[target_zone]['rt']
            target_node = self.dm.zones_gdf.loc[target_zone, 'nearest_node']
            best = None
            max_utility = -float('inf')
            for amb in available:
                moved_ambulances = [{**a, 'nearest_node': target_node} if a['id'] == amb['id'] else a for a in available]
                new_rt = self.calculate_projected_response_time(target_zone, moved_ambulances)
                utility = (original_rt - new_rt) * perf[target_zone]['risk']
                if utility > max_utility:
                    max_utility, best = utility, (amb['id'], self.dm.node_to_zone_map.get(amb['nearest_node'], 'Unknown'), new_rt)
            if best and max_utility > self.params['recommendation_improvement_threshold']:
                amb_id, from_zone, new_rt = best
                return [{"unit": amb_id, "from": from_zone, "to": target_zone, 
                         "reason": f"Reduce projected response time in '{target_zone}' from ~{original_rt:.0f} min to ~{new_rt:.0f} min."}]
            return []
        except Exception as e:
            logger.error("Failed to recommend resource reallocations: %s", e)
            raise

class VisualizationSuite:
    """Handles visualizations."""
    def __init__(self, style_config: Dict):
        self.config = style_config

    def plot_risk_comparison(self, prior_df, posterior_df):
        """Plots prior vs posterior risk comparison."""
        try:
            if prior_df.empty or posterior_df.empty or 'zone' not in prior_df.columns or 'zone' not in posterior_df.columns:
                return alt.Chart().mark_text().encode(text=alt.value("No data"))
            prior_df['type'] = 'Prior (Historical)'
            posterior_df['type'] = 'Posterior (Current + Diffusion)'
            return alt.Chart(pd.concat([prior_df, posterior_df])).mark_bar(opacity=0.8).encode(
                x=alt.X('risk:Q', title='Risk Level'),
                y=alt.Y('zone:N', title='Zone', sort='-x'),
                color=alt.Color('type:N', title='Risk Type', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
                tooltip=['zone', alt.Tooltip('risk', format='.3f')]
            ).properties(title="Bayesian Risk Analysis").interactive()
        except Exception as e:
            logger.error("Failed to plot risk comparison: %s", e)
            raise

    def plot_distribution_comparison(self, hist_df, current_df):
        """Plots historical vs current incident distribution."""
        try:
            if hist_df.empty or current_df.empty or 'zone' not in hist_df.columns or 'zone' not in current_df.columns:
                return alt.Chart().mark_text().encode(text=alt.value("No data"))
            hist_df['type'], current_df['type'] = 'Historical Distribution', 'Current Distribution'
            return alt.Chart(pd.concat([hist_df, current_df])).mark_bar().encode(
                x=alt.X('percentage:Q', title='% of Incidents', axis=alt.Axis(format='%')),
                y=alt.Y('zone:N', title='Zone', sort=alt.EncodingSortField(field="percentage", op="sum", order='descending')),
                color=alt.Color('type:N', title='Distribution', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
                tooltip=['zone', alt.Tooltip('percentage', title='Percentage', format='.1%')]
            ).facet(row=alt.Row('type:N', title="")).properties(title="Distribution Anomaly Analysis").resolve_scale(y='independent')
        except Exception as e:
            logger.error("Failed to plot distribution comparison: %s", e)
            raise

    def plot_risk_forecast(self, df):
        """Plots risk forecast over time."""
        try:
            if df.empty or 'zone' not in df.columns or 'hour' not in df.columns or 'projected_risk' not in df.columns:
                return alt.Chart().mark_text().encode(text=alt.value("No data"))
            return alt.Chart(df).mark_line(color=self.config['colors']['primary'], point=True).encode(
                x=alt.X('hour:Q', title='Hours Ahead'),
                y=alt.Y('projected_risk:Q', title='Projected Risk', scale=alt.Scale(zero=False)),
                tooltip=['hour', alt.Tooltip('projected_risk', format='.3f')]
            ).properties(title="Risk Forecast by Zone").interactive()
        except Exception as e:
            logger.error("Failed to plot risk forecast: %s", e)
            raise

    def plot_incident_trends(self, incidents_df):
        """Plots incident trends by type and zone."""
        try:
            if incidents_df.empty or 'zone' not in incidents_df.columns or 'type' not in incidents_df.columns:
                return alt.Chart().mark_text().encode(text=alt.value("No data"))
            counts = incidents_df.groupby(['type', 'zone']).size().reset_index(name='count')
            return alt.Chart(counts).mark_bar().encode(
                x=alt.X('type:N', title='Incident Type'),
                y=alt.Y('count:Q', title='Number of Incidents'),
                color=alt.Color('zone:N', title='Zone'),
                tooltip=['type', 'zone', 'count']
            ).properties(title="Incident Trends by Type and Zone").interactive()
        except Exception as e:
            logger.error("Failed to plot incident trends: %s", e)
            raise

    def plot_event_probability(self, prob_df):
        """Plots event probability over horizons."""
        try:
            if prob_df.empty or 'zone' not in prob_df.columns or 'horizon' not in prob_df.columns or 'probability' not in prob_df.columns:
                return alt.Chart().mark_text().encode(text=alt.value("No data"))
            return alt.Chart(prob_df).mark_line(point=True).encode(
                x=alt.X('horizon:Q', title='Horizon (Hours)'),
                y=alt.Y('probability:Q', title='Incident Probability', axis=alt.Axis(format='%')),
                color=alt.Color('zone:N', title='Zone'),
                tooltip=['zone', 'horizon', alt.Tooltip('probability', format='.2%')]
            ).properties(title="Incident Probability by Zone and Horizon").interactive()
        except Exception as e:
            logger.error("Failed to plot event probability: %s", e)
            raise

def prepare_visualization_data(data_manager: DataManager, risk_scores: Dict, all_incidents: List, style: Dict) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepares data for visualizations with type sanitization."""
    try:
        hosp_df = pd.DataFrame([
            {'name': f"H: {n}", 'tooltip_text': f"Cap: {d['capacity']} Load: {d['load']}", 'lon': float(d['location'].x), 'lat': float(d['location'].y), 
             'icon_data': {"url": style['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}}
            for n, d in data_manager.hospitals.items() if d.get('location') and not d['location'].is_empty
        ])
        amb_df = pd.DataFrame([
            {'name': f"U: {d['id']}", 'tooltip_text': f"Status: {d['status']}<br>Base: {d['home_base']}", 'lon': float(d['location'].x), 'lat': float(d['location'].y), 
             'icon_data': {"url": style['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, 'size': float(style['sizes']['ambulance'])}
            for d in data_manager.ambulances.values() if d.get('location') and not d['location'].is_empty
        ])
        inc_df = pd.DataFrame([
            {'name': f"I: {i.get('id', 'N/A')}", 'tooltip_text': f"Type: {i.get('type')}<br>Triage: {i.get('triage')}", 'lon': float(i['location'].x), 'lat': float(i['location'].y), 
             'color': style['colors']['hawkes_echo'] if i.get('is_echo') else style['colors']['accent_crit'], 
             'radius': float(style['sizes']['hawkes_echo']) if i.get('is_echo') else float(style['sizes']['incident_base'])}
            for i in all_incidents if i.get('location') and not i['location'].is_empty
        ])
        heat_df = pd.DataFrame([{"lon": float(i['location'].x), "lat": float(i['location'].y)} for i in all_incidents if i.get('location')])
        zones_gdf = data_manager.zones_gdf.copy()
        zones_gdf['risk'] = zones_gdf['node'].map(risk_scores).fillna(0.0).astype(float)
        zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * (r / max(0.01, zones_gdf['risk'].max())))]).tolist()
        return zones_gdf, hosp_df, amb_df, inc_df, heat_df
    except Exception as e:
        logger.error("Failed to prepare visualization data: %s", e)
        raise

def create_deck_gl_map(zones_gdf: gpd.GeoDataFrame, hospital_df: pd.DataFrame, ambulance_df: pd.DataFrame, incident_df: pd.DataFrame, heatmap_df: pd.DataFrame, app_config: Dict) -> pdk.Deck:
    """Creates a Deck.gl map with sanitized data."""
    try:
        style = app_config['styling']
        layers = [
            pdk.Layer("PolygonLayer", data=zones_gdf, filled=True, stroked=False, extruded=True, 
                      get_elevation=f"risk * {style['map_elevation_multiplier']}", get_fill_color="fill_color", opacity=0.1, pickable=True),
            pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style['sizes']['hospital'], size_scale=15, pickable=True),
            pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', size_scale=15, pickable=True)
        ]
        if not heatmap_df.empty:
            layers.insert(0, pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN', threshold=0.1))
        if not incident_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True))
        view_state = pdk.ViewState(latitude=32.5, longitude=-117.02, zoom=11, bearing=0, pitch=50)
        return pdk.Deck(layers=layers, initial_view_state=view_state, map_provider="mapbox" if app_config['mapbox_api_key'] else "carto", 
                        map_style=app_config['map_style'], api_keys={'mapbox': app_config['mapbox_api_key']})
    except Exception as e:
        logger.error("Failed to create Deck.gl map: %s", e)
        raise

@st.cache_resource
def initialize_app_components():
    """Initializes and caches main application components."""
    try:
        warnings.filterwarnings('ignore', category=UserWarning)
        app_config = get_app_config()
        distributions = {k: _normalize_dist(v) for k, v in app_config['data']['distributions'].items()}
        data_manager = DataManager(app_config)
        engine = SimulationEngine(data_manager, app_config['simulation_params'], distributions)
        predictor = PredictiveAnalyticsEngine(data_manager, app_config['model_params'], distributions)
        advisor = StrategicAdvisor(data_manager, predictor, app_config['model_params'])
        sensitivity_analyzer = SensitivityAnalyzer(engine, predictor)
        plotter = VisualizationSuite(app_config['styling'])
        return data_manager, engine, predictor, advisor, sensitivity_analyzer, plotter, app_config
    except Exception as e:
        logger.error("Failed to initialize app components: %s", e)
        raise

def render_intel_briefing(anomaly: float, entropy: float, mutual_info: float, recommendations: List[Dict], app_config: Dict):
    """Renders the intelligence briefing section."""
    try:
        st.subheader("Intel Briefing and Recommendations")
        status = "ANOMALOUS" if anomaly > 0.2 else "ELEVATED" if anomaly > 0.1 else "NOMINAL"
        c1, c2, c3 = st.columns(3)
        c1.metric("System Status", status)
        c2.metric("Anomaly Score (KL)", f"{anomaly:.4f}")
        c3.metric("Mutual Information", f"{mutual_info:.4f}")
        c1.metric("Spatial Entropy", f"{entropy:.4f} bits")
        if recommendations:
            st.warning("Resource Deployment Recommendation:")
            for r in recommendations:
                st.write(f"**Move {r['unit']}** from `{r['from']}` to `{r['to']}`. **Reason:** {r['reason']}")
        else:
            st.success("No resource reallocations required.")
    except Exception as e:
        logger.error("Failed to render intel briefing: %s", e)
        raise

def render_sandbox_tab(dm: DataManager, engine: SimulationEngine, predictor: PredictiveAnalyticsEngine, advisor: StrategicAdvisor, sensitivity_analyzer: SensitivityAnalyzer, plotter: VisualizationSuite, config: Dict):
    """Renders the interactive sandbox tab."""
    try:
        st.header("Command Sandbox: Interactive Simulator")
        c1, c2, c3 = st.columns(3)
        is_holiday = c1.checkbox("Holiday")
        is_payday = c2.checkbox("Payday")
        weather = c3.selectbox("Weather", ["Clear", "Rain", "Fog"])
        c1, c2 = st.columns(2)
        base_rate = c1.slider("Base Rate (μ)",1", 1, 20, 5)
        excitation = c2.slider("Excitation (κ)", 0.0, 1.0, 0.5)
        factors = EnvFactors(is_holiday, is_payday, weather, False, 1.0, base_rate, excitation)
        current_hour = float(st.session_state.get('current_hour', 0.0))
        live_state = engine.get_live_state(factors, current_hour)
        _, risk = predictor.calculate_holistic_risk(live_state)
        anomaly, entropy, _, current_dist, mutual_info = predictor.calculate_information_metrics(live_state)
        recs = advisor.recommend_resource_reallocations(risk)
        render_intel_briefing(anomaly, entropy, mutual_info, recs, config)
        st.subheader("Operations Map")
        vis_data = prepare_visualization_data(dm, risk, live_state["active_incidents"], config['styling'])
        st.pydeck_chart(create_deck_gl_map(*vis_data, config))
    except Exception as e:
        logger.error("Failed to render sandbox tab: %s", e)
        st.error(f"Error rendering sandbox: {e}")

def main():
    """Main application entry point."""
    st.set_page_config(page_title="RedShield AI v10.7", layout="wide")
    st.title("RedShield AI Command Suite")
    st.markdown("**Digital Twin for Emergency Medical Services Management** | Version 10.7")
    try:
        dm, engine, predictor, advisor, sensitivity_analyzer, plotter, config = initialize_app_components()
        if 'current_hour' not in st.session_state:
            st.session_state.current_hour = 0.0
        st.session_state.current_hour += 0.1
        render_sandbox_tab(dm, engine, predictor, advisor, sensitivity_analyzer, plotter, config)
    except Exception as e:
        logger.error("Application failed: %s", e)
        st.error(f"Critical error: {e}. Check logs for details.")

if __name__ == "__main__":
    main()
