# RedShieldAI_Command_Suite.py
# VERSION 10.20 - REFACTORED & PRODUCTION-READY
"""
RedShieldAI_Command_Suite.py
Digital Twin for Emergency Medical Services Management
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import networkx as nx
import os
from pathlib import Path
import altair as alt
import pydeck as pdk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from node2vec import Node2Vec
import logging
import pickle
import warnings
import json
import random

# --- L0: CONFIGURATION & CONSTANTS ---
PROJECTED_CRS = "EPSG:32611"
GEOGRAPHIC_CRS = "EPSG:4326"
DEFAULT_RESPONSE_TIME = 15.0
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CONFIG_FILE = Path("config.json")
FORECAST_HORIZONS = [3, 6, 12, 24, 72, 168]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("redshield_ai.log")])
logger = logging.getLogger(__name__)

# --- L1: DATA STRUCTURES & CONFIGURATION ---

@dataclass(frozen=True)
class EnvFactors:
    """Dataclass to hold environmental factors for the simulation."""
    is_holiday: bool
    is_payday: bool
    weather_condition: str
    major_event_active: bool
    traffic_multiplier: float
    base_rate: int
    self_excitation_factor: float

@st.cache_data(ttl=3600)  # Cache config for an hour
def get_app_config() -> Dict[str, Any]:
    """Loads, validates, and returns application configuration from a JSON file."""
    if not CONFIG_FILE.exists():
        st.error(f"FATAL: Configuration file not found at '{CONFIG_FILE}'. Please create it.")
        logger.error(f"Configuration file not found at '{CONFIG_FILE}'.")
        st.stop()
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # Validate Mapbox key and set fallback
    mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
    if not mapbox_key:
        logger.warning("Mapbox API key not found. Using default Carto map style.")
        config['map_style'] = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    config['mapbox_api_key'] = mapbox_key

    # Validate required sections
    required_sections = ['data', 'model_params', 'simulation_params', 'styling']
    for section in required_sections:
        if section not in config or not config[section]:
            raise ValueError(f"Configuration section '{section}' is missing or empty in {CONFIG_FILE}.")
    return config

def _normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a dictionary of probabilities to sum to 1."""
    if not isinstance(dist, dict):
        logger.error("Distribution must be a dictionary.")
        return {}
    total = sum(v for v in dist.values() if isinstance(v, (int, float)) and v >= 0)
    if total <= 0:
        logger.warning("Invalid or zero-sum distribution encountered. Returning uniform distribution.")
        num_items = len(dist)
        return {k: 1.0 / num_items for k in dist} if num_items > 0 else {}
    return {k: v / total for k, v in dist.items()}

# --- L2: CORE APPLICATION MODULES ---

class DataManager:
    """Manages all static and geospatial data assets with optimized, cached operations."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('data', {})
        self.road_graph = self._build_road_graph()
        self.graph_embeddings = self._load_or_compute_graph_embeddings()
        self.zones_gdf = self._build_zones_gdf()
        self.hospitals = self._initialize_hospitals()
        self.ambulances = self._initialize_ambulances()
        self.city_boundary_poly = self.zones_gdf.unary_union if not self.zones_gdf.empty else None
        self.node_to_zone_map = {data['node']: name for name, data in self.zones_gdf.iterrows() if 'node' in data and pd.notna(data['node'])}
        logger.info(f"DataManager initialized with {len(self.zones_gdf)} zones, {len(self.hospitals)} hospitals, {len(self.ambulances)} ambulances.")

    # Note on _self: In Streamlit, caching instance methods requires a hashable 'self'.
    # The '_self' argument is a common, effective pattern to pass the instance for caching.
    @st.cache_resource
    def _build_road_graph(_self) -> nx.Graph:
        """Builds and caches the NetworkX road graph from config."""
        G = nx.Graph()
        network_config = _self.config.get('road_network', {})
        for node, data in network_config.get('nodes', {}).items():
            G.add_node(node, pos=data['pos'])
        for u, v, weight in network_config.get('edges', []):
            G.add_edge(u, v, weight=float(weight))
        return G

    @st.cache_resource
    def _load_or_compute_graph_embeddings(_self) -> Dict[str, np.ndarray]:
        """Loads graph node embeddings from cache or computes them if not present."""
        cache_file = CACHE_DIR / "graph_embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Recomputing.")
        
        if not _self.road_graph.nodes: return {}
        node2vec = Node2Vec(_self.road_graph, dimensions=8, walk_length=10, num_walks=50, workers=4, quiet=True)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        embeddings = {node: model.wv[node] for node in _self.road_graph.nodes()}
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings

    @st.cache_resource
    def _build_zones_gdf(_self) -> gpd.GeoDataFrame:
        """Builds and caches the GeoDataFrame for city zones."""
        zones = _self.config.get('zones', {})
        valid_zones = []
        for name, data in zones.items():
            try:
                poly = Polygon([(lon, lat) for lat, lon in data['polygon']]).buffer(0)
                if not poly.is_empty:
                    data['name'] = name
                    data['geometry'] = poly
                    valid_zones.append(data)
            except Exception as e:
                logger.warning(f"Skipping invalid polygon for zone '{name}': {e}")
        
        if not valid_zones: return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(valid_zones, crs=GEOGRAPHIC_CRS).set_index('name')
        gdf['centroid'] = gdf.to_crs(PROJECTED_CRS).geometry.centroid.to_crs(GEOGRAPHIC_CRS)
        
        # Associate zones with nearest road graph nodes
        graph_nodes_gdf = gpd.GeoDataFrame(
            geometry=[Point(d['pos'][1], d['pos'][0]) for _, d in _self.road_graph.nodes(data=True)],
            index=list(_self.road_graph.nodes()), crs=GEOGRAPHIC_CRS
        )
        nearest = gpd.sjoin_nearest(gdf, graph_nodes_gdf, how='left', distance_col='distance')
        gdf['nearest_node'] = nearest.groupby(nearest.index)['index_right'].first()
        return gdf.drop(columns=['polygon'], errors='ignore')

    def _initialize_hospitals(self) -> Dict:
        """Initializes hospital data from config."""
        return {name: {**data, 'location': Point(data['location'][1], data['location'][0])}
                for name, data in self.config.get('hospitals', {}).items()}

    def _initialize_ambulances(self) -> Dict:
        """Initializes ambulance data, placing them at their home base centroids."""
        ambulances = {}
        for amb_id, amb_data in self.config.get('ambulances', {}).items():
            home_zone = amb_data.get('home_base')
            if home_zone in self.zones_gdf.index:
                zone_info = self.zones_gdf.loc[home_zone]
                ambulances[amb_id] = {'id': amb_id, **amb_data, 'location': zone_info.centroid, 'nearest_node': zone_info.nearest_node}
            else:
                logger.warning(f"Invalid home base '{home_zone}' for ambulance '{amb_id}'.")
        return ambulances

    def assign_zone_to_point(self, point: Point) -> str:
        """Finds which zone a given point falls into."""
        for name, row in self.zones_gdf.iterrows():
            if row.geometry.contains(point):
                return name
        return None

class SimulationEngine:
    """Generates synthetic incident data based on configurable environmental factors."""
    def __init__(self, data_manager: DataManager, sim_params: Dict, distributions: Dict):
        self.dm = data_manager
        self.sim_params = sim_params
        self.dist = distributions
        self.nhpp_intensity = lambda t: 1 + 0.5 * np.sin((t / 24) * 2 * np.pi)

    def _generate_random_point_in_polygon(self, polygon: Polygon) -> Point:
        """Generates a random point guaranteed to be within a given polygon."""
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            p = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(p):
                return p

    def get_live_state(self, env_factors: EnvFactors, time_hour: float = 0.0) -> Dict[str, Any]:
        """
        Generates a snapshot of the city's state, including new incidents and traffic.
        This version generates incidents realistically within zone polygons.
        """
        mult = self.sim_params.get('multipliers', {})
        intensity = self.nhpp_intensity(time_hour) * float(env_factors.base_rate)
        intensity *= mult.get('holiday', 1.0) if env_factors.is_holiday else 1.0
        intensity *= mult.get('payday', 1.0) if env_factors.is_payday else 1.0
        intensity *= mult.get(env_factors.weather_condition.lower(), 1.0)
        intensity *= mult.get('major_event', 1.0) if env_factors.major_event_active else 1.0

        num_incidents = max(0, int(np.random.poisson(intensity)))
        if num_incidents == 0 or self.dm.zones_gdf.empty:
            return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}

        # Generate incidents based on zone probability distribution
        incident_zones = np.random.choice(
            list(self.dist['zone'].keys()),
            num_incidents,
            p=list(self.dist['zone'].values())
        )
        
        incidents = []
        for i, zone_name in enumerate(incident_zones):
            zone_polygon = self.dm.zones_gdf.loc[zone_name].geometry
            location = self._generate_random_point_in_polygon(zone_polygon)
            incident = {
                'id': f"INC-{int(time_hour*100)}-{i}",
                'type': np.random.choice(list(self.dist['incident_type'].keys()), p=list(self.dist['incident_type'].values())),
                'triage': np.random.choice(list(self.dist['triage'].keys()), p=list(self.dist['triage'].values())),
                'is_echo': False,
                'timestamp': time_hour,
                'location': location,
                'zone': zone_name
            }
            incidents.append(incident)

        # Simplified Hawkes Process: High-severity incidents can trigger "echo" events.
        echo_data = []
        triggers = [inc for inc in incidents if inc['triage'] == 'Rojo']
        for trigger in triggers:
            if np.random.rand() < env_factors.self_excitation_factor:
                for j in range(np.random.randint(1, 3)):
                    echo_loc = Point(trigger['location'].x + np.random.normal(0, 0.005), trigger['location'].y + np.random.normal(0, 0.005))
                    echo_zone = self.dm.assign_zone_to_point(echo_loc)
                    if echo_zone:
                        echo_data.append({'id': f"ECHO-{trigger['id']}-{j}", 'type': "Echo", 'triage': "Verde", 'location': echo_loc, 'is_echo': True, 'zone': echo_zone, 'timestamp': time_hour})

        traffic_conditions = {z: min(1.0, env_factors.traffic_multiplier * np.random.uniform(0.3, 1.0)) for z in self.dm.zones_gdf.index}
        all_incidents = incidents + echo_data
        system_state = "Anomalous" if len(all_incidents) > 10 or any(t['triage'] == 'Rojo' for t in all_incidents) else "Elevated" if len(all_incidents) > 5 else "Normal"
        
        return {"active_incidents": all_incidents, "traffic_conditions": traffic_conditions, "system_state": system_state}

class PredictiveAnalyticsEngine:
    """Handles forecasting, risk calculation, and analytics with ML models."""
    def __init__(self, data_manager: DataManager, model_params: Dict, dist_config: Dict):
        self.dm = data_manager
        self.params = model_params
        self.dist = dist_config
        self.ml_models = {z: GradientBoostingRegressor(n_estimators=50, random_state=42) for z in self.dm.zones_gdf.index}
        self.gp_models = {z: GaussianProcessRegressor(kernel=C(1.0) * RBF(10), n_restarts_optimizer=5, random_state=42) for z in self.dm.zones_gdf.index}
        self._train_models_on_synthetic_data()

    def _train_models_on_synthetic_data(self):
        """
        Trains ML models on synthetic data.
        NOTE: In a real-world scenario, this function would load and train on historical incident data.
        """
        np.random.seed(42)
        hours = np.arange(72)  # Use a longer time series for better training
        nhpp_intensity = lambda t: 1 + 0.5 * np.sin((t / 24) * 2 * np.pi)

        for zone in self.dm.zones_gdf.index:
            node = self.dm.zones_gdf.loc[zone, 'nearest_node']
            embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
            
            # Create more realistic features
            traffic_sim = np.random.uniform(0.3, 1.0, len(hours)) * (1 + 0.2 * np.sin((hours/12)*np.pi))
            prior_risk_sim = np.full_like(hours, self.dist['zone'].get(zone, 0.1))
            
            X = np.hstack([
                hours.reshape(-1, 1),
                traffic_sim.reshape(-1, 1),
                prior_risk_sim.reshape(-1, 1),
                np.tile(embedding, (len(hours), 1))
            ])
            # Target variable y is based on the simulation's own intensity function + noise
            y = (nhpp_intensity(hours) * prior_risk_sim * traffic_sim) + np.random.normal(0, 0.1, len(hours))
            
            self.ml_models[zone].fit(X, y)
            self.gp_models[zone].fit(X, y)

    def calculate_holistic_risk(self, live_state: Dict, prior_risks: Dict) -> Dict:
        """Calculates and diffuses risk across the city graph."""
        df = pd.DataFrame(live_state.get("active_incidents", []))
        counts = df.groupby('zone').size() if not df.empty else pd.Series(dtype=int)
        
        w = self.params['risk_weights']
        inc_load_factor = self.params.get('incident_load_factor', 0.25)

        # Calculate evidence-based risk for each zone
        evidence_risk = {
            zone: (prior_risks.get(zone, 0.5) * w['prior'] + 
                   live_state.get('traffic_conditions', {}).get(zone, 0.5) * w['traffic'] + 
                   counts.get(zone, 0) * inc_load_factor * w['incidents'])
            for zone in self.dm.zones_gdf.index
        }
        
        # Map zone risk to graph nodes and diffuse
        node_risks = {self.dm.zones_gdf.loc[zone, 'nearest_node']: risk for zone, risk in evidence_risk.items()}
        return self._diffuse_risk_on_graph(node_risks)

    def _diffuse_risk_on_graph(self, initial_risks: Dict[str, float]) -> Dict[str, float]:
        """Spreads risk scores across the road network using graph Laplacian."""
        graph = self.dm.road_graph
        if not graph.nodes or not initial_risks: return {}
        
        L = nx.normalized_laplacian_matrix(graph).toarray()
        node_order = list(graph.nodes())
        risks = np.array([initial_risks.get(node, 0) for node in node_order])
        
        for _ in range(self.params.get('risk_diffusion_steps', 3)):
            risks = risks - self.params.get('risk_diffusion_factor', 0.1) * (L @ risks)
            
        return {node: max(0, float(r)) for node, r in zip(node_order, risks)}

    def calculate_information_metrics(self, live_state: Dict) -> Tuple[float, float, Dict, Dict, float]:
        """Calculates KL Divergence, Shannon Entropy, and Mutual Information."""
        hist = self.dist['zone']
        df = pd.DataFrame([i for i in live_state.get("active_incidents", []) if not i.get("is_echo", False)])
        
        if df.empty or 'zone' not in df.columns or df['zone'].isnull().all():
            return 0.0, 0.0, hist, {z: 0.0 for z in self.dm.zones_gdf.index}, 0.0
            
        counts = df.groupby('zone').size()
        total = len(df)
        current = {z: counts.get(z, 0) / total for z in self.dm.zones_gdf.index}
        epsilon = 1e-9
        
        kl_divergence = sum(p * np.log((p + epsilon) / (hist.get(z, 0) + epsilon)) for z, p in current.items() if p > 0)
        shannon_entropy = -sum(p * np.log2(p + epsilon) for p in current.values() if p > 0)
        
        mutual_info = 0.0
        if 'type' in df.columns and not df.dropna(subset=['zone', 'type']).empty:
            joint = pd.crosstab(df['zone'], df['type'], normalize=True)
            p_z = joint.sum(axis=1)
            p_t = joint.sum(axis=0)
            for z in joint.index:
                for t in joint.columns:
                    if joint.loc[z, t] > 0:
                        mutual_info += joint.loc[z, t] * np.log2(joint.loc[z, t] / (p_z[z] * p_t[t]))
                        
        return kl_divergence, shannon_entropy, hist, current, mutual_info

    def forecast_risk_over_time(self, anomaly: float, horizon: int) -> pd.DataFrame:
        """Forecasts risk for future hours using trained ML models."""
        data = []
        for zone in self.dm.zones_gdf.index:
            node = self.dm.zones_gdf.loc[zone, 'nearest_node']
            embedding = self.dm.graph_embeddings.get(node, np.zeros(8))
            
            hours = np.arange(horizon)
            traffic_sim = np.random.uniform(0.3, 1.0, horizon)
            prior_risk_sim = np.full(horizon, self.dist['zone'].get(zone, 0.1))
            
            X = np.hstack([
                hours.reshape(-1, 1),
                traffic_sim.reshape(-1, 1),
                prior_risk_sim.reshape(-1, 1),
                np.tile(embedding, (horizon, 1))
            ])
            
            ml_pred = self.ml_models[zone].predict(X)
            gp_pred, _ = self.gp_models[zone].predict(X, return_std=True)
            combined_pred = np.clip((0.7 * ml_pred + 0.3 * gp_pred) * (1 + 0.2 * anomaly), 0, 2)
            
            for h, pred in enumerate(combined_pred):
                data.append({'zone': zone, 'hour': h, 'projected_risk': float(pred)})
        return pd.DataFrame(data)

class SensitivityAnalyzer:
    """Performs sensitivity analysis on simulation parameters."""
    def __init__(self, simulation_engine: SimulationEngine, predictive_engine: PredictiveAnalyticsEngine):
        self.sim_engine = simulation_engine
        self.pred_engine = predictive_engine
        logger.info("SensitivityAnalyzer initialized.")

    @st.cache_data(ttl=600)
    def analyze_sensitivity(_self, base_env_factors: EnvFactors, parameters_to_test: Dict[str, List[float]], iterations: int = 5) -> pd.DataFrame:
        """Analyzes sensitivity to parameter changes by measuring output variance."""
        results = []
        # Calculate baseline metrics
        base_state = _self.sim_engine.get_live_state(base_env_factors)
        base_risk = _self.pred_engine.calculate_holistic_risk(base_state, _self.pred_engine.dist['zone'])[1]
        base_anomaly = _self.pred_engine.calculate_information_metrics(base_state)[0]
        
        for param, values in parameters_to_test.items():
            for value in values:
                param_risks, param_anomalies = [], []
                for _ in range(iterations):
                    modified_factors_dict = base_env_factors.__dict__.copy()
                    modified_factors_dict[param] = value
                    modified_factors = EnvFactors(**modified_factors_dict)
                    
                    state = _self.sim_engine.get_live_state(modified_factors)
                    # Use static prior for fair comparison across runs
                    risk, _ = _self.pred_engine.calculate_holistic_risk(state, _self.pred_engine.dist['zone'])
                    anomaly = _self.pred_engine.calculate_information_metrics(state)[0]
                    
                    param_risks.append(np.mean(list(risk.values())))
                    param_anomalies.append(anomaly)
                
                results.append({
                    'parameter': param.replace('_', ' ').title(), 'value': value, 
                    'mean_risk': np.mean(param_risks), 'mean_anomaly': np.mean(param_anomalies)
                })
        return pd.DataFrame(results)

class StrategicAdvisor:
    """Provides strategic recommendations for resource allocation."""
    def __init__(self, data_manager: DataManager, model_params: Dict):
        self.dm = data_manager
        self.params = model_params

    def calculate_projected_response_time(self, zone: str, ambulances: List[Dict]) -> float:
        """Calculates the minimum projected response time from available ambulances to a zone."""
        if not zone or zone not in self.dm.zones_gdf.index or not ambulances:
            return DEFAULT_RESPONSE_TIME
            
        zone_node = self.dm.zones_gdf.loc[zone, 'nearest_node']
        if pd.isna(zone_node): return DEFAULT_RESPONSE_TIME
        
        min_time = float('inf')
        for amb in ambulances:
            if amb.get('status') == 'Disponible' and amb.get('nearest_node'):
                try:
                    time = nx.shortest_path_length(self.dm.road_graph, amb['nearest_node'], zone_node, weight='weight')
                    min_time = min(min_time, time)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return (min_time + self.params['response_time_turnout_penalty']) if min_time != float('inf') else DEFAULT_RESPONSE_TIME

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        """Recommends ambulance reallocations based on risk-response time deficits."""
        available_ambs = [{'id': amb_id, **d} for amb_id, d in self.dm.ambulances.items() if d.get('status') == 'Disponible']
        if not available_ambs: return []

        # Calculate current performance (deficit = risk * response_time)
        perf = {
            z: {'risk': risk_scores.get(d['nearest_node'], 0), 'rt': self.calculate_projected_response_time(z, available_ambs)}
            for z, d in self.dm.zones_gdf.iterrows() if pd.notna(d.get('nearest_node'))
        }
        deficits = {z: p['risk'] * p['rt'] for z, p in perf.items()}
        if not deficits or max(deficits.values(), default=0) < self.params['recommendation_deficit_threshold']:
            return []

        # Find zone with highest deficit and its associated graph node
        target_zone = max(deficits, key=deficits.get)
        original_rt = perf[target_zone]['rt']
        target_node = self.dm.zones_gdf.loc[target_zone, 'nearest_node']
        if pd.isna(target_node): return []

        # Find the ambulance move that provides the greatest utility (reduction in deficit)
        best_move = None
        max_utility = -float('inf')
        for amb in available_ambs:
            if not amb.get('nearest_node') or amb['nearest_node'] == target_node:
                continue
            
            # Simulate the move
            moved_ambulances = [{**a, 'nearest_node': target_node} if a['id'] == amb['id'] else a for a in available_ambs]
            new_rt = self.calculate_projected_response_time(target_zone, moved_ambulances)
            
            utility = (original_rt - new_rt) * perf[target_zone]['risk'] # Utility = RT improvement * risk
            if utility > max_utility:
                max_utility = utility
                best_move = (amb['id'], self.dm.node_to_zone_map.get(amb['nearest_node'], 'Unknown'), new_rt)

        if best_move and max_utility > self.params['recommendation_improvement_threshold']:
            amb_id, from_zone, new_rt = best_move
            return [{"unit": amb_id, "from": from_zone, "to": target_zone, "reason": f"Reduce projected response time in high-risk zone '{target_zone}' from ~{original_rt:.0f} to ~{new_rt:.0f} min."}]
        
        return []

# --- L3: VISUALIZATION & UI ---

class VisualizationSuite:
    """Handles the creation of all plots and maps for the UI."""
    def __init__(self, style_config: Dict):
        self.config = style_config

    def plot_risk_profile_comparison(self, prior_df, posterior_df):
        """Plots a comparison of prior (historical) vs. posterior (current) risk."""
        if prior_df.empty or posterior_df.empty:
            return alt.Chart().mark_text(text="No data for risk comparison").encode()
        
        prior_df['type'] = 'Prior (Historical)'
        posterior_df['type'] = 'Posterior (Current)'
        combined = pd.concat([prior_df, posterior_df], ignore_index=True)
        
        return alt.Chart(combined).mark_bar(opacity=0.8).encode(
            x=alt.X('risk:Q', title='Risk Level'),
            y=alt.Y('zone:N', title='Zone', sort='-x'),
            color=alt.Color('type:N', title='Risk Type', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=['zone', 'type', alt.Tooltip('risk', format='.3f')]
        ).properties(title="Risk Profile Comparison").interactive()

    def plot_sensitivity_analysis(self, sensitivity_df: pd.DataFrame) -> alt.Chart:
        """Generates a grouped bar chart for sensitivity analysis results."""
        if sensitivity_df.empty:
            return alt.Chart().mark_text(text="No sensitivity data to display.").encode()

        base = alt.Chart(sensitivity_df).encode(
            x=alt.X('value:Q', title="Parameter Value"),
            y=alt.Y('mean_risk:Q', title="Mean Output Value"),
            color=alt.Color('parameter:N', title="Parameter")
        )
        chart = base.mark_line(point=True) + base.mark_point()
        return chart.properties(
            title="Sensitivity Analysis: Parameter vs. Mean Risk"
        ).facet(
            row=alt.Row('parameter:N', title="Parameter"),
            resolve=alt.Resolve(scale={'x': 'independent'})
        ).interactive()

def prepare_visualization_data(data_manager: DataManager, risk_scores: Dict, all_incidents: List, style: Dict) -> Tuple:
    """Prepares and sanitizes data from various sources into DataFrames for pydeck visualization."""
    zones_gdf = data_manager.zones_gdf.copy()
    zones_gdf['risk'] = zones_gdf['nearest_node'].map(risk_scores).fillna(0.0).astype(float)
    max_risk = max(0.01, zones_gdf['risk'].max())
    zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(220 * (r / max_risk))]).tolist()
    zone_df = pd.DataFrame([
        {'name': idx, 'polygon': list(row.geometry.exterior.coords), 'risk': float(row['risk']), 'fill_color': row['fill_color'], 'tooltip_text': f"Risk: {row['risk']:.3f}"}
        for idx, row in zones_gdf.iterrows() if row.geometry and not row.geometry.is_empty
    ])
    
    hosp_df = pd.DataFrame([
        {'name': f"H: {n}", 'lon': d['location'].x, 'lat': d['location'].y, 'icon_data': {"url": style['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}, 'tooltip_text': f"Capacity: {d['capacity']} Load: {d['load']}"}
        for n, d in data_manager.hospitals.items() if d.get('location')
    ])
    
    amb_df = pd.DataFrame([
        {'name': f"U: {d['id']}", 'lon': d['location'].x, 'lat': d['location'].y, 'icon_data': {"url": style['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, 'tooltip_text': f"Status: {d['status']}<br>Base: {d['home_base']}"}
        for d in data_manager.ambulances.values() if d.get('location')
    ])
    
    inc_df = pd.DataFrame([
        {'lon': i['location'].x, 'lat': i['location'].y, 'color': style['colors']['hawkes_echo'] if i.get('is_echo') else style['colors']['accent_crit'], 'radius': style['sizes']['hawkes_echo'] if i.get('is_echo') else style['sizes']['incident_base'], 'name': f"I: {i.get('id', 'N/A')}", 'tooltip_text': f"Type: {i.get('type')}<br>Triage: {i.get('triage')}"}
        for i in all_incidents if i.get('location')
    ])
    
    heatmap_df = pd.DataFrame([
        {"lon": i['location'].x, "lat": i['location'].y}
        for i in all_incidents if i.get('location') and not i.get('is_echo')
    ])
    
    return zone_df, hosp_df, amb_df, inc_df, heatmap_df

def create_deck_gl_map(zone_df, hosp_df, amb_df, inc_df, heat_df, app_config) -> pdk.Deck:
    """Creates the main Deck.gl map with multiple layers."""
    style = app_config['styling']
    
    layers = [
        pdk.Layer("HeatmapLayer", data=heat_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN', threshold=0.1),
        pdk.Layer("PolygonLayer", data=zone_df, get_polygon="polygon", filled=True, stroked=True, lineWidthMinPixels=1, get_line_color=[255, 255, 255, 50], extruded=True, get_elevation=f"risk * {style['map_elevation_multiplier']}", get_fill_color="fill_color", opacity=0.1, pickable=True),
        pdk.Layer("IconLayer", data=hosp_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style['sizes']['hospital'], size_scale=15, pickable=True),
        pdk.Layer("IconLayer", data=amb_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style['sizes']['ambulance'], size_scale=15, pickable=True),
        pdk.Layer("ScatterplotLayer", data=inc_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True),
    ]
    
    return pdk.Deck(
        layers=[layer for layer in layers if not layer.data.empty],
        initial_view_state=pdk.ViewState(latitude=32.5, longitude=-117.02, zoom=11, bearing=0, pitch=50),
        map_provider="mapbox" if app_config.get('mapbox_api_key') else "carto",
        map_style=app_config['map_style'],
        api_keys={'mapbox': app_config.get('mapbox_api_key')},
        tooltip={"html": "<b>{name}</b><br/>{tooltip_text}"}
    )

# --- L4: APPLICATION ENTRYPOINT & MAIN LOGIC ---

@st.cache_resource
def initialize_app_components():
    """Initializes and caches all core application components."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    app_config = get_app_config()
    distributions = {k: _normalize_dist(v) for k, v in app_config['data']['distributions'].items()}
    
    data_manager = DataManager(app_config)
    engine = SimulationEngine(data_manager, app_config['simulation_params'], distributions)
    predictor = PredictiveAnalyticsEngine(data_manager, app_config['model_params'], distributions)
    advisor = StrategicAdvisor(data_manager, app_config['model_params'])
    sensitivity_analyzer = SensitivityAnalyzer(engine, predictor)
    plotter = VisualizationSuite(app_config['styling'])
    
    return data_manager, engine, predictor, advisor, sensitivity_analyzer, plotter, app_config

def initialize_session_state(config):
    """Initializes session state variables if they don't exist."""
    if 'current_hour' not in st.session_state:
        st.session_state.current_hour = 0.0
    if 'prior_risks' not in st.session_state:
        st.session_state.prior_risks = {
            name: data.get('prior_risk', 0.5)
            for name, data in config['data']['zones'].items()
        }

def update_prior_risks(live_state: Dict, learning_rate: float = 0.05):
    """Updates the prior risk distribution based on new incident data (Bayesian-like update)."""
    df = pd.DataFrame(live_state.get("active_incidents", []))
    if df.empty or 'zone' not in df.columns:
        return
        
    incident_counts = df.groupby('zone').size()
    total_incidents = len(df)
    
    if total_incidents == 0:
        return
        
    # Calculate observed risk (proportion of incidents)
    observed_risk = incident_counts / total_incidents
    
    for zone, risk in observed_risk.items():
        if zone in st.session_state.prior_risks:
            current_prior = st.session_state.prior_risks[zone]
            # Update prior with a weighted average of the old prior and new observation
            new_prior = (1 - learning_rate) * current_prior + learning_rate * risk
            st.session_state.prior_risks[zone] = new_prior


def render_intel_briefing(anomaly: float, entropy: float, mutual_info: float, recommendations: List[Dict]):
    """Renders the main intelligence briefing and recommendations panel."""
    st.subheader("Intel Briefing & Recommendations")
    status = "ANOMALOUS" if anomaly > 0.2 else "ELEVATED" if anomaly > 0.1 else "NOMINAL"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System Status", status)
    c2.metric("Anomaly Score (KL Div.)", f"{anomaly:.4f}")
    c3.metric("Spatial Entropy", f"{entropy:.4f} bits")
    c4.metric("Mutual Information", f"{mutual_info:.4f}")
    
    if recommendations:
        st.warning("Actionable Recommendation:")
        for r in recommendations:
            st.write(f"**Move Unit {r['unit']}** from `{r['from']}` to `{r['to']}`. **Reason:** {r['reason']}")
    else:
        st.success("No resource reallocations required. Current deployment is optimal.")

def main():
    """Main application entry point and render loop."""
    st.set_page_config(page_title="RedShield AI", layout="wide", initial_sidebar_state="expanded")
    
    try:
        dm, engine, predictor, advisor, analyzer, plotter, config = initialize_app_components()
        initialize_session_state(config)
        
        # --- SIDEBAR ---
        st.sidebar.title("RedShield AI v10.20")
        st.sidebar.markdown("**EMS Digital Twin**")
        
        st.sidebar.header("Simulation Control")
        if st.sidebar.button("Advance Time (Simulate)"):
             st.session_state.current_hour = (st.session_state.current_hour + 0.5) % 24
        st.sidebar.metric("Current Simulation Time", f"{st.session_state.current_hour:.1f}h")
        
        st.sidebar.subheader("Environmental Factors")
        is_holiday = st.sidebar.checkbox("Holiday Active")
        is_payday = st.sidebar.checkbox("Is Payday")
        weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Rain", "Fog"], index=0)
        base_rate = st.sidebar.slider("Incident Base Rate (μ)", 1, 20, 5, help="Baseline number of incidents per hour.")
        excitation = st.sidebar.slider("Self-Excitation (κ)", 0.0, 1.0, 0.5, help="Probability of a severe incident triggering an echo.")
        
        factors = EnvFactors(is_holiday, is_payday, weather, False, 1.0, base_rate, excitation)
        
        # --- CORE LOGIC ---
        live_state = engine.get_live_state(factors, st.session_state.current_hour)
        update_prior_risks(live_state)
        
        posterior_risk = predictor.calculate_holistic_risk(live_state, st.session_state.prior_risks)
        anomaly, entropy, hist_dist, curr_dist, mutual_info = predictor.calculate_information_metrics(live_state)
        recs = advisor.recommend_resource_reallocations(posterior_risk)
        
        # --- MAIN PANEL ---
        st.title("RedShield AI Command Suite")
        
        render_intel_briefing(anomaly, entropy, mutual_info, recs)
        st.divider()
        
        st.subheader("Live Operations Map")
        vis_data = prepare_visualization_data(dm, posterior_risk, live_state["active_incidents"], config['styling'])
        deck = create_deck_gl_map(*vis_data, config)
        st.pydeck_chart(deck, use_container_width=True)

        # --- ADVANCED ANALYTICS EXPANDER ---
        with st.expander("Advanced Analytics & Forecasting"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Risk Analysis")
                prior_df = pd.DataFrame(st.session_state.prior_risks.items(), columns=['zone', 'risk'])
                posterior_zone_risk = {dm.node_to_zone_map.get(node): risk for node, risk in posterior_risk.items() if dm.node_to_zone_map.get(node)}
                posterior_df = pd.DataFrame(posterior_zone_risk.items(), columns=['zone', 'risk'])
                st.altair_chart(plotter.plot_risk_profile_comparison(prior_df, posterior_df), use_container_width=True)

            with col2:
                st.subheader("Sensitivity Analysis")
                st.info("Analyze how key factors influence system risk.", icon="🔬")
                if st.button("Run Sensitivity Analysis"):
                    with st.spinner("Running analysis... this may take a moment."):
                        params_to_test = {
                            'base_rate': list(range(1, 11, 2)),
                            'self_excitation_factor': np.linspace(0, 1, 5).tolist()
                        }
                        sensitivity_results = analyzer.analyze_sensitivity(factors, params_to_test)
                        st.altair_chart(plotter.plot_sensitivity_analysis(sensitivity_results), use_container_width=True)

    except Exception as e:
        logger.error(f"Application failed with a critical error: {e}", exc_info=True)
        st.error(f"A critical error occurred: {e}. Please check the `redshield_ai.log` file for details.")

if __name__ == "__main__":
    main()
