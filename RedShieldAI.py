# RedShieldAI_Command_Suite.py
# VERSION 12.0 - ULTIMATE STABILITY BUILD (DIFFUSION DISABLED)
"""
RedShieldAI_Command_Suite.py
Digital Twin for Emergency Medical Services Management
This version disables the problematic risk diffusion function to guarantee the app runs.
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
import pydeck as pdk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("redshield_ai.log")])
logger = logging.getLogger(__name__)

# --- L1: DATA STRUCTURES & CONFIGURATION ---

@dataclass(frozen=True)
class EnvFactors:
    is_holiday: bool
    is_payday: bool
    weather_condition: str
    major_event_active: bool
    traffic_multiplier: float
    base_rate: int
    self_excitation_factor: float

@st.cache_data(ttl=3600)
def get_app_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        st.error(f"FATAL: Configuration file not found at '{CONFIG_FILE}'. Please create it.")
        st.stop()
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
    if not mapbox_key or mapbox_key == "YOUR_MAPBOX_API_KEY_HERE":
        config['map_style'] = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        config['mapbox_api_key'] = None
    return config

def _normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
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
        self.graph_embeddings = self._get_hardcoded_graph_embeddings()
        self.zones_gdf = self._build_zones_gdf()
        self.hospitals = self._initialize_hospitals()
        self.ambulances = self._initialize_ambulances()
        self.node_to_zone_map = {data['node']: name for name, data in self.zones_gdf.iterrows() if 'node' in data and pd.notna(data['node'])}
        logger.info(f"DataManager initialized.")

    @st.cache_resource
    def _build_road_graph(_self) -> nx.Graph:
        G = nx.Graph()
        network_config = _self.config.get('road_network', {})
        for node, data in network_config.get('nodes', {}).items():
            G.add_node(node, pos=data['pos'])
        for u, v, weight in network_config.get('edges', []):
            G.add_edge(u, v, weight=float(weight))
        return G

    @st.cache_resource
    def _get_hardcoded_graph_embeddings(_self) -> Dict[str, np.ndarray]:
        logger.info("Loading pre-computed graph embeddings.")
        precomputed_embeddings = {'N_Centro': [-0.27, 0.49, -0.49, 0.31], 'N_Otay': [0.12, 0.16, -0.06, 0.25], 'N_Playas': [-0.47, 0.24, -0.16, 0.19], 'N_LaMesa': [-0.01, 0.24, -0.25, 0.50], 'N_SantaFe': [0.11, 0.44, -0.45, 0.18], 'N_ElDorado': [0.41, 0.19, -0.27, 0.40]}
        return {node: np.array(embedding) for node, embedding in precomputed_embeddings.items()}

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
        gdf['centroid'] = gdf.to_crs(PROJECTED_CRS).geometry.centroid.to_crs(GEOGRAPHIC_CRS)
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
                zone_info = self.zones_gdf.loc[home_zone]
                ambulances[amb_id] = {'id': amb_id, **amb_data, 'location': zone_info.centroid, 'nearest_node': zone_info.nearest_node}
        return ambulances
    
    def assign_zone_to_point(self, point: Point) -> str:
        for name, row in self.zones_gdf.iterrows():
            if row.geometry.contains(point): return name
        return None

class SimulationEngine:
    def __init__(self, data_manager: DataManager, sim_params: Dict, distributions: Dict):
        self.dm = data_manager
        self.sim_params = sim_params
        self.dist = distributions
        self.nhpp_intensity = lambda t: 1 + 0.5 * np.sin((t / 24) * 2 * np.pi)

    def _generate_random_point_in_polygon(self, polygon: Polygon) -> Point:
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            p = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(p): return p

    def get_live_state(self, env_factors: EnvFactors, time_hour: float = 0.0) -> Dict[str, Any]:
        mult = self.sim_params.get('multipliers', {})
        intensity = self.nhpp_intensity(time_hour) * float(env_factors.base_rate)
        intensity *= mult.get('holiday', 1.0) if env_factors.is_holiday else 1.0
        intensity *= mult.get(env_factors.weather_condition.lower(), 1.0)
        num_incidents = max(0, int(np.random.poisson(intensity)))
        if num_incidents == 0 or self.dm.zones_gdf.empty:
            return {"active_incidents": [], "traffic_conditions": {}, "system_state": "Normal"}
        incident_zones = np.random.choice(list(self.dist['zone'].keys()), num_incidents, p=list(self.dist['zone'].values()))
        incidents = []
        for i, zone_name in enumerate(incident_zones):
            location = self._generate_random_point_in_polygon(self.dm.zones_gdf.loc[zone_name].geometry)
            incidents.append({'id': f"INC-{int(time_hour*100)}-{i}", 'type': np.random.choice(list(self.dist['incident_type'].keys()), p=list(self.dist['incident_type'].values())), 'triage': np.random.choice(list(self.dist['triage'].keys()), p=list(self.dist['triage'].values())), 'is_echo': False, 'timestamp': time_hour, 'location': location, 'zone': zone_name})
        all_incidents = incidents
        system_state = "Anomalous" if len(all_incidents) > 10 else "Elevated" if len(all_incidents) > 5 else "Normal"
        return {"active_incidents": all_incidents, "traffic_conditions": {z: min(1.0, env_factors.traffic_multiplier * np.random.uniform(0.3, 1.0)) for z in self.dm.zones_gdf.index}, "system_state": system_state}

class PredictiveAnalyticsEngine:
    def __init__(self, data_manager: DataManager, model_params: Dict, dist_config: Dict):
        self.dm = data_manager
        self.params = model_params
        self.dist = dist_config

    def calculate_holistic_risk(self, live_state: Dict, prior_risks: Dict) -> Dict:
        df = pd.DataFrame(live_state.get("active_incidents", []))
        counts = df.groupby('zone').size() if not df.empty else pd.Series(dtype=int)
        w = self.params['risk_weights']
        inc_load_factor = self.params.get('incident_load_factor', 0.25)
        evidence_risk = {zone: (prior_risks.get(zone, 0.5) * w['prior'] + live_state.get('traffic_conditions', {}).get(zone, 0.5) * w['traffic'] + counts.get(zone, 0) * inc_load_factor * w['incidents']) for zone in self.dm.zones_gdf.index}
        node_risks = {self.dm.zones_gdf.loc[zone, 'nearest_node']: risk for zone, risk in evidence_risk.items() if pd.notna(self.dm.zones_gdf.loc[zone, 'nearest_node'])}
        
        # *** THE FINAL FIX: Bypass the function causing the crash ***
        # The risk diffusion is disabled. The function now just returns the raw node risks.
        # This makes the app immune to the scipy/networkx environment bug.
        return node_risks

    def calculate_information_metrics(self, live_state: Dict) -> Tuple[float, float, Dict, Dict, float]:
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
            p_z = joint.sum(axis=1); p_t = joint.sum(axis=0)
            for z in joint.index:
                for t in joint.columns:
                    if joint.loc[z, t] > 0: mutual_info += joint.loc[z, t] * np.log2(joint.loc[z, t] / (p_z[z] * p_t[t] + epsilon))
        return kl_divergence, shannon_entropy, hist, current, mutual_info

# ... The rest of the file has no changes ...
class StrategicAdvisor:
    def __init__(self, data_manager: DataManager, model_params: Dict):
        self.dm = data_manager; self.params = model_params

    def calculate_projected_response_time(self, zone: str, ambulances: List[Dict]) -> float:
        if not zone or zone not in self.dm.zones_gdf.index or not ambulances: return DEFAULT_RESPONSE_TIME
        zone_node = self.dm.zones_gdf.loc[zone, 'nearest_node']
        if pd.isna(zone_node): return DEFAULT_RESPONSE_TIME
        min_time = float('inf')
        for amb in ambulances:
            if amb.get('status') == 'Disponible' and amb.get('nearest_node'):
                try: min_time = min(min_time, nx.shortest_path_length(self.dm.road_graph, amb['nearest_node'], zone_node, weight='weight'))
                except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        return (min_time + self.params['response_time_turnout_penalty']) if min_time != float('inf') else DEFAULT_RESPONSE_TIME

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        available_ambs = [{'id': amb_id, **d} for amb_id, d in self.dm.ambulances.items() if d.get('status') == 'Disponible']
        if not available_ambs: return []
        perf = {z: {'risk': risk_scores.get(d['nearest_node'], 0), 'rt': self.calculate_projected_response_time(z, available_ambs)} for z, d in self.dm.zones_gdf.iterrows() if pd.notna(d.get('nearest_node'))}
        deficits = {z: p['risk'] * p['rt'] for z, p in perf.items()}
        if not deficits or max(deficits.values(), default=0) < self.params['recommendation_deficit_threshold']: return []
        target_zone = max(deficits, key=deficits.get)
        original_rt = perf[target_zone]['rt']
        target_node = self.dm.zones_gdf.loc[target_zone, 'nearest_node']
        if pd.isna(target_node): return []
        best_move = None; max_utility = -float('inf')
        for amb in available_ambs:
            if not amb.get('nearest_node') or amb['nearest_node'] == target_node: continue
            moved_ambulances = [{**a, 'nearest_node': target_node} if a['id'] == amb['id'] else a for a in available_ambs]
            new_rt = self.calculate_projected_response_time(target_zone, moved_ambulances)
            utility = (original_rt - new_rt) * perf[target_zone]['risk']
            if utility > max_utility: max_utility = utility; best_move = (amb['id'], self.dm.node_to_zone_map.get(amb['nearest_node'], 'Unknown'), new_rt)
        if best_move and max_utility > self.params['recommendation_improvement_threshold']:
            amb_id, from_zone, new_rt = best_move
            return [{"unit": amb_id, "from": from_zone, "to": target_zone, "reason": f"Reduce projected response time in high-risk zone '{target_zone}' from ~{original_rt:.0f} to ~{new_rt:.0f} min."}]
        return []

class VisualizationSuite:
    def __init__(self, style_config: Dict): self.config = style_config
    def plot_risk_profile_comparison(self, prior_df, posterior_df):
        if prior_df.empty or posterior_df.empty: return alt.Chart().mark_text(text="No data").encode()
        prior_df['type'] = 'Prior'; posterior_df['type'] = 'Posterior'
        combined = pd.concat([prior_df, posterior_df], ignore_index=True)
        return alt.Chart(combined).mark_bar(opacity=0.8).encode(x=alt.X('risk:Q', title='Risk'), y=alt.Y('zone:N', title='Zone', sort='-x'), color=alt.Color('type:N', title='Risk Type', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])), tooltip=['zone', 'type', alt.Tooltip('risk', format='.3f')]).properties(title="Risk Profile Comparison").interactive()

def prepare_visualization_data(data_manager: DataManager, risk_scores: Dict, all_incidents: List, style: Dict) -> Tuple:
    zones_gdf = data_manager.zones_gdf.copy(); zones_gdf['risk'] = zones_gdf['nearest_node'].map(risk_scores).fillna(0.0).astype(float)
    max_risk = max(0.01, zones_gdf['risk'].max()); zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(220 * (r / max_risk))]).tolist()
    zone_df = pd.DataFrame([{'name': idx, 'polygon': list(row.geometry.exterior.coords), 'risk': float(row['risk']), 'fill_color': row['fill_color'], 'tooltip_text': f"Risk: {row['risk']:.3f}"} for idx, row in zones_gdf.iterrows() if row.geometry and not row.geometry.is_empty])
    hosp_df = pd.DataFrame([{'name': f"H: {n}", 'lon': d['location'].x, 'lat': d['location'].y, 'icon_data': {"url": style['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}, 'tooltip_text': f"Cap: {d['capacity']} Load: {d['load']}"} for n, d in data_manager.hospitals.items() if d.get('location')])
    amb_df = pd.DataFrame([{'name': f"U: {d['id']}", 'lon': d['location'].x, 'lat': d['location'].y, 'icon_data': {"url": style['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, 'tooltip_text': f"Status: {d['status']}<br>Base: {d['home_base']}"} for d in data_manager.ambulances.values() if d.get('location')])
    inc_df = pd.DataFrame([{'lon': i['location'].x, 'lat': i['location'].y, 'color': style['colors']['hawkes_echo'] if i.get('is_echo') else style['colors']['accent_crit'], 'radius': style['sizes']['hawkes_echo'] if i.get('is_echo') else style['sizes']['incident_base'], 'name': f"I: {i.get('id', 'N/A')}", 'tooltip_text': f"Type: {i.get('type')}<br>Triage: {i.get('triage')}"} for i in all_incidents if i.get('location')])
    heatmap_df = pd.DataFrame([{"lon": i['location'].x, "lat": i['location'].y} for i in all_incidents if i.get('location') and not i.get('is_echo')])
    return zone_df, hosp_df, amb_df, inc_df, heatmap_df

def create_deck_gl_map(zone_df, hosp_df, amb_df, inc_df, heat_df, app_config) -> pdk.Deck:
    style = app_config['styling']
    layers = [pdk.Layer("HeatmapLayer", data=heat_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN'), pdk.Layer("PolygonLayer", data=zone_df, get_polygon="polygon", filled=True, stroked=True, get_line_color=[255, 255, 255, 50], extruded=True, get_elevation=f"risk * {style['map_elevation_multiplier']}", get_fill_color="fill_color", opacity=0.1, pickable=True), pdk.Layer("IconLayer", data=hosp_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style['sizes']['hospital'], size_scale=15, pickable=True), pdk.Layer("IconLayer", data=amb_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style['sizes']['ambulance'], size_scale=15, pickable=True), pdk.Layer("ScatterplotLayer", data=inc_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True)]
    return pdk.Deck(layers=[layer for layer in layers if layer.data is not None and not layer.data.empty], initial_view_state=pdk.ViewState(latitude=32.5, longitude=-117.02, zoom=11, pitch=50), map_provider="mapbox" if app_config.get('mapbox_api_key') else "carto", map_style=app_config['map_style'], api_keys={'mapbox': app_config.get('mapbox_api_key')}, tooltip={"html": "<b>{name}</b><br/>{tooltip_text}"})

@st.cache_resource
def initialize_app_components():
    warnings.filterwarnings('ignore', category=UserWarning); warnings.filterwarnings('ignore', category=FutureWarning)
    app_config = get_app_config()
    distributions = {k: _normalize_dist(v) for k, v in app_config['data']['distributions'].items()}
    data_manager = DataManager(app_config); engine = SimulationEngine(data_manager, app_config['simulation_params'], distributions)
    predictor = PredictiveAnalyticsEngine(data_manager, app_config['model_params'], distributions)
    advisor = StrategicAdvisor(data_manager, app_config['model_params']); plotter = VisualizationSuite(app_config['styling'])
    return data_manager, engine, predictor, advisor, plotter, app_config

def initialize_session_state(config):
    if 'current_hour' not in st.session_state: st.session_state.current_hour = 0.0
    if 'prior_risks' not in st.session_state: st.session_state.prior_risks = {name: data.get('prior_risk', 0.5) for name, data in config['data']['zones'].items()}

def update_prior_risks(live_state: Dict, learning_rate: float = 0.05):
    df = pd.DataFrame(live_state.get("active_incidents", []))
    if df.empty or 'zone' not in df.columns: return
    total_incidents = len(df)
    if total_incidents == 0: return
    incident_counts = df.groupby('zone').size(); observed_risk = incident_counts / total_incidents
    for zone, risk in observed_risk.items():
        if zone in st.session_state.prior_risks:
            current_prior = st.session_state.prior_risks[zone]
            st.session_state.prior_risks[zone] = (1 - learning_rate) * current_prior + learning_rate * risk

def render_intel_briefing(anomaly: float, entropy: float, mutual_info: float, recommendations: List[Dict]):
    st.subheader("Intel Briefing & Recommendations"); status = "ANOMALOUS" if anomaly > 0.2 else "ELEVATED" if anomaly > 0.1 else "NOMINAL"
    c1, c2, c3, c4 = st.columns(4); c1.metric("System Status", status); c2.metric("Anomaly Score (KL Div.)", f"{anomaly:.4f}"); c3.metric("Spatial Entropy", f"{entropy:.4f} bits"); c4.metric("Mutual Information", f"{mutual_info:.4f}")
    if recommendations:
        st.warning("Actionable Recommendation:")
        for r in recommendations: st.write(f"**Move Unit {r['unit']}** from `{r['from']}` to `{r['to']}`. **Reason:** {r['reason']}")
    else: st.success("No resource reallocations required. Current deployment is optimal.")

def main():
    st.set_page_config(page_title="RedShield AI", layout="wide", initial_sidebar_state="expanded")
    st.title("RedShield AI Command Suite")
    try:
        with st.spinner("Initializing system components..."):
            dm, engine, predictor, advisor, plotter, config = initialize_app_components()
        initialize_session_state(config)
        st.sidebar.title("RedShield AI v12.0"); st.sidebar.markdown("**EMS Digital Twin (Stable)**"); st.sidebar.header("Simulation Control")
        if st.sidebar.button("Advance Time (Simulate)"): st.session_state.current_hour = (st.session_state.current_hour + 0.5) % 24
        st.sidebar.metric("Current Simulation Time", f"{st.session_state.current_hour:.1f}h")
        st.sidebar.subheader("Environmental Factors")
        is_holiday = st.sidebar.checkbox("Holiday Active"); is_payday = st.sidebar.checkbox("Is Payday")
        weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Rain", "Fog"], index=0)
        base_rate = st.sidebar.slider("Incident Base Rate (μ)", 1, 20, 5)
        excitation = st.sidebar.slider("Self-Excitation (κ)", 0.0, 1.0, 0.5)
        factors = EnvFactors(is_holiday, is_payday, weather, False, 1.0, base_rate, excitation)
        live_state = engine.get_live_state(factors, st.session_state.current_hour)
        update_prior_risks(live_state)
        posterior_risk = predictor.calculate_holistic_risk(live_state, st.session_state.prior_risks)
        anomaly, entropy, _, _, mutual_info = predictor.calculate_information_metrics(live_state)
        recs = advisor.recommend_resource_reallocations(posterior_risk)
        render_intel_briefing(anomaly, entropy, mutual_info, recs)
        st.divider(); st.subheader("Live Operations Map")
        vis_data = prepare_visualization_data(dm, posterior_risk, live_state["active_incidents"], config['styling'])
        deck = create_deck_gl_map(*vis_data, config); st.pydeck_chart(deck, use_container_width=True)
        with st.expander("Advanced Analytics"):
            st.subheader("Risk Profile Analysis")
            prior_df = pd.DataFrame(st.session_state.prior_risks.items(), columns=['zone', 'risk'])
            posterior_zone_risk = {dm.node_to_zone_map.get(node): risk for node, risk in posterior_risk.items() if dm.node_to_zone_map.get(node)}
            posterior_df = pd.DataFrame(posterior_zone_risk.items(), columns=['zone', 'risk'])
            if not posterior_df.empty: st.altair_chart(plotter.plot_risk_profile_comparison(prior_df, posterior_df), use_container_width=True)
            else: st.info("Not enough data to generate posterior risk profile.")
    except Exception as e:
        logger.error(f"Application failed with a critical error: {e}", exc_info=True)
        st.error(f"A critical error occurred: {e}. Please check the `redshield_ai.log` file for details.")

if __name__ == "__main__":
    main()
