# RedShieldAI_Sentinel.py
# VERSION 15.0 - SENTINEL ARCHITECTURE
"""
RedShieldAI_Sentinel.py
An advanced, multi-layered emergency incident prediction and operational
intelligence application based on a fusion of stochastic processes, Bayesian
inference, network science, chaos theory, and deep learning.
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

# Advanced Dependencies
import torch
import torch.nn as nn
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- L0: CONFIGURATION & LOGGING ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages all system parameters from an external JSON file."""
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_config(config_path="config.json") -> Dict[str, Any]:
        if not Path(config_path).exists():
            st.error(f"FATAL: Configuration file not found at '{config_path}'.")
            st.stop()
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        mapbox_key = os.environ.get("MAPBOX_API_KEY", config.get("mapbox_api_key", ""))
        if not mapbox_key or mapbox_key.strip() == "" or "YOUR_KEY" in mapbox_key:
            config['mapbox_api_key'] = None
        else:
            config['mapbox_api_key'] = mapbox_key
            
        logger.info("Configuration loaded and validated.")
        return config

# --- L1: DATA & SIMULATION MODULES ---

class DataManager:
    """Manages static geospatial and network data assets."""
    def __init__(self, config: Dict[str, Any]):
        self.data_config = config.get('data', {})
        self.road_graph = self._build_road_graph()
        self.zones_gdf = self._build_zones_gdf()
        self.node_to_zone_map = {data['node']: name for name, data in self.zones_gdf.iterrows() if 'node' in data and pd.notna(data['node'])}
        logger.info("DataManager initialized.")

    @st.cache_resource
    def _build_road_graph(_self) -> nx.Graph:
        G = nx.Graph()
        network_config = _self.data_config.get('road_network', {})
        for node, data in network_config.get('nodes', {}).items(): G.add_node(node, pos=data['pos'])
        for u, v, weight in network_config.get('edges', []): G.add_edge(u, v, weight=float(weight))
        return G

    @st.cache_resource
    def _build_zones_gdf(_self) -> gpd.GeoDataFrame:
        zones = _self.data_config.get('zones', {})
        valid_zones = [{'name': name, 'geometry': Polygon([(lon, lat) for lat, lon in data['polygon']]).buffer(0), **data} for name, data in zones.items()]
        gdf = gpd.GeoDataFrame(valid_zones, crs="EPSG:4326").set_index('name')
        graph_nodes_gdf = gpd.GeoDataFrame(geometry=[Point(d['pos'][1], d['pos'][0]) for _, d in _self.road_graph.nodes(data=True)], index=list(_self.road_graph.nodes()), crs="EPSG:4326")
        nearest = gpd.sjoin_nearest(gdf, graph_nodes_gdf, how='left')
        gdf['nearest_node'] = nearest.groupby(nearest.index)['index_right'].first()
        return gdf.drop(columns=['polygon'], errors='ignore')

class SimulationEngine:
    """Generates synthetic incident data using stochastic processes."""
    def __init__(self, data_manager: DataManager, config: Dict[str, Any]):
        self.dm = data_manager
        self.sim_params = config['simulation_params']
        self.dist = config['data']['distributions']
        self.nhpp_intensity = lambda t: 1 + 0.5 * np.sin((t / 24) * 2 * np.pi)

    def _generate_random_point_in_polygon(self, polygon: Polygon) -> Point:
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            p = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(p): return p

    def get_live_state(self, env_factors: 'EnvFactors', time_hour: float, baseline_rate: float) -> Dict[str, Any]:
        intensity = self.nhpp_intensity(time_hour) * baseline_rate
        intensity *= self.sim_params['multipliers'].get(env_factors.weather.lower(), 1.0)
        intensity *= self.sim_params['multipliers']['holiday'] if env_factors.is_holiday else 1.0
        
        num_incidents = max(0, int(np.random.poisson(intensity)))
        incidents = []
        if num_incidents > 0:
            incident_zones = np.random.choice(list(self.dist['zone'].keys()), num_incidents, p=list(self.dist['zone'].values()))
            for i, zone_name in enumerate(incident_zones):
                location = self._generate_random_point_in_polygon(self.dm.zones_gdf.loc[zone_name].geometry)
                incidents.append({'id': f"INC-{int(time_hour*100)}-{i}", 'type': np.random.choice(list(self.dist['incident_type'].keys()), p=list(self.dist['incident_type'].values())), 'location': location, 'zone': zone_name, 'timestamp': time_hour})
        
        # Hawkes Process for self-excitation (aftershocks)
        hawkes_params = self.sim_params['hawkes_process']
        aftershocks = []
        for inc in incidents:
            if np.random.rand() < hawkes_params['trigger_prob']:
                num_aftershocks = np.random.poisson(hawkes_params['mu'])
                for j in range(num_aftershocks):
                    echo_loc = Point(inc['location'].x + np.random.normal(0, 0.005), inc['location'].y + np.random.normal(0, 0.005))
                    zone_gdf = self.dm.zones_gdf[self.dm.zones_gdf.geometry.contains(echo_loc)]
                    if not zone_gdf.empty:
                        aftershocks.append({'id': f"ECHO-{inc['id']}-{j}", 'type': "Aftershock", 'location': echo_loc, 'zone': zone_gdf.index[0], 'timestamp': time_hour + np.random.exponential(1.0)})

        all_incidents = incidents + aftershocks
        return {"incidents": all_incidents}

# --- L2: PREDICTIVE & STRATEGIC MODULES ---

class TCNN(nn.Module):
    """A simple Temporal Convolutional Network for time-series forecasting."""
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # Conv1d expects (batch, channels, seq_len)
        out = self.network(x)
        out = self.linear(out[:, :, -1])
        return out

class PredictiveAnalyticsEngine:
    """The cognitive core of the system, handling risk, anomaly, and forecasting."""
    def __init__(self, data_manager: DataManager, config: Dict[str, Any]):
        self.dm = data_manager
        self.model_params = config['model_params']
        self.dist = config['data']['distributions']
        self.bn = self._build_bayesian_network(config['bayesian_network'])
        self.tcnn_model, self.tcnn_input_size = self._initialize_tcnn(config['tcnn_params'])
        
    @st.cache_resource
    def _build_bayesian_network(_self, bn_config: Dict) -> BayesianNetwork:
        model = BayesianNetwork(bn_config['structure'])
        cpds = []
        for node, params in bn_config['cpds'].items():
            cpd = TabularCPD(variable=node, variable_card=params['card'], values=params['values'],
                             evidence=params.get('evidence'), evidence_card=params.get('evidence_card'))
            cpds.append(cpd)
        model.add_cpds(*cpds)
        model.check_model()
        return model

    @st.cache_resource
    def _initialize_tcnn(_self, tcnn_params: Dict) -> Tuple[TCNN, int]:
        # Synthetic training for demonstration purposes
        input_size = tcnn_params['input_size']
        model = TCNN(input_size, tcnn_params['output_size'], tcnn_params['channels'])
        # Placeholder for actual training logic on historical data
        # For now, we just return the initialized model
        logger.info("TCNN model initialized (pre-trained model would be loaded here).")
        return model, input_size

    def infer_baseline_rate(self, env_factors: 'EnvFactors') -> float:
        inference = VariableElimination(self.bn)
        evidence = {'Holiday': 1 if env_factors.is_holiday else 0, 'Weather': 0 if env_factors.weather == 'Clear' else 1}
        result = inference.query(variables=['IncidentRate'], evidence=evidence)
        # Return the expected value of the incident rate
        rate_probs = result.values
        rate_values = [1, 5, 10] # Low, Medium, High rates
        expected_rate = np.sum(np.array(rate_probs) * np.array(rate_values))
        return expected_rate

    def calculate_information_metrics(self, incidents: List[Dict]) -> Tuple[float, float]:
        if not incidents: return 0.0, 0.0
        df = pd.DataFrame(incidents)
        counts = df['zone'].value_counts(normalize=True)
        prior_dist = pd.Series(self.dist['zone'])
        # Align series for KL divergence calculation
        p, q = counts.align(prior_dist, fill_value=1e-9)
        kl_divergence = np.sum(p * np.log(p / q))
        shannon_entropy = -np.sum(p * np.log2(p))
        return kl_divergence, shannon_entropy

    def forecast_risk_over_horizons(self, historical_risk: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        # This is a conceptual implementation. In reality, you'd feed a longer history.
        # Create a sample input sequence for the TCNN
        # Shape: (1, seq_len, num_features)
        if historical_risk.empty:
            return pd.DataFrame(columns=['horizon', 'projected_risk'])
            
        # Use last N steps of history as input. Here we just use the most recent.
        latest_features = historical_risk.iloc[-1:].values
        # Pad to match expected input size if needed
        if latest_features.shape[1] < self.tcnn_input_size:
            padding = np.zeros((latest_features.shape[0], self.tcnn_input_size - latest_features.shape[1]))
            latest_features = np.hstack([latest_features, padding])
            
        input_tensor = torch.FloatTensor(latest_features).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            self.tcnn_model.eval()
            prediction = self.tcnn_model(input_tensor).numpy().flatten()
        
        forecast_df = pd.DataFrame({'horizon': horizons, 'projected_risk': prediction})
        return forecast_df

class StrategicAdvisor:
    """Optimizes resource allocation based on future risk profiles."""
    def __init__(self, data_manager: DataManager, config: Dict[str, Any]):
        self.dm = data_manager
        self.params = config['model_params']
        self.ambulances = config['data']['ambulances']

    def recommend_reallocations(self, risk_forecast: pd.DataFrame) -> Dict[int, List[Dict]]:
        # For simplicity, we optimize for the 12-hour horizon
        if risk_forecast.empty: return {}
        target_risk = risk_forecast.set_index('zone')['projected_risk']
        
        # This is a placeholder for a more complex multi-agent optimization.
        # Here we use a greedy approach on the highest-risk zone.
        if target_risk.empty: return {}
        
        highest_risk_zone = target_risk.idxmax()
        
        # Find the closest available ambulance to move
        # This simplifies the original logic for clarity
        recommendations = {
            12: [{
                "unit": "A02", # Placeholder
                "from": "Otay", # Placeholder
                "to": highest_risk_zone,
                "reason": f"Proactively cover projected 12-hour high-risk area in {highest_risk_zone}."
            }]
        }
        return recommendations

# --- L3: UI & VISUALIZATION ---

class VisualizationSuite:
    """Generates all sophisticated visualizations for the command dashboard."""
    @staticmethod
    def plot_operations_map(dm: DataManager, incidents: List[Dict], config: Dict) -> go.Figure:
        inc_df = pd.DataFrame(incidents) if incidents else pd.DataFrame()
        hosp_df = pd.DataFrame([{'lat': h['location'].y, 'lon': h['location'].x, 'name': name} for name, h in config['data']['hospitals'].items()])
        amb_df = pd.DataFrame([{'lat': a['home_base_loc'][0], 'lon': a['home_base_loc'][1], 'name': a_id} for a_id, a in config['data']['ambulances'].items()])
        fig = go.Figure()

        if not inc_df.empty: fig.add_trace(go.Scattermapbox(lat=inc_df['location'].apply(lambda p: p.y), lon=inc_df['location'].apply(lambda p: p.x), mode='markers', marker=go.scattermapbox.Marker(size=14, color='orange', symbol='circle'), text=inc_df['id'], name='Incidents', hoverinfo='text'))
        if not hosp_df.empty: fig.add_trace(go.Scattermapbox(lat=hosp_df['lat'], lon=hosp_df['lon'], mode='markers', marker=go.scattermapbox.Marker(size=18, color='blue', symbol='hospital'), text=hosp_df['name'], name='Hospitals', hoverinfo='text'))
        if not amb_df.empty: fig.add_trace(go.Scattermapbox(lat=amb_df['lat'], lon=amb_df['lon'], mode='markers', marker=go.scattermapbox.Marker(size=12, color='lime', symbol='car'), text=amb_df['name'], name='Ambulances', hoverinfo='text'))
        
        layout_args = {"mapbox": {"center": {"lat": 32.5, "lon": -117.02}, "zoom": 10.5}, "margin": {"r":0,"t":0,"l":0,"b":0}}
        mapbox_token = config.get('mapbox_api_key')
        if mapbox_token:
            layout_args["mapbox_style"] = "dark"; layout_args["mapbox_accesstoken"] = mapbox_token
        else:
            layout_args["mapbox_style"] = "carto-darkmatter"
        fig.update_layout(**layout_args)
        return fig

    @staticmethod
    def plot_risk_forecast(forecast_df: pd.DataFrame) -> alt.Chart:
        if forecast_df.empty: return alt.Chart().mark_text(text="No forecast data.").properties(title="Risk Forecast")
        return alt.Chart(forecast_df).mark_line(point=True).encode(
            x=alt.X('horizon:Q', title='Forecast Horizon (Hours)'),
            y=alt.Y('projected_risk:Q', title='Projected Risk Score'),
            tooltip=['horizon', 'projected_risk']
        ).properties(title="Multi-Horizon Risk Forecast").interactive()
        
    @staticmethod
    def plot_chaos_regime(metrics_history: pd.DataFrame) -> alt.Chart:
        if metrics_history.empty: return alt.Chart().mark_text(text="No data.").properties(title="Chaos Regime Map")
        chart = alt.Chart(metrics_history).mark_point(filled=True, opacity=0.7).encode(
            x=alt.X('anomaly_score:Q', title='Anomaly Score (KL Divergence)', scale=alt.Scale(zero=False)),
            y=alt.Y('chaos_score:Q', title='Chaos Score (Shannon Entropy)', scale=alt.Scale(zero=False)),
            color=alt.Color('time:Q', scale=alt.Scale(scheme='viridis'), title="Time"),
            tooltip=['time', 'anomaly_score', 'chaos_score']
        ).properties(
            title="Chaos Regime Map (Anomaly vs. Chaos)"
        ).interactive()
        return chart

class UIManager:
    """Manages the Streamlit UI components and application state."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        st.set_page_config(page_title="RedShield AI Sentinel", layout="wide")
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = pd.DataFrame(columns=['time', 'anomaly_score', 'chaos_score'])
        if 'risk_history' not in st.session_state:
            # We need to know the number of zones for the risk history columns
            num_zones = len(config['data']['zones'])
            st.session_state.risk_history = pd.DataFrame(columns=[f'zone_{i}' for i in range(num_zones)])

    def render_sidebar(self) -> 'EnvFactors':
        st.sidebar.title("RedShield Sentinel AI")
        st.sidebar.markdown("v15.0 - Proactive Intelligence")
        
        is_holiday = st.sidebar.checkbox("Holiday Period", value=False)
        weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Rain", "Fog"])
        
        st.sidebar.header("Simulation Control")
        run_sim = st.sidebar.button("Advance Time & Re-evaluate")
        return EnvFactors(is_holiday=is_holiday, weather=weather), run_sim

    def render_dashboard(self, kl_div: float, entropy: float, recommendations: Dict):
        st.subheader("System Status & Intelligence Briefing")
        c1, c2, c3 = st.columns(3)
        c1.metric("System Anomaly (KL Div)", f"{kl_div:.4f}", help="How much the current incident pattern deviates from the historical norm.")
        c2.metric("System Chaos (Entropy)", f"{entropy:.4f}", help="The spatial unpredictability of incidents. Higher is more scattered.")
        
        # Markov Chain for System Status
        state_matrix = self.config['markov_chain']['transition_matrix']
        # Simple logic: high anomaly pushes towards a more severe state
        current_state_idx = 0 # Assume Nominal
        if kl_div > 0.5: current_state_idx = 2 # Anomalous
        elif kl_div > 0.1: current_state_idx = 1 # Elevated
        
        status = self.config['markov_chain']['states'][current_state_idx]
        c3.metric("System State", status)
        
        if recommendations:
            st.warning("Strategic Deployment Recommendation (12h Horizon):")
            for rec in recommendations.get(12, []):
                st.write(f"**Move Unit `{rec['unit']}`** from `{rec['from']}` to `{rec['to']}`. **Reason:** {rec['reason']}")
        else:
            st.success("Current resource deployment is optimal for the forecast horizon.")

    def render_analytics(self, forecast_df: pd.DataFrame):
        st.subheader("Predictive Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(VisualizationSuite.plot_risk_forecast(forecast_df), use_container_width=True)
        with col2:
            st.altair_chart(VisualizationSuite.plot_chaos_regime(st.session_state.metrics_history), use_container_width=True)

# --- L4: APPLICATION ENTRYPOINT ---

def main():
    st.title("RedShield AI: Sentinel Command Suite")
    
    # Initialization
    config = ConfigurationManager.get_config()
    dm = DataManager(config)
    predictor = PredictiveAnalyticsEngine(dm, config)
    sim_engine = SimulationEngine(dm, config)
    advisor = StrategicAdvisor(dm, config)
    ui_manager = UIManager(config)

    # UI Controls
    env_factors, run_simulation = ui_manager.render_sidebar()
    
    if run_simulation:
        # 1. Infer baseline from Bayesian Network
        baseline_rate = predictor.infer_baseline_rate(env_factors)
        
        # 2. Simulate the next time step
        current_time = len(st.session_state.metrics_history)
        live_state = sim_engine.get_live_state(env_factors, current_time, baseline_rate)
        
        # 3. Analyze the current state
        kl_div, entropy = predictor.calculate_information_metrics(live_state['incidents'])
        
        # 4. Update and store historical data for models
        new_metrics = pd.DataFrame([{'time': current_time, 'anomaly_score': kl_div, 'chaos_score': entropy}])
        st.session_state.metrics_history = pd.concat([st.session_state.metrics_history, new_metrics], ignore_index=True)
        
        # Create a placeholder for current risk to append to history
        # In a real app, this would be a calculated risk vector
        num_zones = len(config['data']['zones'])
        current_risk_placeholder = pd.DataFrame([np.random.rand(num_zones)], columns=[f'zone_{i}' for i in range(num_zones)])
        st.session_state.risk_history = pd.concat([st.session_state.risk_history, current_risk_placeholder], ignore_index=True)

        # 5. Generate Forecasts
        horizons = config['tcnn_params']['horizons']
        # The TCNN needs a sequence of historical data. We use the risk history.
        forecast_df = predictor.forecast_risk_over_horizons(st.session_state.risk_history, horizons)
        
        # 6. Generate Strategic Recommendations
        recommendations = advisor.recommend_reallocations(forecast_df)
        
        # 7. Render the UI
        ui_manager.render_dashboard(kl_div, entropy, recommendations)
        st.plotly_chart(VisualizationSuite.plot_operations_map(dm, live_state['incidents'], config), use_container_width=True)
        ui_manager.render_analytics(forecast_df)
    else:
        st.info("Press 'Advance Time & Re-evaluate' in the sidebar to run the simulation.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please check the logs. Error: {e}")
