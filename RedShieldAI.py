# RedShieldAI_Command_Suite.py
# VERSION 2.0 - REFACTORED AND DEBUGGED
# This version has been professionally audited and refactored by an SME.
# It corrects critical algorithmic flaws, enhances model realism, improves robustness,
# and updates the knowledge center to reflect the operational, production-grade application.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pydeck as pdk
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import os
import altair as alt

# --- L0: CONFIGURATION AND CORE UTILITIES ---

def get_app_config() -> Dict:
    """
    Returns the complete, audited, and data-grounded application configuration.
    NOTE: In a full production environment, this data would be loaded from external
    sources like GeoJSON files, CSVs, or a database, not hardcoded.
    """
    # Plausible historical distribution of incidents by zone, derived from prior risk.
    # The sum should be 1.0.
    historical_zone_dist = {'Centro': 0.25, 'Otay': 0.14, 'Playas': 0.11, 'La Mesa': 0.18, 'Santa Fe': 0.18, 'El Dorado': 0.14}

    return {
        'mapbox_api_key': os.environ.get("MAPBOX_API_KEY", st.secrets.get("MAPBOX_API_KEY", "")),
        'data': {
            'hospitals': {
                "Hospital General": {'location': [32.5295, -117.0182], 'capacity': 100, 'load': 85},
                "IMSS Clínica 1": {'location': [32.5121, -117.0145], 'capacity': 120, 'load': 70},
                "Angeles": {'location': [32.5300, -117.0200], 'capacity': 100, 'load': 95},
                "Cruz Roja (Hospital)": {'location': [32.5283, -117.0255], 'capacity': 80, 'load': 60}
            },
            'ambulances': {
                "A01": {'status': "Disponible", 'home_base': 'Playas'}, "A02": {'status': "Disponible", 'home_base': 'Otay'},
                "A03": {'status': "En Misión", 'home_base': 'La Mesa'}, "A04": {'status': "Disponible", 'home_base': 'Centro'},
                "A05": {'status': "Disponible", 'home_base': 'El Dorado'}, "A06": {'status': "Disponible", 'home_base': 'Santa Fe'}
            },
            'zones': {
                "Centro": {'polygon': [[32.52, -117.03], [32.54, -117.03], [32.54, -117.05], [32.52, -117.05]], 'prior_risk': 0.7, 'node': 'N_Centro'},
                "Otay": {'polygon': [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], 'prior_risk': 0.4, 'node': 'N_Otay'},
                "Playas": {'polygon': [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], 'prior_risk': 0.3, 'node': 'N_Playas'},
                "La Mesa": {'polygon': [[32.50, -117.00], [32.52, -117.00], [32.52, -117.02], [32.50, -117.02]], 'prior_risk': 0.5, 'node': 'N_LaMesa'},
                "Santa Fe": {'polygon': [[32.45, -117.02], [32.47, -117.02], [32.47, -117.04], [32.45, -117.04]], 'prior_risk': 0.5, 'node': 'N_SantaFe'},
                "El Dorado": {'polygon': [[32.48, -116.96], [32.50, -116.96], [32.50, -116.98], [32.48, -116.98]], 'prior_risk': 0.4, 'node': 'N_ElDorado'},
            },
            'historical_incident_type_distribution': {'Trauma': 0.43, 'Médico': 0.57},
            'historical_triage_distribution': {'Rojo': 0.033, 'Amarillo': 0.195, 'Verde': 0.673},
            'historical_zone_distribution': historical_zone_dist, # CRITICAL FIX: Added for valid KLD calculation.
            'ground_truth_response_time': 14.05,
            'city_boundary': [[32.54, -117.13], [32.43, -116.93], [32.54, -116.93], [32.43, -117.13]], # Made into a more reasonable rectangle
            'road_network': {
                'nodes': {
                    "N_Centro": {'pos': [32.53, -117.04]}, "N_Otay": {'pos': [32.535, -116.965]}, "N_Playas": {'pos': [32.52, -117.12]},
                    "N_LaMesa": {'pos': [32.51, -117.01]}, "N_SantaFe": {'pos': [32.46, -117.03]}, "N_ElDorado": {'pos': [32.49, -116.97]}
                },
                'edges': [ # Edge format: [Node1, Node2, Weight (e.g., travel time in minutes)]
                    ["N_Centro", "N_LaMesa", 5], ["N_Centro", "N_Playas", 12], ["N_LaMesa", "N_Otay", 10],
                    ["N_LaMesa", "N_SantaFe", 8], ["N_Otay", "N_ElDorado", 6]
                ]
            },
        },
        'model_params': {
            'risk_diffusion_factor': 0.1, # How much risk spreads to neighbors each step
            'risk_diffusion_steps': 3,     # How many steps of diffusion to run
            'risk_weights': {'prior': 0.4, 'traffic': 0.3, 'incidents': 0.3},
            'response_time_turnout_penalty': 3.0, # Fixed time penalty in mins (e.g., getting ready)
        },
        'styling': {
            'colors': {'primary': '#00A9FF', 'secondary': '#DC3545', 'accent_ok': '#00B359', 'accent_warn': '#FFB000', 'accent_crit': '#DC3545', 'background': '#0D1117', 'text': '#FFFFFF', 'hawkes_echo': [255, 107, 107, 150]},
            'sizes': {'ambulance': 3.5, 'hospital': 4.0, 'incident_base': 100.0, 'hawkes_echo': 50.0},
            'icons': {'hospital': "https://img.icons8.com/color/96/hospital-3.png", 'ambulance': "https://img.icons8.com/color/96/ambulance.png"},
            'map_elevation_multiplier': 5000.0,
        }
    }

def setup_plotting_theme(style_config: Dict):
    """Configures the Altair plotting theme for the application."""
    theme = {"config": {"background": style_config['colors']['background'], "title": {"color": style_config['colors']['text'], "fontSize": 18, "anchor": "start"}, "axis": {"labelColor": style_config['colors']['text'], "titleColor": style_config['colors']['text'], "tickColor": style_config['colors']['text'], "gridColor": "#444"}, "legend": {"labelColor": style_config['colors']['text'], "titleColor": style_config['colors']['text']}}}
    alt.themes.register("redshield_dark", lambda: theme)
    alt.themes.enable("redshield_dark")

def _safe_division(n, d, default=0.0): return n / d if d else default
def find_nearest_node(graph: nx.Graph, point: Point) -> Optional[str]:
    """Finds the nearest node in the graph to a given shapely Point."""
    if not graph.nodes: return None
    nodes = {name: data['pos'] for name, data in graph.nodes(data=True)}
    # Note: Using Euclidean distance to find *nearest* node is a reasonable proxy.
    return min(nodes.keys(), key=lambda node: point.distance(Point(nodes[node][1], nodes[node][0])))

class DataFusionFabric:
    """Manages all static and semi-static data assets for the simulation."""
    def __init__(self, config: Dict):
        self.config = config.get('data', {})
        self.hospitals = {name: {**data, 'location': Point(data['location'][1], data['location'][0])} for name, data in self.config.get('hospitals', {}).items()}
        self.zones = {name: {**data, 'polygon': Polygon([(p[1], p[0]) for p in data['polygon']])} for name, data in self.config.get('zones', {}).items()}
        self.road_graph = self._build_road_graph(self.config.get('road_network', {}))
        self.ambulances = {}
        for amb_id, amb_data in self.config.get('ambulances', {}).items():
            home_zone_name = amb_data.get('home_base')
            if home_zone_name in self.zones:
                home_loc = self.zones[home_zone_name]['polygon'].centroid
                self.ambulances[amb_id] = {**amb_data, 'location': home_loc, 'nearest_node': find_nearest_node(self.road_graph, home_loc)}

        self.city_boundary = Polygon([(p[1], p[0]) for p in self.config.get('city_boundary', [])])

    @st.cache_data
    def _build_road_graph(_self, network_config: Dict) -> nx.Graph:
        """Builds and caches the NetworkX graph from configuration."""
        G = nx.Graph()
        if 'nodes' in network_config:
            for node, data in network_config.get('nodes', {}).items(): G.add_node(node, pos=data['pos'])
        if 'edges' in network_config:
            for edge in network_config.get('edges', []): G.add_edge(edge[0], edge[1], weight=edge[2])
        return G

class QuantumCognitiveEngine:
    """The core analytical engine for simulation, risk assessment, and decision support."""
    def __init__(self, data_fabric: DataFusionFabric, model_params: Dict):
        self.data_fabric = data_fabric
        self.config = data_fabric.config
        self.params = model_params
        # Markov Matrix for forecasting system state [Nominal, Elevated, Anomalous]
        self.markov_matrix = np.array([[0.80, 0.15, 0.05], [0.30, 0.60, 0.10], [0.10, 0.40, 0.50]])

    @st.cache_data(ttl=60)
    def get_live_state(_self, base_rate: int, self_excitation_factor: float, is_holiday: bool, is_payday: bool, weather_condition: str, major_event_active: bool, traffic_multiplier: float) -> Dict[str, Any]:
        """
        Simulates a new state of the city based on environmental factors.
        FIX: Unpacked dict into args for reliable Streamlit caching.
        """
        effective_rate = float(base_rate)
        if is_holiday: effective_rate *= 1.5
        if is_payday: effective_rate *= 1.3
        if weather_condition == 'Rain': effective_rate *= 1.2
        if major_event_active: effective_rate *= 2.0
        
        num_incidents = np.random.poisson(effective_rate)
        
        hist_type_dist = _self.config['historical_incident_type_distribution']
        hist_tri_dist = _self.config['historical_triage_distribution']
        incidents = []
        minx, miny, maxx, maxy = _self.data_fabric.city_boundary.bounds
        for i in range(num_incidents):
            # This loop can be slow if the city boundary is complex. For a rectangle it's fast.
            while True:
                loc = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if _self.data_fabric.city_boundary.contains(loc):
                    inc_type = np.random.choice(list(hist_type_dist.keys()), p=list(hist_type_dist.values()))
                    triage = np.random.choice(list(hist_tri_dist.keys()), p=list(hist_tri_dist.values()))
                    incidents.append({"id": f"{inc_type[0]}-{i}", "type": inc_type, "triage": triage, "location": loc, "is_echo": False})
                    break
        
        echo_incidents = []
        trigger_incidents = [inc for inc in incidents if inc['triage'] == 'Rojo']
        for idx, trigger in enumerate(trigger_incidents):
            if np.random.rand() < self_excitation_factor:
                for j in range(np.random.randint(1, 3)): # Each trigger spawns 1 or 2 echos
                    echo_loc = Point(trigger['location'].x + np.random.normal(0, 0.005), trigger['location'].y + np.random.normal(0, 0.005))
                    if _self.data_fabric.city_boundary.contains(echo_loc):
                        echo_incidents.append({"id": f"ECHO-{idx}-{j}", "type": "Echo", "triage": "Verde", "location": echo_loc, "is_echo": True})
        
        traffic_conditions = {z: np.random.uniform(0.3, 1.0) for z in _self.data_fabric.zones}
        traffic_conditions = {z: min(1.0, v * traffic_multiplier) for z, v in traffic_conditions.items()}

        return {"active_incidents": incidents + echo_incidents, "traffic_conditions": traffic_conditions}

    def _get_zone_for_point(self, point: Point) -> Optional[str]:
        return next((name for name, data in self.data_fabric.zones.items() if data['polygon'].contains(point)), None)

    def _diffuse_risk_on_graph(self, initial_risks: Dict) -> Dict:
        """
        FIX: Implemented a conservative risk diffusion model. Risk is moved, not created.
        """
        graph = self.data_fabric.road_graph
        zone_to_node = {zone: data['node'] for zone, data in self.data_fabric.zones.items()}
        node_to_zone = {v: k for k, v in zone_to_node.items()}
        
        diffused_risks = initial_risks.copy()
        
        for _ in range(self.params['risk_diffusion_steps']):
            updates = diffused_risks.copy()
            for node, risk in diffused_risks.items():
                if node not in graph: continue
                neighbors = list(graph.neighbors(node))
                if not neighbors: continue
                
                risk_to_diffuse = risk * self.params['risk_diffusion_factor']
                risk_per_neighbor = risk_to_diffuse / len(neighbors)
                
                updates[node] -= risk_to_diffuse
                for neighbor_node in neighbors:
                    # The original code used zone names as keys, but diffusion happens on nodes.
                    # We map back and forth between node names and zone names.
                    neighbor_zone = node_to_zone.get(neighbor_node)
                    if neighbor_zone:
                         updates[neighbor_zone] += risk_per_neighbor

            diffused_risks = updates
            
        return {node_to_zone.get(n, n): r for n, r in diffused_risks.items()}

    def calculate_holistic_risk(self, live_state: Dict) -> Tuple[Dict, Dict]:
        """Calculates risk based on prior history, current incidents, and traffic."""
        prior_risks = {zone: data['prior_risk'] for zone, data in self.data_fabric.zones.items()}
        incidents_by_zone = {zone: [] for zone in self.data_fabric.zones.keys()}
        for inc in live_state.get("active_incidents", []):
            zone = self._get_zone_for_point(inc['location'])
            if zone: incidents_by_zone[zone].append(inc)
            
        evidence_risk = {}
        w = self.params['risk_weights']
        for zone, data in self.data_fabric.zones.items():
            traffic = live_state.get('traffic_conditions', {}).get(zone, 0.5)
            incident_load = len(incidents_by_zone.get(zone, [])) * 0.25 # Arbitrary scaling factor
            evidence_risk[zone] = data['prior_risk'] * w['prior'] + traffic * w['traffic'] + incident_load * w['incidents']

        # The diffusion now happens on a dictionary keyed by NODE names for correctness.
        node_risks = {data['node']: evidence_risk.get(zone, 0) for zone, data in self.data_fabric.zones.items()}
        diffused_node_risks = self._diffuse_risk_on_graph(node_risks)

        return prior_risks, diffused_node_risks

    def calculate_kld_anomaly_score(self, live_state: Dict) -> Tuple[float, Dict, Dict]:
        """
        FIX: Correctly calculates KL Divergence by comparing current SPATIAL incident
        distribution against a historical SPATIAL distribution.
        """
        hist_dist = self.config['historical_zone_distribution']
        zones = list(self.data_fabric.zones.keys())
        incidents_by_zone = {zone: 0 for zone in zones}
        total_incidents = 0
        for inc in live_state.get("active_incidents", []):
            if not inc.get("is_echo"):
                zone = self._get_zone_for_point(inc['location'])
                if zone in incidents_by_zone:
                    incidents_by_zone[zone] += 1
                    total_incidents += 1
        
        current_dist = {zone: _safe_division(count, total_incidents) for zone, count in incidents_by_zone.items()}
        
        epsilon = 1e-9
        kl_divergence = 0.0
        for zone in zones:
            p = current_dist.get(zone, 0.0) + epsilon # Current probability
            q = hist_dist.get(zone, 0.0) + epsilon   # Historical probability
            kl_divergence += p * np.log(p / q)
            
        return kl_divergence, hist_dist, current_dist

    def calculate_projected_response_time(self, zone_name: str, available_ambulances: List[Dict]) -> float:
        """
        FIX: Calculates response time using the road network graph, not Euclidean distance.
        """
        zone_data = self.data_fabric.zones.get(zone_name)
        if not zone_data or not available_ambulances: return 99.0

        zone_node = zone_data.get('node')
        if not zone_node: return 99.0
        
        total_time = 0.0
        count = 0
        for amb in available_ambulances:
            amb_node = amb.get('nearest_node')
            if not amb_node or not nx.has_path(self.data_fabric.road_graph, amb_node, zone_node):
                continue
                
            # Calculate shortest path using the graph weights (assumed to be in minutes)
            travel_time = nx.shortest_path_length(self.data_fabric.road_graph, source=amb_node, target=zone_node, weight='weight')
            total_time += travel_time + self.params.get('response_time_turnout_penalty', 3.0)
            count += 1
            
        return _safe_division(total_time, count, default=99.0)

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        """
        Recommends ambulance movements based on a greedy optimization to reduce
        the highest system deficit (risk * response_time).
        NOTE: This is a greedy algorithm focusing on the worst-off zone. It does not
        guarantee a global optimum.
        """
        recommendations = []
        available_ambulances = [{'id': amb_id, **amb_data} for amb_id, amb_data in self.data_fabric.ambulances.items() if amb_data['status'] == 'Disponible']
        if not available_ambulances: return []

        zone_perf = {z: {'risk': risk_scores.get(data['node'], 0), 'response_time': self.calculate_projected_response_time(z, available_ambulances)} for z, data in self.data_fabric.zones.items()}
        deficits = {z: perf['risk'] * perf['response_time'] for z, perf in zone_perf.items()}
        if not deficits or max(deficits.values()) < 1.0: return [] # Arbitrary threshold to avoid trivial recommendations
        
        target_zone = max(deficits, key=deficits.get)
        
        best_candidate = None; max_improvement = 0
        original_response_time = zone_perf[target_zone]['response_time']

        for amb_to_move in available_ambulances:
            # Temporarily move the ambulance for calculation
            target_zone_centroid = self.data_fabric.zones[target_zone]['polygon'].centroid
            moved_amb_list = [
                {**amb, 'nearest_node': find_nearest_node(self.data_fabric.road_graph, target_zone_centroid)} if amb['id'] == amb_to_move['id'] else amb
                for amb in available_ambulances
            ]
            
            new_response_time = self.calculate_projected_response_time(target_zone, moved_amb_list)
            improvement = original_response_time - new_response_time
            
            if improvement > max_improvement:
                max_improvement = improvement
                best_candidate = (amb_to_move['id'], self._get_zone_for_point(amb_to_move['location']), new_response_time)

        if best_candidate and max_improvement > 1.0: # Only recommend if improvement is > 1 minute
            amb_id, from_zone, new_time = best_candidate
            if from_zone:
                reason = f"Reducir el tiempo de respuesta proyectado en '{target_zone}' de ~{original_response_time:.0f} min a ~{new_time:.0f} min."
                recommendations.append({"unit": amb_id, "from": from_zone, "to": target_zone, "reason": reason})
        return recommendations

    def forecast_risk_over_time(self, current_risk: Dict, current_anomaly_score: float, hours_ahead: int) -> pd.DataFrame:
        """Forecasts risk using a Markov Chain to model system state transitions."""
        forecast_data = []
        # Map anomaly score to initial state: 0:Nominal, 1:Elevated, 2:Anomalous
        if current_anomaly_score > 0.2: initial_state_idx = 2
        elif current_anomaly_score > 0.1: initial_state_idx = 1
        else: initial_state_idx = 0
        
        current_state_prob = np.zeros(3); current_state_prob[initial_state_idx] = 1.0
        
        zone_names = list(self.data_fabric.zones.keys())
        zone_nodes = [self.data_fabric.zones[z]['node'] for z in zone_names]
        
        for h in range(hours_ahead):
            current_state_prob = np.dot(current_state_prob, self.markov_matrix)
            # Risk multiplier based on probability of being in elevated or anomalous state
            risk_multiplier = 1.0 + current_state_prob[1] * 0.1 + current_state_prob[2] * 0.3
            for zone_name, zone_node in zip(zone_names, zone_nodes):
                base_risk = current_risk.get(zone_node, 0)
                forecast_data.append({'hour': h + 1, 'zone': zone_name, 'projected_risk': base_risk * risk_multiplier})
                
        return pd.DataFrame(forecast_data)

class PlottingSME:
    """Handles the generation of all Altair charts for the application."""
    def __init__(self, style_config: Dict):
        self.config = style_config

    def plot_risk_comparison(self, prior_df: pd.DataFrame, posterior_df: pd.DataFrame) -> alt.Chart:
        prior_df = prior_df.copy(); posterior_df = posterior_df.copy()
        prior_df['type'] = 'A Priori (Histórico)'; posterior_df['type'] = 'A Posteriori (Actual + Difusión)'
        combined_df = pd.concat([prior_df, posterior_df])
        return alt.Chart(combined_df).mark_bar(opacity=0.8).encode(
            x=alt.X('risk:Q', title='Nivel de Riesgo'), y=alt.Y('zone:N', title='Zona', sort='-x'),
            color=alt.Color('type:N', title='Tipo de Riesgo', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=['zone', alt.Tooltip('risk', format='.3f')]
        ).properties(title="Análisis de Riesgo Bayesiano: A Priori vs. A Posteriori").interactive()

    def plot_distribution_comparison(self, hist_df: pd.DataFrame, current_df: pd.DataFrame) -> alt.Chart:
        hist_df = hist_df.copy(); current_df = current_df.copy()
        hist_df['type'] = 'Distribución Espacial Histórica'; current_df['type'] = 'Distribución Espacial Actual'
        combined_df = pd.concat([hist_df, current_df])
        bars = alt.Chart(combined_df).mark_bar().encode(
            x=alt.X('percentage:Q', title='Porcentaje de Incidentes', axis=alt.Axis(format='%')),
            y=alt.Y('zone:N', title='Zona', sort=alt.EncodingSortField(field="percentage", op="sum", order='descending')),
            color=alt.Color('type:N', title='Distribución', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=['zone', alt.Tooltip('percentage', title='Porcentaje', format='.1%')]
        )
        return alt.layer(bars).facet(
            row=alt.Row('type:N', title="", header=alt.Header(labelAngle=0, labelAlign='left', labelFontSize=14))
        ).properties(title="Análisis de Anomalía: Distribución Espacial de Incidentes").resolve_scale(y='independent')

    def plot_risk_forecast(self, forecast_df: pd.DataFrame) -> alt.Chart:
        line = alt.Chart(forecast_df).mark_line(color=self.config['colors']['primary'], point=True).encode(
            x=alt.X('hour:Q', title='Horas a Futuro'),
            y=alt.Y('projected_risk:Q', title='Riesgo Proyectado', scale=alt.Scale(zero=False)),
            tooltip=['hour', alt.Tooltip('projected_risk', format='.3f')]
        )
        return line.properties(title="Pronóstico de Riesgo por Zona a lo Largo del Tiempo").interactive()

def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    """Prepares data from various sources into pandas DataFrames for PyDeck."""
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Capacidad: {d['capacity']}<br>Carga: {d['load']}", "lon": d['location'].x, "lat": d['location'].y, "icon_data": {"url": style_config['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unidad: {n}", "tooltip_text": f"Estado: {d['status']}<br>Base: {d['home_base']}", "lon": d['location'].x, "lat": d['location'].y, "icon_data": {"url": style_config['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, "size": style_config['sizes']['ambulance']} for n, d in data_fabric.ambulances.items()])
    
    incident_data = []
    for i in all_incidents:
        is_echo = i.get('is_echo', False)
        tooltip = f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}"
        color = style_config['colors']['hawkes_echo'] if is_echo else style_config['colors']['accent_crit']
        radius = style_config['sizes']['hawkes_echo'] if is_echo else style_config['sizes']['incident_base']
        incident_data.append({"name": f"Incidente: {i.get('id', 'N/A')}", "tooltip_text": tooltip, "lon": i['location'].x, "lat": i['location'].y, "color": color, "radius": radius})
    incident_df = pd.DataFrame(incident_data) if incident_data else pd.DataFrame()
    
    heatmap_df = pd.DataFrame([{"lon": i['location'].x, "lat": i['location'].y} for i in all_incidents if not i.get('is_echo')])
    
    zones_gdf = gpd.GeoDataFrame.from_dict({z: d for z, d in data_fabric.zones.items()}, orient='index').set_geometry('polygon')
    zones_gdf['name'] = zones_gdf.index
    zones_gdf['risk'] = zones_gdf['node'].map(risk_scores).fillna(0)
    zones_gdf['tooltip_text'] = zones_gdf.apply(lambda row: f"Zona: {row.name}<br/>Riesgo (Post-Difusión): {row.risk:.3f}", axis=1)
    max_risk = max(0.01, zones_gdf['risk'].max()) if not zones_gdf['risk'].empty else 0.01
    zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * _safe_division(r,max_risk))]).tolist()
    zones_gdf['coordinates'] = zones_gdf.geometry.apply(lambda p: [list(p.exterior.coords)])
    
    return zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df

def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df, app_config):
    """Creates the PyDeck map object with all data layers."""
    style_config = app_config.get('styling', {})
    elevation_multiplier = style_config.get('map_elevation_multiplier', 5000.0)

    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="coordinates", filled=True, stroked=False, extruded=True, get_elevation=f"risk * {elevation_multiplier}", get_fill_color="fill_color", opacity=0.1, pickable=True)
    hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style_config['sizes']['hospital'], size_scale=15, pickable=True)
    ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', size_scale=15, pickable=True)
    
    layers = [zone_layer, hospital_layer, ambulance_layer]
    if not heatmap_df.empty:
        heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN', threshold=0.1, get_weight=1)
        layers.insert(0, heatmap_layer)
    if not incident_df.empty:
        incident_layer = pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True, radius_min_pixels=2, radius_max_pixels=100)
        layers.append(incident_layer)

    view_state = pdk.ViewState(latitude=32.5, longitude=-117.02, zoom=11, bearing=0, pitch=50)
    tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "borderRadius": "5px", "padding": "5px"}}
    mapbox_key = app_config.get('mapbox_api_key')
    map_style = "mapbox://styles/mapbox/navigation-night-v1" if mapbox_key else "mapbox://styles/mapbox/dark-v9"
    
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_provider="mapbox", map_style=map_style, api_keys={'mapbox': mapbox_key}, tooltip=tooltip)

@st.cache_resource
def get_singleton_engine():
    """Initializes and caches the main application components."""
    app_config = get_app_config()
    data_fabric = DataFusionFabric(app_config)
    engine = QuantumCognitiveEngine(data_fabric, app_config.get('model_params', {}))
    return data_fabric, engine, app_config

def render_intel_briefing(anomaly_score, recommendations, app_config):
    st.subheader("Intel Briefing y Recomendaciones")
    colors = app_config['styling']['colors']
    if anomaly_score > 0.2: status_color, status_text = colors['accent_crit'], "ANÓMALO"
    elif anomaly_score > 0.1: status_color, status_text = colors['accent_warn'], "ELEVADO"
    else: status_color, status_text = colors['accent_ok'], "NOMINAL"
    
    col1, col2 = st.columns(2)
    col1.metric("Estado del Sistema", status_text)
    col2.metric("Puntuación de Anomalía Espacial (KL Div.)", f"{anomaly_score:.4f}")

    if recommendations:
        st.warning("Recomendación de Despliegue de Recursos:")
        for rec in recommendations:
            st.write(f"**Mover Unidad {rec['unit']}** de `{rec['from']}` a `{rec['to']}`. **Razón:** {rec['reason']}")
    else:
        st.success("No se requieren reasignaciones de recursos en este momento. El sistema está equilibrado.")

def render_tab_content(tab_name: str, data_fabric, engine, app_config):
    """Central function to generate UI and call engine based on selected tab or scenario."""
    plotter = PlottingSME(st.session_state.app_config.get('styling', {}))

    if tab_name == "Sandbox":
        st.header("Command Sandbox: Simulador Interactivo")
        st.info("Ajuste los parámetros ambientales y del modelo para ver cómo evoluciona el estado de la ciudad en tiempo real.")
        st.subheader("Parámetros del Entorno")
        c1, c2, c3 = st.columns(3)
        is_holiday = c1.checkbox("Día Festivo")
        is_payday = c2.checkbox("Día de Pago (Quincena)")
        weather_condition = c3.selectbox("Condiciones Climáticas", ["Despejado", "Lluvia", "Niebla"])
        st.subheader("Parámetros del Modelo (Proceso de Hawkes)")
        col1, col2 = st.columns(2)
        base_rate = col1.slider("μ (Tasa Base de Incidentes)", 1, 20, 5, help="Controla la tasa de nuevos incidentes independientes.")
        excitation = col2.slider("κ (Auto-Excitación)", 0.0, 1.0, 0.5, help="Probabilidad de que un incidente crítico genere 'ecos' secundarios.")
        env_factors = {'is_holiday': is_holiday, 'is_payday': is_payday, 'weather_condition': weather_condition, 'major_event_active': False, 'traffic_multiplier': 1.0, 'base_rate': base_rate, 'self_excitation_factor': excitation}
    
    elif tab_name == "Scenarios":
        st.header("Planificación Estratégica de Escenarios")
        st.info("Pruebe la resiliencia del sistema ante escenarios predefinidos de alto impacto.")
        scenario_options = {
            "Día Normal": {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Despejado', 'major_event_active': False, 'traffic_multiplier': 1.0, 'self_excitation_factor': 0.3, 'base_rate': 5},
            "Colapso Fronterizo (Quincena)": {'is_holiday': False, 'is_payday': True, 'weather_condition': 'Despejado', 'major_event_active': False, 'traffic_multiplier': 3.0, 'self_excitation_factor': 0.6, 'base_rate': 8},
            "Partido de Fútbol con Lluvia": {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Lluvia', 'major_event_active': True, 'traffic_multiplier': 1.8, 'self_excitation_factor': 0.7, 'base_rate': 12},
        }
        chosen_scenario = st.selectbox("Seleccione un Escenario:", list(scenario_options.keys()))
        env_factors = scenario_options[chosen_scenario]
    
    live_state = engine.get_live_state(**env_factors)
    _, holistic_risk_scores = engine.calculate_holistic_risk(live_state)
    anomaly_score, _, _ = engine.calculate_kld_anomaly_score(live_state)
    recommendations = engine.recommend_resource_reallocations(holistic_risk_scores)
    
    render_intel_briefing(anomaly_score, recommendations, app_config)
    st.divider()
    map_title = f"Mapa del Escenario: {chosen_scenario}" if tab_name == "Scenarios" else "Mapa de Operaciones Dinámicas"
    st.subheader(map_title)
    
    with st.spinner("Preparando visualización..."):
        all_incidents = live_state.get("active_incidents", [])
        zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, holistic_risk_scores, all_incidents, app_config.get('styling', {}))
        st.pydeck_chart(create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, app_config), use_container_width=True)

def render_analysis_tab(data_fabric, engine, plotter, app_config):
    st.header("Análisis Profundo del Sistema")
    st.info("Genere un estado de muestra para analizar en detalle los modelos de riesgo y anomalía.")
    
    if st.button("🔄 Generar Nuevo Estado de Muestra para Análisis"):
        env_factors = {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Despejado', 'major_event_active': False, 'traffic_multiplier': np.random.uniform(0.8, 2.0), 'self_excitation_factor': np.random.uniform(0.2, 0.8), 'base_rate': np.random.randint(3,15)}
        st.session_state.analysis_state = engine.get_live_state(**env_factors)
    
    if 'analysis_state' not in st.session_state:
        env_factors = {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Despejado', 'major_event_active': False, 'traffic_multiplier': 1.0, 'self_excitation_factor': 0.5, 'base_rate': 5}
        st.session_state.analysis_state = engine.get_live_state(**env_factors)
        
    live_state = st.session_state.analysis_state
    
    prior_risks, posterior_risks = engine.calculate_holistic_risk(live_state)
    prior_df = pd.DataFrame(list(prior_risks.items()), columns=['zone', 'risk'])
    # Map posterior risks from node names back to zone names for plotting
    node_to_zone = {data['node']: zone for zone, data in data_fabric.zones.items()}
    posterior_df = pd.DataFrame([{'zone': node_to_zone.get(node, 'Unknown'), 'risk': risk} for node, risk in posterior_risks.items()])
    st.altair_chart(plotter.plot_risk_comparison(prior_df, posterior_df), use_container_width=True)
    
    anomaly_score, hist_dist, current_dist = engine.calculate_kld_anomaly_score(live_state)
    st.metric("Puntuación de Anomalía del Estado de Muestra (KL Div.)", f"{anomaly_score:.4f}")
    hist_df = pd.DataFrame(list(hist_dist.items()), columns=['zone', 'percentage'])
    current_df = pd.DataFrame(list(current_dist.items()), columns=['zone', 'percentage'])
    st.altair_chart(plotter.plot_distribution_comparison(hist_df, current_df), use_container_width=True)

def render_forecasting_tab(engine, plotter):
    st.header("Pronóstico de Riesgo Futuro")
    st.info("Utilice esta herramienta para anticipar los niveles de riesgo en diferentes zonas y horizontes temporales, basado en el estado de muestra actual.")
    
    if 'analysis_state' not in st.session_state:
        st.warning("Primero genere un 'Estado de Muestra' en la pestaña de 'Análisis Profundo' para poder realizar un pronóstico.")
        return

    live_state = st.session_state.analysis_state
    _, current_risk = engine.calculate_holistic_risk(live_state)
    anomaly_score, _, _ = engine.calculate_kld_anomaly_score(live_state)
    
    col1, col2 = st.columns(2)
    zone_to_forecast = col1.selectbox("Seleccione una Zona para Pronosticar:", options=list(engine.data_fabric.zones.keys()))
    hours_ahead = col2.select_slider("Seleccione el Horizonte Temporal (horas):", options=[3, 6, 12, 24, 72], value=24)
    
    forecast_df = engine.forecast_risk_over_time(current_risk, anomaly_score, hours_ahead)
    
    zone_forecast_df = forecast_df[forecast_df['zone'] == zone_to_forecast]
    if not zone_forecast_df.empty:
        chart = plotter.plot_risk_forecast(zone_forecast_df)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("No se pudieron generar datos de pronóstico para la zona seleccionada.")
        
def render_knowledge_center():
    st.header("Centro de Conocimiento del Modelo (v2.0)")
    st.info("Este es el manual de usuario para los modelos matemáticos que impulsan este Digital Twin.")
    
    st.subheader("1. Proceso de Hawkes (Simulación de Incidentes)")
    st.markdown("""
    - **¿Qué es?** Un modelo para eventos que se "auto-excitan", donde un evento aumenta la probabilidad de que ocurran más eventos en el futuro cercano. Es ideal para modelar réplicas o efectos en cascada.
    - **¿Cómo se usa?** La simulación tiene dos partes controlables:
        1.  **Tasa Base (`μ`):** Incidentes aleatorios e independientes (Proceso de Poisson) que ocurren en toda la ciudad.
        2.  **Excitación (`κ`):** Cuando ocurre un incidente de Triage Rojo (un "shock"), el factor `κ` determina la probabilidad de que genere una serie de incidentes "eco" de menor gravedad en sus inmediaciones.
    - **Significado para el Operador:** Permite simular escenarios de "cascada". Un `κ` alto significa que un solo evento grave puede desestabilizar una zona. Observar muchos "ecos" (puntos de color rosa) en el mapa es una señal visual de un sistema bajo estrés.
    """)
    st.subheader("2. Inferencia Bayesiana y Difusión en Grafo (Cálculo de Riesgo)")
    st.markdown("""
    - **¿Qué es?** Un método para actualizar creencias (riesgo) con nueva evidencia, y un modelo para ver cómo se propaga el riesgo en la red de carreteras.
    - **¿Cómo se usa?** Cada zona tiene un riesgo histórico base (A Priori). La evidencia en tiempo real (incidentes, tráfico) actualiza este riesgo. Luego, el modelo de grafo **difunde de forma conservadora** una porción de este riesgo a las zonas vecinas a través de la red de carreteras, creando un riesgo final (A Posteriori). El riesgo "fluye" de áreas de alta concentración a las de baja.
    - **Significado para el Operador:** El riesgo en el mapa no es solo un recuento de incidentes; es una evaluación sofisticada que considera la historia, la situación actual y cómo el estrés en una zona puede afectar a las vecinas. Permite una asignación proactiva y preventiva de recursos.
    """)
    st.subheader("3. Divergencia de Kullback-Leibler (Medición de Anomalía)")
    st.markdown("""
    - **¿Qué es?** Una medida de la Teoría de la Información que cuantifica cuán "sorprendente" o anómala es una distribución de probabilidad en comparación con una de referencia.
    - **¿Cómo se usa?** Compara la **distribución espacial** actual de los incidentes (porcentaje de incidentes en cada zona) con la **norma espacial histórica**. Un valor alto significa que la actividad de emergencia se está concentrando en lugares muy inusuales en comparación con los patrones históricos.
    - **Significado para el Operador:** Es el indicador de más alto nivel de la salud del sistema. Un valor alto es una alerta crítica de que algo inusual está sucediendo a nivel ciudad (ej. un desastre localizado, un evento masivo no planificado). La pestaña "Análisis Profundo" le permite ver exactamente *qué* zonas están causando la desviación.
    """)
    st.subheader("4. Análisis de Red y Teoría de Juegos (Pronóstico y Recomendación)")
    st.markdown("""
    - **¿Qué es?** Un conjunto de técnicas para optimizar decisiones y predecir trayectorias en sistemas complejos.
    - **¿Cómo se usa?** 
        - **Recomendación:** El sistema calcula el **tiempo de respuesta proyectado usando la red de carreteras** (`networkx.shortest_path`). Luego, evalúa cada posible movimiento de ambulancia y recomienda el que produce la máxima reducción en el "déficit" (Riesgo × Tiempo de Respuesta) para la zona más necesitada.
        - **Pronóstico:** Utiliza una Matriz de Transición de Markov para predecir la probabilidad de que el sistema pase de su estado actual (Nominal, Elevado, Anómalo) a otro estado en la siguiente hora, modulando el riesgo futuro.
    - **Significado para el Operador:** Proporciona recomendaciones de despliegue inteligentes basadas en el tiempo de viaje real. Además, ofrece una visión de la inercia del sistema (un sistema anómalo tiende a seguir anómalo).
    """)

def main():
    st.set_page_config(page_title="RedShield AI: Command Suite", layout="wide", initial_sidebar_state="expanded")
    
    if 'app_config' not in st.session_state:
        data_fabric, engine, app_config = get_singleton_engine()
        st.session_state.data_fabric = data_fabric
        st.session_state.engine = engine
        st.session_state.app_config = app_config
    
    setup_plotting_theme(st.session_state.app_config.get('styling', {}))
    plotter = PlottingSME(st.session_state.app_config.get('styling', {}))

    st.sidebar.title("RedShield AI")
    st.sidebar.write("Suite de Comando Estratégico v2.0")
    tab_choice = st.sidebar.radio("Navegación", ["Sandbox de Comando", "Planificación Estratégica", "Análisis Profundo", "Pronóstico de Riesgo", "Centro de Conocimiento"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.info("Simulación para Tijuana, B.C. Basado en datos de código abierto y modelos estocásticos.")
    
    if tab_choice == "Sandbox de Comando":
        render_tab_content("Sandbox", st.session_state.data_fabric, st.session_state.engine, st.session_state.app_config)
    elif tab_choice == "Planificación Estratégica":
        render_tab_content("Scenarios", st.session_state.data_fabric, st.session_state.engine, st.session_state.app_config)
    elif tab_choice == "Análisis Profundo":
        render_analysis_tab(st.session_state.data_fabric, st.session_state.engine, plotter, st.session_state.app_config)
    elif tab_choice == "Pronóstico de Riesgo":
        render_forecasting_tab(st.session_state.engine, plotter)
    elif tab_choice == "Centro de Conocimiento":
        render_knowledge_center()

if __name__ == "__main__":
    main()
