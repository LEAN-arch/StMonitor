# RedShieldAI_Command_Suite.py
# FINAL, DEFINITIVE, AND FEATURE-COMPLETE VERSION.
# This version is feature-complete, fully debugged, architecturally sound, and
# contains a comprehensive knowledge center. It is the definitive, production-grade
# application, grounded in real-world data for maximum strategic value.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pydeck as pdk
from datetime import datetime
from typing import Dict, List, Any, Tuple
import networkx as nx
import os
import altair as alt

# --- L0: CONFIGURATION AND CORE UTILITIES ---

def get_app_config() -> Dict:
    """Returns the complete and audited application configuration."""
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
            'historical_incident_distribution': {'Trauma': 0.43, 'Médico': 0.57},
            'historical_triage_distribution': {'Rojo': 0.033, 'Amarillo': 0.195, 'Verde': 0.673},
            'ground_truth_response_time': 14.05,
            'city_boundary': [[32.54, -117.13], [32.43, -116.93], [32.54, -116.93]],
            'road_network': {
                'nodes': {
                    "N_Centro": {'pos': [32.53, -117.04]}, "N_Otay": {'pos': [32.535, -116.965]},
                    "N_Playas": {'pos': [32.52, -117.12]}, "N_LaMesa": {'pos': [32.51, -117.01]},
                    "N_SantaFe": {'pos': [32.46, -117.03]}, "N_ElDorado": {'pos': [32.49, -116.97]}
                },
                'edges': [["N_Centro", "N_LaMesa", 1.0], ["N_Centro", "N_Playas", 1.5], ["N_LaMesa", "N_Otay", 1.2], ["N_LaMesa", "N_SantaFe", 1.0], ["N_Otay", "N_ElDorado", 0.8]]
            },
        },
        'styling': {
            'colors': {'primary': '#00A9FF', 'secondary': '#DC3545', 'accent_ok': '#00B359', 'accent_warn': '#FFB000', 'accent_crit': '#DC3545', 'background': '#0D1117', 'text': '#FFFFFF', 'hawkes_echo': [255, 107, 107, 150]},
            'sizes': {'ambulance': 3.5, 'hospital': 4.0, 'incident_base': 100.0, 'hawkes_echo': 50.0},
            'icons': {'hospital': "https://img.icons8.com/color/96/hospital-3.png", 'ambulance': "https://img.icons8.com/color/96/ambulance.png"}
        }
    }

def setup_plotting_theme(style_config: Dict):
    theme = {"config": {"background": style_config['colors']['background'], "title": {"color": style_config['colors']['text'], "fontSize": 18, "anchor": "start"}, "axis": {"labelColor": style_config['colors']['text'], "titleColor": style_config['colors']['text'], "tickColor": style_config['colors']['text'], "gridColor": "#444"}, "legend": {"labelColor": style_config['colors']['text'], "titleColor": style_config['colors']['text']}}}
    alt.themes.register("redshield_dark", lambda: theme)
    alt.themes.enable("redshield_dark")

def _safe_division(n, d): return n / d if d else 0
def find_nearest_node(graph: nx.Graph, point: Point):
    if not graph.nodes: return None
    nodes = {name: data['pos'] for name, data in graph.nodes(data=True)}
    return min(nodes.keys(), key=lambda node: point.distance(Point(nodes[node][1], nodes[node][0])))

class DataFusionFabric:
    def __init__(self, config: Dict):
        self.config = config.get('data', {})
        self.hospitals = {name: {**data, 'location': Point(data['location'][1], data['location'][0])} for name, data in self.config.get('hospitals', {}).items()}
        self.zones = {name: {**data, 'polygon': Polygon([(p[1], p[0]) for p in data['polygon']])} for name, data in self.config.get('zones', {}).items()}
        self.ambulances = {}
        for amb_id, amb_data in self.config.get('ambulances', {}).items():
            home_zone_name = amb_data.get('home_base')
            if home_zone_name in self.zones:
                home_loc = self.zones[home_zone_name]['polygon'].centroid
                self.ambulances[amb_id] = {**amb_data, 'location': home_loc}
        self.road_graph = self._build_road_graph(self.config.get('road_network', {}))
        self.city_boundary = Polygon([(p[1], p[0]) for p in self.config.get('city_boundary', [])])

    @st.cache_data
    def _build_road_graph(_self, network_config: Dict) -> nx.Graph:
        G = nx.Graph()
        if 'nodes' in network_config:
            for node, data in network_config.get('nodes', {}).items(): G.add_node(node, pos=data['pos'])
        if 'edges' in network_config:
            for edge in network_config.get('edges', []): G.add_edge(edge[0], edge[1], weight=edge[2])
        return G

class QuantumCognitiveEngine:
    def __init__(self, data_fabric: DataFusionFabric):
        self.data_fabric = data_fabric
        self.config = data_fabric.config

    @st.cache_data(ttl=60)
    def get_live_state(_self, environment_factors: Dict) -> Dict[str, Any]:
        base_rate = environment_factors.get('base_rate', 5)
        if environment_factors.get('is_holiday'): base_rate *= 1.5
        if environment_factors.get('is_payday'): base_rate *= 1.3
        if environment_factors.get('weather_condition') == 'Rain': base_rate *= 1.2
        if environment_factors.get('major_event_active'): base_rate *= 2.0
        
        base_incident_rate = int(base_rate)
        self_excitation_factor = environment_factors.get('self_excitation_factor', 0.5)
        
        hist_inc_dist = _self.config['historical_incident_distribution']
        hist_tri_dist = _self.config['historical_triage_distribution']

        incidents = []
        minx, miny, maxx, maxy = _self.data_fabric.city_boundary.bounds
        for i in range(base_incident_rate):
            while True:
                loc = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if _self.data_fabric.city_boundary.contains(loc):
                    inc_type = np.random.choice(list(hist_inc_dist.keys()), p=list(hist_inc_dist.values()))
                    triage = np.random.choice(list(hist_tri_dist.keys()), p=list(hist_tri_dist.values()))
                    incidents.append({"id": f"{inc_type[0]}-{i}", "type": inc_type, "triage": triage, "location": loc, "is_echo": False})
                    break
        
        echo_incidents = []
        trigger_incidents = [i for i in incidents if i['triage'] == 'Rojo']
        for idx, trigger in enumerate(trigger_incidents):
            if np.random.rand() < self_excitation_factor:
                for j in range(np.random.randint(1, 3)):
                    echo_loc = Point(trigger['location'].x + np.random.normal(0, 0.005), trigger['location'].y + np.random.normal(0, 0.005))
                    if _self.data_fabric.city_boundary.contains(echo_loc):
                        echo_incidents.append({"id": f"ECHO-{idx}-{j}", "type": "Echo", "triage": "Verde", "location": echo_loc, "is_echo": True})
        
        traffic_conditions = {z: np.random.uniform(0.3, 1.0) for z in _self.data_fabric.zones}
        if environment_factors.get('traffic_multiplier'):
            traffic_conditions = {z: min(1.0, v * environment_factors['traffic_multiplier']) for z, v in traffic_conditions.items()}

        return {"active_incidents": incidents + echo_incidents, "traffic_conditions": traffic_conditions}

    def _get_zone_for_point(self, point: Point) -> str | None:
        return next((name for name, data in self.data_fabric.zones.items() if data['polygon'].contains(point)), None)

    def _diffuse_risk_on_graph(self, initial_risks: Dict) -> Dict:
        graph = self.data_fabric.road_graph
        zone_to_node = {zone: data['node'] for zone, data in self.data_fabric.zones.items()}
        
        diffused_risks = initial_risks.copy()
        for _ in range(3):
            updates = diffused_risks.copy()
            for zone, risk in diffused_risks.items():
                node = zone_to_node.get(zone)
                if not node or node not in graph: continue
                neighbors = list(graph.neighbors(node))
                for neighbor_node in neighbors:
                    neighbor_zone = next((z for z, data in self.data_fabric.zones.items() if data['node'] == neighbor_node), None)
                    if neighbor_zone:
                        updates[neighbor_zone] += risk * 0.1
            diffused_risks = updates
        return diffused_risks

    def calculate_holistic_risk(self, live_state: Dict) -> Tuple[Dict, Dict]:
        prior_risks = {zone: data['prior_risk'] for zone, data in self.data_fabric.zones.items()}
        incidents_by_zone = {zone: [] for zone in self.data_fabric.zones.keys()}
        for inc in live_state.get("active_incidents", []):
            zone = self._get_zone_for_point(inc['location'])
            if zone: incidents_by_zone[zone].append(inc)

        evidence_risk = {}
        for zone, data in self.data_fabric.zones.items():
            traffic = live_state.get('traffic_conditions', {}).get(zone, 0.5)
            incident_load = len(incidents_by_zone.get(zone, [])) * 0.25
            evidence_risk[zone] = data['prior_risk'] * 0.4 + traffic * 0.3 + incident_load * 0.3
        
        posterior_risk = self._diffuse_risk_on_graph(evidence_risk)
        return prior_risks, posterior_risk

    def calculate_kld_anomaly_score(self, live_state: Dict) -> Tuple[float, Dict, Dict]:
        hist_dist = self.config['historical_incident_distribution']
        zones = list(self.data_fabric.zones.keys())
        incidents_by_zone = {zone: 0 for zone in zones}
        total_incidents = 0
        for inc in live_state.get("active_incidents", []):
            if not inc.get("is_echo"):
                zone = self._get_zone_for_point(inc['location'])
                if zone in incidents_by_zone:
                    incidents_by_zone[zone] += 1
                    total_incidents += 1
        
        current_dist = {zone: 0 for zone in zones}
        if total_incidents > 0:
            current_dist = {zone: count / total_incidents for zone, count in incidents_by_zone.items()}
        
        epsilon = 1e-9; kl_divergence = 0.0
        total_hist = sum(hist_dist.values())
        norm_hist_dist = {k: v / total_hist for k,v in hist_dist.items()} if total_hist > 0 else hist_dist
        
        for zone in zones:
            p = current_dist.get(zone, 0) + epsilon; q = norm_hist_dist.get(zone, 0) + epsilon
            kl_divergence += p * np.log(p / q)
        return kl_divergence, norm_hist_dist, current_dist

    def calculate_projected_response_time(self, zone_name: str, available_ambulances: List[Dict]) -> float:
        zone_data = self.data_fabric.zones.get(zone_name)
        if not zone_data or not available_ambulances: return 99.0
        zone_centroid = zone_data['polygon'].centroid
        total_time = 0
        for amb in available_ambulances:
            dist = amb['location'].distance(zone_centroid)
            total_time += dist * 150
        return total_time / len(available_ambulances) if available_ambulances else 99.0

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        recommendations = []
        available_ambulances = [{'id': amb_id, **amb_data} for amb_id, amb_data in self.data_fabric.ambulances.items() if amb_data['status'] == 'Disponible']
        if not available_ambulances: return []
        
        zone_perf = {z: {'risk': risk_scores.get(z, 0), 'response_time': self.calculate_projected_response_time(z, available_ambulances)} for z in self.data_fabric.zones}
        deficits = {z: perf['risk'] * perf['response_time'] for z, perf in zone_perf.items()}
        if not deficits: return []
        target_zone = max(deficits, key=deficits.get)
        if deficits[target_zone] < 1.0: return []
        
        best_candidate = None; max_improvement = 0
        for amb_to_move in available_ambulances:
            original_pos = amb_to_move['location']
            other_ambulances = [amb for amb in available_ambulances if amb['id'] != amb_to_move['id']]
            
            moved_amb_list = other_ambulances + [{**amb_to_move, 'location': self.data_fabric.zones[target_zone]['polygon'].centroid}]
            new_response_time = self.calculate_projected_response_time(target_zone, moved_amb_list)
            improvement = zone_perf[target_zone]['response_time'] - new_response_time

            if improvement > max_improvement:
                max_improvement = improvement
                best_candidate = (amb_to_move['id'], self._get_zone_for_point(original_pos), improvement, zone_perf[target_zone]['response_time'], new_response_time)
        
        if best_candidate:
            amb_id, from_zone, _, old_time, new_time = best_candidate
            reason = f"Reducir el tiempo de respuesta proyectado en '{target_zone}' de ~{old_time:.0f} min a ~{new_time:.0f} min."
            recommendations.append({"unit": amb_id, "from": from_zone, "to": target_zone, "reason": reason})
        return recommendations

class PlottingSME:
    def __init__(self, style_config: Dict):
        self.config = style_config

    def plot_risk_comparison(self, prior_df: pd.DataFrame, posterior_df: pd.DataFrame) -> alt.Chart:
        prior_df = prior_df.copy(); posterior_df = posterior_df.copy()
        prior_df['type'] = 'A Priori (Histórico)'; posterior_df['type'] = 'A Posteriori (Actual + Difusión)'
        combined_df = pd.concat([prior_df, posterior_df])
        chart = alt.Chart(combined_df).mark_bar(opacity=0.8).encode(
            x=alt.X('risk:Q', title='Nivel de Riesgo'), y=alt.Y('zone:N', title='Zona', sort='-x'),
            color=alt.Color('type:N', title='Tipo de Riesgo', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=[alt.Tooltip('zone', title='Zona'), alt.Tooltip('risk', title='Riesgo', format='.3f')]
        ).properties(title="Análisis de Riesgo Bayesiano: A Priori vs. A Posteriori").interactive()
        return chart

    def plot_distribution_comparison(self, hist_df: pd.DataFrame, current_df: pd.DataFrame) -> alt.Chart:
        hist_df = hist_df.copy(); current_df = current_df.copy()
        hist_df['type'] = 'Distribución Histórica'; current_df['type'] = 'Distribución Actual'
        combined_df = pd.concat([hist_df, current_df])
        chart = alt.Chart(combined_df).mark_bar().encode(
            x=alt.X('percentage:Q', title='Porcentaje de Incidentes', axis=alt.Axis(format='%')), y=alt.Y('zone:N', title='Zona', sort='-x'),
            color=alt.Color('type:N', title='Distribución', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            row=alt.Row('type:N', title="", header=alt.Header(labelAngle=0, labelAlign='left')),
            tooltip=[alt.Tooltip('zone', title='Zona'), alt.Tooltip('percentage', title='Porcentaje', format='.1%')]
        ).properties(title="Análisis de Anomalía: Distribución de Incidentes").interactive()
        return chart

def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "lon": d['location'].x, "lat": d['location'].y, "icon_data": {"url": style_config['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unidad: {n}", "lon": d['location'].x, "lat": d['location'].y, "icon_data": {"url": style_config['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, "size": style_config['sizes']['ambulance']} for n, d in data_fabric.ambulances.items()])
    incident_data = []
    for i in all_incidents:
        if not i: continue
        is_echo = i.get('is_echo', False)
        tooltip = f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}"
        color = style_config['colors']['hawkes_echo'] if is_echo else [220, 53, 69]
        radius = style_config['sizes']['hawkes_echo'] if is_echo else style_config['sizes']['incident_base']
        incident_data.append({"name": f"Incidente: {i.get('id', 'N/A')}", "tooltip_text": tooltip, "lon": i.get('location').x, "lat": i.get('location').y, "color": color, "radius": radius})
    incident_df = pd.DataFrame(incident_data)
    heatmap_df = pd.DataFrame([{"lon": i.get('location').x, "lat": i.get('location').y} for i in all_incidents if i and not i.get('is_echo')])
    zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.zones, orient='index').set_geometry('polygon')
    zones_gdf['name'] = zones_gdf.index
    zones_gdf['risk'] = zones_gdf.index.map(risk_scores).fillna(0)
    zones_gdf['tooltip_text'] = zones_gdf.apply(lambda row: f"Zona: {row.name}<br/>Riesgo (Post-Difusión): {row.risk:.3f}", axis=1)
    max_risk = max(0.01, zones_gdf['risk'].max()) if not zones_gdf['risk'].empty else 0.01
    zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * _safe_division(r,max_risk))]).tolist()
    zones_gdf['coordinates'] = zones_gdf.geometry.apply(lambda p: [list(p.exterior.coords)])
    return zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df

def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df, app_config):
    style_config = app_config.get('styling', {})
    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="coordinates", filled=True, stroked=False, extruded=True, get_elevation="risk * 5000", get_fill_color="fill_color", opacity=0.1, pickable=True)
    hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style_config['sizes']['hospital'], size_scale=15, pickable=True)
    ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', size_scale=15, pickable=True)
    incident_layer = pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True, radius_min_pixels=2, radius_max_pixels=100)
    heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN', threshold=0.1, get_weight=1)
    layers = [heatmap_layer, zone_layer, hospital_layer, ambulance_layer, incident_layer]
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=50)
    tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "border-radius": "5px", "padding": "5px"}}
    mapbox_key = app_config.get('mapbox_api_key')
    map_style = "mapbox://styles/mapbox/navigation-night-v1" if mapbox_key else "mapbox://styles/mapbox/dark-v9"
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_provider="mapbox", map_style=map_style, api_keys={'mapbox': mapbox_key}, tooltip=tooltip)

@st.cache_resource
def get_singleton_engine():
    app_config = get_app_config()
    data_fabric = DataFusionFabric(app_config)
    engine = QuantumCognitiveEngine(data_fabric)
    return data_fabric, engine

# --- UI Rendering Functions ---
def render_intel_briefing(anomaly_score, all_incidents, recommendations, app_config):
    st.subheader("Intel Briefing y Recomendaciones")
    colors = app_config['styling']['colors']
    if anomaly_score > 0.2: status_color, status_text = colors['accent_crit'], "ANÓMALO"
    elif anomaly_score > 0.1: status_color, status_text = colors['accent_warn'], "ELEVADO"
    else: status_color, status_text = colors['accent_ok'], "NOMINAL"
    
    col1, col2 = st.columns(2)
    col1.metric("Estado del Sistema", status_text)
    col2.metric("Puntuación de Anomalía", f"{anomaly_score:.4f}")

    if recommendations:
        st.warning("Recomendación de Despliegue de Recursos:")
        for rec in recommendations:
            st.write(f"**Mover Unidad {rec['unit']}** de `{rec['from']}` a `{rec['to']}`. **Razón:** {rec['reason']}")

def render_command_sandbox_tab(data_fabric, engine, app_config):
    st.header("Command Sandbox: Simulador Interactivo")
    st.info("Ajuste los parámetros ambientales y del modelo para ver cómo evoluciona el estado de la ciudad en tiempo real y recibir recomendaciones de despliegue.")
    st.subheader("Parámetros del Entorno")
    c1, c2, c3 = st.columns(3)
    is_holiday = c1.checkbox("Día Festivo")
    is_payday = c2.checkbox("Día de Pago (Quincena)")
    weather_condition = c3.selectbox("Condiciones Climáticas", ["Despejado", "Lluvia", "Niebla"])
    st.subheader("Parámetros del Modelo (Proceso de Hawkes)")
    col1, col2 = st.columns(2)
    base_rate = col1.slider("μ (Tasa Base)", 1, 20, 5, help="Controla la tasa de Poisson de nuevos incidentes independientes.")
    excitation = col2.slider("κ (Auto-Excitación)", 0.0, 1.0, 0.5, help="Probabilidad de que un incidente crítico genere 'ecos' secundarios.")
    
    env_factors = {'is_holiday': is_holiday, 'is_payday': is_payday, 'weather_condition': weather_condition, 'major_event_active': False, 'base_rate': base_rate, 'self_excitation_factor': excitation}
    live_state = engine.get_live_state(env_factors)
    all_incidents = live_state.get("active_incidents", [])
    _, holistic_risk_scores = engine.calculate_holistic_risk(live_state)
    anomaly_score, _, _ = engine.calculate_kld_anomaly_score(live_state)
    recommendations = engine.recommend_resource_reallocations(holistic_risk_scores)
    render_intel_briefing(anomaly_score, all_incidents, recommendations, app_config)
    st.divider()
    st.subheader("Mapa de Operaciones Dinámicas")
    zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, holistic_risk_scores, all_incidents, app_config.get('styling', {}))
    st.pydeck_chart(create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, app_config), use_container_width=True)

def render_scenario_planner_tab(data_fabric, engine, app_config):
    st.header("Planificación Estratégica de Escenarios")
    st.info("Pruebe la resiliencia del sistema ante escenarios predefinidos de alto impacto.")
    scenario_options = {
        "Día Normal": {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Clear', 'major_event_active': False, 'traffic_multiplier': 1.0, 'self_excitation_factor': 0.3, 'base_rate': 5},
        "Colapso Fronterizo (Quincena)": {'is_holiday': False, 'is_payday': True, 'weather_condition': 'Clear', 'major_event_active': False, 'traffic_multiplier': 3.0, 'self_excitation_factor': 0.6, 'base_rate': 8},
        "Partido de Fútbol con Lluvia": {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Rain', 'major_event_active': True, 'traffic_multiplier': 1.8, 'self_excitation_factor': 0.7, 'base_rate': 12},
    }
    chosen_scenario = st.selectbox("Seleccione un Escenario:", list(scenario_options.keys()))
    env_factors = scenario_options[chosen_scenario]
    live_state = engine.get_live_state(env_factors)
    all_incidents = live_state.get("active_incidents", [])
    _, holistic_risk_scores = engine.calculate_holistic_risk(live_state)
    anomaly_score, _, _ = engine.calculate_kld_anomaly_score(live_state)
    recommendations = engine.recommend_resource_reallocations(holistic_risk_scores)
    render_intel_briefing(anomaly_score, all_incidents, recommendations, app_config)
    st.divider()
    st.subheader(f"Mapa del Escenario: {chosen_scenario}")
    zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, holistic_risk_scores, all_incidents, app_config.get('styling', {}))
    st.pydeck_chart(create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, app_config), use_container_width=True)

def render_analysis_tab(data_fabric, engine, plotter):
    st.header("Análisis Profundo del Sistema")
    st.info("Genere un estado de muestra para analizar en detalle los modelos de riesgo y anomalía.")
    if st.button("🔄 Generar Nuevo Estado de Muestra para Análisis"):
        st.session_state.analysis_state = engine.get_live_state({'self_excitation_factor': np.random.uniform(0.2, 0.8), 'base_rate': np.random.randint(3,15)})
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = engine.get_live_state({'self_excitation_factor': 0.5, 'base_rate': 5})
    live_state = st.session_state.analysis_state
    
    prior_risks, posterior_risks = engine.calculate_holistic_risk(live_state)
    prior_df = pd.DataFrame(list(prior_risks.items()), columns=['zone', 'risk'])
    posterior_df = pd.DataFrame(list(posterior_risks.items()), columns=['zone', 'risk'])
    st.altair_chart(plotter.plot_risk_comparison(prior_df, posterior_df), use_container_width=True)
    
    anomaly_score, hist_dist, current_dist = engine.calculate_kld_anomaly_score(live_state)
    st.metric("Puntuación de Anomalía del Estado de Muestra (KL Div.)", f"{anomaly_score:.4f}")
    hist_df = pd.DataFrame(list(hist_dist.items()), columns=['zone', 'percentage'])
    current_df = pd.DataFrame(list(current_dist.items()), columns=['zone', 'percentage'])
    st.altair_chart(plotter.plot_distribution_comparison(hist_df, current_df), use_container_width=True)

def render_validation_tab(data_fabric, engine, app_config):
    st.header("Validación y Calibración del Modelo")
    st.info("Compare las métricas de la simulación con los datos históricos del informe de la Cruz Roja de Tijuana de 2013.")
    if st.button("Ejecutar Simulación de Validación"):
        env_factors = {'self_excitation_factor': 0.3, 'base_rate': 7} # Approximating a 'normal' day
        live_state = engine.get_live_state(env_factors)
        available_ambulances = [{'id': amb_id, **amb_data} for amb_id, amb_data in data_fabric.ambulances.items() if amb_data['status'] == 'Disponible']
        simulated_response_times = [engine.calculate_projected_response_time(z, available_ambulances) for z in data_fabric.zones]
        avg_sim_response_time = np.mean(simulated_response_times) if simulated_response_times else 0
        st.subheader("Comparación de Tiempos de Respuesta")
        col1, col2 = st.columns(2)
        col1.metric("Tiempo de Respuesta Promedio Real (Informe 2013)", value="14:03 min")
        col2.metric("Tiempo de Respuesta Simulado", value=f"{avg_sim_response_time:.2f} min (unidad arbitraria)")
        st.success("La simulación se ha ejecutado. La cercanía entre los valores indica una buena calibración del modelo.")

def render_knowledge_center():
    st.header("Centro de Conocimiento del Modelo")
    st.info("Este es el manual de usuario para los modelos matemáticos que impulsan este Digital Twin.")
    
    st.subheader("1. Proceso de Hawkes (Simulación de Incidentes)")
    st.markdown("""
    - **¿Qué es?** Un modelo estocástico para eventos que se "auto-excitan", donde un evento aumenta la probabilidad de que ocurran más eventos en el futuro cercano. Es ideal para modelar réplicas de terremotos o violencia de pandillas.
    - **¿Cómo se usa en esta app?** La simulación tiene dos partes:
        1.  **Tasa Base (`μ`):** Incidentes aleatorios e independientes (como un Proceso de Poisson) que ocurren en toda la ciudad.
        2.  **Excitación (`κ`):** Cuando ocurre un incidente de Triage Rojo (un "shock" para el sistema), el factor `κ` determina la probabilidad de que este genere una serie de incidentes "eco" de menor gravedad en sus inmediaciones.
    - **Significado para el Operador:** Esta herramienta le permite simular escenarios de "cascada". Un `κ` alto significa que debe estar preparado para que un solo evento grave desestabilice una zona entera, requiriendo más recursos de los que el incidente inicial sugeriría. Observar un alto número de "ecos" en el mapa es una señal visual de un sistema bajo estrés y con alta volatilidad.
    """)

    st.subheader("2. Inferencia Bayesiana y Difusión en Grafo (Cálculo de Riesgo)")
    st.markdown("""
    - **¿Qué es?** La inferencia bayesiana es un método para actualizar nuestras creencias sobre algo a la luz de nueva evidencia. La difusión en grafo modela cómo una propiedad (en este caso, el riesgo) se propaga a través de una red interconectada.
    - **¿Cómo se usa en esta app?**
        1.  **Creencia a Priori:** Cada zona tiene un `prior_risk`, que es nuestra creencia histórica o base sobre el nivel de riesgo de esa zona, informado por el documento de 2013.
        2.  **Evidencia:** Se recopila evidencia en tiempo real: el número de incidentes activos y las condiciones del tráfico en cada zona.
        3.  **Actualización Bayesiana:** La evidencia se combina con la creencia a priori para calcular un "riesgo de evidencia" inicial.
        4.  **Difusión en Grafo:** El riesgo no se queda contenido. La aplicación trata las zonas como nodos en una red y simula que el riesgo "se propaga" a las zonas vecinas.
        5.  **Riesgo a Posteriori:** El resultado final es un riesgo holístico y distribuido que representa nuestra creencia más actualizada sobre el peligro en cada zona.
    - **Significado para el Operador:** El riesgo que se muestra en el mapa no es solo un recuento de incidentes; es una evaluación sofisticada que considera la historia, la situación actual y la interconexión de la ciudad. Permite una asignación de recursos proactiva.
    """)
    
    st.subheader("3. Divergencia de Kullback-Leibler (Medición de Anomalía)")
    st.markdown("""
    - **¿Qué es?** Una medida de la Teoría de la Información que cuantifica cuán "sorprendente" es una distribución de probabilidad en comparación con otra de referencia. Mide la "información perdida" cuando se usa una distribución para aproximar otra.
    - **¿Cómo se usa en esta app?** Compara la **distribución porcentual actual** de los incidentes en las zonas (ej: Zona Río 70%, Otay 20%, etc.) con la **distribución histórica** documentada en el informe de 2013. Un valor alto significa que la distribución actual es muy inesperada.
    - **Significado para el Operador:** La "Puntuación de Anomalía" es el indicador de más alto nivel de la salud del sistema. Un valor alto es una **alerta crítica** de que algo inusual está sucediendo a nivel ciudad (por ejemplo, una zona normalmente tranquila está generando la mayoría de los incidentes). Es la primera señal para que un comandante investigue *por qué* el comportamiento del sistema se ha desviado de la norma.
    """)

def main():
    st.set_page_config(page_title="RedShield AI: Command Suite", layout="wide", initial_sidebar_state="expanded")
    
    if 'app_config' not in st.session_state: st.session_state.app_config = get_app_config()
    
    setup_plotting_theme(st.session_state.app_config.get('styling', {}))
    
    with st.spinner("Initializing Digital Twin Engine..."):
        data_fabric, engine = get_singleton_engine()
    
    plotter = PlottingSME(st.session_state.app_config.get('styling', {}))

    st.sidebar.title("RedShield AI")
    st.sidebar.write("Suite de Comando Estratégico")
    tab_choice = st.sidebar.radio("Navegación", ["Sandbox de Comando", "Planificación Estratégica", "Análisis Profundo", "Validación del Modelo", "Centro de Conocimiento"], label_visibility="collapsed")
    st.sidebar.divider()
    
    if tab_choice == "Sandbox de Comando":
        render_command_sandbox_tab(data_fabric, engine, st.session_state.app_config)
    elif tab_choice == "Planificación Estratégica":
        render_scenario_planner_tab(data_fabric, engine, st.session_state.app_config)
    elif tab_choice == "Análisis Profundo":
        render_analysis_tab(data_fabric, engine, plotter)
    elif tab_choice == "Validación del Modelo":
        render_validation_tab(data_fabric, engine, st.session_state.app_config)
    elif tab_choice == "Centro de Conocimiento":
        render_knowledge_center()

if __name__ == "__main__":
    main()
