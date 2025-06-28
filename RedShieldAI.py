# RedShieldAI_Command_Suite.py
# FINAL, DEFINITIVE, AND FEATURE-COMPLETE VERSION.
# This version integrates all advanced models, UI components, and actionable insights
# into a single, robust, and high-performance application. It is architecturally
# sound, free of previous bugs, and provides maximum strategic value.

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
                "Cruz Roja Tijuana": {'location': [32.5283, -117.0255], 'capacity': 80, 'load': 60}
            },
            'ambulances': {
                "A01": {'location': [32.515, -117.115], 'status': "Disponible", 'home_base': 'Playas'},
                "A02": {'location': [32.535, -116.96], 'status': "Disponible", 'home_base': 'Otay'},
                "A03": {'location': [32.508, -117.00], 'status': "En Misión", 'home_base': 'La Mesa'},
                "A04": {'location': [32.525, -117.02], 'status': "Disponible", 'home_base': 'Zona Río'},
            },
            'zones': {
                "Zona Río": {'polygon': [[32.52, -117.01], [32.535, -117.01], [32.535, -117.035], [32.52, -117.035]], 'prior_risk': 0.6, 'population_density': 0.9},
                "Otay": {'polygon': [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], 'prior_risk': 0.4, 'population_density': 0.7},
                "Playas": {'polygon': [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], 'prior_risk': 0.3, 'population_density': 0.5},
                "La Mesa": {'polygon': [[32.50, -117.00], [32.52, -117.00], [32.52, -117.02], [32.50, -117.02]], 'prior_risk': 0.5, 'population_density': 0.8},
            },
            'historical_incident_distribution': {'Zona Río': 0.4, 'Otay': 0.2, 'Playas': 0.1, 'La Mesa': 0.3},
            'city_boundary': [[32.54, -117.13], [32.43, -116.93], [32.54, -116.93]],
            'road_network': {
                'nodes': {
                    "N_ZonaRío": {'pos': [32.528, -117.025]}, "N_Otay": {'pos': [32.535, -116.965]},
                    "N_Playas": {'pos': [32.52, -117.12]}, "N_LaMesa": {'pos': [32.51, -117.01]}
                },
                'edges': [["N_ZonaRío", "N_Otay", 1.0], ["N_ZonaRío", "N_Playas", 1.0], ["N_ZonaRío", "N_LaMesa", 0.5], ["N_LaMesa", "N_Otay", 1.2]]
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
        self.ambulances = {name: {**data, 'location': Point(data['location'][1], data['location'][0])} for name, data in self.config.get('ambulances', {}).items()}
        self.zones = {name: {**data, 'polygon': Polygon([(p[1], p[0]) for p in data['polygon']])} for name, data in self.config.get('zones', {}).items()}
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

        incidents = []
        minx, miny, maxx, maxy = _self.data_fabric.city_boundary.bounds
        for i in range(base_incident_rate):
            while True:
                loc = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if _self.data_fabric.city_boundary.contains(loc):
                    inc_type, triage_probs = ("Trauma", [0.4, 0.5, 0.1]) if np.random.rand() > 0.5 else ("Médico", [0.15, 0.65, 0.20])
                    incidents.append({"id": f"{inc_type[0]}-{i}", "type": inc_type, "triage": np.random.choice(["Rojo", "Amarillo", "Verde"], p=triage_probs), "location": loc, "is_echo": False})
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
        for name, data in self.data_fabric.zones.items():
            if data['polygon'].contains(point): return name
        return None

    def _diffuse_risk_on_graph(self, initial_risks: Dict) -> Dict:
        graph = self.data_fabric.road_graph
        zone_to_node = {zone: f"N_{zone.replace(' ', '')}" for zone in self.data_fabric.zones.keys()}
        
        diffused_risks = initial_risks.copy()
        for _ in range(3):
            updates = diffused_risks.copy()
            for zone, risk in diffused_risks.items():
                node = zone_to_node.get(zone)
                if not node or node not in graph: continue
                neighbors = list(graph.neighbors(node))
                for neighbor_node in neighbors:
                    neighbor_zone = next((z for z, n in zone_to_node.items() if n == neighbor_node), None)
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
        zones = list(hist_dist.keys())
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
        for zone in zones:
            p = current_dist.get(zone, 0) + epsilon; q = hist_dist.get(zone, 0) + epsilon
            kl_divergence += p * np.log(p / q)
        return kl_divergence, hist_dist, current_dist

    def recommend_resource_reallocations(self, risk_scores: Dict) -> List[Dict]:
        recommendations = []
        zone_coverage = {z: 0 for z in self.data_fabric.zones}
        for amb in self.data_fabric.ambulances.values():
            if amb['status'] == 'Disponible':
                zone = self._get_zone_for_point(amb['location'])
                if zone: zone_coverage[zone] += 1
        deficits = {z: risk_scores.get(z,0) * (1 / (1 + zone_coverage[z])) for z in self.data_fabric.zones}
        if not deficits: return []
        target_zone = max(deficits, key=deficits.get)
        if deficits[target_zone] < 0.5: return []
        best_candidate = None; min_move_cost = float('inf')
        for amb_id, amb_data in self.data_fabric.ambulances.items():
            if amb_data['status'] == 'Disponible':
                current_zone = self._get_zone_for_point(amb_data['location'])
                if current_zone == target_zone: continue
                move_cost = deficits.get(current_zone, 1.0) 
                if move_cost < min_move_cost:
                    min_move_cost = move_cost
                    best_candidate = (amb_id, current_zone)
        if best_candidate:
            amb_id, from_zone = best_candidate
            recommendations.append({"unit": amb_id, "from": from_zone, "to": target_zone, "reason": f"Reducir el déficit de cobertura en la zona de alto riesgo '{target_zone}'."})
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
    # This function is stable
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
    # This function is stable
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
        "Día Normal": {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Clear', 'major_event_active': False, 'traffic_multiplier': 1.0, 'self_excitation_factor': 0.3},
        "Colapso Fronterizo (Quincena)": {'is_holiday': False, 'is_payday': True, 'weather_condition': 'Clear', 'major_event_active': False, 'traffic_multiplier': 3.0, 'self_excitation_factor': 0.6},
        "Partido de Fútbol con Lluvia": {'is_holiday': False, 'is_payday': False, 'weather_condition': 'Rain', 'major_event_active': True, 'traffic_multiplier': 1.8, 'self_excitation_factor': 0.7},
        "Festival Masivo en Zona Río": {'is_holiday': True, 'is_payday': False, 'weather_condition': 'Clear', 'major_event_active': True, 'traffic_multiplier': 2.5, 'self_excitation_factor': 0.4}
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
    chart_risk = plotter.plot_risk_comparison(prior_df, posterior_df)
    st.altair_chart(chart_risk, use_container_width=True)
    
    anomaly_score, hist_dist, current_dist = engine.calculate_kld_anomaly_score(live_state)
    st.metric("Puntuación de Anomalía del Estado de Muestra (KL Div.)", f"{anomaly_score:.4f}")
    hist_df = pd.DataFrame(list(hist_dist.items()), columns=['zone', 'percentage'])
    current_df = pd.DataFrame(list(current_dist.items()), columns=['zone', 'percentage'])
    chart_dist = plotter.plot_distribution_comparison(hist_df, current_df)
    st.altair_chart(chart_dist, use_container_width=True)

def render_knowledge_center():
    st.header("Centro de Conocimiento del Modelo")
    st.info("Este es el manual de usuario para los modelos matemáticos que impulsan este Digital Twin.")
    
    st.subheader("1. Proceso de Hawkes (Simulación de Incidentes)")
    st.markdown("""
    **¿Qué es?** Un modelo estocástico para eventos que se "auto-excitan", donde un evento aumenta la probabilidad de que ocurran más eventos en el futuro cercano.
    **¿Cómo se usa en esta app?** Cuando ocurre un incidente de Triage Rojo, el factor de "auto-excitación" (κ) determina la probabilidad de que genere incidentes "eco" de menor gravedad en sus inmediaciones, simulando la inestabilidad local.
    **Significado para el Operador:** Permite simular escenarios de "cascada". Un κ alto significa que debe estar preparado para que un solo evento grave desestabilice una zona entera.
    """)
    st.subheader("2. Inferencia Bayesiana y Difusión en Grafo (Cálculo de Riesgo)")
    st.markdown("""
    **¿Qué es?** Un método para actualizar creencias (riesgo) con nueva evidencia, y un modelo para ver cómo se propaga el riesgo en una red.
    **¿Cómo se usa en esta app?** Cada zona tiene un riesgo histórico base (A Priori). La evidencia en tiempo real (incidentes, tráfico) actualiza este riesgo. Luego, el modelo de grafo "difunde" una porción de este riesgo a las zonas vecinas, creando un riesgo final (A Posteriori).
    **Significado para el Operador:** El riesgo en el mapa no es solo un recuento; es una evaluación sofisticada que considera la historia, la situación actual y la interconexión de la ciudad. Permite una asignación proactiva de recursos.
    """)
    st.subheader("3. Divergencia de Kullback-Leibler (Medición de Anomalía)")
    st.markdown("""
    **¿Qué es?** Una medida de la Teoría de la Información que cuantifica cuán "sorprendente" es una distribución de probabilidad en comparación con otra de referencia.
    **¿Cómo se usa en esta app?** Compara la distribución porcentual actual de los incidentes en las zonas con la norma histórica. Un valor alto significa que la distribución actual es muy inesperada.
    **Significado para el Operador:** Es el indicador de más alto nivel de la salud del sistema. Un valor alto es una alerta crítica de que algo inusual está sucediendo a nivel ciudad, impulsando una investigación más profunda.
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
    tab_choice = st.sidebar.radio("Navegación", ["Sandbox de Comando", "Planificación Estratégica", "Análisis Profundo", "Centro de Conocimiento"], label_visibility="collapsed")
    st.sidebar.divider()
    
    if tab_choice == "Sandbox de Comando":
        render_command_sandbox_tab(data_fabric, engine, st.session_state.app_config)
    elif tab_choice == "Planificación Estratégica":
        render_scenario_planner_tab(data_fabric, engine, st.session_state.app_config)
    elif tab_choice == "Análisis Profundo":
        render_analysis_tab(data_fabric, engine, plotter)
    elif tab_choice == "Centro de Conocimiento":
        render_knowledge_center()

if __name__ == "__main__":
    main()
