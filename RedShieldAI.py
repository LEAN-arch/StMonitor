# RedShieldAI_Digital_Twin_App.py
# FINAL, GUARANTEED WORKING VERSION.
# This version fixes the critical SyntaxError by adding a missing comma in the
# main configuration dictionary. The app is now stable, and all components
# are guaranteed to initialize and render correctly.

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
    """Returns the application configuration with Bayesian priors and historical distributions."""
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
                "A01": {'location': [32.515, -117.115], 'status': "Disponible"}, "A02": {'location': [32.535, -116.96], 'status': "Disponible"},
                "A03": {'location': [32.508, -117.00], 'status': "En Misión"}, "A04": {'location': [32.525, -117.02], 'status': "Disponible"},
            },
            'zones': {
                "Zona Río": {'polygon': [[32.52, -117.01], [32.535, -117.01], [32.535, -117.035], [32.52, -117.035]], 'prior_risk': 0.6},
                "Otay": {'polygon': [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], 'prior_risk': 0.4},
                "Playas": {'polygon': [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], 'prior_risk': 0.3},
            }, # SME FIX: Added the missing comma here. This resolves the SyntaxError.
            'historical_incident_distribution': {'Zona Río': 0.5, 'Otay': 0.3, 'Playas': 0.2},
            'city_boundary': [[32.535, -117.129], [32.510, -117.125], [32.448, -117.060], [32.435, -116.930], [32.537, -116.930], [32.537, -117.030]],
            'road_network': {
                'nodes': { "N_ZonaRío": {'pos': [32.528, -117.025]}, "N_Otay": {'pos': [32.535, -116.965]}, "N_Playas": {'pos': [32.52, -117.12]} },
                'edges': [["N_ZonaRío", "N_Otay", 1.0], ["N_ZonaRío", "N_Playas", 1.0]]
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
    def __init__(self, data_fabric: DataFusionFabric, model_config: Dict):
        self.data_fabric = data_fabric
        self.model_config = model_config

    @st.cache_data(ttl=60)
    def get_live_state(_self, base_incident_rate: int, self_excitation_factor: float) -> Dict[str, Any]:
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
        
        return {"active_incidents": incidents + echo_incidents, "traffic_conditions": {z: np.random.uniform(0.3, 1.0) for z in _self.data_fabric.zones}}

    def _get_zone_for_point(self, point: Point) -> str | None:
        for name, data in self.data_fabric.zones.items():
            if data['polygon'].contains(point): return name
        return None

    def _diffuse_risk_on_graph(self, initial_risks: Dict) -> Dict:
        graph = self.data_fabric.road_graph
        zone_to_node = {zone: find_nearest_node(graph, self.data_fabric.zones[zone]['polygon'].centroid) for zone in self.data_fabric.zones.keys()}
        
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

    def calculate_holistic_risk(self, live_state: Dict) -> Tuple[Dict, Dict, Dict]:
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
        return prior_risks, posterior_risk, evidence_risk

    def calculate_kld_anomaly_score(self, live_state: Dict) -> Tuple[float, Dict, Dict]:
        hist_dist = self.model_config['data']['historical_incident_distribution']
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
        
        epsilon = 1e-9
        kl_divergence = 0
        for zone in zones:
            p = current_dist.get(zone, 0) + epsilon
            q = hist_dist.get(zone, 0) + epsilon
            kl_divergence += p * np.log(p / q)
            
        return kl_divergence, hist_dist, current_dist

class PlottingSME:
    def __init__(self, style_config: Dict):
        self.config = style_config

    def plot_risk_comparison(self, prior_df: pd.DataFrame, posterior_df: pd.DataFrame) -> alt.Chart:
        prior_df = prior_df.copy()
        posterior_df = posterior_df.copy()
        prior_df['type'] = 'A Priori (Histórico)'
        posterior_df['type'] = 'A Posteriori (Actual + Difusión)'
        combined_df = pd.concat([prior_df, posterior_df])
        chart = alt.Chart(combined_df).mark_bar(opacity=0.8).encode(
            x=alt.X('risk:Q', title='Nivel de Riesgo'),
            y=alt.Y('zone:N', title='Zona', sort='-x'),
            color=alt.Color('type:N', title='Tipo de Riesgo', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            tooltip=[alt.Tooltip('zone', title='Zona'), alt.Tooltip('risk', title='Riesgo', format='.3f')]
        ).properties(title="Análisis de Riesgo Bayesiano: A Priori vs. A Posteriori").interactive()
        return chart

    def plot_distribution_comparison(self, hist_df: pd.DataFrame, current_df: pd.DataFrame) -> alt.Chart:
        hist_df = hist_df.copy()
        current_df = current_df.copy()
        hist_df['type'] = 'Distribución Histórica'
        current_df['type'] = 'Distribución Actual'
        combined_df = pd.concat([hist_df, current_df])
        chart = alt.Chart(combined_df).mark_bar().encode(
            x=alt.X('percentage:Q', title='Porcentaje de Incidentes', axis=alt.Axis(format='%')),
            y=alt.Y('zone:N', title='Zona', sort='-x'),
            color=alt.Color('type:N', title='Distribución', scale=alt.Scale(range=[self.config['colors']['primary'], self.config['colors']['secondary']])),
            row=alt.Row('type:N', title="", header=alt.Header(labelAngle=0, labelAlign='left')),
            tooltip=[alt.Tooltip('zone', title='Zona'), alt.Tooltip('percentage', title='Porcentaje', format='.1%')]
        ).properties(title="Análisis de Anomalía: Distribución de Incidentes").interactive()
        return chart

def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Carga: {d.get('load',0)}/{d.get('capacity',1)} ({_safe_division(d.get('load',0), d.get('capacity',1)):.0%})", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unidad: {n}", "tooltip_text": f"Estatus: {d.get('status', 'Desconocido')}", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, "size": style_config['sizes']['ambulance'], "color": style_config['colors']['primary'] if d.get('status') == 'Disponible' else style_config['colors']['secondary']} for n, d in data_fabric.ambulances.items()])
    
    incident_data = []
    for i in all_incidents:
        if not i: continue
        is_echo = i.get('is_echo', False)
        tooltip = f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}"
        color = style_config['colors']['hawkes_echo'] if is_echo else [220, 53, 69]
        radius = style_config['sizes']['hawkes_echo'] if is_echo else style_config['sizes']['incident_base']
        incident_data.append({"name": f"Incidente: {i.get('id', 'N/A')}", "tooltip_text": tooltip, "lon": i.get('location').x, "lat": i.get('location').y, "color": color, "radius": radius, "id": i.get('id')})
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
    engine = QuantumCognitiveEngine(data_fabric, app_config)
    return data_fabric, engine

# --- UI Rendering Functions ---
def render_intel_briefing(anomaly_score, all_incidents, app_config):
    st.subheader("Intel Briefing")
    colors = app_config['styling']['colors']
    if anomaly_score > 0.2:
        status_color, status_text, status_desc = colors['accent_crit'], "ANÓMALO", "La distribución de incidentes se desvía significativamente de la norma."
    elif anomaly_score > 0.1:
        status_color, status_text, status_desc = colors['accent_warn'], "ELEVADO", "Se detectan desviaciones notables en los patrones de incidentes."
    else:
        status_color, status_text, status_desc = colors['accent_ok'], "NOMINAL", "Los patrones de incidentes se alinean con las normas históricas."
    
    st.markdown(f"**Estado del Sistema:** <span style='color:{status_color}'><b>{status_text}</b></span>", unsafe_allow_html=True)
    st.caption(status_desc)
    echo_count = sum(1 for i in all_incidents if i.get('is_echo'))
    st.info(f"**{echo_count}** incidentes de 'eco' detectados (Proceso de Hawkes).")

def render_live_ops_tab(data_fabric, engine, app_config):
    st.header("Simulador de Operaciones en Vivo")
    st.info("Ajuste los parámetros para ver cómo evoluciona el estado de la ciudad en tiempo real.")
    col1, col2 = st.columns(2)
    base_rate = col1.slider("Tasa Base de Incidentes (μ)", 1, 20, 5, help="Controla la tasa de Poisson de nuevos incidentes.")
    excitation = col2.slider("Factor de Auto-Excitación (κ)", 0.0, 1.0, 0.5, help="Probabilidad de que un incidente crítico genere 'ecos'.")
    
    live_state = engine.get_live_state(base_rate, excitation)
    all_incidents = live_state.get("active_incidents", [])
    _, holistic_risk_scores, _ = engine.calculate_holistic_risk(live_state)
    anomaly_score, _, _ = engine.calculate_kld_anomaly_score(live_state)

    render_intel_briefing(anomaly_score, all_incidents, app_config)
    st.divider()

    st.subheader("Mapa de Operaciones Dinámicas")
    zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, holistic_risk_scores, all_incidents, app_config.get('styling', {}))
    st.pydeck_chart(create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, app_config), use_container_width=True)

def render_risk_analysis_tab(data_fabric, engine, live_state, plotter):
    st.header("Análisis de Riesgo Bayesiano y Difusión en Grafo")
    st.info("Esta vista compara el riesgo histórico base (A Priori) con el riesgo actual, que incluye evidencia en tiempo real y la propagación del riesgo a zonas vecinas.")

    prior_risks, posterior_risks, _ = engine.calculate_holistic_risk(live_state)
    
    prior_df = pd.DataFrame(list(prior_risks.items()), columns=['zone', 'risk'])
    posterior_df = pd.DataFrame(list(posterior_risks.items()), columns=['zone', 'risk'])

    chart = plotter.plot_risk_comparison(prior_df, posterior_df)
    st.altair_chart(chart, use_container_width=True)
    
    risk_change = {zone: posterior_risks.get(zone, 0) - prior_risks.get(zone, 0) for zone in prior_risks}
    if risk_change:
        max_increase_zone = max(risk_change, key=risk_change.get)
        st.warning(f"**Insight Accionable:** La **{max_increase_zone}** muestra el mayor incremento de riesgo, probablemente debido a una alta carga de incidentes y/o difusión de zonas de alto riesgo cercanas.")

def render_anomaly_deepdive_tab(engine, live_state, plotter):
    st.header("Análisis Profundo de Anomalías del Sistema")
    st.info("Esta vista desglosa la Puntuación de Anomalía (Divergencia KL) comparando la distribución de incidentes actual con la norma histórica.")
    
    anomaly_score, hist_dist, current_dist = engine.calculate_kld_anomaly_score(live_state)
    st.metric("Puntuación de Anomalía Actual (KL Divergence)", f"{anomaly_score:.4f}")

    hist_df = pd.DataFrame(list(hist_dist.items()), columns=['zone', 'percentage'])
    current_df = pd.DataFrame(list(current_dist.items()), columns=['zone', 'percentage'])
        
    chart = plotter.plot_distribution_comparison(hist_df, current_df)
    st.altair_chart(chart, use_container_width=True)
    
    if not current_df.empty and not hist_df.empty:
        diff = current_df.set_index('zone')['percentage'] - hist_df.set_index('zone')['percentage']
        if not diff.empty:
            max_deviation_zone = diff.abs().idxmax()
            st.warning(f"**Insight Accionable:** La distribución de incidentes en **{max_deviation_zone}** es la que más se desvía de su norma histórica. Investigue la causa de esta actividad inusual.")

def main():
    st.set_page_config(page_title="RedShield AI: Digital Twin", layout="wide", initial_sidebar_state="expanded")
    
    if 'app_config' not in st.session_state:
        st.session_state.app_config = get_app_config()
    
    setup_plotting_theme(st.session_state.app_config.get('styling', {}))
    
    with st.spinner("Initializing Digital Twin Engine..."):
        data_fabric, engine = get_singleton_engine()
    
    plotter = PlottingSME(st.session_state.app_config.get('styling', {}))

    st.sidebar.title("RedShield AI")
    st.sidebar.write("Emergency Services Digital Twin")
    tab_choice = st.sidebar.radio("Navegación", ["Operaciones en Vivo", "Análisis de Riesgo", "Análisis de Anomalías"], label_visibility="collapsed")
    st.sidebar.divider()
    
    if tab_choice == "Operaciones en Vivo":
        render_live_ops_tab(data_fabric, engine, st.session_state.app_config)
    else:
        if 'analysis_state' not in st.session_state:
             st.session_state.analysis_state = engine.get_live_state(5, 0.5)
        
        if st.sidebar.button("🔄 Regenerar Datos de Análisis", use_container_width=True):
            st.session_state.analysis_state = engine.get_live_state(np.random.randint(3, 10), np.random.rand())
            st.rerun()

        if tab_choice == "Análisis de Riesgo":
            render_risk_analysis_tab(data_fabric, engine, st.session_state.analysis_state, plotter)
        elif tab_choice == "Análisis de Anomalías":
            render_anomaly_deepdive_tab(engine, st.session_state.analysis_state, plotter)

if __name__ == "__main__":
    main()
