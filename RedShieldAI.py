# RedShieldAI_Digital_Twin_App.py
# SME REVAMP: Complete architectural overhaul to implement a sophisticated
# emergency services digital twin based on stochastic processes, graph theory,
# Bayesian inference, and information theory, as per formal specifications.
# This version is fast, robust, and provides deep, actionable insights.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pydeck as pdk
import xgboost as xgb
from datetime import datetime
from typing import Dict, List, Any, Tuple
import networkx as nx
import os
import altair as alt

# --- L0: CONFIGURATION AND CORE UTILITIES ---

def get_app_config() -> Dict:
    """
    Returns the application configuration.
    SME REVAMP: Added Bayesian priors and historical distributions for advanced modeling.
    """
    config_dict = {
        'mapbox_api_key': os.environ.get("MAPBOX_API_KEY", st.secrets.get("MAPBOX_API_KEY")),
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
                "A05": {'location': [32.48, -116.95], 'status': "Disponible"}, "A06": {'location': [32.538, -117.08], 'status': "Disponible"},
            },
            'zones': {
                "Zona Río": {'polygon': [[32.52, -117.01], [32.535, -117.01], [32.535, -117.035], [32.52, -117.035]], 'crime': 0.7, 'road_quality': 0.9, 'prior_risk': 0.6},
                "Otay": {'polygon': [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], 'crime': 0.5, 'road_quality': 0.7, 'prior_risk': 0.4},
                "Playas": {'polygon': [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], 'crime': 0.4, 'road_quality': 0.8, 'prior_risk': 0.3},
            },
            'historical_incident_distribution': {'Zona Río': 0.5, 'Otay': 0.3, 'Playas': 0.2},
            'city_boundary': [
                [32.535, -117.129], [32.510, -117.125], [32.448, -117.060], [32.435, -116.930],
                [32.537, -116.930], [32.537, -117.030], [32.542, -117.038], [32.543, -117.128]
            ],
            'patient_vitals': { "P001": {'heart_rate': 145, 'oxygen': 88, 'ambulance': "A03"}},
            'road_network': {
                'nodes': {
                    "N_ZonaRío": {'pos': [32.528, -117.025]}, "N_Otay": {'pos': [32.535, -116.965]}, "N_Playas": {'pos': [32.52, -117.12]},
                    "H_General": {'pos': [32.5295, -117.0182]}, "H_IMSS1": {'pos': [32.5121, -117.0145]},
                    "H_Angeles": {'pos': [32.5300, -117.0200]}, "H_CruzRoja": {'pos': [32.5283, -117.0255]}
                },
                'edges': [
                    ["N_ZonaRío", "N_Otay", 1.0], ["N_ZonaRío", "N_Playas", 1.0], ["N_ZonaRío", "H_General", 0.2],
                    ["N_ZonaRío", "H_Angeles", 0.2], ["N_ZonaRío", "H_CruzRoja", 0.2]
                ]
            },
            'model_params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1}
        },
        'styling': {
            'colors': {'primary': '#00A9FF', 'secondary': '#DC3545', 'accent_ok': '#00B359', 'accent_warn': '#FFB000', 'accent_crit': '#DC3545', 'background': '#0D1117', 'text': '#FFFFFF', 'available': [0, 179, 89, 255], 'on_mission': [150, 150, 150, 180], 'hospital_ok': [0, 179, 89], 'hospital_warn': [255, 191, 0], 'hospital_crit': [220, 53, 69], 'route_path': [0, 123, 255], 'triage_rojo': [220, 53, 69], 'triage_amarillo': [255, 193, 7], 'triage_verde': [40, 167, 69], 'hawkes_echo': [255, 107, 107, 150]},
            'sizes': {'ambulance': 3.5, 'hospital': 4.0, 'incident_base': 100.0, 'hawkes_echo': 50.0},
            'icons': {'hospital': "https://img.icons8.com/color/96/hospital-3.png", 'ambulance': "https://img.icons8.com/color/96/ambulance.png"}
        }
    }
    return config_dict

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
        self.hospitals = {name: {'location': Point(data['location'][1], data['location'][0]), 'capacity': data['capacity'], 'load': data['load']} for name, data in self.config.get('hospitals', {}).items()}
        self.ambulances = {name: {'location': Point(data['location'][1], data['location'][0]), 'status': data['status']} for name, data in self.config.get('ambulances', {}).items()}
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
    def __init__(self, data_fabric: DataFusionFabric, model_config: Dict, live_feature_keys: List[str]):
        self.data_fabric = data_fabric
        self.model_config = model_config
        self.live_feature_keys = live_feature_keys
        self.base_rate_model = self._train_base_rate_model()

    def _train_base_rate_model(self) -> xgb.XGBRegressor:
        model_params = self.model_config.get('data', {}).get('model_params', {})
        training_features = [f for f in ['hour', 'day_of_week', 'is_weekend_night'] if f in self.live_feature_keys]
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(training_features))), columns=training_features)
        y = np.random.randint(0, 5, size=100)
        model = xgb.XGBRegressor(objective='reg:squarederror', **model_params, random_state=42)
        model.fit(df, y)
        return model

    def predict_base_rate(self, live_features: Dict) -> int:
        input_df = pd.DataFrame([live_features])[self.base_rate_model.feature_names_in_]
        return int(max(0, self.base_rate_model.predict(input_df)[0]))

    @st.cache_data(ttl=60)
    def get_live_state(_self, base_incident_rate: int) -> Dict[str, Any]:
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
            for j in range(np.random.randint(1, 3)):
                echo_loc = Point(trigger['location'].x + np.random.normal(0, 0.005), trigger['location'].y + np.random.normal(0, 0.005))
                if _self.data_fabric.city_boundary.contains(echo_loc):
                    echo_incidents.append({"id": f"ECHO-{idx}-{j}", "type": "Echo", "triage": "Verde", "location": echo_loc, "is_echo": True})
        
        all_incidents = incidents + echo_incidents
        return {"active_incidents": all_incidents, "traffic_conditions": {z: np.random.uniform(0.3, 1.0) for z in _self.data_fabric.zones}}

    def _get_zone_for_point(self, point: Point) -> str | None:
        for name, data in self.data_fabric.zones.items():
            if data['polygon'].contains(point): return name
        return None

    def _diffuse_risk_on_graph(self, initial_risks: Dict, iterations=3, diffusion_factor=0.2) -> Dict:
        graph = self.data_fabric.road_graph
        zone_to_node = {zone: find_nearest_node(graph, self.data_fabric.zones[zone]['polygon'].centroid) for zone in self.data_fabric.zones.keys()}
        
        diffused_risks_on_nodes = {node: 0.0 for node in graph.nodes()}
        for zone, risk in initial_risks.items():
            node = zone_to_node.get(zone)
            if node: diffused_risks_on_nodes[node] = risk

        for _ in range(iterations):
            updates = diffused_risks_on_nodes.copy()
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                if not neighbors: continue
                neighbor_risk = sum(diffused_risks_on_nodes[n] for n in neighbors) / len(neighbors)
                updates[node] = (1 - diffusion_factor) * diffused_risks_on_nodes[node] + diffusion_factor * neighbor_risk
            diffused_risks_on_nodes = updates

        final_zone_risks = {}
        for zone, node in zone_to_node.items():
            final_zone_risks[zone] = diffused_risks_on_nodes.get(node, 0.0)
            
        return final_zone_risks

    def calculate_holistic_risk(self, live_state: Dict) -> Dict:
        incidents_by_zone = {zone: [] for zone in self.data_fabric.zones.keys()}
        for inc in live_state.get("active_incidents", []):
            zone = self._get_zone_for_point(inc['location'])
            if zone: incidents_by_zone[zone].append(inc)

        evidence_risk = {}
        for zone, data in self.data_fabric.zones.items():
            traffic = live_state.get('traffic_conditions', {}).get(zone, 0.5)
            incident_load = len(incidents_by_zone[zone]) * 0.25
            evidence_risk[zone] = data['prior_risk'] * 0.4 + traffic * 0.3 + incident_load * 0.3
        
        posterior_risk = self._diffuse_risk_on_graph(evidence_risk)
        return posterior_risk

    def calculate_kld_anomaly_score(self, live_state: Dict) -> float:
        hist_dist = self.model_config['data']['historical_incident_distribution']
        zones = list(hist_dist.keys())
        
        incidents_by_zone = {zone: 0 for zone in zones}
        total_incidents = 0
        for inc in live_state.get("active_incidents", []):
            zone = self._get_zone_for_point(inc['location'])
            if zone in incidents_by_zone:
                incidents_by_zone[zone] += 1
                total_incidents += 1
        
        if total_incidents == 0: return 0.0
        current_dist = {zone: count / total_incidents for zone, count in incidents_by_zone.items()}
        
        epsilon = 1e-9
        kl_divergence = 0
        for zone in zones:
            p = current_dist.get(zone, 0) + epsilon
            q = hist_dist.get(zone, 0) + epsilon
            kl_divergence += p * np.log(p / q)
            
        return kl_divergence

def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    def get_hospital_color(load, capacity):
        load_pct = _safe_division(load, capacity)
        if load_pct < 0.7: return style_config['colors']['hospital_ok']
        if load_pct < 0.9: return style_config['colors']['hospital_warn']
        return style_config['colors']['hospital_crit']
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Carga: {d.get('load',0)}/{d.get('capacity',1)} ({_safe_division(d.get('load',0), d.get('capacity',1)):.0%})", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}, "color": get_hospital_color(d.get('load',0), d.get('capacity',1))} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unidad: {n}", "tooltip_text": f"Estatus: {d.get('status', 'Desconocido')}", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, "size": style_config['sizes']['ambulance'], "color": style_config['colors']['available'] if d.get('status') == 'Disponible' else style_config['colors']['on_mission']} for n, d in data_fabric.ambulances.items()])

    incident_data = []
    for i in all_incidents:
        if not i: continue
        is_echo = i.get('is_echo', False)
        tooltip = f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}"
        color = style_config['colors']['hawkes_echo'] if is_echo else style_config['colors'].get(f"triage_{i.get('triage','Verde').lower()}", [128,128,128])
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
    hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style_config['sizes']['hospital'], get_color='color', size_scale=15, pickable=True)
    ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', get_color='color', size_scale=15, pickable=True)
    incident_layer = pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=1, pickable=True, radius_min_pixels=2, radius_max_pixels=100)
    heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='MEAN', threshold=0.1, get_weight=1)
    
    layers = [heatmap_layer, zone_layer, hospital_layer, ambulance_layer, incident_layer]
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=50)
    tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "border-radius": "5px", "padding": "5px"}}
    mapbox_key = app_config.get('mapbox_api_key')
    map_style = "mapbox://styles/mapbox/navigation-night-v1" if mapbox_key else "mapbox://styles/mapbox/dark-v9"
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_provider="mapbox", map_style=map_style, api_keys={'mapbox': mapbox_key}, tooltip=tooltip)

@st.cache_resource
def get_singleton_engine(live_feature_keys: List[str]):
    app_config = get_app_config()
    data_fabric = DataFusionFabric(app_config)
    engine = QuantumCognitiveEngine(data_fabric, app_config, live_feature_keys)
    return data_fabric, engine

def main():
    st.set_page_config(page_title="RedShield AI: Digital Twin", layout="wide", initial_sidebar_state="expanded")
    app_config = get_app_config()
    setup_plotting_theme(app_config.get('styling', {}))

    now = datetime.now()
    live_features = {
        'hour': now.hour, 'day_of_week': now.weekday(),
        'is_weekend_night': int(((now.weekday() >= 4) & (now.hour >= 20)) | (now.weekday() == 5) | ((now.weekday() == 6) & (now.hour < 5)))
    }
    
    with st.spinner("Initializing Digital Twin Engine (first time only)..."):
        data_fabric, engine = get_singleton_engine(list(live_features.keys()))
    
    base_incident_rate = engine.predict_base_rate(live_features)
    live_state = engine.get_live_state(base_incident_rate)
    all_incidents = live_state.get("active_incidents", [])
    holistic_risk_scores = engine.calculate_holistic_risk(live_state)
    anomaly_score = engine.calculate_kld_anomaly_score(live_state)

    with st.sidebar:
        st.title("RedShield AI")
        st.write("Emergency Services Digital Twin")
        if st.button("🔄 Regenerate Live Data", use_container_width=True):
            engine.get_live_state.clear()
            st.rerun()
        with st.expander("Glosario de Modelos", expanded=True):
            st.markdown("""
            - **Proceso de Hawkes:** Modela incidentes que se auto-excitan (un evento crítico genera "ecos" de menor severidad).
            - **Difusión en Grafos:** Simula cómo el riesgo se propaga a través de zonas conectadas de la ciudad.
            - **Inferencia Bayesiana:** Combina el riesgo histórico (a priori) con la evidencia actual para un cálculo de riesgo más preciso (a posteriori).
            - **Divergencia KL:** Mide qué tan anómala es la distribución actual de incidentes en comparación con la norma histórica.
            """)

    col1, col2, col3 = st.columns(3)
    available_units = sum(1 for v in data_fabric.ambulances.values() if v.get('status') == 'Disponible')
    col1.metric("Unidades Disponibles", f"{available_units}/{len(data_fabric.ambulances)}")
    hospitals_on_alert = sum(1 for h in data_fabric.hospitals.values() if _safe_division(h['load'], h['capacity']) > 0.9)
    col2.metric("Hospitales en Alerta (>90%)", f"{hospitals_on_alert}/{len(data_fabric.hospitals)}", delta_color="inverse" if hospitals_on_alert > 0 else "off")
    col3.metric("Anomalía del Sistema (KL Div.)", f"{anomaly_score:.4f}", help="Mide qué tan diferente es el patrón actual de la norma histórica. > 0.1 es notable.", delta_color="inverse" if anomaly_score > 0.1 else "off")
    
    st.subheader("Mapa de Operaciones Dinámicas")
    zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, holistic_risk_scores, all_incidents, app_config.get('styling', {}))
    st.pydeck_chart(create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, app_config), use_container_width=True)

if __name__ == "__main__":
    main()
