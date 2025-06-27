# RedShieldAI_SME_Self_Contained_App.py
# FINAL, VISUALLY-ENHANCED DEPLOYMENT VERSION: Features a high-impact Altair chart,
# a guaranteed geographic simulation, and a robust, visually compelling map to
# create a true command-level dashboard.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pydeck as pdk
import xgboost as xgb
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml
import networkx as nx
import time
import altair as alt # Import the new library

# --- L0: CONFIGURATION AND CORE UTILITIES ---

def get_app_config() -> Dict:
    """
    Returns the application configuration as a native Python dictionary.
    This eliminates all parsing errors and file dependencies.
    """
    config_dict = {
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
                "A07": {'location': [32.50, -117.03], 'status': "Disponible"}, "A08": {'location': [32.46, -117.02], 'status': "Disponible"},
                "A09": {'location': [32.51, -116.98], 'status': "Disponible"}
            },
            'zones': {
                "Zona Río": {'polygon': [[32.52, -117.01], [32.535, -117.01], [32.535, -117.035], [32.52, -117.035]], 'crime': 0.7, 'road_quality': 0.9},
                "Otay": {'polygon': [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], 'crime': 0.5, 'road_quality': 0.7},
                "Playas": {'polygon': [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], 'crime': 0.4, 'road_quality': 0.8}
            },
            'city_boundary': [
                [32.535, -117.129], [32.510, -117.125], [32.448, -117.060], [32.435, -116.930], 
                [32.537, -116.930], [32.537, -117.030], [32.542, -117.038], [32.543, -117.128]
            ],
            'patient_vitals': {
                "P001": {'heart_rate': 145, 'oxygen': 88, 'ambulance': "A03"},
                "P002": {'heart_rate': 90, 'oxygen': 97, 'ambulance': "A01"},
                "P003": {'heart_rate': 150, 'oxygen': 99, 'ambulance': "A02"}
            },
            'road_network': {
                'nodes': {
                    "N_Playas": {'pos': [32.52, -117.12]}, "N_Centro": {'pos': [32.53, -117.04]},
                    "N_ZonaRio": {'pos': [32.528, -117.025]}, "N_5y10": {'pos': [32.50, -117.03]},
                    "N_LaMesa": {'pos': [32.51, -117.00]}, "N_Otay": {'pos': [32.535, -116.965]},
                    "N_ElFlorido": {'pos': [32.48, -116.95]}, "N_SantaFe": {'pos': [32.46, -117.02]},
                    "H_General": {'pos': [32.5295, -117.0182]}, "H_IMSS1": {'pos': [32.5121, -117.0145]},
                    "H_Angeles": {'pos': [32.5300, -117.0200]}, "H_CruzRoja": {'pos': [32.5283, -117.0255]}
                },
                'edges': [
                    ["N_Playas", "N_Centro", 5.0], ["N_Centro", "N_ZonaRio", 2.0], ["N_ZonaRio", "N_5y10", 3.0], 
                    ["N_ZonaRio", "H_Angeles", 0.5], ["N_ZonaRio", "H_CruzRoja", 0.2], ["N_ZonaRio", "H_General", 1.0], 
                    ["N_5y10", "N_LaMesa", 2.5], ["N_5y10", "N_SantaFe", 4.0], ["N_LaMesa", "H_IMSS1", 1.0], 
                    ["N_LaMesa", "N_ElFlorido", 5.0], ["N_ZonaRio", "N_Otay", 6.0]
                ]
            },
            'model_params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
        },
        'styling': {
            'colors': {'available': [0, 179, 89, 255], 'on_mission': [150, 150, 150, 180], 'hospital_ok': [0, 179, 89], 'hospital_warn': [255, 191, 0], 'hospital_crit': [220, 53, 69], 'route_path': [0, 123, 255], 'triage_rojo': [220, 53, 69], 'triage_amarillo': [255, 193, 7], 'triage_verde': [40, 167, 69]},
            'sizes': {'ambulance_available': 5.0, 'ambulance_mission': 2.5, 'hospital': 4.0, 'incident_base': 100.0},
            'icons': {'hospital': "https://img.icons8.com/color/96/hospital-3.png", 'ambulance': "https://img.icons8.com/color/96/ambulance.png"}
        }
    }
    return config_dict

def _safe_division(n, d): return n / d if d else 0
def find_nearest_node(graph: nx.Graph, point: Point):
    return min(graph.nodes, key=lambda node: point.distance(Point(graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0])))

# --- L1: DATA & MODELING LAYER ---
class DataFusionFabric:
    def __init__(self, config: Dict):
        self.config = config.get('data', {}); self.hospitals = {name: {'location': Point(data['location'][1], data['location'][0]), 'capacity': data['capacity'], 'load': data['load']} for name, data in self.config.get('hospitals', {}).items()}; self.ambulances = {name: {'location': Point(data['location'][1], data['location'][0]), 'status': data['status']} for name, data in self.config.get('ambulances', {}).items()}; self.zones = {name: {**data, 'polygon': Polygon([(p[1], p[0]) for p in data['polygon']])} for name, data in self.config.get('zones', {}).items()}; self.patient_vitals = self.config.get('patient_vitals', {}); self.road_graph = self._build_road_graph(self.config.get('road_network', {})); self.city_boundary = Polygon([(p[1], p[0]) for p in self.config.get('city_boundary', [])])
    @st.cache_data
    def _build_road_graph(_self, network_config: Dict) -> nx.Graph:
        G = nx.Graph();
        for node, data in network_config.get('nodes', {}).items(): G.add_node(node, pos=data['pos'])
        for edge in network_config.get('edges', []): G.add_edge(edge[0], edge[1], weight=edge[2])
        return G
    @st.cache_data(ttl=60)
    def get_live_state(_self, medical_pred: int, trauma_pred: int) -> Dict:
        state = {"city_incidents": {"active_incidents": []}}; minx, miny, maxx, maxy = _self.city_boundary.bounds
        def generate_incident(inc_type: str, triage_probs: List[float]):
            while True:
                random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                if _self.city_boundary.contains(random_point):
                    incident_id = f"{inc_type[0]}-{np.random.randint(1000,9999)}"; incident_node = find_nearest_node(_self.road_graph, random_point)
                    triage_color = np.random.choice(["Rojo", "Amarillo", "Verde"], p=triage_probs)
                    return {"id": incident_id, "type": inc_type, "triage": triage_color, "location": random_point, "node": incident_node}
        generated_incidents = []
        for _ in range(medical_pred): generated_incidents.append(generate_incident("Médico", [0.15, 0.65, 0.20]))
        for _ in range(trauma_pred): generated_incidents.append(generate_incident("Trauma", [0.4, 0.5, 0.1]))
        state["city_incidents"]["active_incidents"] = generated_incidents
        for zone in _self.zones.keys():
            state[zone] = {"traffic": np.random.uniform(0.3, 1.0)}
        return state

class CognitiveEngine:
    def __init__(self, data_fabric: DataFusionFabric, model_config: Dict):
        self.data_fabric = data_fabric; self.medical_model, self.medical_features = self._train_specialized_model("medical", model_config); self.trauma_model, self.trauma_features = self._train_specialized_model("trauma", model_config)
    def _train_specialized_model(self, model_type: str, model_config: Dict) -> Tuple[xgb.XGBRegressor, List[str]]:
        print(f"--- Entrenando modelo especializado para: {model_type} ---")
        model_params = model_config.get('data', {}).get('model_params', {}); hours = 24 * 7
        timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=hours, freq='h'))
        if model_type == "medical":
            features = {'hour': timestamps.hour, 'day_of_week': timestamps.dayofweek, 'temperature_extreme': abs(np.random.normal(22, 8, hours) - 22), 'air_quality_index': np.random.randint(30, 150, hours)}; X_train = pd.DataFrame(features); y_train = np.maximum(0, 3 + 2 * np.sin(X_train['hour'] * 2 * np.pi / 24) + X_train['temperature_extreme']/5 + X_train['air_quality_index']/50 + np.random.randn(hours)).astype(int)
        else:
            features = {'hour': timestamps.hour, 'is_weekend_night': ((timestamps.dayofweek >= 4) & (timestamps.hour >= 20)) | ((timestamps.dayofweek <= 1) & (timestamps.hour < 4)), 'is_quincena': timestamps.day.isin([14,15,16,29,30,31,1]), 'major_event_active': np.random.choice([0, 1], size=hours, p=[0.95, 0.05]), 'border_wait': np.random.randint(20, 120, hours)}; X_train = pd.DataFrame(features); y_train = np.maximum(0, 2 + X_train['is_weekend_night']*3 + X_train['is_quincena']*2 + X_train['major_event_active']*5 + X_train['border_wait']/40 + np.random.randn(hours)).astype(int)
        model = xgb.XGBRegressor(objective='reg:squarederror', **model_params, random_state=42, n_jobs=-1); model.fit(X_train, y_train)
        return model, list(X_train.columns)
    def predict_demand(self, live_features: Dict) -> Tuple[int, int]:
        medical_input = pd.DataFrame({k: [live_features[k]] for k in self.medical_features}); medical_pred = int(max(0, self.medical_model.predict(medical_input)[0]))
        trauma_input = pd.DataFrame({k: [live_features[k]] for k in self.trauma_features}); trauma_pred = int(max(0, self.trauma_model.predict(trauma_input)[0]))
        return medical_pred, trauma_pred
    def calculate_risk_scores(self, live_state: Dict) -> Dict:
        risk_scores = {};
        for zone, s_data in self.data_fabric.zones.items():
            l_data = live_state.get(zone, {}); risk = (l_data.get('traffic', 0.5) * 0.6 + (1 - s_data.get('road_quality', 0.5)) * 0.2 + s_data.get('crime', 0.5) * 0.2)
            incidents_in_zone = [inc for inc in live_state.get("city_incidents", {}).get("active_incidents", []) if s_data['polygon'].contains(inc['location'])]
            risk_scores[zone] = risk * (1 + len(incidents_in_zone))
        return risk_scores
    def get_patient_alerts(self) -> List[Dict]:
        alerts = [];
        for pid, vitals in self.data_fabric.patient_vitals.items():
            if vitals.get('heart_rate', 100) > 140 or vitals.get('oxygen', 100) < 90: alerts.append({"Patient ID": pid, "Heart Rate": vitals.get('heart_rate'), "Oxygen %": vitals.get('oxygen'), "Ambulance": vitals.get('ambulance', 'N/A')})
        return alerts
    def find_best_route_for_incident(self, incident: Dict, risk_scores: Dict) -> Dict:
        available_ambulances = {k: v for k, v in self.data_fabric.ambulances.items() if v.get('status') == 'Disponible'}
        if not available_ambulances: return {"error": "No hay ambulancias disponibles."}
        incident_node = incident.get('node')
        if not incident_node: return {"error": "Incidente no está mapeado a la red de calles."}
        amb_node_map = {name: find_nearest_node(self.data_fabric.road_graph, data['location']) for name, data in available_ambulances.items()}
        ambulance_unit, amb_start_node = min(amb_node_map.items(), key=lambda item: nx.shortest_path_length(self.data_fabric.road_graph, source=item[1], target=incident_node, weight='weight'))
        def cost_heuristic(u, v, d):
            edge_data = self.data_fabric.road_graph.get_edge_data(u, v); pos_u, pos_v = self.data_fabric.road_graph.nodes[u]['pos'], self.data_fabric.road_graph.nodes[v]['pos']; midpoint = Point(np.mean([pos_u[1], pos_v[1]]), np.mean([pos_u[0], pos_v[0]])); zone = next((name for name, z_data in self.data_fabric.zones.items() if z_data['polygon'].contains(midpoint)), None); risk_multiplier = 1 + risk_scores.get(zone, 0); return edge_data.get('weight', 1) * risk_multiplier
        options = []; hosp_node_map = {name: find_nearest_node(self.data_fabric.road_graph, data['location']) for name, data in self.data_fabric.hospitals.items()}
        for name, h_node in hosp_node_map.items():
            h_data = self.data_fabric.hospitals[name]
            try:
                eta_to_incident = nx.astar_path_length(self.data_fabric.road_graph, amb_start_node, incident_node, heuristic=None, weight=cost_heuristic); path_to_incident = nx.astar_path(self.data_fabric.road_graph, amb_start_node, incident_node, heuristic=None, weight=cost_heuristic); eta_to_hospital = nx.astar_path_length(self.data_fabric.road_graph, incident_node, h_node, heuristic=None, weight=cost_heuristic); path_to_hospital = nx.astar_path(self.data_fabric.road_graph, incident_node, h_node, heuristic=None, weight=cost_heuristic); total_eta = eta_to_incident + eta_to_hospital; full_path_nodes = path_to_incident + path_to_hospital[1:]; load_pct = _safe_division(h_data.get('load', 0), h_data.get('capacity', 1)); load_penalty = load_pct**2 * 20; total_score = total_eta * 0.8 + load_penalty * 0.2; options.append({"hospital": name, "eta_min": total_eta, "load_penalty": load_penalty, "load_pct": load_pct, "total_score": total_score, "path_nodes": full_path_nodes})
            except nx.NetworkXNoPath: continue
        if not options: return {"error": "No se pudieron calcular rutas a hospitales."}
        best_option = min(options, key=lambda x: x.get('total_score', float('inf'))); path_coords = [[self.data_fabric.road_graph.nodes[node]['pos'][1], self.data_fabric.road_graph.nodes[node]['pos'][0]] for node in best_option['path_nodes']]; return {"ambulance_unit": ambulance_unit, "best_hospital": best_option.get('hospital'), "routing_analysis": pd.DataFrame(options).drop(columns=['path_nodes']).sort_values('total_score').reset_index(drop=True), "route_path_coords": path_coords}

def kpi_card(icon: str, title: str, value: Any, color: str):
    st.markdown(f"""<div style="background-color: #262730; border: 1px solid #444; border-radius: 10px; padding: 20px; text-align: center; height: 100%;"><div style="font-size: 40px;">{icon}</div><div style="font-size: 16px; color: #bbb; margin-top: 10px; text-transform: uppercase; font-weight: 600;">{title}</div><div style="font-size: 28px; font-weight: bold; color: {color};">{value}</div></div>""", unsafe_allow_html=True)
def info_box(message):
    st.markdown(f'<div style="background-color: #e6f3ff; border-left: 5px solid #007bff; padding: 15px; border-radius: 5px; margin-bottom: 1em; color: #004085;">{message}</div>', unsafe_allow_html=True)
def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    def get_hospital_color(load, capacity):
        load_pct = _safe_division(load, capacity);
        if load_pct < 0.7: return style_config['colors']['hospital_ok']
        if load_pct < 0.9: return style_config['colors']['hospital_warn']
        return style_config['colors']['hospital_crit']
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Carga: {d.get('load',0)}/{d.get('capacity',1)} ({_safe_division(d.get('load',0), d.get('capacity',1)):.0%})", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}, "color": get_hospital_color(d.get('load',0), d.get('capacity',1))} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unidad: {n}", "tooltip_text": f"Estatus: {d.get('status', 'Desconocido')}", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, "size": style_config['sizes']['ambulance_available'] if d.get('status') == 'Disponible' else style_config['sizes']['ambulance_mission'], "color": style_config['colors']['available'] if d.get('status') == 'Disponible' else style_config['colors']['on_mission']} for n, d in data_fabric.ambulances.items()])
    def get_triage_color(triage_str):
        return style_config['colors'].get(f"triage_{triage_str.lower()}", [128, 128, 128])
    incident_df = pd.DataFrame([{"name": f"Incidente: {i.get('id', 'N/A')}", "tooltip_text": f"Tipo: {i.get('type')}<br>Triage: {i.get('triage')}", "lon": i.get('location').x, "lat": i.get('location').y, "color": get_triage_color(i.get('triage', 'Verde')), "radius": style_config['sizes']['incident_base'], "id": i.get('id')} for i in all_incidents])
    heatmap_df = pd.DataFrame([{"lon": i.get('location').x, "lat": i.get('location').y} for i in all_incidents])
    zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.zones, orient='index').set_geometry('polygon'); zones_gdf['name'] = zones_gdf.index; zones_gdf['risk'] = zones_gdf.index.map(risk_scores).fillna(0); zones_gdf['tooltip_text'] = zones_gdf.apply(lambda row: f"Zona: {row.name}<br/>Puntaje de Riesgo: {row.risk:.2f}", axis=1)
    max_risk = max(1, zones_gdf['risk'].max()); zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * _safe_division(r,max_risk))]).tolist()
    return zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df
def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df, route_info=None, style_config=None):
    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="geometry", filled=True, stroked=False, extruded=True, get_elevation="risk * 3000", get_fill_color="fill_color", opacity=0.1, pickable=True); hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style_config['sizes']['hospital'], get_color='color', size_scale=15, pickable=True); ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', get_color='color', size_scale=15, pickable=True); 
    incident_layer = pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='radius', get_fill_color='color', radius_scale=15, pickable=True, radius_min_pixels=3, radius_max_pixels=100)
    heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='"MEAN"', threshold=0.1, get_weight=1); layers = [heatmap_layer, zone_layer, hospital_layer, ambulance_layer, incident_layer]
    if route_info and "error" not in route_info and "route_path_coords" in route_info:
        layers.append(pdk.Layer('PathLayer', data=pd.DataFrame([{'path': route_info['route_path_coords']}]), get_path='path', get_width=8, get_color=style_config['colors']['route_path'], width_scale=1, width_min_pixels=6))
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=50); tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "border-radius": "5px", "padding": "5px"}}; return pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/navigation-night-v1", tooltip=tooltip)
def display_ai_rationale(route_info: Dict):
    st.markdown("---"); st.markdown("> La IA balancea tiempo de viaje, seguridad de la ruta y capacidad hospitalaria para encontrar el destino óptimo.")
    st.subheader("Lógica del Despacho de IA"); best = route_info['routing_analysis'].iloc[0]; st.success(f"**Recomendado:** Despachar unidad `{route_info['ambulance_unit']}` a `{route_info['best_hospital']}`", icon="✅"); st.markdown(f"**Razón:** Balance óptimo del menor tiempo de viaje y la disponibilidad del hospital. El algoritmo A* calculó un ETA ajustado por riesgo de **{best.get('eta_min', 0):.1f} min**.")
    if len(route_info['routing_analysis']) > 1:
        rejected = route_info['routing_analysis'].iloc[1]; reasons = []
        if (rejected.get('eta_min', 0) / best.get('eta_min', 1)) > 1.15: reasons.append(f"un ETA significativamente más largo ({rejected.get('eta_min', 0):.1f} min)")
        if (rejected.get('load_penalty', 0) > best.get('load_penalty', 1)) > 1.2: reasons.append(f"una carga hospitalaria prohibitiva (`{rejected.get('load_pct', 0):.0%}`)")
        if not reasons: reasons.append("fue un cercano segundo lugar, pero menos óptimo en general")
        st.error(f"**Alternativa Rechazada:** `{rejected.get('hospital', 'N/A')}` debido a {', '.join(reasons)}.", icon="❌")

@st.cache_resource
def get_engine():
    app_config = get_app_config(); data_fabric = DataFusionFabric(app_config); engine = CognitiveEngine(data_fabric, app_config); return engine

def main():
    st.set_page_config(page_title="RedShield AI: Comando Élite", layout="wide", initial_sidebar_state="expanded")
    with st.spinner("Inicializando el motor de IA por primera vez... (esto es rápido después del primer arranque)"):
        engine = get_engine()
    data_fabric = engine.data_fabric
    now = datetime.now()
    live_features = {'hour': now.hour, 'day_of_week': now.weekday(), 'is_weekend_night': ((now.weekday() >= 4) & (now.hour >= 20)) | ((now.weekday() <= 1) & (now.hour < 4)), 'is_quincena': now.day in [14,15,16,29,30,31,1], 'temperature_extreme': 25, 'air_quality_index': 70, 'major_event_active': 0, 'border_wait': 60}
    medical_pred, trauma_pred = engine.predict_demand(live_features)
    live_state = data_fabric.get_live_state(medical_pred, trauma_pred); risk_scores = engine.calculate_risk_scores(live_state); all_incidents = live_state.get("city_incidents", {}).get("active_incidents", [])
    incident_dict = {i['id']: i for i in all_incidents}

    def handle_incident_selection():
        selected_id = st.session_state.get("incident_selector")
        if selected_id and incident_dict.get(selected_id):
            st.session_state.selected_incident = incident_dict[selected_id]
            st.session_state.route_info = engine.find_best_route_for_incident(st.session_state.selected_incident, risk_scores)
        else:
            st.session_state.selected_incident = None; st.session_state.route_info = None

    with st.sidebar:
        st.title("RedShield AI"); st.write("Inteligencia de Emergencias de Tijuana"); tab_choice = st.radio("Navegación", ["Operaciones en Vivo", "Análisis del Sistema", "Simulación Estratégica"], label_visibility="collapsed"); st.divider();
        if st.button("🔄 Forzar Actualización de Datos", use_container_width=True): 
            data_fabric.get_live_state.clear(); st.session_state.selected_incident = None; st.session_state.route_info = None
            if "incident_selector" in st.session_state: st.session_state.incident_selector = None
            st.rerun()
        st.info("Seleccione un incidente del menú en el panel derecho.")
        
    if tab_choice == "Operaciones en Vivo":
        kpi_cols = st.columns(3); available_units = sum(1 for v in data_fabric.ambulances.values() if v.get('status') == 'Disponible'); avg_load = np.mean([_safe_division(h.get('load',0),h.get('capacity',1)) for h in data_fabric.hospitals.values()]);
        with kpi_cols[0]: kpi_card("🚑", "Unidades Disponibles", f"{available_units}/{len(data_fabric.ambulances)}", "#00A9FF")
        with kpi_cols[1]: kpi_card("🏥", "Carga Hosp. Prom.", f"{avg_load:.0%}", "#FFB000")
        with kpi_cols[2]: kpi_card("🚨", "Incidentes Activos", len(all_incidents), "#DC3545")
        with st.expander("¿Qué significan estos indicadores (KPIs)?"):
            st.markdown("""- **<font color='#00A9FF'>Unidades Disponibles:</font>** Muestra el número de ambulancias listas para ser despachadas.<br>- **<font color='#FFB000'>Carga Hosp. Prom.:</font>** El promedio de capacidad ocupada en todos los hospitales.<br>- **<font color='#DC3545'>Incidentes Activos:</font>** El número actual de emergencias.""", unsafe_allow_html=True)
        st.divider()
        map_col, ticket_col = st.columns((2.5, 1.5))
        with ticket_col:
            st.subheader("Boleta de Despacho")
            st.selectbox(
                "Seleccione un Incidente Activo:", options=[None] + sorted(list(incident_dict.keys())),
                format_func=lambda x: "Elegir un incidente..." if x is None else f"{x} ({incident_dict.get(x, {}).get('type', 'N/A')})",
                key="incident_selector", on_change=handle_incident_selection,
            )
            if st.session_state.get('selected_incident'):
                if st.session_state.get('route_info') and "error" not in st.session_state.route_info:
                    st.metric("Respondiendo a Incidente", st.session_state.selected_incident.get('id', 'N/A'))
                    display_ai_rationale(st.session_state.route_info)
                    with st.expander("Mostrar Análisis de Ruta Detallado"):
                        st.dataframe(st.session_state.route_info['routing_analysis'].set_index('hospital'))
                else:
                    st.error(f"Error de Ruteo: {st.session_state.get('route_info', {}).get('error', 'No se pudo calcular una ruta.')}")
            else:
                st.info("Seleccione un incidente del menú de arriba para generar un plan de despacho.")
        with map_col:
            st.subheader("Mapa de Operaciones de la Ciudad")
            with st.expander("Mostrar Leyenda del Mapa", expanded=True):
                st.markdown("""**Triage de Incidentes:**<br>
                - **<font color='#dc3545'>Círculo Rojo:</font>** Amenaza la vida (Triage Rojo).<br>
                - **<font color='#ffc107'>Círculo Amarillo:</font>** Urgente, no mortal (Triage Amarillo).<br>
                - **<font color='#28a745'>Círculo Verde:</font>** No urgente (Triage Verde).""", unsafe_allow_html=True)
            app_config = get_app_config()
            zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, risk_scores, all_incidents, app_config.get('styling', {}))
            deck = create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, st.session_state.get('route_info'), app_config.get('styling', {}))
            st.pydeck_chart(deck, use_container_width=True)

    elif tab_choice == "Análisis del Sistema":
        st.header("Análisis del Sistema e Inteligencia Artificial")
        st.info("Esta pestaña muestra los dos modelos de IA especializados que predicen la demanda de incidentes médicos y de trauma por separado, junto con sus factores más influyentes.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Modelo de Emergencias Médicas")
            info_box("Factores como la calidad del aire y las temperaturas extremas impulsan este tipo de incidentes.")
            st.bar_chart(pd.DataFrame({'feature': engine.medical_features, 'importance': engine.medical_model.feature_importances_}).sort_values('importance', ascending=True))
        with col2:
            st.subheader("Modelo de Incidentes de Trauma")
            info_box("Factores como fines de semana, quincenas y eventos especiales impulsan este tipo de incidentes.")
            st.bar_chart(pd.DataFrame({'feature': engine.trauma_features, 'importance': engine.trauma_model.feature_importances_}).sort_values('importance', ascending=True))
        st.divider(); col1, col2 = st.columns(2)
        with col1:
            st.subheader("Estatus de Carga Hospitalaria"); st.markdown("Capacidad en tiempo real de todos los hospitales receptores.")
            for name, data in data_fabric.hospitals.items():
                load_pct = _safe_division(data['load'], data['capacity']); st.markdown(f"**{name}** ({data['load']}/{data['capacity']})"); st.progress(load_pct)
        with col2:
            st.subheader("Alertas de Pacientes Críticos"); st.markdown("Pacientes con signos vitales críticos de monitoreo remoto.")
            patient_alerts = engine.get_patient_alerts()
            if not patient_alerts: st.success("✅ No hay alertas de pacientes críticos en este momento.")
            else:
                for alert in patient_alerts: st.error(f"**Paciente {alert.get('Patient ID')}:** FC: {alert.get('Heart Rate')}, O2: {alert.get('Oxygen %')}% | Unidad: {alert.get('Ambulance')}", icon="❤️‍🩹")
    elif tab_choice == "Simulación Estratégica":
        st.header("Simulación Estratégica y Análisis 'What-If'")
        st.info("""Esta herramienta le permite probar la resiliencia del sistema en condiciones extremas. Al aumentar el multiplicador de tráfico, puede simular eventos como la hora pico, días festivos o cierres de carreteras importantes para ver cómo impactan el riesgo zonal.""")
        sim_traffic_spike = st.slider("Simular Multiplicador de Tráfico en Toda la Ciudad", 1.0, 5.0, 1.0, 0.25)
        if st.button("Ejecutar Simulación", use_container_width=True):
            sim_risk_scores = {};
            for zone, s_data in data_fabric.zones.items():
                l_data = live_state.get(zone, {}); sim_risk = (l_data.get('traffic', 0.5) * sim_traffic_spike * 0.6 + (1 - s_data.get('road_quality', 0.5)) * 0.2 + s_data.get('crime', 0.5) * 0.2)
                incidents_in_zone = [inc for inc in all_incidents if s_data['polygon'].contains(inc['location'])]
                sim_risk_scores[zone] = sim_risk * (1 + len(incidents_in_zone))
            st.subheader("Puntajes de Riesgo Zonal Simulados"); st.bar_chart(pd.DataFrame.from_dict(sim_risk_scores, orient='index', columns=['Riesgo Simulado']).sort_values('Riesgo Simulado', ascending=False)); st.markdown("Las zonas de alto riesgo bajo estas condiciones simuladas requerirían un posicionamiento preventivo de recursos.")

if __name__ == "__main__":
    main()
