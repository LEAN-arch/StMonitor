# RedShieldAI_SME_Self_Contained_App.py
# FINAL, ROBUST DEPLOYMENT VERSION 10: Implements a standard callback-driven
# state management pattern for the dropdown, which is the most reliable method
# in Streamlit. This guarantees correct behavior on user selection.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pydeck as pdk
import xgboost as xgb
from datetime import datetime
from typing import Dict, List, Any
import yaml
import networkx as nx
import os
import json
import time

# --- L0: CONFIGURATION, PATHS, AND SELF-SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.yaml')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'demand_model.xgb')
FEATURES_FILE = os.path.join(SCRIPT_DIR, 'model_features.json')
LOCK_FILE = os.path.join(SCRIPT_DIR, '.model_lock')

def _train_and_save_model():
    try:
        with open(LOCK_FILE, 'x') as f: f.write(str(os.getpid()))
    except FileExistsError:
        st.info("Another user is initializing the AI model. Please wait a moment...")
        while os.path.exists(LOCK_FILE): time.sleep(2)
        return
    try:
        with st.spinner("First-time setup: Training demand forecast model... This may take a minute."):
            print("--- Starting one-time RedShield AI Model Training ---")
            with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
            model_params = config.get('data', {}).get('model_params', {})
            hours = 24 * 365; timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=hours, freq='h'))
            X_train = pd.DataFrame({'hour': timestamps.hour, 'day_of_week': timestamps.dayofweek, 'is_quincena': timestamps.day.isin([14,15,16,29,30,31,1]), 'temperature': np.random.normal(22, 5, hours), 'border_wait': np.random.randint(20, 120, hours)})
            y_train = np.maximum(0, 5 + 3 * np.sin(X_train['hour'] * 2 * np.pi / 24) + X_train['is_quincena'] * 5 + X_train['border_wait']/20 + np.random.randn(hours)).astype(int)
            model = xgb.XGBRegressor(objective='reg:squarederror', **model_params, random_state=42, n_jobs=-1); model.fit(X_train, y_train)
            model.save_model(MODEL_FILE); features = list(X_train.columns)
            with open(FEATURES_FILE, 'w') as f: json.dump(features, f)
            print("--- Model Training Successful ---")
    finally:
        if os.path.exists(LOCK_FILE): os.remove(LOCK_FILE)

@st.cache_data
def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

@st.cache_resource
def load_demand_model() -> tuple:
    if not os.path.exists(MODEL_FILE):
        _train_and_save_model()
        if not os.path.exists(MODEL_FILE):
             st.error("Model training failed. Please check the logs."); st.stop()
        st.rerun()
    model = xgb.XGBRegressor(); model.load_model(MODEL_FILE)
    with open(FEATURES_FILE, 'r') as f: features = json.load(f)
    return model, features

def _safe_division(n, d): return n / d if d else 0
def find_nearest_node(graph: nx.Graph, point: Point):
    return min(graph.nodes, key=lambda node: point.distance(Point(graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0])))

# --- L1 & L2 (Unchanged) ---
class DataFusionFabric:
    def __init__(self, config: Dict):
        self.config = config.get('data', {}); self.hospitals = {name: {'location': Point(data['location'][1], data['location'][0]), 'capacity': data['capacity'], 'load': data['load']} for name, data in self.config.get('hospitals', {}).items()}; self.ambulances = {name: {'location': Point(data['location'][1], data['location'][0]), 'status': data['status']} for name, data in self.config.get('ambulances', {}).items()}; self.zones = {name: {**data, 'polygon': Polygon([(p[1], p[0]) for p in data['polygon']])} for name, data in self.config.get('zones', {}).items()}; self.patient_vitals = self.config.get('patient_vitals', {}); self.road_graph = self._build_road_graph(self.config.get('road_network', {}))
    @st.cache_data
    def _build_road_graph(_self, network_config: Dict) -> nx.Graph:
        G = nx.Graph();
        for node, data in network_config.get('nodes', {}).items(): G.add_node(node, pos=data['pos'])
        for edge in network_config.get('edges', []): G.add_edge(edge[0], edge[1], weight=edge[2])
        return G
    @st.cache_data(ttl=60)
    def get_live_state(_self) -> Dict:
        state = {}; all_nodes = list(_self.road_graph.nodes)
        for zone, data in _self.zones.items():
            incidents = []
            for _ in range(np.random.randint(0, 4)):
                node = np.random.choice(all_nodes); loc_coords = _self.road_graph.nodes[node]['pos']; incident_point = Point(loc_coords[1], loc_coords[0])
                if data['polygon'].contains(incident_point): incidents.append({"id": f"I-{zone[:2].upper()}{np.random.randint(100,999)}", "location": incident_point, "priority": np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]), "node": node})
            state[zone] = {"traffic": np.random.uniform(0.3, 1.0), "active_incidents": incidents}
        return state

class CognitiveEngine:
    def __init__(self, data_fabric: DataFusionFabric):
        self.data_fabric = data_fabric; self.demand_model, self.model_features = load_demand_model()
    def predict_citywide_demand(self, features: Dict) -> float:
        input_df = pd.DataFrame([features], columns=self.model_features); return max(0, self.demand_model.predict(input_df)[0])
    def calculate_risk_scores(self, live_state: Dict) -> Dict:
        risk_scores = {};
        for zone, s_data in self.data_fabric.zones.items():
            l_data = live_state.get(zone, {}); risk = (l_data.get('traffic', 0.5) * 0.6 + (1 - s_data.get('road_quality', 0.5)) * 0.2 + s_data.get('crime', 0.5) * 0.2); risk_scores[zone] = risk * (1 + len(l_data.get('active_incidents', [])))
        return risk_scores
    def get_patient_alerts(self) -> List[Dict]:
        alerts = []
        for pid, vitals in self.data_fabric.patient_vitals.items():
            if vitals.get('heart_rate', 100) > 140 or vitals.get('oxygen', 100) < 90: alerts.append({"Patient ID": pid, "Heart Rate": vitals.get('heart_rate'), "Oxygen %": vitals.get('oxygen'), "Ambulance": vitals.get('ambulance', 'N/A')})
        return alerts
    def find_best_route_for_incident(self, incident: Dict, risk_scores: Dict) -> Dict:
        available_ambulances = {k: v for k, v in self.data_fabric.ambulances.items() if v.get('status') == 'Available'}
        if not available_ambulances: return {"error": "No available ambulances."}
        incident_node = incident.get('node')
        if not incident_node: return {"error": "Incident is not mapped to the road network."}
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
        if not options: return {"error": "No valid hospital routes could be calculated."}
        best_option = min(options, key=lambda x: x.get('total_score', float('inf'))); path_coords = [[self.data_fabric.road_graph.nodes[node]['pos'][1], self.data_fabric.road_graph.nodes[node]['pos'][0]] for node in best_option['path_nodes']]; return {"ambulance_unit": ambulance_unit, "best_hospital": best_option.get('hospital'), "routing_analysis": pd.DataFrame(options).drop(columns=['path_nodes']).sort_values('total_score').reset_index(drop=True), "route_path_coords": path_coords}

def kpi_card(icon: str, title: str, value: Any, color: str):
    st.markdown(f"""<div style="background-color: #262730; border: 1px solid #444; border-radius: 10px; padding: 20px; text-align: center; height: 100%;"><div style="font-size: 40px;">{icon}</div><div style="font-size: 16px; color: #bbb; margin-top: 10px; text-transform: uppercase; font-weight: 600;">{title}</div><div style="font-size: 28px; font-weight: bold; color: {color};">{value}</div></div>""", unsafe_allow_html=True)
def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    def get_hospital_color(load, capacity):
        load_pct = _safe_division(load, capacity);
        if load_pct < 0.7: return style_config['colors']['hospital_ok']
        if load_pct < 0.9: return style_config['colors']['hospital_warn']
        return style_config['colors']['hospital_crit']
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Load: {d.get('load',0)}/{d.get('capacity',1)} ({_safe_division(d.get('load',0), d.get('capacity',1)):.0%})", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['hospital'], "width": 128, "height": 128, "anchorY": 128}, "color": get_hospital_color(d.get('load',0), d.get('capacity',1))} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unit: {n}", "tooltip_text": f"Status: {d.get('status', 'Unknown')}", "lon": d.get('location').x, "lat": d.get('location').y, "icon_data": {"url": style_config['icons']['ambulance'], "width": 128, "height": 128, "anchorY": 128}, "size": style_config['sizes']['ambulance_available'] if d.get('status') == 'Available' else style_config['sizes']['ambulance_mission'], "color": style_config['colors']['available'] if d.get('status') == 'Available' else style_config['colors']['on_mission']} for n, d in data_fabric.ambulances.items()])
    incident_df = pd.DataFrame([{"name": f"Incident: {i.get('id', 'N/A')}", "tooltip_text": f"Priority: {i.get('priority', 1)}", "lon": i.get('location').x, "lat": i.get('location').y, "size": style_config['sizes']['incident_base'] + i.get('priority', 1)**2, "id": i.get('id')} for i in all_incidents])
    heatmap_df = pd.DataFrame([{"lon": i.get('location').x, "lat": i.get('location').y} for i in all_incidents])
    zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.zones, orient='index').set_geometry('polygon'); zones_gdf['name'] = zones_gdf.index; zones_gdf['risk'] = zones_gdf.index.map(risk_scores).fillna(0); zones_gdf['tooltip_text'] = zones_gdf.apply(lambda row: f"Zone: {row.name}<br/>Risk Score: {row.risk:.2f}", axis=1)
    max_risk = max(1, zones_gdf['risk'].max()); zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [220, 53, 69, int(200 * _safe_division(r,max_risk))]).tolist()
    return zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df
def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, heatmap_df, route_info=None, style_config=None):
    # This map is now for visualization only. 'pickable' is still useful for tooltips.
    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="geometry", filled=True, stroked=False, extruded=True, get_elevation="risk * 3000", get_fill_color="fill_color", opacity=0.1, pickable=True); hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style_config['sizes']['hospital'], get_color='color', size_scale=15, pickable=True); ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', get_color='color', size_scale=15, pickable=True); incident_layer = pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='size*20', get_fill_color=style_config['colors']['incident_halo'], pickable=True, radius_min_pixels=5, stroked=True, get_line_width=100, get_line_color=[*style_config['colors']['incident_halo'], 100]); heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position='[lon, lat]', opacity=0.3, aggregation='"MEAN"', threshold=0.1, get_weight=1); layers = [heatmap_layer, zone_layer, hospital_layer, ambulance_layer, incident_layer]
    if route_info and "error" not in route_info and "route_path_coords" in route_info:
        layers.append(pdk.Layer('PathLayer', data=pd.DataFrame([{'path': route_info['route_path_coords']}]), get_path='path', get_width=5, get_color=style_config['colors']['route_path'], width_scale=1, width_min_pixels=5))
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=50); tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border": "1px solid #555", "border-radius": "5px", "padding": "5px"}}; return pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip=tooltip)
def display_ai_rationale(route_info: Dict):
    st.subheader("AI Dispatch Rationale"); best = route_info['routing_analysis'].iloc[0]; st.success(f"**Recommended:** Dispatch `{route_info['ambulance_unit']}` to `{route_info['best_hospital']}`", icon="✅"); st.markdown(f"**Reason:** Optimal balance of lowest travel time and hospital readiness. The A* algorithm calculated a total risk-adjusted ETA of **{best.get('eta_min', 0):.1f} min** via the city's road network, while accounting for a hospital load penalty of `{best.get('load_penalty',0):.1f}`.")
    if len(route_info['routing_analysis']) > 1:
        rejected = route_info['routing_analysis'].iloc[1]; reasons = []
        if (rejected.get('eta_min', 0) / best.get('eta_min', 1)) > 1.15: reasons.append(f"a significantly longer ETA ({rejected.get('eta_min', 0):.1f} min)")
        if (rejected.get('load_penalty', 0) > best.get('load_penalty', 1)) > 1.2: reasons.append(f"prohibitive hospital load (`{rejected.get('load_pct', 0):.0%}`)")
        if not reasons: reasons.append("it was a close second but less optimal overall")
        st.error(f"**Alternative Rejected:** `{rejected.get('hospital', 'N/A')}` due to {', '.join(reasons)}.", icon="❌")

# --- L3: MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="RedShield AI: Elite Command", layout="wide", initial_sidebar_state="expanded")
    
    # Initialize stateful objects
    config = load_config(CONFIG_FILE)
    if 'cognitive_engine' not in st.session_state:
        st.session_state.cognitive_engine = CognitiveEngine(DataFusionFabric(config))
    engine = st.session_state.cognitive_engine
    data_fabric = engine.data_fabric

    # Get live data for this run
    live_state = data_fabric.get_live_state()
    risk_scores = engine.calculate_risk_scores(live_state)
    all_incidents = [inc for zone_data in live_state.values() for inc in zone_data.get('active_incidents', [])]
    
    # Map incident IDs to the incident data for easy lookup
    incident_dict = {i['id']: i for i in all_incidents}

    # ##################################################################
    # ###############     ROBUST CALLBACK-BASED LOGIC      ###############
    # ##################################################################
    def handle_incident_selection():
        """This function is called when the user selects an incident from the dropdown."""
        selected_id = st.session_state.incident_selector # Get the selected ID
        if selected_id:
            # If an ID is selected, find the incident and calculate the route
            st.session_state.selected_incident = incident_dict.get(selected_id)
            st.session_state.route_info = engine.find_best_route_for_incident(
                st.session_state.selected_incident, risk_scores
            )
        else:
            # If the user deselects (chooses placeholder), clear the info
            st.session_state.selected_incident = None
            st.session_state.route_info = None

    with st.sidebar:
        st.title("RedShield AI"); st.write("Tijuana Emergency Intelligence"); tab_choice = st.radio("Navigation", ["Live Operations", "System Analytics", "Strategic Simulation"], label_visibility="collapsed"); st.divider();
        if st.button("🔄 Force Refresh Live Data", use_container_width=True): data_fabric.get_live_state.clear(); st.rerun()
        st.info("Select an incident from the dropdown in the right panel to generate a dispatch plan.")
        
    if tab_choice == "Live Operations":
        kpi_cols = st.columns(3); available_units = sum(1 for v in data_fabric.ambulances.values() if v.get('status') == 'Available'); avg_load = np.mean([_safe_division(h.get('load',0),h.get('capacity',1)) for h in data_fabric.hospitals.values()]);
        with kpi_cols[0]: kpi_card("🚑", "Units Available", f"{available_units}/{len(data_fabric.ambulances)}", "#00A9FF")
        with kpi_cols[1]: kpi_card("🏥", "Avg. Hospital Load", f"{avg_load:.0%}", "#FFB000")
        with kpi_cols[2]: kpi_card("🚨", "Active Incidents", len(all_incidents), "#DC3545")
        st.divider(); map_col, ticket_col = st.columns((2.5, 1.5))
        
        with ticket_col:
            st.subheader("Dispatch Ticket")
            
            # Use st.selectbox with the on_change callback for reliable state handling.
            st.selectbox(
                "Select an Active Incident:",
                options=[None] + list(incident_dict.keys()), # Add None for a placeholder
                format_func=lambda x: "Choose an incident..." if x is None else f"{x} (Priority {incident_dict[x]['priority']})",
                key="incident_selector", # The key that links to st.session_state
                on_change=handle_incident_selection,
            )

            # The rest of the UI is now declarative: it just displays what's in the state.
            if st.session_state.get('selected_incident'):
                if st.session_state.get('route_info') and "error" not in st.session_state.route_info:
                    st.metric("Responding to Incident", st.session_state.selected_incident.get('id', 'N/A'))
                    display_ai_rationale(st.session_state.route_info)
                    with st.expander("Show Detailed Routing Analysis"):
                        st.dataframe(st.session_state.route_info['routing_analysis'].set_index('hospital'))
                else:
                    st.error(f"Routing Error: {st.session_state.get('route_info', {}).get('error', 'Could not calculate a route.')}")
            else:
                st.info("Select an incident from the dropdown above to generate a dispatch plan.")
        
        with map_col:
            zones_gdf, hosp_df, amb_df, inc_df, heat_df = prepare_visualization_data(data_fabric, risk_scores, all_incidents, config.get('styling', {}))
            deck = create_deck_gl_map(zones_gdf, hosp_df, amb_df, inc_df, heat_df, st.session_state.get('route_info'), config.get('styling', {}))
            st.pydeck_chart(deck, use_container_width=True)

    # ... (Rest of the app is unchanged) ...
    elif tab_choice == "System Analytics":
        st.header("System-Wide Analytics & AI Insights"); forecast_col, feature_col = st.columns(2)
        with forecast_col:
            st.subheader("24-Hour Probabilistic Demand Forecast");
            with st.spinner("Calculating 24-hour forecast..."):
                future_hours = pd.date_range(start=datetime.now(), periods=24, freq='h'); forecast_data = []
                for ts in future_hours:
                    features = {"hour": ts.hour, "day_of_week": ts.weekday(), "is_quincena": ts.day in [14,15,16,29,30,31,1], 'temperature': 22, 'border_wait': 75}; mean_pred = engine.predict_citywide_demand(features); std_dev = mean_pred * 0.10; forecast_data.append({'time': ts, 'Predicted Calls': mean_pred, 'Upper Bound': mean_pred + 1.96 * std_dev, 'Lower Bound': np.maximum(0, mean_pred - 1.96 * std_dev)})
                st.area_chart(pd.DataFrame(forecast_data).set_index('time'))
        with feature_col:
            st.subheader("Demand Model: Feature Importance (XAI)"); st.info("This chart shows which factors have the biggest impact on our call demand forecast. It's the 'why' behind the AI's prediction."); feature_importance = pd.DataFrame({'feature': engine.model_features, 'importance': engine.demand_model.feature_importances_}).sort_values('importance', ascending=True); st.bar_chart(feature_importance.set_index('feature'))
        st.divider(); col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hospital Load Status")
            for name, data in data_fabric.hospitals.items():
                load_pct = _safe_division(data['load'], data['capacity']); st.markdown(f"**{name}** ({data['load']}/{data['capacity']})"); st.progress(load_pct)
        with col2:
            st.subheader("Critical Patient Alerts"); patient_alerts = engine.get_patient_alerts()
            if not patient_alerts: st.success("✅ No critical patient alerts at this time.")
            else:
                for alert in patient_alerts: st.error(f"**Patient {alert.get('Patient ID')}:** HR: {alert.get('Heart Rate')}, O2: {alert.get('Oxygen %')}% | Unit: {alert.get('Ambulance')}", icon="❤️‍🩹")
    elif tab_choice == "Strategic Simulation":
        st.header("Strategic Simulation & 'What-If' Analysis"); st.info("Test system resilience by simulating extreme conditions."); sim_traffic_spike = st.slider("Simulate City-Wide Traffic Multiplier", 1.0, 5.0, 1.0, 0.25); st.warning("Running a simulation will use the current live incident map. A high number of incidents combined with a high traffic multiplier will show significant risk increases.")
        if st.button("Run Simulation", use_container_width=True):
            sim_risk_scores = {};
            for zone, s_data in data_fabric.zones.items():
                l_data = live_state.get(zone, {}); sim_risk = (l_data.get('traffic', 0.5) * sim_traffic_spike * 0.6 + (1 - s_data.get('road_quality', 0.5)) * 0.2 + s_data.get('crime', 0.5) * 0.2); sim_risk_scores[zone] = sim_risk * (1 + len(l_data.get('active_incidents', [])))
            st.subheader("Simulated Zonal Risk Scores"); st.bar_chart(pd.DataFrame.from_dict(sim_risk_scores, orient='index', columns=['Simulated Risk']).sort_values('Simulated Risk', ascending=False)); st.markdown("High-risk zones under these simulated conditions would require pre-emptive resource staging.")

if __name__ == "__main__":
    main()
