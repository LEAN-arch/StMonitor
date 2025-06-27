# RedShieldAI_Elite_Command_Center.py
# SME LEVEL: The definitive, UX/DX-focused version with a fully interactive map,
# external configuration, and unparalleled visual clarity.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from typing import Dict, Any, Tuple
import yaml

# --- L0: CONFIGURATION & UTILITIES ---
@st.cache_data
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _safe_division(n, d): return n / d if d else 0

# --- L1: DATA & MODELING LAYER ---
class DataFusionFabric:
    """Manages all static and dynamic data for the city."""
    def __init__(self, config: Dict):
        self.config = config['data']
        self.hospitals = {name: {'location': Point(data['location']), 'capacity': data['capacity'], 'load': data['load']} for name, data in self.config['hospitals'].items()}
        self.ambulances = {name: {'location': Point(data['location']), 'status': data['status']} for name, data in self.config['ambulances'].items()}
        self.zones = {name: {**data, 'polygon': Polygon(data['polygon'])} for name, data in self.config['zones'].items()}

    @st.cache_data(ttl=60)
    def get_live_state(_self) -> Dict:
        state = {}
        for zone, data in _self.zones.items():
            incidents = [{"id": f"I-{zone[:2]}{i}", "location": Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)), "priority": np.random.choice([1,2,3], p=[0.6, 0.3, 0.1])} for i in range(np.random.randint(0, 4)) for minx, miny, maxx, maxy in [data['polygon'].bounds]]
            state[zone] = {"traffic": np.random.uniform(0.3, 1.0), "active_incidents": incidents}
        return state

class CognitiveEngine:
    """The 'brain' of the system, with enhanced analysis."""
    def __init__(self, data_fabric: DataFusionFabric):
        self.data_fabric = data_fabric
    
    def calculate_risk_scores(self, live_state: Dict) -> Dict:
        risk_scores = {};
        for zone, s_data in self.data_fabric.zones.items():
            l_data = live_state.get(zone, {}); risk = (l_data.get('traffic', 0.5)*0.6 + (1 - s_data.get('road_quality', 0.5))*0.2 + s_data.get('crime', 0.5)*0.2)
            risk_scores[zone] = risk * (1 + len(l_data.get('active_incidents', [])))
        return risk_scores

    def find_best_route_for_incident(self, incident: Dict, risk_gdf: gpd.GeoDataFrame) -> Dict:
        available_ambulances = {k: v for k, v in self.data_fabric.ambulances.items() if v.get('status') == 'Available'}
        if not available_ambulances: return {"error": "No available ambulances."}
        incident_location = incident.get('location');
        if not incident_location: return {"error": f"Incident {incident.get('id', 'Unknown')} has no location data."}
        ambulance_unit, amb_data = min(available_ambulances.items(), key=lambda item: incident_location.distance(item[1].get('location', Point(0,0))))
        options = []
        for name, h_data in self.data_fabric.hospitals.items():
            amb_loc = amb_data.get('location', Point(0,0)); hosp_loc = h_data.get('location', Point(0,0))
            distance = amb_loc.distance(hosp_loc) * 111; base_eta = distance * 1.5
            path = LineString([amb_loc, hosp_loc])
            path_risk = sum(zone_row.iloc[0]['risk'] for i in range(11) if not (zone_row := risk_gdf[risk_gdf.contains(path.interpolate(i/10, normalized=True))]).empty)
            load_pct = _safe_division(h_data.get('load', 0), h_data.get('capacity', 1)); load_penalty = load_pct**2 * 20
            total_score = base_eta * 0.5 + path_risk * 0.3 + load_penalty * 0.2
            options.append({"hospital": name, "eta_min": base_eta, "path_risk_cost": path_risk, "load_penalty": load_penalty, "load_pct": load_pct, "total_score": total_score})
        if not options: return {"error": "No valid hospital options."}
        best_option = min(options, key=lambda x: x.get('total_score', float('inf')))
        return {"ambulance_unit": ambulance_unit, "ambulance_location": amb_loc, "incident_location": incident_location, "best_hospital": best_option.get('hospital'), "hospital_location": self.data_fabric.hospitals.get(best_option.get('hospital'), {}).get('location'), "routing_analysis": pd.DataFrame(options).sort_values('total_score').reset_index(drop=True)}

# --- L2: PRESENTATION & VISUALIZATION LAYER ---
def prepare_visualization_data(data_fabric, risk_scores, all_incidents, style_config):
    """DX ENHANCEMENT: Encapsulates all data-to-dataframe logic for clean, readable code."""
    def get_hospital_color(load, capacity):
        load_pct = _safe_division(load, capacity)
        if load_pct < 0.7: return style_config['hospital_ok']
        if load_pct < 0.9: return style_config['hospital_warn']
        return style_config['hospital_crit']
    
    hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Load: {d.get('load',0)}/{d.get('capacity',1)} ({_safe_division(d.get('load',0), d.get('capacity',1)):.0%})", "lon": d.get('location', Point(0,0)).x, "lat": d.get('location', Point(0,0)).y, "color": get_hospital_color(d.get('load',0), d.get('capacity',1))} for n, d in data_fabric.hospitals.items()])
    ambulance_df = pd.DataFrame([{"name": f"Unit: {n}", "tooltip_text": f"Status: {d.get('status', 'Unknown')}", "lon": d.get('location', Point(0,0)).x, "lat": d.get('location', Point(0,0)).y, "size": style_config['ambulance_available'] if d.get('status') == 'Available' else style_config['ambulance_mission'], "color": style_config['available'] if d.get('status') == 'Available' else style_config['on_mission']} for n, d in data_fabric.ambulances.items()])
    incident_df = pd.DataFrame([{"name": f"Incident: {i.get('id', 'N/A')}", "tooltip_text": f"Priority: {i.get('priority', 1)}", "lon": i.get('location', Point(0,0)).x, "lat": i.get('location', Point(0,0)).y, "id": i.get('id'), "size": style_config['incident_base'] + i.get('priority', 1)**2} for i in all_incidents])
    
    zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.zones, orient='index').set_geometry('polygon')
    zones_gdf['name'] = zones_gdf.index; zones_gdf['risk'] = zones_gdf.index.map(risk_scores).fillna(0); zones_gdf['tooltip_text'] = ""
    max_risk = max(1, zones_gdf['risk'].max()); zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [255, int(255*(1-_safe_division(r,max_risk))), 0, 140]).tolist()
    
    return zones_gdf, hospital_df, ambulance_df, incident_df

def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info=None, style_config=None):
    """Creates a rich, multi-layered, 3D PyDeck map."""
    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="geometry", filled=True, stroked=False, extruded=True, get_elevation="risk * 2000", get_fill_color="fill_color", opacity=0.15, pickable=True)
    hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="icon_data", get_position='[lon, lat]', get_size=style_config['hospital'], get_color='color', size_scale=15, pickable=True)
    ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="icon_data", get_position='[lon, lat]', get_size='size', get_color='color', size_scale=15, pickable=True)
    incident_layer = pdk.Layer("ScatterplotLayer", data=incident_df, get_position='[lon, lat]', get_radius='size*20', get_fill_color=style_config['incident_halo'], pickable=True, radius_min_pixels=5, stroked=True, get_line_width=100, get_line_color=[*style_config['incident_halo'], 100])
    layers = [zone_layer, hospital_layer, ambulance_layer, incident_layer]
    if route_info and "error" not in route_info:
        route_path = LineString([route_info['ambulance_location'], route_info['hospital_location']])
        route_df = pd.DataFrame([{'path': [list(p) for p in route_path.coords]}])
        layers.append(pdk.Layer('PathLayer', data=route_df, get_path='path', get_width=5, get_color=style_config['route_path'], width_scale=1, width_min_pixels=5))
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=50)
    tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border-radius": "5px", "padding": "5px"}}
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", tooltip=tooltip)

def display_ai_rationale(route_info: Dict):
    st.subheader("AI Dispatch Rationale"); best = route_info['routing_analysis'].iloc[0]
    st.success(f"**Recommended:** `{best.get('hospital', 'N/A')}`", icon="✅")
    st.markdown(f"**Reason:** Lowest composite score (`{best.get('total_score', 0):.1f}`). Achieves the best balance of fast ETA, low-risk travel path, and manageable hospital load.")
    if len(route_info['routing_analysis']) > 1:
        rejected = route_info['routing_analysis'].iloc[1]; st.error(f"**Alternative Rejected:** `{rejected.get('hospital', 'N/A')}`", icon="❌")
        reasons = []
        if rejected.get('load_penalty', 0) > best.get('load_penalty', 0) * 1.2: reasons.append(f"high hospital load (`{rejected.get('load_pct', 0):.0%}`)")
        if rejected.get('path_risk_cost', 0) > best.get('path_risk_cost', 0) * 1.2: reasons.append("a high-risk travel path")
        if not reasons: reasons.append("it was a close second but less optimal overall.")
        st.markdown(f"Rejected primarily due to {', '.join(reasons)}.")

# --- L4: MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="RedShield AI: Elite Command", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style> .block-container { padding-top: 1rem; } [data-testid="stSidebar"] {background-color: #111;} </style>""", unsafe_allow_html=True)

    config = load_config()
    if 'data_fabric' not in st.session_state: st.session_state.data_fabric = DataFusionFabric(config)
    if 'cognitive_engine' not in st.session_state: st.session_state.cognitive_engine = CognitiveEngine(st.session_state.data_fabric)
    data_fabric, engine = st.session_state.data_fabric, st.session_state.cognitive_engine

    with st.sidebar:
        st.title("RedShield AI")
        st.header("Master Controls")
        if st.button("🔄 Force Refresh Live Data", use_container_width=True): data_fabric.get_live_state.clear()
        st.header("Incident Queue")
        live_state = data_fabric.get_live_state()
        all_incidents = [inc for zone_data in live_state.values() for inc in zone_data.get('active_incidents', [])]
        if not all_incidents: st.info("No active incidents.")
        else:
            for incident in sorted(all_incidents, key=lambda x: x.get('priority', 1), reverse=True):
                if st.button(f"🚨 Priority {incident.get('priority', 1)}: Incident {incident.get('id', 'N/A')}", key=incident.get('id'), use_container_width=True, type="primary" if incident.get('priority') == 3 else "secondary"):
                    st.session_state.selected_incident = incident
    
    risk_scores = engine.calculate_risk_scores(live_state)
    
    # --- TOP KPI DASHBOARD ---
    kpi_cols = st.columns(3)
    available_units = sum(1 for v in data_fabric.ambulances.values() if v.get('status') == 'Available')
    kpi_cols[0].markdown(create_gauge(available_units, "Units Available", max_val=len(data_fabric.ambulances)), unsafe_allow_html=True)
    avg_load = np.mean([_safe_division(h.get('load',0),h.get('capacity',1)) for h in data_fabric.hospitals.values()]) * 100
    kpi_cols[1].markdown(create_gauge(avg_load, "Avg. Hospital Load"), unsafe_allow_html=True)
    kpi_cols[2].markdown(create_gauge(len(engine.get_patient_alerts()), "Critical Patients", max_val=5), unsafe_allow_html=True)
    st.divider()

    # --- MAIN LAYOUT: MAP + DISPATCH TICKET ---
    map_col, ticket_col = st.columns((2.5, 1.5))
    with map_col:
        zones_gdf, hospital_df, ambulance_df, incident_df = prepare_visualization_data(data_fabric, risk_scores, all_incidents, config['styling'])
        route_info = engine.find_best_route_for_incident(st.session_state.selected_incident, zones_gdf) if st.session_state.get('selected_incident') else None
        st.pydeck_chart(create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info, config['styling']))

    with ticket_col:
        st.subheader("Dispatch Ticket")
        if not st.session_state.get('selected_incident'): st.info("Select an incident from the queue to generate a dispatch plan.")
        elif not route_info or "error" in route_info: st.error(route_info.get("error", "Could not calculate a route."))
        else:
            st.metric("Responding to Incident", st.session_state.selected_incident.get('id', 'N/A'))
            display_ai_rationale(route_info)
            with st.expander("Show Detailed Routing Analysis"): st.dataframe(route_info['routing_analysis'].set_index('hospital'))

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
