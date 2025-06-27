# RedShieldAI_Elite_Command_Center_FIXED.py
# SME LEVEL: The definitive, fully operational version with the critical KeyError fixed
# by implementing robust, defensive data handling throughout the application.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from typing import Dict, List, Any

# --- L1: DATA & MODELING LAYER ---
class DataFusionFabric:
    """Manages all static and dynamic data for the city with enhanced attributes."""
    def __init__(self):
        self.static_zonal_data = {
            "Zona Río": {"polygon": Polygon([(32.52, -117.01), (32.53, -117.01), (32.53, -117.03), (32.52, -117.03)]), "crime": 0.7, "road_quality": 0.9},
            "Otay": {"polygon": Polygon([(32.53, -116.95), (32.54, -116.95), (32.54, -116.98), (32.53, -116.98)]), "crime": 0.5, "road_quality": 0.7},
            "Playas": {"polygon": Polygon([(32.51, -117.11), (32.53, -117.11), (32.53, -117.13), (32.51, -117.13)]), "crime": 0.4, "road_quality": 0.8}
        }
        self.hospitals = { "Hospital General": {"location": Point(32.5295, -117.0182), "capacity": 100, "load": 85}, "IMSS Clínica 1": {"location": Point(32.5121, -117.0145), "capacity": 120, "load": 70}, "Hospital Angeles": {"location": Point(32.5300, -117.0200), "capacity": 100, "load": 95}}
        self.ambulances = { "A01": {"location": Point(32.515, -117.04), "status": "Available"}, "A02": {"location": Point(32.535, -116.96), "status": "Available"}, "A03": {"location": Point(32.52, -117.12), "status": "On Mission"}}
        self.patient_vitals = { "P001": {'heart_rate': 145, 'oxygen': 88}, "P002": {'heart_rate': 90, 'oxygen': 97}, "P003": {'heart_rate': 150, 'oxygen': 99}}

    @st.cache_data(ttl=60)
    def get_live_state(_self) -> Dict:
        state = {}
        for zone, data in _self.static_zonal_data.items():
            incidents = []
            for i in range(np.random.randint(0, 4)):
                minx, miny, maxx, maxy = data['polygon'].bounds
                incidents.append({"id": f"I-{zone[:2]}{i}", "location": Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)), "priority": np.random.choice([1,2,3], p=[0.6, 0.3, 0.1])})
            state[zone] = {"traffic": np.random.uniform(0.3, 1.0), "active_incidents": incidents}
        return state

class CognitiveEngine:
    """The 'brain' of the system, with enhanced analysis."""
    def __init__(self, data_fabric: DataFusionFabric):
        self.data_fabric = data_fabric
        self.demand_model = self._get_demand_model()

    @st.cache_resource
    def _get_demand_model(_self):
        hours = 24 * 90; timestamps = pd.to_datetime(pd.date_range(start='2024-01-01', periods=hours, freq='H'))
        X_train = pd.DataFrame({'hour': timestamps.hour, 'day_of_week': timestamps.dayofweek, 'is_quincena': timestamps.day.isin([14,15,16,29,30,31,1]), 'temperature': np.random.normal(22, 5, hours), 'border_wait': np.random.randint(20, 120, hours)})
        y_train = np.maximum(0, 5 + 3 * np.sin(X_train['hour'] * 2 * np.pi / 24) + X_train['is_quincena'] * 5 + X_train['border_wait']/20 + np.random.randn(hours)).astype(int)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1); model.fit(X_train, y_train)
        return model, list(X_train.columns)

    def predict_citywide_demand(self, features: Dict) -> float:
        model, feature_names = self.demand_model; input_df = pd.DataFrame([features], columns=feature_names)
        return max(0, model.predict(input_df)[0])

    def calculate_risk_scores(self, live_state: Dict) -> Dict:
        risk_scores = {}
        for zone, s_data in self.data_fabric.static_zonal_data.items():
            l_data = live_state.get(zone, {}); risk = (l_data.get('traffic', 0.5)*0.6 + (1 - s_data.get('road_quality', 0.5))*0.2 + s_data.get('crime', 0.5)*0.2)
            risk_scores[zone] = risk * (1 + len(l_data.get('active_incidents', [])))
        return risk_scores

    def get_patient_alerts(self) -> List[Dict]:
        alerts = []; 
        for pid, vitals in self.data_fabric.patient_vitals.items():
            if vitals.get('heart_rate', 100) > 140 or vitals.get('oxygen', 100) < 90:
                alerts.append({"Patient ID": pid, "Heart Rate": vitals.get('heart_rate'), "Oxygen %": vitals.get('oxygen')})
        return alerts

    def find_best_route_for_incident(self, incident: Dict, risk_gdf: gpd.GeoDataFrame) -> Dict:
        available_ambulances = {k: v for k, v in self.data_fabric.ambulances.items() if v.get('status') == 'Available'}
        if not available_ambulances: return {"error": "No available ambulances."}

        # Defensive access to incident location
        incident_location = incident.get('location')
        if not incident_location: return {"error": f"Incident {incident.get('id', 'Unknown')} has no location data."}

        ambulance_unit, amb_data = min(available_ambulances.items(), key=lambda item: incident_location.distance(item[1].get('location', Point(0,0))))

        options = []
        for name, h_data in self.data_fabric.hospitals.items():
            amb_loc = amb_data.get('location', Point(0,0))
            hosp_loc = h_data.get('location', Point(0,0))
            distance = amb_loc.distance(hosp_loc) * 111; base_eta = distance * 1.5
            path = LineString([amb_loc, hosp_loc])
            path_risk = sum(zone_row.iloc[0]['risk'] for i in range(11) if not (zone_row := risk_gdf[risk_gdf.contains(path.interpolate(i/10, normalized=True))]).empty)
            load_pct = h_data.get('load', 0) / h_data.get('capacity', 1); load_penalty = load_pct**2 * 20
            total_score = base_eta * 0.5 + path_risk * 0.3 + load_penalty * 0.2
            options.append({"hospital": name, "eta_min": base_eta, "path_risk_cost": path_risk, "load_penalty": load_penalty, "load_pct": load_pct, "total_score": total_score})
        
        if not options: return {"error": "No valid hospital options."}
        
        best_option = min(options, key=lambda x: x.get('total_score', float('inf')))
        return {"ambulance_unit": ambulance_unit, "ambulance_location": amb_loc, "incident_location": incident_location, "best_hospital": best_option.get('hospital'), "hospital_location": self.data_fabric.hospitals[best_option.get('hospital')].get('location'), "routing_analysis": pd.DataFrame(options).sort_values('total_score').reset_index(drop=True)}

# --- L3: HIGH-IMPACT VISUALIZATION & UI ---
def create_gauge(value: float, label: str, max_val: int = 100) -> str:
    """Generates an SVG for a gauge meter."""
    value = min(max(value, 0), max_val); percent = value / max_val; angle = percent * 180 - 90
    return f"""<div style="text-align: center;">
        <svg height="80" width="160">
            <defs><linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#28a745" /><stop offset="70%" stop-color="#ffc107" /><stop offset="90%" stop-color="#dc3545" /></linearGradient></defs>
            <path d="M 10 70 A 60 60 0 0 1 150 70" stroke="url(#grad1)" stroke-width="20" fill="none" />
            <line x1="80" y1="70" x2="{80 + 55 * np.cos(np.deg2rad(angle))}" y2="{70 + 55 * np.sin(np.deg2rad(angle))}" stroke="white" stroke-width="3" />
            <text x="80" y="60" text-anchor="middle" font-size="20" fill="white" font-weight="bold">{int(value)}{'%' if max_val == 100 else ''}</text>
        </svg>
        <div style="font-size: 14px; font-weight: bold; color: #ccc;">{label}</div>
    </div>"""

def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info=None):
    """Creates a rich, multi-layered, 3D PyDeck map with an open-source base map."""
    CARTO_DARK_MATTER_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="geometry", filled=True, stroked=False, extruded=True, get_elevation="risk * 2000", get_fill_color="fill_color", opacity=0.15, pickable=True)
    def create_icon_layer(data, get_size_func, get_color_func=None):
        return pdk.Layer("IconLayer", data=data, get_icon="icon_url", get_position='[lon, lat]', get_size=get_size_func, get_color=get_color_func, size_scale=15, pickable=True)
    layers = [zone_layer, create_icon_layer(hospital_df, 4), create_icon_layer(ambulance_df, 'size', 'color'), create_icon_layer(incident_df, 'size')]
    if route_info and "error" not in route_info:
        route_path = LineString([route_info['ambulance_location'], route_info['hospital_location']])
        route_df = pd.DataFrame([{'path': [list(p) for p in route_path.coords]}])
        layers.append(pdk.Layer('PathLayer', data=route_df, get_path='path', get_width=5, get_color=[251, 192, 45], width_scale=1, width_min_pixels=5))
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=50)
    tooltip = {"html": "<b>{name}</b><br/>{tooltip_text}", "style": {"backgroundColor": "#333", "color": "white", "border-radius": "5px", "padding": "5px"}}
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=CARTO_DARK_MATTER_STYLE, tooltip=tooltip)

def display_ai_rationale(route_info: Dict):
    st.subheader("AI Rationale")
    best = route_info['routing_analysis'].iloc[0]
    st.success(f"**Recommended:** `{best.get('hospital', 'N/A')}`")
    st.markdown(f"✅ **Lowest Composite Score:** `{best.get('total_score', 0):.1f}`\n"
                f"✅ **Favorable ETA:** `{best.get('eta_min', 0):.1f}` min\n"
                f"✅ **Acceptable Path Risk:** `{best.get('path_risk_cost', 0):.1f}`\n"
                f"✅ **Manageable Hospital Load:** `{best.get('load_pct', 0):.0%}`")
    if len(route_info['routing_analysis']) > 1:
        rejected = route_info['routing_analysis'].iloc[1]
        st.error(f"**Alternative Rejected:** `{rejected.get('hospital', 'N/A')}`")
        st.markdown(f"❌ **Higher Composite Score:** `{rejected.get('total_score', 0):.1f}`")
        reasons = []
        if rejected.get('load_penalty', 0) > best.get('load_penalty', 0) * 1.2: reasons.append(f"high hospital load penalty (`{rejected.get('load_penalty', 0):.1f}`)")
        if rejected.get('path_risk_cost', 0) > best.get('path_risk_cost', 0) * 1.2: reasons.append(f"high-risk travel path (`{rejected.get('path_risk_cost', 0):.1f}`)")
        if not reasons: reasons.append("it was a close second but less optimal overall.")
        st.markdown(f"Rejected primarily due to {', '.join(reasons)}.")

# --- MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="RedShield AI: Elite Command", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style> .block-container { padding-top: 1rem; } </style>""", unsafe_allow_html=True)

    if 'data_fabric' not in st.session_state: st.session_state.data_fabric = DataFusionFabric()
    if 'cognitive_engine' not in st.session_state: st.session_state.cognitive_engine = CognitiveEngine(st.session_state.data_fabric)
    data_fabric, engine = st.session_state.data_fabric, st.session_state.cognitive_engine

    live_state = data_fabric.get_live_state()
    risk_scores = engine.calculate_risk_scores(live_state)
    all_incidents = [inc for zone_data in live_state.values() for inc in zone_data.get('active_incidents', [])]

    st.header("RedShield AI: Elite Command Center")
    kpi1, kpi2, kpi3 = st.columns(3)
    available_units = sum(1 for v in data_fabric.ambulances.values() if v.get('status') == 'Available')
    kpi1.markdown(create_gauge(available_units, "Units Available", max_val=len(data_fabric.ambulances)), unsafe_allow_html=True)
    avg_load = np.mean([h.get('load',0)/h.get('capacity',1) for h in data_fabric.hospitals.values()]) * 100
    kpi2.markdown(create_gauge(avg_load, "Avg. Hospital Load"), unsafe_allow_html=True)
    critical_alerts = len(engine.get_patient_alerts())
    kpi3.markdown(create_gauge(critical_alerts, "Critical Patients", max_val=5), unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns((2.5, 1.5))
    with col1:
        # BUG FIX: Defensive data handling using .get() to prevent KeyErrors
        hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Load: {d.get('load',0)}/{d.get('capacity',1)}", "lon": d.get('location', Point(0,0)).x, "lat": d.get('location', Point(0,0)).y, "icon_url": "https://img.icons8.com/color/48/hospital-3.png"} for n, d in data_fabric.hospitals.items()])
        ambulance_df = pd.DataFrame([{"name": f"Unit: {n}", "tooltip_text": f"Status: {d.get('status', 'Unknown')}", "lon": d.get('location', Point(0,0)).x, "lat": d.get('location', Point(0,0)).y, "icon_url": "https://img.icons8.com/officel/48/ambulance.png", "size": 4 if d.get('status') == 'Available' else 2.5, "color": [0, 255, 0, 255] if d.get('status') == 'Available' else [128, 128, 128, 180]} for n, d in data_fabric.ambulances.items()])
        incident_df = pd.DataFrame([{"name": f"Incident: {i.get('id', 'N/A')}", "tooltip_text": f"Priority: {i.get('priority', 1)}", "lon": i.get('location', Point(0,0)).x, "lat": i.get('location', Point(0,0)).y, "icon_url": "https://img.icons8.com/fluency/48/siren.png", "size": 2 + i.get('priority', 1)} for i in all_incidents])
        
        zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.static_zonal_data, orient='index').set_geometry('polygon')
        zones_gdf['name'] = zones_gdf.index; zones_gdf['risk'] = zones_gdf.index.map(risk_scores).fillna(0); zones_gdf['tooltip_text'] = ""
        max_risk = max(1, zones_gdf['risk'].max()); zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [255, int(255*(1-r/max_risk)), 0, 140]).tolist()
        
        route_info = engine.find_best_route_for_incident(st.session_state.selected_incident, zones_gdf) if st.session_state.get('selected_incident') else None
        st.pydeck_chart(create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info))

    with col2:
        st.subheader("Active Incidents")
        if not all_incidents:
            st.info("No active incidents. System is clear.")
        else:
            for incident in sorted(all_incidents, key=lambda x: x.get('priority', 1), reverse=True):
                if st.button(f"🚨 Priority {incident.get('priority', 1)}: Incident {incident.get('id', 'N/A')}", key=incident.get('id'), use_container_width=True):
                    st.session_state.selected_incident = incident
                    st.rerun()

        st.divider()
        st.subheader("Dispatch Ticket")
        if not st.session_state.get('selected_incident'):
            st.info("Select an incident to generate a dispatch plan.")
        elif not route_info or "error" in route_info:
            st.error(route_info.get("error", "Could not calculate a route."))
        else:
            display_ai_rationale(route_info)
            with st.expander("Show Detailed Routing Analysis"):
                st.dataframe(route_info['routing_analysis'].set_index('hospital'))

if __name__ == "__main__":
    main()
