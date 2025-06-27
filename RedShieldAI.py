# RedShieldAI_Definitive_Command_Center_OSS_MAP.py
# SME LEVEL: The definitive, fully integrated version with high-quality visualizations,
# 3D maps, probabilistic plots, and a robust, open-source CARTO map provider.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from typing import Dict, List, Any

# --- L1: DATA & MODELING LAYER ---
class DataFusionFabric:
    """Manages all static and dynamic data for the city."""
    def __init__(self):
        self.static_zonal_data = {
            "Zona Río": {"polygon": Polygon([(32.52, -117.01), (32.53, -117.01), (32.53, -117.03), (32.52, -117.03)]), "crime": 0.7, "road_quality": 0.9},
            "Otay": {"polygon": Polygon([(32.53, -116.95), (32.54, -116.95), (32.54, -116.98), (32.53, -116.98)]), "crime": 0.5, "road_quality": 0.7},
            "Playas": {"polygon": Polygon([(32.51, -117.11), (32.53, -117.11), (32.53, -117.13), (32.51, -117.13)]), "crime": 0.4, "road_quality": 0.8}
        }
        self.hospitals = {
            "Hospital General": {"location": Point(32.5295, -117.0182), "capacity": 100, "load": 85},
            "IMSS Clínica 1": {"location": Point(32.5121, -117.0145), "capacity": 120, "load": 70},
            "Hospital Angeles": {"location": Point(32.5300, -117.0200), "capacity": 100, "load": 95}
        }
        self.ambulances = { "A01": {"location": Point(32.515, -117.04), "status": "Available"}, "A02": {"location": Point(32.535, -116.96), "status": "Available"}, "A03": {"location": Point(32.52, -117.12), "status": "On Mission"}}
        self.patient_vitals = { "P001": {'heart_rate': 145, 'oxygen': 88}, "P002": {'heart_rate': 90, 'oxygen': 97}, "P003": {'heart_rate': 150, 'oxygen': 99}}

    @st.cache_data(ttl=300)
    def get_live_state(_self) -> Dict:
        state = {}
        for zone, data in _self.static_zonal_data.items():
            incidents = []
            for i in range(np.random.randint(0, 3)):
                minx, miny, maxx, maxy = data['polygon'].bounds
                incidents.append({"id": f"I-{zone[:2]}{i}", "location": Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))})
            state[zone] = {"traffic": np.random.uniform(0.3, 1.0), "active_incidents": incidents}
        return state

class CognitiveEngine:
    """The 'brain' of the system, handling predictions, routing, and alerts."""
    def __init__(self, data_fabric: DataFusionFabric):
        self.data_fabric = data_fabric
        self.demand_model = self._get_demand_model()

    @st.cache_resource
    def _get_demand_model(_self):
        hours = 24 * 90
        timestamps = pd.to_datetime(pd.date_range(start='2024-01-01', periods=hours, freq='H'))
        X_train = pd.DataFrame({'hour': timestamps.hour, 'day_of_week': timestamps.dayofweek, 'is_quincena': timestamps.day.isin([14,15,16,29,30,31,1]), 'temperature': np.random.normal(22, 5, hours), 'border_wait': np.random.randint(20, 120, hours)})
        y_train = np.maximum(0, 5 + 3 * np.sin(X_train['hour'] * 2 * np.pi / 24) + X_train['is_quincena'] * 5 + X_train['border_wait']/20 + np.random.randn(hours)).astype(int)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, list(X_train.columns)

    def predict_citywide_demand(self, features: Dict) -> float:
        model, feature_names = self.demand_model
        input_df = pd.DataFrame([features], columns=feature_names)
        return max(0, model.predict(input_df)[0])

    def calculate_risk_scores(self, live_state: Dict) -> Dict:
        risk_scores = {}
        for zone, s_data in self.data_fabric.static_zonal_data.items():
            l_data = live_state.get(zone, {})
            risk = (l_data.get('traffic', 0.5)*0.6 + (1 - s_data.get('road_quality', 0.5))*0.2 + s_data.get('crime', 0.5)*0.2)
            risk_scores[zone] = risk * (1 + len(l_data.get('active_incidents', [])))
        return risk_scores

    def get_patient_alerts(self) -> List[Dict]:
        alerts = []
        for pid, vitals in self.data_fabric.patient_vitals.items():
            if vitals.get('heart_rate', 100) > 140 or vitals.get('oxygen', 100) < 90:
                alerts.append({"Patient ID": pid, "Heart Rate": vitals['heart_rate'], "Oxygen %": vitals['oxygen']})
        return alerts

    def find_best_route_for_incident(self, incident: Dict, risk_gdf: gpd.GeoDataFrame) -> Dict:
        available_ambulances = {k: v for k, v in self.data_fabric.ambulances.items() if v['status'] == 'Available'}
        if not available_ambulances: return {"error": "No available ambulances."}

        ambulance_unit, amb_data = min(available_ambulances.items(), key=lambda item: incident['location'].distance(item[1]['location']))

        options = []
        for name, h_data in self.data_fabric.hospitals.items():
            distance = amb_data['location'].distance(h_data['location']) * 111
            base_eta = distance * 1.5
            path = LineString([amb_data['location'], h_data['location']])
            path_risk = sum(zone_row.iloc[0]['risk'] for i in range(11) if not (zone_row := risk_gdf[risk_gdf.contains(path.interpolate(i/10, normalized=True))]).empty)
            load_penalty = (h_data['load'] / h_data['capacity'])**2 * 20
            total_score = base_eta * 0.5 + path_risk * 0.3 + load_penalty * 0.2
            options.append({"hospital": name, "eta_min": base_eta, "path_risk_cost": path_risk, "load_penalty": load_penalty, "total_score": total_score})

        if not options: return {"error": "No valid hospital options."}
        
        best_option = min(options, key=lambda x: x['total_score'])
        return {
            "ambulance_unit": ambulance_unit, "ambulance_location": amb_data['location'], "incident_location": incident['location'],
            "best_hospital": best_option['hospital'], "hospital_location": self.data_fabric.hospitals[best_option['hospital']]['location'],
            "routing_analysis": pd.DataFrame(options).sort_values('total_score').reset_index(drop=True)
        }

# --- L3: HIGH-QUALITY VISUALIZATION & UI ---
def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info=None):
    """Creates a rich, multi-layered, 3D PyDeck map with an open-source base map."""
    # SME FIX: Switched to a key-less open-source map provider (CARTO) to ensure rendering.
    # The map_provider and api_keys arguments are no longer needed.
    CARTO_DARK_MATTER_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    
    zone_layer = pdk.Layer(
        "PolygonLayer",
        data=zones_gdf,
        get_polygon="geometry",
        filled=True,
        stroked=True,
        extruded=True,
        get_elevation="risk * 2000",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        get_line_width=50,
        opacity=0.3,
        pickable=True,
        auto_highlight=True,
    )
    
    def create_icon_layer(data, icon_url):
        data['icon_data'] = [{"url": icon_url, "width": 128, "height": 128, "anchorY": 128}] * len(data)
        return pdk.Layer("IconLayer", data=data, get_icon="icon_data", get_position='[lon, lat]', get_size=4, size_scale=15, pickable=True)

    layers = [
        zone_layer,
        create_icon_layer(hospital_df, "https://img.icons8.com/color/48/hospital-3.png"),
        create_icon_layer(ambulance_df, "https://img.icons8.com/officel/48/ambulance.png"),
        create_icon_layer(incident_df, "https://img.icons8.com/fluency/48/siren.png")
    ]
    
    if route_info and "error" not in route_info:
        route_path = LineString([route_info['ambulance_location'], route_info['hospital_location']])
        route_df = pd.DataFrame([{'path': [list(p) for p in route_path.coords]}])
        route_layer = pdk.Layer('PathLayer', data=route_df, get_path='path', get_width=5, get_color=[251, 192, 45], width_scale=1, width_min_pixels=5)
        layers.append(route_layer)
        
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=45)
    
    tooltip = {
        "html": "<b>{name}</b><br/>{tooltip_text}",
        "style": {"backgroundColor": "#333", "color": "white", "border-radius": "5px", "padding": "5px"}
    }
    
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=CARTO_DARK_MATTER_STYLE, tooltip=tooltip)

def display_hospital_load_bars(hospitals_data):
    st.subheader("Hospital Load Status")
    for name, data in hospitals_data.items():
        load_pct = data['load'] / data['capacity']
        color = "#28a745" if load_pct < 0.7 else ("#ffc107" if load_pct < 0.9 else "#dc3545")
        st.markdown(f"**{name}**")
        st.progress(load_pct)

# --- MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="RedShield AI: Unified Command", layout="wide")
    st.title("🚑 RedShield AI: Unified Command & Control Dashboard")

    if 'data_fabric' not in st.session_state: st.session_state.data_fabric = DataFusionFabric()
    if 'cognitive_engine' not in st.session_state: st.session_state.cognitive_engine = CognitiveEngine(st.session_state.data_fabric)
    data_fabric, engine = st.session_state.data_fabric, st.session_state.cognitive_engine

    st.sidebar.header("Master Controls")
    if st.sidebar.button("Force Refresh Live Data"): data_fabric.get_live_state.clear()
    
    st.sidebar.subheader("Demand Forecast Inputs")
    forecast_features = {"temperature": st.sidebar.slider("Temperature (°C)", -5, 45, 25), "border_wait": st.sidebar.slider("Border Wait Time (min)", 0, 180, 75)}

    live_state = data_fabric.get_live_state()
    risk_scores = engine.calculate_risk_scores(live_state)
    patient_alerts = engine.get_patient_alerts()
    all_incidents = [inc for zone_data in live_state.values() for inc in zone_data['active_incidents']]
    
    tab1, tab2, tab3 = st.tabs(["**Live Operations Command**", "**System-Wide Analytics**", "**Strategic Simulation**"])

    with tab1:
        st.sidebar.subheader("Incident Response")
        if not all_incidents: st.sidebar.info("No active incidents.")
        else:
            selected_id = st.sidebar.selectbox("Select Incident:", [inc['id'] for inc in all_incidents])
            st.session_state.selected_incident = next((inc for inc in all_incidents if inc['id'] == selected_id), None)

        hospital_df = pd.DataFrame([{"name": f"Hospital: {n}", "tooltip_text": f"Load: {d['load']}/{d['capacity']}", "lon": d['location'].x, "lat": d['location'].y} for n, d in data_fabric.hospitals.items()])
        ambulance_df = pd.DataFrame([{"name": f"Unit: {n}", "tooltip_text": f"Status: {d['status']}", "lon": d['location'].x, "lat": d['location'].y} for n, d in data_fabric.ambulances.items()])
        incident_df = pd.DataFrame([{"name": f"Incident: {i['id']}", "tooltip_text": "Click for details", "lon": i['location'].x, "lat": i['location'].y} for i in all_incidents])
        zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.static_zonal_data, orient='index').set_geometry('polygon')
        zones_gdf['name'] = zones_gdf.index
        zones_gdf['risk'] = zones_gdf.index.map(risk_scores).fillna(0)
        zones_gdf['tooltip_text'] = "" # Tooltip from DeckGL works on gdf columns
        max_risk = max(1, zones_gdf['risk'].max())
        zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [255, int(255*(1-r/max_risk)), 0, 140]).tolist()
        
        col1, col2 = st.columns((2.5, 1.5))
        with col1:
            st.subheader("Operational 3D Risk Map")
            route_info = engine.find_best_route_for_incident(st.session_state.selected_incident, zones_gdf) if st.session_state.get('selected_incident') else None
            st.pydeck_chart(create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info))
        with col2:
            st.subheader("Dispatch Decision Engine")
            if not route_info: st.info("Select an incident from the sidebar to see the routing plan.")
            elif "error" in route_info: st.error(route_info["error"])
            else:
                st.success(f"**Plan for Incident {st.session_state.selected_incident['id']}**")
                st.metric("Dispatch Unit", route_info['ambulance_unit'])
                st.metric("Optimal Destination", route_info['best_hospital'])
                st.dataframe(route_info['routing_analysis'].set_index('hospital').style.highlight_min(axis=0, props="color:white; background-color:green;"))

    with tab2:
        st.header("System-Wide Analytics & KPIs")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Active Incidents", len(all_incidents))
        avg_load = np.mean([h['load']/h['capacity'] for h in data_fabric.hospitals.values()])
        kpi2.metric("Avg. Hospital Load", f"{avg_load:.0%}")
        kpi3.metric("Critical Patient Alerts", len(patient_alerts), delta_color="inverse")
        st.divider()

        display_hospital_load_bars(data_fabric.hospitals)
        st.divider()

        st.subheader("24-Hour Probabilistic Demand Forecast")
        with st.spinner("Calculating 24-hour forecast..."):
            future_hours = pd.date_range(start=datetime.now(), periods=24, freq='H')
            forecast_data = []
            for ts in future_hours:
                features = {"hour": ts.hour, "day_of_week": ts.weekday(), "is_quincena": ts.day in [14,15,16,29,30,31,1], **forecast_features}
                mean_pred = engine.predict_citywide_demand(features)
                std_dev = mean_pred * 0.15
                forecast_data.append({'time': ts, 'Predicted Calls': mean_pred, 'Upper Bound': mean_pred + 1.96 * std_dev, 'Lower Bound': np.maximum(0, mean_pred - 1.96 * std_dev)})
            forecast_df = pd.DataFrame(forecast_data).set_index('time')
            st.area_chart(forecast_df)
        
        st.subheader("Critical Patient Alerts")
        if not patient_alerts: st.success("✅ No critical patient alerts.")
        else:
            for alert in patient_alerts: st.error(f"**CRITICAL ALERT - Patient {alert['Patient ID']}:** Heart Rate: {alert['Heart Rate']}, Oxygen: {alert['Oxygen %']}%", icon="🚨")

    with tab3:
        st.header("Strategic Simulation")
        sim_traffic_spike = st.slider("Simulate Traffic Spike Across All Zones", 1.0, 3.0, 1.0, 0.1)
        if st.button("Run Simulation"):
            sim_state = data_fabric.get_live_state()
            sim_risk_scores = {}
            for zone, s_data in data_fabric.static_zonal_data.items():
                l_data = sim_state.get(zone, {}); sim_risk = (l_data.get('traffic', 0.5) * sim_traffic_spike * 0.6 + (1-s_data.get('road_quality',0.5))*0.2)
                sim_risk_scores[zone] = sim_risk * (1 + len(l_data.get('active_incidents', [])))
            sim_risk_df = pd.DataFrame.from_dict(sim_risk_scores, orient='index', columns=['Simulated Risk']).sort_values('Simulated Risk', ascending=False)
            st.subheader("Simulated Zonal Risk"); st.bar_chart(sim_risk_df)
            st.success("Simulation complete.")

if __name__ == "__main__":
    main()
