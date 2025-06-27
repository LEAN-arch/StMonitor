# RedShieldAI_Operational_Twin_FIXED.py
# SME Note: This version fixes the critical TypeError by sanitizing data before passing it
# to PyDeck, ensuring all objects are JSON-serializable.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pydeck as pdk
from datetime import datetime
from typing import Dict, List, Any

# --- L1: GEOSPATIALLY-AWARE DATA FABRIC ---
class DataFusionFabric:
    """Manages all data, now with rich geospatial context."""
    def __init__(self):
        self.static_zonal_data = {
            "Zona Río": {"polygon": Polygon([(32.52, -117.01), (32.53, -117.01), (32.53, -117.03), (32.52, -117.03)]), "crime": 0.7, "road_quality": 0.9, "base_demand": 5.0},
            "Otay": {"polygon": Polygon([(32.53, -116.95), (32.54, -116.95), (32.54, -116.98), (32.53, -116.98)]), "crime": 0.5, "road_quality": 0.7, "base_demand": 7.0},
            "Playas": {"polygon": Polygon([(32.51, -117.11), (32.53, -117.11), (32.53, -117.13), (32.51, -117.13)]), "crime": 0.4, "road_quality": 0.8, "base_demand": 3.0}
        }
        self.hospitals = {
            "Hospital General": {"location": Point(32.5295, -117.0182), "capacity": 100, "load": 85},
            "IMSS Clínica 1": {"location": Point(32.5121, -117.0145), "capacity": 120, "load": 70},
            "Hospital Angeles": {"location": Point(32.5300, -117.0200), "capacity": 100, "load": 95}
        }
        self.ambulances = {
            "A01": {"location": Point(32.515, -117.04), "status": "Available"},
            "A02": {"location": Point(32.535, -116.96), "status": "Available"},
            "A03": {"location": Point(32.52, -117.12), "status": "On Mission"}
        }

    @st.cache_data(ttl=300)
    def get_live_state(_self) -> Dict:
        """Simulates fetching real-time data, now including incident locations."""
        state = {}
        for zone, data in _self.static_zonal_data.items():
            incidents = []
            num_incidents = np.random.randint(0, 3)
            for i in range(num_incidents):
                minx, miny, maxx, maxy = data['polygon'].bounds
                incident_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
                incidents.append({"id": f"I-{zone[:2]}{i}", "location": incident_point})

            state[zone] = {
                "traffic": np.random.uniform(0.3, 1.0),
                "active_incidents": incidents,
            }
        return state

# --- L2: COGNITIVE ENGINE WITH ROUTING LOGIC ---
class DigitalTwinCore:
    def __init__(self, data_fabric: DataFusionFabric):
        self.data_fabric = data_fabric
        self.model_confidence = 0.95

    def _calculate_risk_scores(self, live_state: Dict) -> Dict:
        """Calculates a dynamic risk score for each zone."""
        risk_scores = {}
        for zone, s_data in self.data_fabric.static_zonal_data.items():
            l_data = live_state.get(zone, {})
            risk = (l_data.get('traffic', 0.5) * 0.6 + 
                    (1 - s_data.get('road_quality', 0.5)) * 0.2 + 
                    s_data.get('crime', 0.5) * 0.2)
            risk_scores[zone] = risk * (1 + len(l_data.get('active_incidents', [])))
        return risk_scores

    def find_best_route_for_incident(self, incident: Dict, risk_gdf: gpd.GeoDataFrame) -> Dict:
        """Calculates the optimal hospital destination for a given incident."""
        available_ambulances = {k: v for k, v in self.data_fabric.ambulances.items() if v['status'] == 'Available'}
        if not available_ambulances: return {"error": "No available ambulances."}

        ambulance_unit, amb_data = min(available_ambulances.items(), key=lambda item: incident['location'].distance(item[1]['location']))

        options = []
        for name, h_data in self.data_fabric.hospitals.items():
            distance = amb_data['location'].distance(h_data['location']) * 111
            base_eta = distance * 1.5
            
            path = LineString([amb_data['location'], h_data['location']])
            path_risk = 0
            for i in range(11):
                point_on_path = path.interpolate(i / 10, normalized=True)
                zone_row = risk_gdf[risk_gdf.contains(point_on_path)]
                if not zone_row.empty: path_risk += zone_row.iloc[0]['risk']
            
            load_penalty = (h_data['load'] / h_data['capacity'])**2 * 20
            total_score = base_eta * 0.5 + path_risk * 0.3 + load_penalty * 0.2
            options.append({
                "hospital": name, "eta_min": base_eta, "path_risk_cost": path_risk,
                "load_penalty": load_penalty, "total_score": total_score
            })

        if not options: return {"error": "No valid hospital options."}
        
        best_option = min(options, key=lambda x: x['total_score'])
        return {
            "ambulance_unit": ambulance_unit, "ambulance_location": amb_data['location'],
            "incident_location": incident['location'], "best_hospital": best_option['hospital'],
            "hospital_location": self.data_fabric.hospitals[best_option['hospital']]['location'],
            "routing_analysis": pd.DataFrame(options).sort_values('total_score').reset_index(drop=True)
        }

# --- L3: VISUALIZATION & UI ---
def create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info=None):
    """Creates a rich, multi-layered PyDeck map using sanitized DataFrames."""
    ICON_MAPPING = {
        "hospital": {"url": "https://img.icons8.com/color/48/hospital-3.png", "width": 256, "height": 256, "anchorY": 256},
        "ambulance": {"url": "https://img.icons8.com/officel/48/ambulance.png", "width": 256, "height": 256, "anchorY": 256},
        "incident": {"url": "https://img.icons8.com/ios-filled/50/FA5252/error.png", "width": 256, "height": 256, "anchorY": 256},
    }
    
    zone_layer = pdk.Layer("PolygonLayer", data=zones_gdf, get_polygon="geometry", filled=True, stroked=True, get_fill_color="fill_color", get_line_color=[255, 255, 255], get_line_width=100, opacity=0.2, pickable=True, auto_highlight=True)
    
    # These layers now receive clean, serializable data.
    hospital_layer = pdk.Layer("IconLayer", data=hospital_df, get_icon="hospital", get_position='[lon, lat]', get_size=40, size_scale=1, pickable=True, icon_atlas=ICON_MAPPING['hospital']['url'], icon_mapping=ICON_MAPPING)
    ambulance_layer = pdk.Layer("IconLayer", data=ambulance_df, get_icon="ambulance", get_position='[lon, lat]', get_size=35, size_scale=1, pickable=True, icon_atlas=ICON_MAPPING['ambulance']['url'], icon_mapping=ICON_MAPPING)
    incident_layer = pdk.Layer("IconLayer", data=incident_df, get_icon="incident", get_position='[lon, lat]', get_size=30, size_scale=1, pickable=True, icon_atlas=ICON_MAPPING['incident']['url'], icon_mapping=ICON_MAPPING)
    
    layers = [zone_layer, hospital_layer, ambulance_layer, incident_layer]
    
    if route_info and "error" not in route_info:
        route_path = LineString([route_info['ambulance_location'], route_info['hospital_location']])
        route_df = pd.DataFrame([{'path': [list(p) for p in route_path.coords]}])
        route_layer = pdk.Layer('PathLayer', data=route_df, get_path='path', get_width=5, get_color=[251, 192, 45], width_scale=1, width_min_pixels=5)
        layers.append(route_layer)
        
    view_state = pdk.ViewState(latitude=32.525, longitude=-117.02, zoom=11.5, bearing=0, pitch=45)
    
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v9', tooltip={"text": "Zone: {name}\nRisk: {risk:.2f}"})

def main():
    st.set_page_config(page_title="RedShield AI: Operational Twin", layout="wide")
    st.title("🚑 RedShield AI: Operational Digital Twin")

    if 'data_fabric' not in st.session_state: st.session_state.data_fabric = DataFusionFabric()
    if 'cognitive_engine' not in st.session_state: st.session_state.cognitive_engine = DigitalTwinCore(st.session_state.data_fabric)

    data_fabric, cognitive_engine = st.session_state.data_fabric, st.session_state.cognitive_engine
    
    if st.sidebar.button("Force Refresh Live Data"): data_fabric.get_live_state.clear()
    live_state = data_fabric.get_live_state()
    risk_scores = cognitive_engine._calculate_risk_scores(live_state)
    
    # --- BUG FIX: DATA SANITIZATION STEP ---
    # Create clean, serializable lists of dictionaries for Pydeck DataFrames.
    hospital_data = [{"name": name, "lon": data['location'].x, "lat": data['location'].y} for name, data in data_fabric.hospitals.items()]
    ambulance_data = [{"name": name, "lon": data['location'].x, "lat": data['location'].y} for name, data in data_fabric.ambulances.items()]
    all_incidents = [inc for zone_data in live_state.values() for inc in zone_data['active_incidents']]
    incident_data = [{"name": inc['id'], "lon": inc['location'].x, "lat": inc['location'].y} for inc in all_incidents]
    
    # Create the final DataFrames for visualization
    hospital_df = pd.DataFrame(hospital_data)
    ambulance_df = pd.DataFrame(ambulance_data)
    incident_df = pd.DataFrame(incident_data)

    zones_gdf = gpd.GeoDataFrame.from_dict(data_fabric.static_zonal_data, orient='index').set_geometry('polygon')
    zones_gdf['name'] = zones_gdf.index
    zones_gdf['risk'] = zones_gdf.index.map(risk_scores)
    max_risk = zones_gdf['risk'].max() if not zones_gdf['risk'].empty else 1.0
    zones_gdf['fill_color'] = zones_gdf['risk'].apply(lambda r: [255, int(255 * (1 - r / max_risk)), 0, 140] if max_risk > 0 else [0, 255, 0, 140]).tolist()

    st.sidebar.subheader("Live Incidents")
    if not all_incidents:
        st.sidebar.info("No active incidents.")
        st.session_state.selected_incident = None
    else:
        incident_ids = [inc['id'] for inc in all_incidents]
        selected_id = st.sidebar.selectbox("Select Incident to Route:", incident_ids, key="incident_selector")
        st.session_state.selected_incident = next((inc for inc in all_incidents if inc['id'] == selected_id), None)

    col1, col2 = st.columns((2, 1))

    with col1:
        st.subheader("Operational Map")
        route_info = None
        if st.session_state.get('selected_incident'):
            with st.spinner("Calculating optimal risk-aware route..."):
                route_info = cognitive_engine.find_best_route_for_incident(st.session_state.selected_incident, zones_gdf)
        
        # Pass the sanitized DataFrames to the map function
        deck_map = create_deck_gl_map(zones_gdf, hospital_df, ambulance_df, incident_df, route_info)
        st.pydeck_chart(deck_map)

    with col2:
        st.subheader("Routing Decision Engine")
        if not st.session_state.get('selected_incident'):
            st.info("Select an incident from the sidebar to see the routing plan.")
        elif route_info and "error" not in route_info:
            st.success(f"**Dispatch Plan for Incident {st.session_state.selected_incident['id']}**")
            st.metric("Recommended Unit", route_info['ambulance_unit'])
            st.metric("Optimal Destination", route_info['best_hospital'])
            st.markdown("**Routing Analysis (Lower Score is Better):**")
            st.dataframe(route_info['routing_analysis'].set_index('hospital').style.highlight_min(axis=0, color='#006400'))
        else:
            st.error(route_info.get("error", "Could not calculate a route."))
    
if __name__ == "__main__":
    main()
