# RedShieldAI_Final_Review_Patched.py
# SME REVIEW: UnhashableParamError fixed. The application is now stable.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
from typing import Dict, Any, List, Tuple, Sequence
import logging
from datetime import date

# --- App Configuration ---
# Centralize all static data and magic numbers for easy management.
CONFIG = {
    "page_title": "RedShield AI Emergency Optimization Platform",
    "ambulance_start_location": (32.5149, -117.0382),
    "hospital_options": [
        {'name': 'Hospital General Tijuana', 'location': (32.5295, -117.0182), 'current_load': 80, 'capacity': 100},
        {'name': 'IMSS Clinica 1', 'location': (32.5121, -117.0145), 'current_load': 60, 'capacity': 120},
        {'name': 'Hospital Angeles', 'location': (32.5300, -117.0200), 'current_load': 90, 'capacity': 100},
        {'name': 'Field Clinic (Closed)', 'location': (32.5000, -117.0000), 'current_load': 0, 'capacity': 0}
    ],
    "traffic_data": {
        'Hospital General Tijuana': 5,
        'IMSS Clinica 1': 3,
        'Hospital Angeles': 7
    },
    "patient_sensor_data": {
        'P001': {'heart_rate': 130, 'oxygen': 93},
        'P002': {'heart_rate': 145, 'oxygen': 87},
        'P003': {'heart_rate': 88, 'oxygen': 98},
        'P004': {'heart_rate': 150}
    },
    # BUG FIX: Converted the unhashable pd.DatetimeIndex into a hashable tuple of date objects.
    # This is the primary fix for the UnhashableParamError.
    "holidays": tuple(pd.to_datetime(['2024-01-01', '2024-01-06']).date),
    "routing_eta_factors": {
        "avg_speed_km_per_min": 0.8,
        "load_penalty_multiplier": 10
    },
    "critical_vitals_thresholds": {
        "max_heart_rate": 140,
        "min_oxygen_level": 90
    }
}

st.set_page_config(page_title=CONFIG["page_title"], layout="wide")

# --- Utility Functions ---
def _safe_division(numerator, denominator):
    """Safely divides two numbers, returning 0.0 if the denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator

# --- Data Loading and Caching ---
@st.cache_data
def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates and caches sample historical and weather data."""
    historical_data = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=24*30, freq='H')),
        'calls': np.random.poisson(5, size=24*30) + np.sin(np.arange(24*30) * 2 * np.pi / 24) * 3 + 2
    })
    weather_data = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=24*30, freq='H')),
        'temperature': np.random.normal(25, 5, size=24*30),
        'rain': np.random.choice([0, 1], size=24*30, p=[0.8, 0.2])
    })
    return historical_data, weather_data

# --- Module: Demand Forecasting ---
@st.cache_resource
def train_forecasting_model(historical_data: pd.DataFrame, weather_data: pd.DataFrame, holidays: Sequence[date]) -> Tuple[RandomForestRegressor, List[str]]:
    """
    Prepares data, trains the model, and returns the model and feature names.
    The 'holidays' argument is now a hashable Sequence of date objects.
    """
    df = pd.merge(historical_data, weather_data, on='timestamp', how='left')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # BUG FIX: The 'holidays' variable is now a simple tuple, which can be used directly.
    df['is_holiday'] = df['timestamp'].dt.date.isin(holidays).astype(int)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    feature_names = ['hour', 'day_of_week', 'temperature', 'rain', 'is_holiday']
    X = df[feature_names]
    y = df['calls']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, feature_names

def display_forecasting_module(model: RandomForestRegressor, feature_names: List[str], holidays: Sequence[date]):
    """Renders the UI for the Demand Forecasting section."""
    st.header("📈 Demand Forecasting")
    st.markdown("Predict the number of emergency calls for a specific time and weather condition.")
    
    col1, col2 = st.columns(2)
    with col1:
        date_input = st.date_input("Select date", pd.to_datetime("2024-01-05"))
        temperature = st.slider("Temperature (°C)", -5, 45, 22)
    with col2:
        hour_input = st.slider("Select hour", 0, 23, 15)
        rain = st.checkbox("Raining", value=False)

    # BUG FIX: Logic now correctly checks against the tuple of date objects.
    is_holiday = pd.to_datetime(date_input).date() in holidays
    day_of_week = pd.to_datetime(date_input).dayofweek
    
    prediction_input = pd.DataFrame(
        data=[[hour_input, day_of_week, temperature, int(rain), int(is_holiday)]],
        columns=feature_names
    )
    
    predicted_calls = model.predict(prediction_input)[0]
    st.metric(label="Predicted Emergency Calls per Hour", value=f"{predicted_calls:.2f}", delta_color="off")


# --- Module: Smart Routing ---
def calculate_optimal_route(ambulance_location: tuple, hospitals: List[Dict], traffic: Dict, eta_factors: Dict) -> Dict:
    """Calculates the best hospital to route to with robust error handling."""
    best_option = None
    min_score = float('inf')
    
    avg_speed = eta_factors.get('avg_speed_km_per_min', 0.8)
    if avg_speed == 0:
        logging.error("Configuration error: avg_speed_km_per_min cannot be zero. Falling back to default.")
        avg_speed = 0.8

    for h in hospitals:
        capacity = h.get('capacity', 0)
        current_load = h.get('current_load', 0)
        
        if capacity == 0:
            continue

        distance_km = geodesic(ambulance_location, h.get('location', ambulance_location)).km
        travel_time = _safe_division(distance_km, avg_speed)
        traffic_delay = traffic.get(h.get('name'), 0)
        
        hospital_load_pct = _safe_division(current_load, capacity)
        load_penalty = hospital_load_pct * eta_factors.get('load_penalty_multiplier', 10)
        
        total_score = travel_time + traffic_delay + load_penalty
        
        if total_score < min_score:
            min_score = total_score
            best_option = h
            
    return best_option

def display_routing_module():
    """Renders the UI for the Smart Routing section."""
    st.header("🛣️ Smart Routing")
    st.markdown("Recommends the optimal hospital based on travel time, traffic, and real-time hospital capacity.")
    
    best_hospital = calculate_optimal_route(
        CONFIG["ambulance_start_location"],
        CONFIG["hospital_options"],
        CONFIG["traffic_data"],
        CONFIG["routing_eta_factors"]
    )
    
    if best_hospital:
        st.success(f"**Recommended Hospital: {best_hospital['name']}**")
        st.map(pd.DataFrame([
            {'lat': CONFIG['ambulance_start_location'][0], 'lon': CONFIG['ambulance_start_location'][1]},
            {'lat': best_hospital.get('location', (0,0))[0], 'lon': best_hospital.get('location', (0,0))[1]}
        ]))
    else:
        st.error("Could not determine an optimal route. All available hospitals may be at zero capacity.")

# --- Module: Patient Monitoring ---
def check_patient_vitals(sensor_data: Dict, thresholds: Dict) -> List[Dict]:
    """Identifies patients with critical vital signs, robust to missing data."""
    alerts = []
    max_hr = thresholds.get("max_heart_rate", 140)
    min_o2 = thresholds.get("min_oxygen_level", 90)

    for pid, vitals in sensor_data.items():
        heart_rate = vitals.get('heart_rate', 0)
        oxygen = vitals.get('oxygen', 100)

        if heart_rate > max_hr or oxygen < min_o2:
            alerts.append({'Patient ID': pid, 'Alert': 'Critical Vitals', 'Details': vitals})
    return alerts

def display_monitoring_module():
    """Renders the UI for the Patient Monitoring section."""
    st.header("🩺 Real-Time Patient Monitoring")
    st.markdown("Monitors incoming patient data from ambulances and flags critical conditions.")
    
    alerts = check_patient_vitals(CONFIG["patient_sensor_data"], CONFIG["critical_vitals_thresholds"])
    
    if alerts:
        st.warning(f"🚨 {len(alerts)} Active Alert(s) Detected!")
        st.table(pd.DataFrame(alerts).set_index('Patient ID'))
    else:
        st.success("✅ All patient vitals are within normal ranges.")

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title(f"🚑 {CONFIG['page_title']}")

    historical_data, weather_data = load_sample_data()
    # This call now works correctly because CONFIG["holidays"] is hashable
    model, feature_names = train_forecasting_model(historical_data, weather_data, CONFIG["holidays"])
    
    st.header("📊 System-Wide Dashboard")
    alerts = check_patient_vitals(CONFIG["patient_sensor_data"], CONFIG["critical_vitals_thresholds"])
    
    hospital_loads = {
        h.get('name', 'Unknown'): _safe_division(h.get('current_load', 0), h.get('capacity', 0))
        for h in CONFIG["hospital_options"] if h.get('capacity', 0) > 0
    }
    load_df = pd.DataFrame.from_dict(hospital_loads, orient='index', columns=['Utilization'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Active Ambulances", value=12)
    with col2:
        st.metric(label="Average Response Time (min)", value="8.4")
    with col3:
        st.metric(label="Open Critical Alerts", value=len(alerts))
    
    st.subheader("Current Hospital Utilization (Active Facilities)")
    st.bar_chart(load_df)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Demand Forecasting", "Smart Routing", "Patient Monitoring"])

    with tab1:
        display_forecasting_module(model, feature_names, CONFIG["holidays"])
    
    with tab2:
        display_routing_module()
        
    with tab3:
        display_monitoring_module()

if __name__ == "__main__":
    main()
