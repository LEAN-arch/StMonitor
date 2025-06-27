# RedShieldAI_Streamlit_Optimized.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
from typing import Dict, Any, List, Tuple

# --- App Configuration ---
# Centralize all static data and magic numbers for easy management.
CONFIG = {
    "page_title": "RedShield AI Emergency Optimization Platform",
    "ambulance_start_location": (32.5149, -117.0382),
    "hospital_options": [
        {'name': 'Hospital General Tijuana', 'location': (32.5295, -117.0182), 'current_load': 80, 'capacity': 100},
        {'name': 'IMSS Clinica 1', 'location': (32.5121, -117.0145), 'current_load': 60, 'capacity': 120},
        {'name': 'Hospital Angeles', 'location': (32.5300, -117.0200), 'current_load': 90, 'capacity': 100}
    ],
    "traffic_data": { # In a real app, this would come from a live API
        'Hospital General Tijuana': 5, # minutes
        'IMSS Clinica 1': 3,
        'Hospital Angeles': 7
    },
    "patient_sensor_data": {
        'P001': {'heart_rate': 130, 'oxygen': 93},
        'P002': {'heart_rate': 145, 'oxygen': 87},
        'P003': {'heart_rate': 88, 'oxygen': 98}
    },
    "holidays": pd.to_datetime(['2024-01-01', '2024-01-06']),
    "routing_eta_factors": {
        "avg_speed_km_per_min": 0.8, # 48 km/h
        "load_penalty_multiplier": 10 # Multiplier for hospital load percentage
    },
    "critical_vitals_thresholds": {
        "max_heart_rate": 140,
        "min_oxygen_level": 90
    }
}

st.set_page_config(page_title=CONFIG["page_title"], layout="wide")

# --- Data Loading and Caching ---
@st.cache_data
def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates and caches sample historical and weather data.
    Using @st.cache_data for dataframes that can be safely hashed.
    """
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
def train_forecasting_model(historical_data: pd.DataFrame, weather_data: pd.DataFrame, holidays: pd.Series) -> RandomForestRegressor:
    """
    Prepares data and trains the demand forecasting model.
    Using @st.cache_resource as a trained model is a complex object
    that should not be hashed on every run. It is created once and reused.
    """
    df = pd.merge(historical_data, weather_data, on='timestamp', how='left')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering (done efficiently)
    df['is_holiday'] = df['timestamp'].dt.date.isin([d.date() for d in holidays]).astype(int)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    X = df[['hour', 'day_of_week', 'temperature', 'rain', 'is_holiday']]
    y = df['calls']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def display_forecasting_module(model: RandomForestRegressor, holidays: pd.Series):
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

    # Prepare input for prediction
    is_holiday = pd.to_datetime(date_input).date() in [d.date() for d in holidays]
    day_of_week = pd.to_datetime(date_input).dayofweek
    
    input_features = np.array([[hour_input, day_of_week, temperature, int(rain), int(is_holiday)]])
    
    predicted_calls = model.predict(input_features)[0]
    st.metric(label="Predicted Emergency Calls per Hour", value=f"{predicted_calls:.2f}", delta_color="off")


# --- Module: Smart Routing ---
def calculate_optimal_route(ambulance_location: tuple, hospitals: List[Dict], traffic: Dict, eta_factors: Dict) -> Dict:
    """
    Calculates the best hospital to route to based on distance, traffic, and hospital load.
    
    Returns: The dictionary of the best hospital option.
    """
    best_option = None
    min_score = float('inf')
    
    for h in hospitals:
        distance_km = geodesic(ambulance_location, h['location']).km
        travel_time = distance_km / eta_factors['avg_speed_km_per_min']
        traffic_delay = traffic.get(h['name'], 0)
        
        # Calculate load penalty: higher load means higher penalty
        hospital_load_pct = h['current_load'] / h['capacity']
        load_penalty = hospital_load_pct * eta_factors['load_penalty_multiplier']
        
        # Total score is a combination of ETA and penalty
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
            {'lat': best_hospital['location'][0], 'lon': best_hospital['location'][1]}
        ]))
    else:
        st.error("Could not determine optimal route.")


# --- Module: Patient Monitoring ---
def check_patient_vitals(sensor_data: Dict, thresholds: Dict) -> List[Dict]:
    """Identifies patients with critical vital signs."""
    return [
        {'Patient ID': pid, 'Alert': 'Critical Vitals', 'Details': vitals}
        for pid, vitals in sensor_data.items()
        if vitals['heart_rate'] > thresholds['max_heart_rate'] or vitals['oxygen'] < thresholds['min_oxygen_level']
    ]

def display_monitoring_module():
    """Renders the UI for the Patient Monitoring section."""
    st.header("🩺 Real-Time Patient Monitoring")
    st.markdown("Monitors incoming patient data from ambulances and flags critical conditions.")
    
    alerts = check_patient_vitals(CONFIG["patient_sensor_data"], CONFIG["critical_vitals_thresholds"])
    
    if alerts:
        st.warning("🚨 Active Alerts Detected!")
        st.table(pd.DataFrame(alerts).set_index('Patient ID'))
    else:
        st.success("✅ All patient vitals are within normal ranges.")


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title(f"🚑 {CONFIG['page_title']}")

    # --- Load data and train model (will be cached after first run) ---
    historical_data, weather_data = load_sample_data()
    model = train_forecasting_model(historical_data, weather_data, CONFIG["holidays"])
    
    # --- Dashboard Summary ---
    st.header("📊 System-Wide Dashboard")
    alerts = check_patient_vitals(CONFIG["patient_sensor_data"], CONFIG["critical_vitals_thresholds"])
    
    # Hospital load data for chart
    hospital_loads = {h['name']: h['current_load'] / h['capacity'] for h in CONFIG["hospital_options"]}
    load_df = pd.DataFrame.from_dict(hospital_loads, orient='index', columns=['Utilization'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Active Ambulances", value=12)
    with col2:
        st.metric(label="Average Response Time (min)", value="8.4")
    with col3:
        st.metric(label="Open Critical Alerts", value=len(alerts))
    
    st.subheader("Current Hospital Utilization")
    st.bar_chart(load_df)
    st.divider()

    # --- Main Application Tabs ---
    tab1, tab2, tab3 = st.tabs(["Demand Forecasting", "Smart Routing", "Patient Monitoring"])

    with tab1:
        display_forecasting_module(model, CONFIG["holidays"])
    
    with tab2:
        display_routing_module()
        
    with tab3:
        display_monitoring_module()

if __name__ == "__main__":
    main()
