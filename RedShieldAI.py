# RedShieldAI_Streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic

st.set_page_config(page_title="RedShield AI Platform", layout="wide")

# --- Module: Demand Forecasting ---
def predict_demand_model(historical_data, weather_data, holidays):
    df = pd.merge(historical_data, weather_data, on='timestamp', how='left')
    df['is_holiday'] = df['timestamp'].isin(holidays).astype(int)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    X = df[['hour', 'day_of_week', 'temperature', 'rain', 'is_holiday']]
    y = df['calls']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# --- UI ---
st.title("🚑 RedShield AI Emergency Optimization Platform")

historical_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
    'calls': np.random.poisson(5, size=100)
})
weather_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
    'temperature': np.random.normal(25, 5, size=100),
    'rain': np.random.choice([0, 1], size=100)
})
holidays = pd.to_datetime(['2024-01-06', '2024-12-25'])

# --- Forecasting ---
st.header("📈 Demand Forecasting")
date_input = st.date_input("Select date", pd.to_datetime("2024-12-25"))
hour_input = st.slider("Select hour", 0, 23, 15)
temperature = st.slider("Temperature (°C)", 5, 45, 22)
rain = st.checkbox("Rain", value=True)
is_holiday = pd.to_datetime(date_input) in holidays

model = predict_demand_model(historical_data, weather_data, holidays)
predicted = model.predict([[hour_input, pd.to_datetime(date_input).dayofweek, temperature, int(rain), int(is_holiday)]])[0]
st.metric(label="Predicted Emergency Calls per Hour", value=f"{predicted:.2f}")

# --- Smart Routing ---
st.header("🛣️ Smart Routing")
ambulance_location = (32.5149, -117.0382)
hospital_options = [
    {'name': 'Hospital General Tijuana', 'location': (32.5295, -117.0182), 'current_load': 80, 'capacity': 100},
    {'name': 'IMSS Clinica 1', 'location': (32.5121, -117.0145), 'current_load': 60, 'capacity': 120},
    {'name': 'Hospital Angeles', 'location': (32.5300, -117.0200), 'current_load': 90, 'capacity': 100}
]
traffic_data = {
    'Hospital General Tijuana': 5,
    'IMSS Clinica 1': 3,
    'Hospital Angeles': 7
}

def calculate_fastest_route(location, hospitals, traffic):
    best_option = None
    min_eta = float('inf')
    for h in hospitals:
        dist_km = geodesic(location, h['location']).km
        delay = traffic.get(h['name'], 0)
        load_penalty = (h['current_load'] / h['capacity']) * 10
        eta = dist_km / 0.6 + delay + load_penalty
        if eta < min_eta:
            min_eta = eta
            best_option = h
    return best_option

best = calculate_fastest_route(ambulance_location, hospital_options, traffic_data)
st.success(f"🏥 Recommended Hospital: {best['name']}")

# --- Patient Monitoring ---
st.header("🩺 Patient Monitoring")
sensor_data = {
    'P001': {'heart_rate': 130, 'oxygen': 93},
    'P002': {'heart_rate': 145, 'oxygen': 87}
}
alerts = []
for pid, vitals in sensor_data.items():
    if vitals['heart_rate'] > 140 or vitals['oxygen'] < 90:
        alerts.append({'patient_id': pid, 'alert': 'Critical vital signs', 'status': vitals})

if alerts:
    st.warning("🚨 Active Alerts")
    st.table(pd.DataFrame(alerts))
else:
    st.success("No critical alerts.")

# --- Dashboard Summary ---
st.header("📊 Dashboard Summary")
dashboard = {
    'active_ambulances': 2,
    'average_response_time': np.mean([8, 10]),
    'open_alerts': len(alerts),
    'hospital_loads': {h['name']: round(h['current_load'] / h['capacity'], 2) for h in hospital_options}
}
st.json(dashboard)
