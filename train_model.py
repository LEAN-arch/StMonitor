# train_model.py
# SME-Grade Offline Model Training Script
# SME FIX: Uses absolute paths to guarantee files are saved in the correct location.

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import yaml
import os

print("--- Starting RedShield AI Model Training ---")

# --- SME FIX: Use absolute paths to make script location-independent ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, 'demand_model.xgb')
FEATURES_FILE = os.path.join(SCRIPT_DIR, 'model_features.json')
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.yaml')

print(f"Project directory detected: {SCRIPT_DIR}")

# Load configuration to get model parameters
print(f"Loading configuration from {CONFIG_FILE}...")
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)
model_params = config.get('data', {}).get('model_params', {})
print(f"Using XGBoost parameters: {model_params}")

# 1. Generate Training Data
print("Generating synthetic training data (1 year, hourly)...")
hours = 24 * 365
timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=hours, freq='h'))
X_train = pd.DataFrame({
    'hour': timestamps.hour,
    'day_of_week': timestamps.dayofweek,
    'is_quincena': timestamps.day.isin([14, 15, 16, 29, 30, 31, 1]),
    'temperature': np.random.normal(22, 5, hours),
    'border_wait': np.random.randint(20, 120, hours)
})
y_train = np.maximum(0, 5 + 3 * np.sin(X_train['hour'] * 2 * np.pi / 24) + X_train['is_quincena'] * 5 + X_train['border_wait']/20 + np.random.randn(hours)).astype(int)
print("Data generation complete.")

# 2. Train the XGBoost Model
print("Training XGBoost Regressor model... (This may take a moment)")
model = xgb.XGBRegressor(objective='reg:squarederror', **model_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# 3. Save the Model and Feature List to Disk
print(f"Saving model to '{MODEL_FILE}'...")
model.save_model(MODEL_FILE)

print(f"Saving feature list to '{FEATURES_FILE}'...")
features = list(X_train.columns)
with open(FEATURES_FILE, 'w') as f:
    json.dump(features, f)

print("\n--- Model Training Successful ---")
print(f"Artifacts '{MODEL_FILE}' and '{FEATURES_FILE}' have been created.")
print("The main RedShieldAI application is now ready to be launched.")
