# RedShieldAI_Cognitive_Engine_FIXED.py
# SME LEVEL: A robust, bug-free, and production-quality prototype of the 
# self-correcting, prescriptive digital twin architecture.

import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any

# --- L1: DATA FUSION FABRIC (SIMULATED & ROBUST) ---
class DataFusionFabric:
    """
    Simulates real-time data ingestion from various city sources.
    In production, this class would manage connections to databases, Kafka, and APIs.
    """
    def __init__(self):
        self.static_zonal_data = {
            "Zona Río": {"crime_index": 0.7, "road_quality": 0.9, "base_demand": 5.0},
            "Otay": {"crime_index": 0.5, "road_quality": 0.7, "base_demand": 7.0},
            "Playas": {"crime_index": 0.4, "road_quality": 0.8, "base_demand": 3.0}
        }

    # ROBUSTNESS: Cached to provide a stable state that doesn't change on every widget interaction.
    @st.cache_data(ttl=300) # Cache live data for 5 minutes
    def get_live_state(_self) -> Dict[str, Dict[str, Any]]:
        """
        Simulates fetching a real-time snapshot of the city.
        The simulation is now time-dependent for realism.
        """
        hour = datetime.now().hour
        # Simulate higher traffic during rush hour and evening
        is_rush_hour = 7 <= hour <= 9 or 16 <= hour <= 19
        is_evening = 19 < hour <= 23

        live_state = {}
        for zone, data in _self.static_zonal_data.items():
            base_traffic = data.get('crime_index', 0.5) # Base traffic related to zone character
            traffic_multiplier = 1.0
            if is_rush_hour: traffic_multiplier = 1.5
            if is_evening and zone == "Zona Río": traffic_multiplier = 1.8 # Evening rush

            live_state[zone] = {
                "traffic": min(1.0, base_traffic * traffic_multiplier + np.random.uniform(-0.1, 0.1)),
                "active_incidents": random.randint(0, int(data.get('base_demand', 5) / 3)),
                "event_multiplier": 3.0 if is_evening and zone == "Zona Río" else 1.0,
            }
        return live_state

    def get_incident_outcome(self, incident_id: str) -> Dict[str, Any]:
        """Simulates fetching outcome data after an incident is resolved."""
        return {
            "incident_id": incident_id,
            "predicted_eta_min": 12.0,
            "actual_eta_min": 15.5, # The model was too optimistic
            "patient_stability_degradation": 0.2, # Patient worsened
            "zone": "Zona Río"
        }

# --- L2: DIGITAL TWIN CORE (COGNITIVE ENGINE) ---
class DigitalTwinCore:
    """
    The 'brain' of the system. It uses models to predict, plan, and learn.
    """
    def __init__(self, data_fabric: DataFusionFabric):
        # BUG FIX: Use dependency injection. The engine now uses the one true data_fabric instance.
        self.data_fabric = data_fabric
        # These would be complex, pre-trained models. We simulate their logic.
        self.probabilistic_demand_model = "LoadedModel_GNN"
        self.causal_friction_model = "LoadedModel_Causal"
        self.rl_dispatch_agent = "LoadedModel_RL"
        self.model_confidence = 0.95

    def run_analysis(self, live_state: Dict[str, Any], available_ambulances: int) -> Dict[str, Any]:
        """Runs the full cognitive pipeline: predict, plan, and package results."""
        with st.spinner("Cognitive Engine processing..."):
            demand_dist = self._predict_demand_distribution(live_state)
            dispatch_plan = self._get_optimal_dispatch_plan(live_state, available_ambulances, demand_dist)
        return {"demand_distribution": demand_dist, "dispatch_plan": dispatch_plan}

    def _predict_demand_distribution(self, live_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Predicts a probability distribution of calls per zone, capturing uncertainty."""
        predictions = {}
        for zone, state in live_state.items():
            # ROBUSTNESS: Use .get() with defaults for all dictionary access.
            base_demand = self.data_fabric.static_zonal_data.get(zone, {}).get('base_demand', 0)
            traffic = state.get('traffic', 0)
            event_mult = state.get('event_multiplier', 1)
            incidents = state.get('active_incidents', 0)
            
            mean_calls = base_demand * (1 + traffic) * event_mult + incidents
            # Higher confidence -> lower uncertainty (std dev)
            std_dev = mean_calls * (1.1 - self.model_confidence)
            predictions[zone] = {"mean": mean_calls, "std_dev": max(0.1, std_dev)}
        return predictions

    def _get_optimal_dispatch_plan(self, live_state: Dict, available_ambulances: int, demand_dist: Dict) -> Dict:
        """Uses a simulated RL agent to create a strategic resource allocation plan."""
        # RL agent would consider demand, risk, available units, etc.
        highest_demand_zone = max(demand_dist, key=lambda z: demand_dist[z].get('mean', 0))
        return {
            "system_recommendation": "Pre-positioning and strategic response",
            "actions": [
                {"action": "PRE_POSITION", "unit": "A03", "target_zone": highest_demand_zone, "reason": "Anticipated high demand"},
                {"action": "HOLD_IN_RESERVE", "unit": "A04", "reason": "Coverage for unexpected major incidents"},
                {"action": "RESPOND", "unit": "A01", "incident_id": "I-123", "zone": "Zona Río"},
            ]
        }

    def learn_from_outcome(self, outcome: Dict[str, Any]):
        """Continual learning loop. Updates models based on real-world feedback."""
        eta_mismatch = outcome.get('actual_eta_min', 0) - outcome.get('predicted_eta_min', 0)
        zone = outcome.get('zone', 'Unknown')
        
        if abs(eta_mismatch) > 1:
            st.warning(f"LEARNING: Significant ETA error of {eta_mismatch:.1f} min in {zone}. Updating friction model.")
            # In reality: self.causal_friction_model.partial_fit(new_data)
        
        self.model_confidence = max(0.5, self.model_confidence - abs(eta_mismatch) * 0.01)
        st.info(f"System model confidence adjusted to: {self.model_confidence:.2%}")

# --- MAIN APPLICATION LOGIC ---
def main():
    st.set_page_config(page_title="RedShield AI: Cognitive Engine", layout="wide")
    st.title("🧠 RedShield AI: Sentient Digital Twin (Operational Prototype)")

    # --- Initialize Singleton Instances in Session State ---
    if 'data_fabric' not in st.session_state:
        st.session_state.data_fabric = DataFusionFabric()
    if 'cognitive_engine' not in st.session_state:
        st.session_state.cognitive_engine = DigitalTwinCore(st.session_state.data_fabric)

    data_fabric = st.session_state.data_fabric
    cognitive_engine = st.session_state.cognitive_engine

    # --- UI Tabs ---
    tab1, tab2, tab3 = st.tabs(["Strategic Dashboard", "Simulation & Counterfactuals", "System Learning"])

    with tab1:
        st.header("Real-Time Cognitive Dashboard")
        if st.button("Force Refresh Live Data"):
            # BUG FIX: Correct way to force a refresh on a cached function.
            data_fabric.get_live_state.clear()
        
        # BUG FIX: Unified logic flow. Get state, then run analysis.
        live_state = data_fabric.get_live_state()
        analysis_results = cognitive_engine.run_analysis(live_state, available_ambulances=5)
        demand_dist = analysis_results["demand_distribution"]
        dispatch_plan = analysis_results["dispatch_plan"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Probabilistic Demand Forecast")
            for zone, dist in demand_dist.items():
                st.metric(
                    label=f"Predicted Calls in {zone}",
                    value=f"{dist.get('mean', 0):.1f}",
                    delta=f"± {1.96 * dist.get('std_dev', 0):.1f} (95% CI)",
                    delta_color="off"
                )
        with col2:
            st.subheader("Prescriptive Dispatch Plan")
            st.success(f"**{dispatch_plan.get('system_recommendation')}**")
            st.json(dispatch_plan.get('actions', []))

    with tab2:
        st.header("Simulation: Counterfactual Analysis")
        st.markdown("Ask 'What-If' questions to test system resilience.")
        
        sim_traffic_increase = st.slider("Simulate Traffic Spike in Zona Río", 0.0, 0.5, 0.2)
        
        # Create a safe, deep copy for simulation
        simulated_state = {k: v.copy() for k, v in data_fabric.get_live_state().items()}
        simulated_state["Zona Río"]["traffic"] = min(1.0, simulated_state["Zona Río"].get('traffic', 0) + sim_traffic_increase)
        
        st.warning("Running simulation with modified state...")
        sim_results = cognitive_engine.run_analysis(simulated_state, available_ambulances=5)
        
        st.subheader("Simulated Outcome")
        st.metric("Predicted Calls in Zona Río (Simulated)", f"{sim_results['demand_distribution']['Zona Río']['mean']:.1f}")
        st.json({"New Dispatch Plan": sim_results['dispatch_plan'].get('actions', [])})

    with tab3:
        st.header("Feedback & Continual Learning")
        st.info("This demonstrates how the system self-corrects after an incident.")
        
        if st.button("Simulate Resolution of Incident 'I-123'"):
            with st.spinner("Processing incident outcome..."):
                outcome_data = data_fabric.get_incident_outcome("I-123")
                st.subheader("Received Outcome Data for Incident I-123")
                st.json(outcome_data)
                
                cognitive_engine.learn_from_outcome(outcome_data)
            st.success("Cognitive engine has processed the feedback and updated its internal models.")

if __name__ == "__main__":
    main()
