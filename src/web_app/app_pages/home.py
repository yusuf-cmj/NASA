"""
Home page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.predictor import predict_exoplanet
from config.settings import FEATURE_NAMES, FEATURE_LABELS, PERFORMANCE_METRICS, DATASET_INFO

def home_page(model, scaler, label_encoder):
    """Home page with overview and quick prediction"""
    
    st.markdown("# NASA Exoplanet Detection System")
    st.markdown("AI-powered classification of exoplanet candidates using NASA mission data")
    st.markdown("---")
    
    # Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### System Overview")
        
        st.markdown(f"""
        This AI system classifies exoplanet candidates as **confirmed exoplanets** or **false positives** 
        with **{PERFORMANCE_METRICS['accuracy']}% accuracy** using NASA mission data.
        """)
        
        st.markdown("### Input Features")
        st.markdown("""
        - **Orbital Period** (days)
        - **Transit Duration** (hours)
        - **Planet Radius** (Earth radii)
        - **Stellar Temperature** (Kelvin)
        - **Stellar Radius** (Solar radii)
        - **Transit Depth** (parts per million)
        """)
        
        st.markdown("### Training Data")
        st.markdown(f"""
        - **Kepler Mission:** {DATASET_INFO['kepler_records']:,} records
        - **TESS Mission:** {DATASET_INFO['tess_records']:,} records
        - **K2 Mission:** {DATASET_INFO['k2_records']:,} records
        - **Total:** {DATASET_INFO['total_records']:,} exoplanet candidates
        """)
        
        st.markdown("### Model Performance")
        st.markdown(f"""
        - **Accuracy:** {PERFORMANCE_METRICS['accuracy']}%
        - **ROC-AUC:** {PERFORMANCE_METRICS['roc_auc']}
        - **Method:** Stacking Ensemble (Random Forest + XGBoost + Extra Trees)
        """)
    
    with col2:
        st.markdown("### Quick Prediction")
        
        with st.form("quick_prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                orbital_period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0)
                transit_duration = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=50.0, value=3.0)
                planet_radius = st.number_input("Planet Radius (Earth radii)", min_value=0.1, max_value=50.0, value=2.0)
            
            with col2:
                stellar_temp = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5500)
                stellar_radius = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=50.0, value=1.0)
                transit_depth = st.number_input("Transit Depth (ppm)", min_value=1, max_value=100000, value=1000)
            
            submitted = st.form_submit_button("Analyze Candidate", use_container_width=True, type="primary")
            
            if submitted:
                features = np.array([
                    orbital_period, transit_duration, planet_radius,
                    stellar_temp, stellar_radius, transit_depth
                ])
                
                prediction, confidence, prob = predict_exoplanet(features, model, scaler, label_encoder)
                
                if prediction:
                    if prediction == 'CONFIRMED':
                        st.markdown(f"""
                        <div class="prediction-result confirmed">
                        CONFIRMED EXOPLANET<br>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-result false-positive">
                        FALSE POSITIVE<br>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Next Steps")
    st.markdown("For detailed analysis with confidence intervals and feature explanations, use the **Single Prediction** page.")
