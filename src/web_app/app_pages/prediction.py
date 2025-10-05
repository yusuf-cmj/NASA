"""
Prediction page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.predictor import predict_exoplanet
from config.settings import FEATURE_NAMES, FEATURE_LABELS

def prediction_page(model, scaler, label_encoder):
    """Detailed prediction page with comprehensive inputs and outputs"""
    
    st.markdown("# Single Exoplanet Prediction")
    
    # Show active model indicator
    active_model = st.session_state.get('active_model', 'NebulaticAI')
    st.info(f"**Using Model:** {active_model} | [Switch Model](?page=models)")
    
    st.markdown("Detailed analysis of individual exoplanet candidates")
    st.markdown("---")
    
    st.markdown("## Input Parameters")
    
    with st.form("detailed_prediction"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Planetary Characteristics")
            orbital_period = st.number_input(
                "Orbital Period (days)", 
                min_value=0.1, max_value=1000.0, 
                value=10.0, step=0.000001, format="%.6f",
                help="Time for planet to orbit its star (high precision)"
            )
            transit_duration = st.number_input(
                "Transit Duration (hours)", 
                min_value=0.1, max_value=50.0, 
                value=3.0, step=0.000001, format="%.6f",
                help="Time planet passes in front of star (high precision)"
            )
            planet_radius = st.number_input(
                "Planet Radius (Earth radii)", 
                min_value=0.1, max_value=50.0, 
                value=2.0, step=0.000001, format="%.6f",
                help="Planet size relative to Earth (high precision)"
            )
        
        with col2:
            st.markdown("### Stellar Characteristics")
            stellar_temp = st.number_input(
                "Stellar Temperature (K)", 
                min_value=2000.0, max_value=10000.0, 
                value=5500.0, step=0.000001, format="%.6f",
                help="Star surface temperature (high precision)"
            )
            stellar_radius = st.number_input(
                "Stellar Radius (Solar radii)", 
                min_value=0.1, max_value=50.0, 
                value=1.0, step=0.000001, format="%.6f",
                help="Star size relative to Sun (high precision)"
            )
            transit_depth = st.number_input(
                "Transit Depth (parts per million)", 
                min_value=1.0, max_value=100000.0, 
                value=1000.0, step=0.000001, format="%.6f",
                help="How much star light is blocked (high precision)"
            )
        
        submitted = st.form_submit_button("Analyze Candidate", use_container_width=True, type="primary")
    
    if submitted:
        # Make prediction
        features = np.array([
            orbital_period, transit_duration, planet_radius,
            stellar_temp, stellar_radius, transit_depth
        ])
        
        prediction, confidence, prob = predict_exoplanet(features, model, scaler, label_encoder)
        
        if prediction:
            st.markdown("## Analysis Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 'CONFIRMED':
                    st.markdown(f"""
                    <div class="prediction-result confirmed">
                    CONFIRMED EXOPLANET<br>
                    Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    ### Classification Result
                    - **Prediction:** CONFIRMED EXOPLANET
                    - **Confidence:** {confidence:.1f}%
                    - **Interpretation:** Strong evidence of genuine planetary companion
                    """)
                    
                    st.markdown("### Scientific Analysis")
                    st.markdown(f"""
                    The orbital parameters, transit characteristics, and stellar properties 
                    indicate a **{confidence:.1f}% probability** of this being a confirmed exoplanet.
                    This classification is based on patterns learned from NASA mission data.
                    """)
                    
                else:  # FALSE_POSITIVE
                    st.markdown(f"""
                    <div class="prediction-result false-positive">
                    FALSE POSITIVE<br>
                    Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    ### Classification Result
                    - **Prediction:** FALSE POSITIVE
                    - **Confidence:** {confidence:.1f}%
                    - **Interpretation:** Signal inconsistent with planetary transit
                    """)
                    
                    st.markdown("### Scientific Analysis")
                    st.markdown(f"""
                    The input parameters suggest a **{confidence:.1f}% probability** of this being 
                    a false positive signal. This could result from instrumental noise, stellar activity, 
                    or other non-planetary phenomena.
                    """)
            
            with col2:
                st.markdown("### Model Confidence")
                
                prob_df = pd.DataFrame({
                    'Class': ['CONFIRMED', 'FALSE_POSITIVE'],
                    'Probability': [prob[0]*100, prob[1]*100] if prob is not None else [50, 50]
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    color='Class',
                    color_discrete_map={'CONFIRMED': '#2E8B57', 'FALSE_POSITIVE': '#DC143C'},
                    title="Classification Probabilities"
                )
                fig.update_layout(showlegend=False, height=300)
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
            