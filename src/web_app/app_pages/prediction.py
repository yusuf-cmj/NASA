"""
Prediction page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    
    st.markdown("# 🔮 Single Exoplanet Prediction")
    
    # Input form
    st.markdown("## 📥 Enter Exoplanet Candidate Data")
    
    with st.form("detailed_prediction"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 Planetary Characteristics")
            orbital_period = st.number_input(
                "Orbital Period (days)", 
                min_value=0.1, max_value=1000.0, 
                value=10.0, help="Time for planet to orbit its star"
            )
            transit_duration = st.number_input(
                "Transit Duration (hours)", 
                min_value=0.1, max_value=50.0, 
                value=3.0, help="Time planet passes in front of star"
            )
            planet_radius = st.number_input(
                "Planet Radius (Earth radii)", 
                min_value=0.1, max_value=50.0, 
                value=2.0, help="Planet size relative to Earth"
            )
        
        with col2:
            st.markdown("### ⭐ Stellar Characteristics") 
            stellar_temp = st.number_input(
                "Stellar Temperature (K)", 
                min_value=2000, max_value=10000, 
                value=5500, help="Star surface temperature"
            )
            stellar_radius = st.number_input(
                "Stellar Radius (Solar radii)", 
                min_value=0.1, max_value=50.0, 
                value=1.0, help="Star size relative to Sun"
            )
            transit_depth = st.number_input(
                "Transit Depth (parts per million)", 
                min_value=1, max_value=100000, 
                value=1000, help="How much star light is blocked"
            )
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Analyze Exoplanet Candidate", use_container_width=True, type="primary")
    
    if submitted:
        # Make prediction
        features = np.array([
            orbital_period, transit_duration, planet_radius,
            stellar_temp, stellar_radius, transit_depth
        ])
        
        prediction, confidence, prob = predict_exoplanet(features, model, scaler, label_encoder)
        
        if prediction:
            # Results display
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 'CONFIRMED':
                    st.markdown(f"""
                    <div class="prediction-result confirmed">
                    🌍 CONFIRMED EXOPLANET<br>
                    Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    ### ✅ What This Means:
                    - This candidate shows strong evidence of being a **real exoplanet**
                    - Our model is **{confidence:.1f}% confident** in this prediction
                    - This result is based on patterns learned from **21,271 NASA records**
                    
                    ### 🔬 Scientific Interpretation:
                    The combination of orbital parameters, transit characteristics, and stellar properties 
                    strongly suggests this is a genuine planetary companion orbiting its host star.
                    """)
                    
                else:  # FALSE_POSITIVE
                    st.markdown(f"""
                    <div class="prediction-result false-positive">
                    ⚠️ FALSE POSITIVE<br>
                    Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    ### ❌ What This Means:
                    - This candidate shows characteristics consistent with **false positive signals**
                    - Our model is **{confidence:.1f}% confident** this is not a genuine exoplanet
                    - This could be instrumental noise, stellar activity, or other false signals
                    
                    ### 🔬 Scientific Interpretation:
                    While interesting, this signal does not appear to represent a true planetary transit
                    based on our analysis of the input parameters.
                    """)
            
            with col2:
                st.markdown("### 📊 Prediction Probabilities")
                
                # Probability visualization
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
                    title="Model Confidence"
                )
                fig.update_layout(showlegend=False, height=300)
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance section
            st.markdown("### 🔍 Feature Analysis")
            st.markdown("This section would show feature importance analysis (coming in Phase 2)")
