"""
About page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PERFORMANCE_METRICS, DATASET_INFO

def about_page():
    """About page with project information"""
    
    st.markdown("# About NASA Exoplanet Detection System")
    st.markdown("Comprehensive AI-powered exoplanet classification system")
    st.markdown("---")
    
    st.markdown("## Project Overview")
    
    st.markdown(f"""
    This AI-powered system classifies exoplanet candidates as confirmed exoplanets or false positives 
    with **{PERFORMANCE_METRICS['accuracy']}% accuracy**. Built using NASA mission data from Kepler, TESS, and K2.
    """)
    st.markdown("## Mission Statement")
    
    st.markdown("""
    Our goal is to assist astronomers and researchers in efficiently identifying real exoplanets from 
    NASA mission data. By automating the classification process, we aim to accelerate exoplanet 
    discovery and reduce manual analysis time.
    """)
    st.markdown("## Technical Specifications")
    
    st.markdown(f"""
    ### Machine Learning Approach:
    - **Binary Classification:** CONFIRMED vs FALSE_POSITIVE
    - **Ensemble Method:** Stacking (Random Forest + XGBoost + Extra Trees)
    - **Dataset:** {DATASET_INFO['total_records']:,} exoplanet candidates from NASA missions
    - **Features:** 6 key astronomical parameters
    
    ### Performance Metrics:
    - **Accuracy:** {PERFORMANCE_METRICS['accuracy']}%
    - **ROC-AUC:** {PERFORMANCE_METRICS['roc_auc']}
    - **F1-Score:** {PERFORMANCE_METRICS['f1_score']}
    - **Improvement:** +23.39% over ternary classification
    """)
    st.markdown("## Data Sources")
    
    st.markdown(f"""
    Our model is trained on three comprehensive NASA datasets:
    
    1. **Kepler Objects of Interest (KOI):** {DATASET_INFO['kepler_records']:,} records
    2. **TESS Objects of Interest (TOI):** {DATASET_INFO['tess_records']:,} records
    3. **K2 Planets and Candidates:** {DATASET_INFO['k2_records']:,} records
    """)
    st.markdown("## System Features")
    
    st.markdown("""
    ### Advanced Capabilities:
    - **Batch Processing:** Upload CSV files for multiple predictions
    - **Model Comparison:** Compare different ML models
    - **Analytics Dashboard:** Feature importance and pattern analysis
    - **Export Results:** Download predictions as CSV
    """)
    st.markdown("## Development Approach")
    
    st.markdown("""
    This project emphasizes:
    - User-friendly web interface
    - High-accuracy machine learning models
    - Comprehensive data analysis
    - Real-world applicability
    """)
    st.markdown("## Future Enhancements")
    
    st.markdown("""
    - Real-time data ingestion
    - Hyperparameter optimization interface
    - Model retraining with new data
    - API integration
    - Mobile application
    """)
