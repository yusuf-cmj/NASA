"""
About page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PERFORMANCE_METRICS, DATASET_INFO

def about_page():
    """About page with project information"""
    
    st.markdown("# ℹ️ About NASA Exoplanet Detection System")
    
    st.markdown(f"""
    ## 🚀 Project Overview
    
    This web application represents a comprehensive AI-powered system for detecting confirmed exoplanets 
    from candidate observations. Built as part of NASA's hackathon challenge, our system achieves 
    **{PERFORMANCE_METRICS['accuracy']}% accuracy** in distinguishing genuine exoplanets from false positives.
    
    ## 🎯 Mission Statement
    
    Our goal is to assist astronomers and researchers in efficiently identifying real exoplanets from 
    the massive amounts of data collected by NASA's Kepler, TESS, and K2 missions. By automating 
    the classification process, we aim to accelerate exoplanet discovery and reduce manual analysis time.
    
    ## 📊 Technical Details
    
    ### 🤖 Machine Learning Approach:
    - **Binary Classification:** CONFIRMED vs FALSE_POSITIVE
    - **Ensemble Method:** Stacking (Random Forest + XGBoost + Extra Trees)
    - **Dataset:** {DATASET_INFO['total_records']:,} exoplanet candidates from NASA missions
    - **Features:** 6 key astronomical parameters
    
    ### 🏆 Performance Metrics:
    - **Accuracy:** {PERFORMANCE_METRICS['accuracy']}%
    - **ROC-AUC:** {PERFORMANCE_METRICS['roc_auc']}
    - **F1-Score:** {PERFORMANCE_METRICS['f1_score']}
    - **Improvement:** +23.39% over ternary classification
    
    ## 🔬 Data Sources
    
    Our model is trained on three comprehensive NASA datasets:
    
    1. **Kepler Objects of Interest (KOI):** {DATASET_INFO['kepler_records']:,} records
    2. **TESS Objects of Interest (TOI):** {DATASET_INFO['tess_records']:,} records  
    3. **K2 Planets and Candidates:** {DATASET_INFO['k2_records']:,} records
    
    ## 🌟 Phase 3 Features
    
    ### ✅ Advanced Capabilities:
    - **Batch Processing:** Upload CSV files for multiple predictions
    - **Model Comparison:** Compare different ML models
    - **Analytics Dashboard:** Feature importance and pattern analysis
    - **Export Results:** Download predictions as CSV
    
    ## 👥 Development Team
    
    This project was developed following NASA's hackathon requirements, emphasizing:
    - User-friendly web interface
    - High-accuracy machine learning models
    - Comprehensive data analysis
    - Real-world applicability
    
    ## 🔮 Future Enhancements
    
    - Real-time data ingestion
    - Hyperparameter optimization interface
    - Model retraining with new data
    - API integration
    - Mobile application
    
    ---
    
    *Built with ❤️ for advancing exoplanet science and discovery*
    """)
