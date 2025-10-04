"""
Models page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_manager import get_available_models, get_model_info
from config.settings import PERFORMANCE_METRICS

def models_page():
    """Model management page"""
    
    st.markdown("# Model Management")
    st.markdown("Overview of available models and performance metrics")
    st.markdown("---")
    
    st.markdown("## Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{PERFORMANCE_METRICS['accuracy']}%")
    
    with col2:
        st.metric("ROC-AUC", f"{PERFORMANCE_METRICS['roc_auc']}")
    
    with col3:
        st.metric("F1-Score", f"{PERFORMANCE_METRICS['f1_score']}")
    
    with col4:
        st.metric("Precision", f"{PERFORMANCE_METRICS['precision']}")
    
    st.markdown("## Model Details")
    
    models = get_available_models()
    
    if models:
        for model in models:
            with st.expander(f"{model['name']}"):
                st.write(f"**Description:** {model['description']}")
                st.write(f"**Created:** {model['created_at']}")
                st.write(f"**Accuracy:** {model['accuracy']:.2f}%")
                st.write(f"**ROC-AUC:** {model['roc_auc']:.4f}")
    else:
        st.info("No additional models found. Only the default model is available.")
    
    st.markdown("## Architecture Overview")
    
    st.markdown("""
    ### Current Model: Binary Stacking Ensemble
    
    **Components:**
    - Random Forest Classifier
    - XGBoost Classifier
    - Extra Trees Classifier
    - Meta-learner: Logistic Regression
    
    **Performance:**
    - High accuracy (88.21%)
    - Robust to overfitting
    - Good generalization
    - Handles feature interactions well
    
    **Training Data:**
    - 21,271 exoplanet candidates
    - 6 key features
    - Binary classification (CONFIRMED vs FALSE_POSITIVE)
    """)
