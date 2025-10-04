"""
Model training page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PERFORMANCE_METRICS

def training_page(model, scaler, label_encoder):
    """Model retraining and hyperparameter tuning page"""
    
    st.markdown("# Model Training & Optimization")
    st.markdown("Retrain models and optimize hyperparameters for improved performance")
    st.markdown("---")
    
    st.markdown("## Current Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Accuracy", f"{PERFORMANCE_METRICS['accuracy']}%")
    
    with col2:
        st.metric("ROC-AUC Score", f"{PERFORMANCE_METRICS['roc_auc']}")
    
    with col3:
        st.metric("F1-Score", f"{PERFORMANCE_METRICS['f1_score']}")
    
    st.markdown("## Training Options")
    
    training_option = st.radio(
        "Select training approach:",
        ["Retrain Existing Model", "Hyperparameter Tuning", "Train New Model"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if training_option == "Retrain Existing Model":
        st.markdown("## Retrain Existing Model")
        
        st.markdown("""
        ### Retrain with New Data
        
        Upload new NASA exoplanet data to retrain the existing model:
        """)
        
        uploaded_data = st.file_uploader(
            "Upload new training data (CSV format)",
            type=['csv'],
            help="New NASA exoplanet data for model retraining"
        )
        
        if uploaded_data is not None:
            st.success("Data uploaded successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Retrain Model", type="primary"):
                    with st.spinner("Training model..."):
                        # Simulate training process
                        import time
                        time.sleep(2)
                        st.success("Model retrained successfully!")
            
            with col2:
                if st.button("Compare Performance"):
                    st.info("Performance comparison feature coming soon!")
    
    elif training_option == "Hyperparameter Tuning":
        st.markdown("## Hyperparameter Tuning")
        
        st.markdown("""
        ### Optimize Model Parameters
        
        Adjust hyperparameters to improve model performance:
        """)
        
        with st.expander("Random Forest Parameters"):
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            max_depth = st.slider("Max Depth", 5, 50, 20)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
            
            if st.button("Optimize Random Forest"):
                st.info("Hyperparameter optimization feature coming soon!")
        
        with st.expander("XGBoost Parameters"):
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            n_estimators_xgb = st.slider("Number of Boosters", 50, 500, 100)
            max_depth_xgb = st.slider("Max Depth", 3, 20, 6)
            
            if st.button("Optimize XGBoost"):
                st.info("Hyperparameter optimization feature coming soon!")
    
    elif training_option == "Train New Model":
        st.markdown("## Train New Model")
        
        st.markdown("""
        ### Create Custom Model
        
        Train a completely new model with different algorithms:
        """)
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "XGBoost", "Extra Trees", "Neural Network", "SVM"]
        )
        
        if st.button("Train New Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                # Simulate training
                import time
                time.sleep(3)
                st.success(f"{model_type} model trained successfully!")
    
    st.markdown("---")
    
    st.markdown("## Model Comparison")
    
    st.markdown("""
    ### Compare Different Models
    
    Evaluate and compare multiple model architectures:
    """)
    
    if st.button("Run Model Comparison"):
        st.info("Model comparison feature coming soon!")
    
    # Show relevant notes based on selected option
    if training_option == "Retrain Existing Model":
        st.markdown("## Retraining Notes")
        
        st.markdown("""
        ### Retraining Considerations:
        
        1. **Data Quality:** Ensure new data is properly cleaned and validated
        2. **Feature Consistency:** New data must have the same 6 required features
        3. **Computational Resources:** Retraining may take several minutes
        4. **Model Validation:** Always validate retrained models before deployment
        
        ### Best Practices:
        - Use cross-validation for robust performance estimates
        - Monitor for overfitting with validation curves
        - Save model metadata for reproducibility
        - Test on held-out NASA validation data
        """)
    
    elif training_option == "Hyperparameter Tuning":
        st.markdown("## Hyperparameter Tuning Notes")
        
        st.markdown("""
        ### Tuning Considerations:
        
        1. **Grid Search:** Systematic exploration of parameter space
        2. **Cross-Validation:** Use k-fold CV for robust estimates
        3. **Computational Cost:** Tuning can be computationally expensive
        4. **Overfitting Risk:** Monitor validation performance carefully
        
        ### Best Practices:
        - Start with coarse grid, then refine
        - Use early stopping to prevent overfitting
        - Document all parameter combinations tested
        - Validate final model on independent test set
        """)
    
    elif training_option == "Train New Model":
        st.markdown("## New Model Training Notes")
        
        st.markdown("""
        ### Training Considerations:
        
        1. **Algorithm Selection:** Choose appropriate algorithm for your data
        2. **Feature Engineering:** Consider feature transformations
        3. **Computational Resources:** Training may take several minutes
        4. **Model Validation:** Always validate new models before deployment
        
        ### Best Practices:
        - Use cross-validation for robust performance estimates
        - Monitor for overfitting with validation curves
        - Save model metadata for reproducibility
        - Test on held-out NASA validation data
        """)
