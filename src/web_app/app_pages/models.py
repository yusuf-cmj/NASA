"""
Models page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_manager import get_available_models, get_model_info, load_model_by_name
from config.settings import PERFORMANCE_METRICS

def models_page():
    """Model management page"""
    
    st.markdown("# Model Management")
    st.markdown("Manage your trained models and switch between them")
    st.markdown("---")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.info("No custom models found. Train a new model to see it here.")
        return
    
    st.markdown(f"### Available Models ({len(available_models)})")
    
    # Show currently active model
    active_model = st.session_state.get('active_model', 'NebulaticAI')
    st.info(f"**Currently Active Model:** {active_model}")
    
    # Display models in cards
    for i, model in enumerate(available_models):
        # Show if this model is currently active
        is_active = model['name'] == active_model
        expander_title = f"{model['name']} - {model['accuracy']*100:.2f}% Accuracy"
        if is_active:
            expander_title += " ✅ (ACTIVE)"
        
        with st.expander(expander_title, expanded=True):
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.write(f"**Description:** {model['description'] or 'No description'}")
                st.write(f"**Created:** {model['created_at'][:19].replace('T', ' ')}")
                st.write(f"**ROC-AUC:** {model['roc_auc']:.4f}")
            
            with col2:
                if st.button(f"Use", key=f"use_{i}", type="primary", disabled=is_active):
                    model_obj, scaler, encoder, metadata = load_model_by_name(model['filename'])
                    if model_obj is not None:
                        st.session_state['active_model'] = model['name']
                        st.session_state['loaded_model'] = model_obj
                        st.session_state['loaded_scaler'] = scaler
                        st.session_state['loaded_encoder'] = encoder
                        st.success(f"Now using '{model['name']}' for predictions")
                        st.rerun()
                    else:
                        st.error("Failed to load model")
            
            with col3:
                if st.button(f"Delete", key=f"delete_{i}", type="secondary", disabled=is_active):
                    # Actually delete the model
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.join(current_dir, '..', '..')
                    models_dir = os.path.join(project_root, 'data', 'models')
                    
                    try:
                        model_filename = model['filename'].replace(' ', '_').replace('/', '_')
                        files_to_delete = [
                            f'{model_filename}_model.pkl',
                            f'{model_filename}_scaler.pkl',
                            f'{model_filename}_encoder.pkl',
                            f'{model_filename}_metadata.pkl'
                        ]
                        
                        deleted_count = 0
                        for file in files_to_delete:
                            file_path = os.path.join(models_dir, file)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"Deleted {deleted_count} files for '{model['name']}'")
                            # Clear session state if deleted model was active
                            if model['name'] == active_model:
                                st.session_state['active_model'] = 'NebulaticAI'
                                st.session_state['loaded_model'] = None
                                st.session_state['loaded_scaler'] = None
                                st.session_state['loaded_encoder'] = None
                            st.rerun()
                        else:
                            st.error("No files found to delete")
                            
                    except Exception as e:
                        st.error(f"Error deleting model: {e}")
    
    st.markdown("---")
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
