"""
Model Management for NASA Exoplanet Detection Web App
"""

import os
import joblib
import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_DIR, MODEL_FILES

def get_available_models():
    """Get list of available models"""
    models = []
    
    # Add default NebulaticAI model first
    models.append({
        'filename': 'default_nebulaticai',
        'name': 'NebulaticAI',
        'description': 'Default NASA Exoplanet Detection Model - Binary Stacking Ensemble',
        'created_at': '2024-01-01T00:00:00',
        'accuracy': 0.8821,
        'roc_auc': 0.9448,
        'is_default': True
    })
    
    # Add custom trained models
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if file.endswith('_metadata.pkl'):
                try:
                    metadata = joblib.load(os.path.join(MODEL_DIR, file))
                    models.append({
                        'filename': file.replace('_metadata.pkl', ''),
                        'name': metadata.get('name', file.replace('_metadata.pkl', '')),
                        'description': metadata.get('description', ''),
                        'created_at': metadata.get('created_at', ''),
                        'accuracy': metadata.get('accuracy', 0),
                        'roc_auc': metadata.get('roc_auc', 0),
                        'is_default': False
                    })
                except:
                    continue
    
    # Sort by creation date (newest first), but keep default first
    models.sort(key=lambda x: (not x.get('is_default', False), x['created_at']), reverse=True)
    return models

def load_models():
    """Load trained models and preprocessing objects with enhanced caching"""
    try:
        # Define paths
        model_path = os.path.join(MODEL_DIR, MODEL_FILES['model'])
        scaler_path = os.path.join(MODEL_DIR, MODEL_FILES['scaler'])
        encoder_path = os.path.join(MODEL_DIR, MODEL_FILES['encoder'])
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
            st.error("❌ Model files not found. Please run training first.")
            return None, None, None, None
        
        # Load with spinner
        with st.spinner("🔄 Loading models..."):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
        
        # Model info
        model_info = {
            'name': 'NebulaticAI',
            'type': 'Binary Classification',
            'features': 6,
            'classes': ['CONFIRMED', 'FALSE_POSITIVE'],
            'accuracy': 0.8821,
            'roc_auc': 0.9448,
            'description': 'Default NASA Exoplanet Detection Model - Binary Stacking Ensemble'
        }
        
        return model, scaler, label_encoder, model_info
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

def load_model_by_name(model_name):
    """Load model by name"""
    # Handle default NebulaticAI model
    if model_name == 'default_nebulaticai':
        return load_models()
    
    model_filename = model_name.replace(' ', '_').replace('/', '_')
    
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_scaler.pkl'))
        encoder = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_encoder.pkl'))
        metadata = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_metadata.pkl'))
        
        return model, scaler, encoder, metadata
    except:
        return None, None, None, None

def save_model(model, scaler, encoder, metadata, model_name):
    """Save model with metadata"""
    model_filename = model_name.replace(' ', '_').replace('/', '_')
    
    try:
        joblib.dump(model, os.path.join(MODEL_DIR, f'{model_filename}_model.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'{model_filename}_scaler.pkl'))
        joblib.dump(encoder, os.path.join(MODEL_DIR, f'{model_filename}_encoder.pkl'))
        joblib.dump(metadata, os.path.join(MODEL_DIR, f'{model_filename}_metadata.pkl'))
        return True
    except Exception as e:
        st.error(f"❌ Error saving model: {e}")
        return False

def get_model_info(model_name):
    """Get model information"""
    models = get_available_models()
    for model in models:
        if model['filename'] == model_name:
            return model
    return None
