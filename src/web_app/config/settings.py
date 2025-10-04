"""
Configuration settings for NASA Exoplanet Detection Web App
"""

import os

# Page Configuration
PAGE_CONFIG = {
    'page_title': "NASA Exoplanet Detection",
    'page_icon': "🚀",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Model Configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'data', 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'data')

# Available Models
AVAILABLE_MODELS = [
    'binary_model_binary_stacking.pkl',
    'binary_model_xgboost.pkl', 
    'binary_model_random_forest.pkl',
    'binary_model_extra_trees.pkl'
]

# Model Files
MODEL_FILES = {
    'model': 'binary_model_binary_stacking.pkl',
    'scaler': 'binary_scaler.pkl',
    'encoder': 'binary_label_encoder.pkl'
}

# Feature Names
FEATURE_NAMES = [
    'orbital_period',
    'transit_duration', 
    'planet_radius',
    'stellar_temp',
    'stellar_radius',
    'transit_depth'
]

# Feature Labels for UI
FEATURE_LABELS = {
    'orbital_period': 'Orbital Period (days)',
    'transit_duration': 'Transit Duration (hours)',
    'planet_radius': 'Planet Radius (Earth radii)',
    'stellar_temperature': 'Stellar Temperature (Kelvin)',
    'stellar_radius': 'Stellar Radius (Solar radii)',
    'transit_depth': 'Transit Depth (parts per million)'
}

# Prediction Labels
PREDICTION_LABELS = {
    'CONFIRMED': '✅ CONFIRMED',
    'FALSE_POSITIVE': '❌ FALSE_POSITIVE'
}

# Performance Metrics
PERFORMANCE_METRICS = {
    'accuracy': 88.21,
    'roc_auc': 0.9448,
    'f1_score': 0.88,
    'precision': 0.89,
    'recall': 0.87
}

# Dataset Info
DATASET_INFO = {
    'total_records': 21271,
    'kepler_records': 9564,
    'tess_records': 7703,
    'k2_records': 4004
}

# UI Settings
UI_SETTINGS = {
    'page_size': 100,
    'chart_height': 400,
    'sidebar_width': 300
}
