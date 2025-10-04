"""
Data Processing utilities for NASA Exoplanet Detection Web App
"""

import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import FEATURE_NAMES

def detect_nasa_format(df):
    """Detect NASA dataset format automatically"""
    
    # TESS format detection (more comprehensive)
    tess_indicators = ['pl_orbper', 'pl_trandurh', 'pl_rade', 'st_teff', 'st_rad', 'pl_trandep', 'pl_disposition']
    if any(col in df.columns for col in tess_indicators):
        # Additional TESS-specific checks
        if 'toi' in df.columns or 'tfopwg_disp' in df.columns or 'pl_disposition' in df.columns:
            return 'tess'
        # If it has TESS-like columns but no specific indicators, still likely TESS
        elif sum(1 for col in tess_indicators if col in df.columns) >= 4:
            return 'tess'
    
    # Kepler format detection
    elif 'koi_period' in df.columns or 'koi_disposition' in df.columns:
        return 'kepler'
    
    # K2 format detection (more flexible)
    elif any(col in df.columns for col in ['pl_name', 'disposition', 'epic_hostname', 'k2_name']):
        return 'k2'
    
    # Standard format detection
    elif 'orbital_period' in df.columns:
        return 'standard'
    
    # Unknown format
    else:
        return 'unknown'

def get_auto_mapping(format_type):
    """Get automatic column mapping for detected format"""
    
    mappings = {
        'tess': {
            'orbital_period': 'pl_orbper',
            'transit_duration': 'pl_trandurh',
            'planet_radius': 'pl_rade',
            'stellar_temp': 'st_teff',
            'stellar_radius': 'st_rad',
            'transit_depth': 'pl_trandep',
            'disposition': 'tfopwg_disp'  # Primary TESS disposition column
        },
        'kepler': {
            'orbital_period': 'koi_period',
            'transit_duration': 'koi_duration',
            'planet_radius': 'koi_prad',
            'stellar_temp': 'koi_steff',
            'stellar_radius': 'koi_srad',
            'transit_depth': 'koi_depth',
            'disposition': 'koi_disposition'  # Primary Kepler disposition column
        },
        'k2': {
            'orbital_period': 'pl_orbper',
            'transit_duration': 'pl_trandur',
            'planet_radius': 'pl_rade',
            'stellar_temp': 'st_teff',
            'stellar_radius': 'st_rad',
            'transit_depth': 'pl_trandep',
            'disposition': 'pl_disposition'  # Primary K2 disposition column
        },
        'standard': {
            'orbital_period': 'orbital_period',
            'transit_duration': 'transit_duration',
            'planet_radius': 'planet_radius',
            'stellar_temp': 'stellar_temp',
            'stellar_radius': 'stellar_radius',
            'transit_depth': 'transit_depth',
            'disposition': 'disposition'
        }
    }
    
    return mappings.get(format_type, {})

def find_disposition_column(df, format_type):
    """Find disposition column in NASA datasets with multiple possible names"""
    
    # Define possible disposition column names for each format
    disposition_candidates = {
        'tess': ['tfopwg_disp', 'pl_disposition', 'disposition', 'Disposition', 'DISPOSITION', 
                'tfopwg_disposition', 'pl_status'],
        'kepler': ['koi_disposition', 'disposition', 'Disposition', 'DISPOSITION',
                  'koi_status', 'status'],
        'k2': ['pl_disposition', 'disposition', 'Disposition', 'DISPOSITION',
              'pl_status', 'status', 'k2_disposition'],
        'standard': ['disposition', 'Disposition', 'DISPOSITION', 'label', 'Label', 'LABEL']
    }
    
    candidates = disposition_candidates.get(format_type, ['disposition', 'Disposition', 'DISPOSITION'])
    
    # Find the first matching column
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    return None

def show_mapping_interface(df, format_type, auto_mapping, include_disposition=False):
    """Show interactive column mapping interface"""
    
    # Show detected format
    st.info(f"📋 Detected format: **{format_type.upper()}**")
    
    # Show available columns
    st.write("**Available columns in your data:**")
    st.write(list(df.columns))
    
    # Create mapping interface
    mapping = {}
    
    for feature in FEATURE_NAMES:
        feature_label = feature.replace('_', ' ').title()
        
        # Get default mapping
        default_col = auto_mapping.get(feature, '')
        
        # Create selectbox
        available_cols = [''] + list(df.columns)
        selected_col = st.selectbox(
            f"Map to **{feature_label}**:",
            available_cols,
            index=available_cols.index(default_col) if default_col in available_cols else 0,
            key=f"mapping_{feature}"
        )
        
        if selected_col:
            mapping[feature] = selected_col
    
    # Disposition mapping (only if requested)
    if include_disposition:
        disposition_col = st.selectbox(
            "Map to **Disposition** (optional):",
            [''] + list(df.columns),
            key="mapping_disposition"
        )
        
        if disposition_col:
            mapping['disposition'] = disposition_col
    
    return mapping

def validate_and_process_mapping(df, final_mapping):
    """Validate mapping and create processed dataframe"""
    
    try:
        # Check if all required features are mapped
        required_features = FEATURE_NAMES
        missing_features = [f for f in required_features if f not in final_mapping]
        
        if missing_features:
            st.error(f"❌ Missing required features: {', '.join(missing_features)}")
            return None
        
        # Create processed dataframe
        processed_df = pd.DataFrame()
        
        for feature, column in final_mapping.items():
            if feature in required_features and column in df.columns:
                processed_df[feature] = df[column]
            elif feature == 'disposition' and column in df.columns:
                processed_df[feature] = df[column]
        
        # Check for missing values
        missing_count = processed_df.isnull().sum().sum()
        if missing_count > 0:
            st.warning(f"⚠️ Found {missing_count} missing values. These will be handled during processing.")
        
        # Show preview
        st.success("✅ Mapping validated successfully!")
        st.write("**Preview of mapped data:**")
        st.dataframe(processed_df.head())
        
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error processing mapping: {e}")
        return None

def process_predictions(mapped_df, model, scaler, label_encoder):
    """Process predictions for mapped dataframe using fast batch processing"""
    
    try:
        # Prepare features
        feature_cols = FEATURE_NAMES
        features_df = mapped_df[feature_cols].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        # Make predictions
        from models.predictor import predict_batch
        predictions, confidences, probabilities = predict_batch(features_df, model, scaler, label_encoder)
        
        if predictions is None:
            return None
        
        # Create results dataframe
        results_df = mapped_df.copy()
        results_df['prediction'] = predictions
        results_df['confidence'] = confidences
        
        # Add probability columns if available
        if probabilities is not None:
            results_df['prob_confirmed'] = probabilities[:, 0]  # Assuming 0 = CONFIRMED
            results_df['prob_false_positive'] = probabilities[:, 1]  # Assuming 1 = FALSE_POSITIVE
        
        return results_df
        
    except Exception as e:
        st.error(f"❌ Error processing predictions: {e}")
        return None
