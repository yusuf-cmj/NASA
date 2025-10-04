"""
Validation utilities for NASA Exoplanet Detection Web App
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
from config.settings import FEATURE_NAMES, FEATURE_LABELS

def validate_feature_input(feature_name, value):
    """Validate individual feature input"""
    
    if pd.isna(value) or value is None:
        return False, f"{FEATURE_LABELS.get(feature_name, feature_name)} cannot be empty"
    
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except ValueError:
            return False, f"{FEATURE_LABELS.get(feature_name, feature_name)} must be a number"
    
    # Feature-specific validation
    if feature_name == 'orbital_period' and (value <= 0 or value > 10000):
        return False, "Orbital period must be between 0 and 10,000 days"
    
    elif feature_name == 'transit_duration' and (value <= 0 or value > 100):
        return False, "Transit duration must be between 0 and 100 hours"
    
    elif feature_name == 'planet_radius' and (value <= 0 or value > 50):
        return False, "Planet radius must be between 0 and 50 Earth radii"
    
    elif feature_name == 'stellar_temperature' and (value <= 0 or value > 100000):
        return False, "Stellar temperature must be between 0 and 100,000 Kelvin"
    
    elif feature_name == 'stellar_radius' and (value <= 0 or value > 100):
        return False, "Stellar radius must be between 0 and 100 Solar radii"
    
    elif feature_name == 'transit_depth' and (value <= 0 or value > 1000000):
        return False, "Transit depth must be between 0 and 1,000,000 ppm"
    
    return True, "Valid"

def validate_dataframe(df, required_features=None):
    """Validate entire dataframe"""
    
    if required_features is None:
        required_features = FEATURE_NAMES
    
    errors = []
    warnings = []
    
    # Check if all required features are present
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        errors.append(f"Missing required features: {', '.join(missing_features)}")
    
    # Check for empty dataframe
    if df.empty:
        errors.append("Dataframe is empty")
        return False, errors, warnings
    
    # Check each feature
    for feature in required_features:
        if feature in df.columns:
            # Check for missing values
            missing_count = df[feature].isnull().sum()
            if missing_count > 0:
                warnings.append(f"{feature}: {missing_count} missing values")
            
            # Check for invalid values
            invalid_mask = ~df[feature].apply(lambda x: validate_feature_input(feature, x)[0])
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                errors.append(f"{feature}: {invalid_count} invalid values")
    
    return len(errors) == 0, errors, warnings

def validate_file_upload(uploaded_file):
    """Validate uploaded file"""
    
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    if not uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
        return False, "File must be CSV or Excel format"
    
    # Check file size (limit to 50MB)
    if uploaded_file.size > 50 * 1024 * 1024:
        return False, "File size must be less than 50MB"
    
    return True, "Valid file"

def validate_model_files():
    """Validate that required model files exist"""
    
    import os
    import os
    from config.settings import MODEL_DIR, MODEL_FILES
    
    missing_files = []
    
    for file_type, filename in MODEL_FILES.items():
        file_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {filename}")
    
    if missing_files:
        return False, f"Missing model files: {', '.join(missing_files)}"
    
    return True, "All model files present"

def show_validation_results(is_valid, errors, warnings):
    """Show validation results in UI"""
    
    if not is_valid:
        st.error("❌ Validation Failed:")
        for error in errors:
            st.error(f"• {error}")
    
    if warnings:
        st.warning("⚠️ Warnings:")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    if is_valid and not warnings:
        st.success("✅ Validation passed successfully!")

def sanitize_input(value):
    """Sanitize user input"""
    
    if isinstance(value, str):
        # Remove extra whitespace
        value = value.strip()
        
        # Handle common input formats
        if value.lower() in ['', 'none', 'null', 'nan']:
            return None
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            return value
    
    return value
