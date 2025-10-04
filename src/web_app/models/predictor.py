"""
Prediction functions for NASA Exoplanet Detection Web App
"""

import numpy as np
import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def predict_exoplanet(features, model, scaler, label_encoder):
    """Make prediction for a single exoplanet candidate"""
    try:
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        # Convert back to original labels
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probability) * 100
        
        return prediction_label, confidence, probability
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, None

def predict_batch(features_df, model, scaler, label_encoder):
    """Make predictions for a batch of exoplanet candidates"""
    try:
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
        
        # Convert back to original labels
        prediction_labels = label_encoder.inverse_transform(predictions)
        
        # Calculate confidence scores
        if probabilities is not None:
            confidences = np.max(probabilities, axis=1) * 100
        else:
            confidences = np.full(len(predictions), 50.0)
        
        return prediction_labels, confidences, probabilities
        
    except Exception as e:
        st.error(f"❌ Batch prediction error: {e}")
        return None, None, None

def get_prediction_explanation(prediction, confidence, features, feature_names):
    """Generate explanation for prediction"""
    explanation = f"The model predicts this candidate is **{prediction}** with {confidence:.1f}% confidence. "
    
    if confidence > 80:
        explanation += "This is a high-confidence prediction."
    elif confidence > 60:
        explanation += "This is a moderate-confidence prediction."
    else:
        explanation += "This is a low-confidence prediction. Consider additional validation."
    
    return explanation

def format_prediction_result(prediction, confidence):
    """Format prediction result for display"""
    if prediction == 'CONFIRMED':
        return f"✅ **CONFIRMED** (Confidence: {confidence:.1f}%)"
    else:
        return f"❌ **FALSE_POSITIVE** (Confidence: {confidence:.1f}%)"
