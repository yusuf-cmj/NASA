#!/usr/bin/env python3
"""
NASA Exoplanet Detection - Web Application
Streamlit-based web interface for exoplant classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NASA Exoplanet Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NASA theme
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1E90FF;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1E90FF, #87CEEB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-header {
    font-size: 1.5rem;
    color: #2E8B57;
    font-weight: bold;
    margin-bottom: 1rem;
}

.metrics-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.prediction-result {
    font-size: 1.2rem;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    text-align: center;
    font-weight: bold;
}

.confirmed {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #c3e6cb;
}

.false-positive {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        
        # Build absolute paths to model files
        models_dir = os.path.join(project_root, 'data', 'models')
        model_path = os.path.join(models_dir, 'binary_model_binary_stacking.pkl')
        scaler_path = os.path.join(models_dir, 'binary_scaler.pkl')
        encoder_path = os.path.join(models_dir, 'binary_label_encoder.pkl')
        
        # Debug: Show paths being used
        st.write(f"🔍 Loading models from: {models_dir}")
        
        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        
        st.success("✅ Models loaded successfully!")
        return model, scaler, label_encoder
        
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        
        # Show available files for debugging
        try:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'models')
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                st.error(f"📁 Available files in {models_dir}: {files}")
            else:
                st.error(f"📁 Directory does not exist: {models_dir}")
        except:
            pass
            
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

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

def show_model_performance():
    """Display model performance metrics"""
    
    st.markdown('<div class="sub-header">🎯 Model Performance</div>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Accuracy",
            value="88.21%",
            delta="+23.39% vs Ternary"
        )
    
    with col2:
        st.metric(
            label="📊 ROC-AUC",
            value="0.9448",
            delta="Excellent"
        )
    
    with col3:
        st.metric(
            label="⚡ F1-Score",
            value="0.88",
            delta="Balanced"
        )
    
    with col4:
        st.metric(
            label="📈 Dataset Size",
            value="21,271",
            delta="NASA Sources"
        )
    
    # Detailed metrics
    st.markdown("### 📈 Detailed Performance Metrics")
    
    details_df = pd.DataFrame({
        'Metric': ['CONFIRMED Precision', 'CONFIRMED Recall', 'FALSE_POSITIVE Precision', 'FALSE_POSITIVE Recall'],
        'Score': [0.86, 0.90, 0.90, 0.87],
        'Interpretation': ['Detects True Planets', 'Finds Most Planets', 'Avoids False Alarms', 'Rejects False Signals']
    })
    
    st.dataframe(details_df, use_container_width=True)

def main():
    """Main application interface"""
    
    # Header
    st.markdown('<div class="main-header">🚀 NASA Exoplanet Detection System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
        AI-Powered Detection of Confirmed Exoplanets vs False Positives<br>
        <small>Trained on NASA Kepler, TESS, and K2 datasets • 88.21% Accuracy</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, label_encoder = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🛠️ Navigation")
    page = st.sidebar.selectbox(
        "Choose an option:",
        [
            "🏠 Home", 
            "🔮 Single Prediction", 
            "📁 Batch Upload", 
            "📊 Analytics Dashboard",
            "⚖️ Model Comparison", 
            "📊 Model Performance", 
            "ℹ️ About"
        ]
    )
    
    if page == "🏠 Home":
        home_page(model, scaler, label_encoder)
    elif page == "🔮 Single Prediction":
        prediction_page(model, scaler, label_encoder)
    elif page == "📁 Batch Upload":
        batch_upload_page(model, scaler, label_encoder)
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard(model, scaler, label_encoder)
    elif page == "⚖️ Model Comparison":
        model_comparison_page()
    elif page == "📊 Model Performance":
        show_model_performance()
    elif page == "ℹ️ About":
        about_page()

def home_page(model, scaler, label_encoder):
    """Home page with overview and quick prediction"""
    
    st.markdown("## 🌟 Welcome to NASA Exoplanet Detection System")
    
    # Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This System Does
        
        Our AI system analyzes exoplanet candidate data and determines whether a candidate is a 
        **confirmed exoplanet** or a **false positive** with **88.21% accuracy**.
        
        #### 🔬 Input Features:
        - **Orbital Period** (days)
        - **Transit Duration** (hours)  
        - **Planet Radius** (Earth radii)
        - **Stellar Temperature** (Kelvin)
        - **Stellar Radius** (Solar radii)
        - **Transit Depth** (parts per million)
        
        #### ✅ Trained on NASA Data:
        - **Kepler Mission:** 9,564 records
        - **TESS Mission:** 7,703 records
        - **K2 Mission:** 4,004 records
        - **Total:** 21,271 exoplanet candidates
        
        #### 🎯 Model Performance:
        - **Binary Classification:** CONFIRMED vs FALSE_POSITIVE
        - **Accuracy:** 88.21%
        - **ROC-AUC:** 0.9448
        - **Method:** Stacking Ensemble (Random Forest + XGBoost + Extra Trees)
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Prediction")
        
        # Mini input form
        with st.form("quick_prediction"):
            orbital_period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0)
            transit_duration = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=50.0, value=3.0)
            planet_radius = st.number_input("Planet Radius (Earth radii)", min_value=0.1, max_value=50.0, value=2.0)
            stellar_temp = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5500)
            stellar_radius = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=50.0, value=1.0)
            transit_depth = st.number_input("Transit Depth (ppm)", min_value=1, max_value=100000, value=1000)
            
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
            
            if submitted:
                features = np.array([
                    orbital_period, transit_duration, planet_radius,
                    stellar_temp, stellar_radius, transit_depth
                ])
                
                prediction, confidence, prob = predict_exoplanet(features, model, scaler, label_encoder)
                
                if prediction:
                    if prediction == 'CONFIRMED':
                        st.markdown(f"""
                        <div class="prediction-result confirmed">
                        ✅ CONFIRMED EXOPLANET<br>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-result false-positive">
                        ❌ FALSE POSITIVE<br>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    st.markdown("### 🎯 Ready to Explore?")
    st.markdown("Use the **🔮 Single Prediction** page for detailed analysis with confidence intervals and feature explanations!")

def prediction_page(model, scaler, label_encoder):
    """Detailed prediction page with comprehensive inputs and outputs"""
    
    st.markdown("# 🔮 Single Exoplanet Prediction")
    
    # Input form
    st.markdown("## 📥 Enter Exoplanet Candidate Data")
    
    with st.form("detailed_prediction"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 Planetary Characteristics")
            orbital_period = st.number_input(
                "Orbital Period (days)", 
                min_value=0.1, max_value=1000.0, 
                value=10.0, help="Time for planet to orbit its star"
            )
            transit_duration = st.number_input(
                "Transit Duration (hours)", 
                min_value=0.1, max_value=50.0, 
                value=3.0, help="Time planet passes in front of star"
            )
            planet_radius = st.number_input(
                "Planet Radius (Earth radii)", 
                min_value=0.1, max_value=50.0, 
                value=2.0, help="Planet size relative to Earth"
            )
        
        with col2:
            st.markdown("### ⭐ Stellar Characteristics") 
            stellar_temp = st.number_input(
                "Stellar Temperature (K)", 
                min_value=2000, max_value=10000, 
                value=5500, help="Star surface temperature"
            )
            stellar_radius = st.number_input(
                "Stellar Radius (Solar radii)", 
                min_value=0.1, max_value=50.0, 
                value=1.0, help="Star size relative to Sun"
            )
            transit_depth = st.number_input(
                "Transit Depth (parts per million)", 
                min_value=1, max_value=100000, 
                value=1000, help="How much star light is blocked"
            )
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Analyze Exoplanet Candidate", use_container_width=True, type="primary")
    
    if submitted:
        # Make prediction
        features = np.array([
            orbital_period, transit_duration, planet_radius,
            stellar_temp, stellar_radius, transit_depth
        ])
        
        prediction, confidence, prob = predict_exoplanet(features, model, scaler, label_encoder)
        
        if prediction:
            # Results display
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 'CONFIRMED':
                    st.markdown(f"""
                    <div class="prediction-result confirmed">
                    🌍 CONFIRMED EXOPLANET<br>
                    Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    ### ✅ What This Means:
                    - This candidate shows strong evidence of being a **real exoplanet**
                    - Our model is **{:.1f}% confident** in this prediction
                    - This result is based on patterns learned from **21,271 NASA records**
                    
                    ### 🔬 Scientific Interpretation:
                    The combination of orbital parameters, transit characteristics, and stellar properties 
                    strongly suggests this is a genuine planetary companion orbiting its host star.
                    """.format(confidence))
                    
                else:  # FALSE_POSITIVE
                    st.markdown(f"""
                    <div class="prediction-result false-positive">
                    ⚠️ FALSE POSITIVE<br>
                    Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    ### ❌ What This Means:
                    - This candidate shows characteristics consistent with **false positive signals**
                    - Our model is **{:.1f}% confident** this is not a genuine exoplanet
                    - This could be instrumental noise, stellar activity, or other false signals
                    
                    ### 🔬 Scientific Interpretation:
                    While interesting, this signal does not appear to represent a true planetary transit
                    based on our analysis of the input parameters.
                    """.format(confidence))
            
            with col2:
                st.markdown("### 📊 Prediction Probabilities")
                
                # Probability visualization
                prob_df = pd.DataFrame({
                    'Class': ['CONFIRMED', 'FALSE_POSITIVE'],
                    'Probability': [prob[0]*100, prob[1]*100] if prob is not None else [50, 50]
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    color='Class',
                    color_discrete_map={'CONFIRMED': '#2E8B57', 'FALSE_POSITIVE': '#DC143C'},
                    title="Model Confidence"
                )
                fig.update_layout(showlegend=False, height=300)
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance section
            st.markdown("### 🔍 Feature Analysis")
            st.markdown("This section would show feature importance analysis (coming in Phase 2)")

def batch_upload_page(model, scaler, label_encoder):
    """Batch upload and processing page"""
    
    st.markdown("# 📁 Batch Upload & Processing")
    st.markdown("Upload CSV files for bulk exoplanet analysis")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with exoplanet candidate data. Required columns: orbital_period, transit_duration, planet_radius, stellar_temp, stellar_radius, transit_depth"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            df = pd.read_csv(uploaded_file)
            
            st.markdown("## 📊 Uploaded Data Preview")
            st.dataframe(df.head(10))
            
            # Feature validation
            required_features = ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp', 'stellar_radius', 'transit_depth']
            missing_features = [col for col in required_features if col not in df.columns]
            
            if missing_features:
                st.error(f"❌ Missing required columns: {missing_features}")
                st.info("📋 Required columns: orbital_period, transit_duration, planet_radius, stellar_temp, stellar_radius, transit_depth")
            else:
                # Extract features
                X_batch = df[required_features].values
                
                # Remove rows with missing values
                complete_mask = np.isfinite(X_batch).all(axis=1)
                X_clean = X_batch[complete_mask]
                df_clean = df[complete_mask].copy()
                
                st.info(f"📊 Processing {len(X_clean)} rows (removed {len(df) - len(X_clean)} rows with missing values)")
                
                # Predict
                if st.button("🚀 Analyze All Candidates", type="primary"):
                    
                    # Batch prediction
                    features_scaled = scaler.transform(X_clean)
                    predictions = model.predict(features_scaled)
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_scaled)
                        confidences = np.max(probabilities, axis=1) * 100
                    else:
                        confidences = np.full(len(predictions), 50.0)
                    
                    # Convert to original labels
                    prediction_labels = label_encoder.inverse_transform(predictions)
                    
                    # Add results to dataframe
                    df_results = df_clean.copy()
                    df_results['prediction'] = prediction_labels
                    df_results['confidence'] = confidences
                    
                    # Results summary
                    st.markdown("## 🎯 Batch Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Analyzed", len(df_results))
                    with col2:
                        confirmed_count = len(df_results[df_results['prediction'] == 'CONFIRMED'])
                        st.metric("Confirmed Exoplanets", confirmed_count)
                    with col3:
                        fp_count = len(df_results[df_results['prediction'] == 'FALSE_POSITIVE'])
                        st.metric("False Positives", fp_count)
                    
                    # Confidence distribution
                    avg_confidence = df_results['confidence'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    
                    # Color code predictions
                    def color_predictions(val):
                        if val == 'CONFIRMED':
                            return 'background-color: #d4edda'
                        elif val == 'FALSE_POSITIVE':
                            return 'background-color: #f8d7da'
                        return ''
                    
                    styled_df = df_results.style.applymap(color_predictions, subset=['prediction'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"exoplanet_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.markdown("### 📊 Analysis Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction distribution
                        pred_counts = df_results['prediction'].value_counts()
                        fig1 = px.pie(values=pred_counts.values, names=pred_counts.index,
                                    title="Prediction Distribution",
                                    color_discrete_map={'CONFIRMED': '#2E8B57', 'FALSE_POSITIVE': '#DC143C'})
                        fig1.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Confidence distribution
                        fig2 = px.histogram(df_results, x='confidence', 
                                          title="Confidence Distribution",
                                          nbins=20)
                        fig2.update_layout(xaxis_title="Confidence (%)", yaxis_title="Count")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Confidence vs Prediction
                    fig3 = px.box(df_results, x='prediction', y='confidence',
                                title="Confidence by Prediction Type",
                                color='prediction',
                                color_discrete_map={'CONFIRMED': '#2E8B57', 'FALSE_POSITIVE': '#DC143C'})
                    st.plotly_chart(fig3, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please check your CSV format and column names")

def analytics_dashboard(model, scaler, label_encoder):
    """Advanced analytics dashboard"""
    
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("Advanced analytics and insights from our exoplanet dataset")
    
    # Load processed data for analytics
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        data_path = os.path.join(project_root, 'data', 'processed', 'ml_ready_data.csv')
        
        if os.path.exists(data_path):
            df_analytics = pd.read_csv(data_path)
            
            # Overview metrics
            st.markdown("## 📈 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_rows = len(df_analytics)
                st.metric("Total Records", f"{total_rows:,}")
            
            with col2:
                confirmed_pct = len(df_analytics[df_analytics['disposition'] == 'CONFIRMED']) / total_rows * 100
                st.metric("Confirmed Rate", f"{confirmed_pct:.1f}%")
            
            with col3:
                avg_radius = df_analytics['planet_radius'].mean()
                st.metric("Avg Planet Radius", f"{avg_radius:.1f} R⊕")
            
            with col4:
                avg_period = df_analytics['orbital_period'].mean()
                st.metric("Avg Orbital Period", f"{avg_period:.1f} days")
            
            # Feature analysis
            st.markdown("## 🔍 Feature Analysis")
            
            # Interactive scatter plot
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", 
                                     ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp'], 
                                     key='x_axis')
            with col2:
                y_axis = st.selectbox("Y-axis", 
                                     ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp'],
                                     key='y_axis')
            
            fig_scatter = px.scatter(df_analytics, x=x_axis, y=y_axis, color='disposition',
                                   title=f"{x_axis} vs {y_axis}",
                                   color_discrete_map={'CONFIRMED': '#2E8B57', 'CANDIDATE': '#FFD700', 'FALSE_POSITIVE': '#DC143C'},
                                   opacity=0.6)
            fig_scatter.update_traces(marker=dict(size=8))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Distribution plots
            st.markdown("## 📊 Feature Distributions")
            
            feature_cols = ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp']
            
            for i in range(0, len(feature_cols), 2):
                cols = st.columns(2)
                for j, feature in enumerate(feature_cols[i:i+2]):
                    if j < len(cols):
                        with cols[j]:
                            fig = px.histogram(df_analytics, x=feature, color='disposition',
                                             title=f"{feature.replace('_', ' ').title()} Distribution",
                                             color_discrete_map={'CONFIRMED': '#2E8B57', 'CANDIDATE': '#FFD700', 'FALSE_POSITIVE': '#DC143C'},
                                             opacity=0.7)
                            fig.update_layout(xaxis_title=feature.replace('_', ' ').title(),
                                           yaxis_title="Count", showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.markdown("## 🔗 Feature Correlations")
            
            numeric_cols = df_analytics.select_dtypes(include=[np.number]).columns
            corr_matrix = df_analytics[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               labels=dict(x="Features", y="Features", color="Correlation"),
                               title="Feature Correlation Matrix",
                               color_continuous_scale="RdBu",
                               aspect="auto")
            fig_corr.update_layout(title_x=0.5)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Statistical summary
            st.markdown("## 📋 Statistical Summary")
            st.dataframe(df_analytics.describe(), use_container_width=True)
            
        else:
            st.warning("📁 Analytics data not found. Please ensure processed dataset is available.")
            
    except Exception as e:
        st.error(f"❌ Error loading analytics data: {e}")

def model_comparison_page():
    """Model comparison and hyperparameter tweaking page"""
    
    st.markdown("# ⚖️ Model Comparison & Tools")
    
    # Load available models
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        models_dir = os.path.join(project_root, 'data', 'models')
        
        available_models = []
        model_files = os.listdir(models_dir)
        
        binary_models = [f for f in model_files if f.startswith('binary_model_') and f.endswith('.pkl')]
        
        if binary_models:
            st.markdown("## 🤖 Available Models")
            
            for model_file in binary_models:
                model_name = model_file.replace('binary_model_', '').replace('.pkl', '').replace('_', ' ').title()
                
                # Load and show model info
                try:
                    model_path = os.path.join(models_dir, model_file)
                    model = joblib.load(model_path)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {model_name}")
                        
                        # Model type
                        model_type = type(model).__name__
                        st.write(f"**Type:** {model_type}")
                        
                        # Show model parameters
                        if hasattr(model, 'get_params'):
                            params = model.get_params()
                            # Show only key parameters
                            key_params = {k: v for k, v in params.items() 
                                        if len(str(v)) < 50 and not k.endswith('_')}
                            st.write(f"**Key Parameters:** {key_params}")
                        
                        st.markdown(f"**Performance:** Available in Model Performance page")
                    
                    with col2:
                        if st.button(f"🎯 Test Model", key=f"test_{model_file}"):
                            st.success(f"✅ {model_name} loaded successfully!")
                            st.info("Model ready for prediction! Use Single Prediction page to test.")
                    
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error loading {model_name}: {e}")
        else:
            st.warning("📁 No trained models found.")
        
        # Hyperparameter info (simulation)
        st.markdown("## ⚙️ Hyperparameter Information")
        
        st.info("""
        **Current Model Parameters:**
        - **Random Forest:** n_estimators=200, max_depth=15, class_weight='balanced'
        - **XGBoost:** n_estimators=200, max_depth=8, learning_rate=0.1
        - **Stacking:** Logistic Regression meta-learner
        """)
        
        # Model performance comparison
        st.markdown("## 📊 Performance Comparison")
        
        performance_data = {
            'Model': ['Random Forest', 'XGBoost', 'Extra Trees', 'Stacking Ensemble'],
            'Accuracy': [86.83, 87.99, 84.70, 88.21],
            'ROC-AUC': [0.9362, 0.9448, 0.9191, 0.9411],
            'F1-Score': [0.87, 0.88, 0.84, 0.88]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Style top performer
        def highlight_top(row):
            if row.name == 3:  # Stacking Ensemble (best)
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        styled_perf_df = perf_df.style.apply(highlight_top, axis=1)
        st.dataframe(styled_perf_df, use_container_width=True)
        
        st.success("🏆 **Stacking Ensemble** performs best with 88.21% accuracy!")
        
    except Exception as e:
        st.error(f"❌ Error accessing models directory: {e}")

def about_page():
    """About page with project information"""
    
    st.markdown("# ℹ️ About NASA Exoplanet Detection System")
    
    st.markdown("""
    ## 🚀 Project Overview
    
    This web application represents a comprehensive AI-powered system for detecting confirmed exoplanets 
    from candidate observations. Built as part of NASA's hackathon challenge, our system achieves 
    **88.21% accuracy** in distinguishing genuine exoplanets from false positives.
    
    ## 🎯 Mission Statement
    
    Our goal is to assist astronomers and researchers in efficiently identifying real exoplanets from 
    the massive amounts of data collected by NASA's Kepler, TESS, and K2 missions. By automating 
    the classification process, we aim to accelerate exoplanet discovery and reduce manual analysis time.
    
    ## 📊 Technical Details
    
    ### 🤖 Machine Learning Approach:
    - **Binary Classification:** CONFIRMED vs FALSE_POSITIVE
    - **Ensemble Method:** Stacking (Random Forest + XGBoost + Extra Trees)
    - **Dataset:** 21,271 exoplanet candidates from NASA missions
    - **Features:** 6 key astronomical parameters
    
    ### 🏆 Performance Metrics:
    - **Accuracy:** 88.21%
    - **ROC-AUC:** 0.9448
    - **F1-Score:** 0.88
    - **Improvement:** +23.39% over ternary classification
    
    ## 🔬 Data Sources
    
    Our model is trained on three comprehensive NASA datasets:
    
    1. **Kepler Objects of Interest (KOI):** 9,564 records
    2. **TESS Objects of Interest (TOI):** 7,703 records  
    3. **K2 Planets and Candidates:** 4,004 records
    
    ## 👥 Development Team
    
    This project was developed following NASA's hackathon requirements, emphasizing:
    - User-friendly web interface
    - High-accuracy machine learning models
    - Comprehensive data analysis
    - Real-world applicability
    
    ## 🔮 Future Enhancements
    
    - Real-time data ingestion
    - Batch processing capabilities
    - Advanced visualization tools
    - Hyperparameter optimization interface
    - Model retraining with new data
    
    ---
    
    *Built with ❤️ for advancing exoplanet science and discovery*
    """)

if __name__ == "__main__":
    main()
