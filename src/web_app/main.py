"""
Main application file for NASA Exoplanet Detection Web App
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config.settings import PERFORMANCE_METRICS
from config.theme import get_theme_css, toggle_theme

# Import model management
from models.model_manager import load_models, get_available_models, load_model_by_name

# Import pages
from app_pages.home import home_page
from app_pages.prediction import prediction_page
from app_pages.batch_upload import batch_upload_page
from app_pages.analytics import analytics_dashboard
from app_pages.models import models_page
from app_pages.training import training_page
from app_pages.about import about_page

# Import utilities
from utils.ui_components import show_model_performance

def main():
    """Main application interface"""
    
    # Page configuration
    st.set_page_config(
        page_title="NASA Exoplanet Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dark mode
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    
    # Load models with loading indicator
    with st.spinner("Loading AI models..."):
        # Check if there's an active model in session state
        active_model_name = st.session_state.get('active_model', 'NebulaticAI')
        loaded_model = st.session_state.get('loaded_model')
        loaded_scaler = st.session_state.get('loaded_scaler')
        loaded_encoder = st.session_state.get('loaded_encoder')
        
        if loaded_model is not None and loaded_scaler is not None and loaded_encoder is not None:
            # Use loaded model from session state
            model, scaler, label_encoder = loaded_model, loaded_scaler, loaded_encoder
            model_info = {'name': active_model_name, 'type': 'Binary Classification', 'features': 6, 'classes': ['CONFIRMED', 'FALSE_POSITIVE']}
        else:
            # Load default model
            model, scaler, label_encoder, model_info = load_models()
            # Set default as active if not set
            if 'active_model' not in st.session_state:
                st.session_state['active_model'] = 'NebulaticAI'
                st.session_state['loaded_model'] = model
                st.session_state['loaded_scaler'] = scaler
                st.session_state['loaded_encoder'] = label_encoder
    
    if model is None:
        st.error("Failed to load models. Please check the model files.")
        st.stop()
    
    # Show model info in sidebar
    with st.sidebar.expander("Active Model"):
        st.write(f"**Current Model:** {active_model_name}")
        if model_info:
            st.write(f"**Type:** {model_info.get('type', 'Binary Classification')}")
            st.write(f"**Features:** {model_info.get('features', 6)}")
            st.write(f"**Classes:** {model_info.get('classes', ['CONFIRMED', 'FALSE_POSITIVE'])}")
            st.write("**Status:** Ready")
        
        # Quick model switch
        if st.button("Switch Model", help="Go to Model Management to switch models"):
            st.query_params.page = "models"
            st.rerun()
    
    # Sidebar navigation - URL Routing Sistemi
    st.sidebar.markdown("## NASA Exoplanet Detection")
    
    # Get current page from URL parameters
    current_page = st.query_params.get("page", "home")
    
    # Navigation options with their corresponding page values
    nav_options = [
        ("Home", "home"),
        ("Single Prediction", "prediction"), 
        ("Batch Upload", "batch_upload"),
        ("Analytics Dashboard", "analytics"),
        ("Model Management", "models"),
        ("Model Performance", "performance"),
        ("Model Training", "training"),
        ("About", "about")
    ]
    
    # Create navigation buttons
    for display_name, page_value in nav_options:
        # Highlight current page
        button_type = "primary" if page_value == current_page else "secondary"
        
        if st.sidebar.button(display_name, key=f"nav_{page_value}", use_container_width=True, type=button_type):
            st.query_params.page = page_value
            st.rerun()
    
    # Route to appropriate page based on URL parameter
    if current_page == "home":
        home_page(model, scaler, label_encoder)
    elif current_page == "prediction":
        prediction_page(model, scaler, label_encoder)
    elif current_page == "batch_upload":
        batch_upload_page(model, scaler, label_encoder)
    elif current_page == "analytics":
        analytics_dashboard(model, scaler, label_encoder)
    elif current_page == "models":
        models_page()
    elif current_page == "performance":
        show_model_performance()
    elif current_page == "training":
        training_page(model, scaler, label_encoder)
    elif current_page == "about":
        about_page()
    else:
        # Default to home if invalid page
        st.query_params.page = "home"
        st.rerun()

if __name__ == "__main__":
    main()
