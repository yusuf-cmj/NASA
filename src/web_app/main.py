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
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dark mode
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    
    # Load models with loading indicator
    with st.spinner("🔄 Loading AI models..."):
        model, scaler, label_encoder, model_info = load_models()
    
    if model is None:
        st.error("❌ Failed to load models. Please check the model files.")
        st.stop()
    
    # Show model info in sidebar
    with st.sidebar.expander("🤖 Model Info"):
        # Get available models
        available_models = get_available_models()
        
        if available_models:
            # Model selection
            model_names = [f"{m['name']} ({m['accuracy']*100:.1f}%)" for m in available_models]
            selected_model_idx = st.selectbox(
                "Select Model:",
                range(len(model_names)),
                format_func=lambda x: model_names[x]
            )
            
            if selected_model_idx is not None:
                selected_model = available_models[selected_model_idx]
                
                # Load selected model
                if st.button("🔄 Load Selected Model"):
                    model, scaler, encoder, metadata = load_model_by_name(selected_model['filename'])
                    if model is not None:
                        st.session_state['active_model'] = selected_model['name']
                        st.session_state['loaded_model'] = model
                        st.session_state['loaded_scaler'] = scaler
                        st.session_state['loaded_encoder'] = encoder
                        st.success(f"✅ Loaded '{selected_model['name']}'")
                        st.rerun()
                
                # Show model details
                st.write(f"**Model:** {selected_model['name']}")
                st.write(f"**Algorithm:** Stacking Ensemble")
                st.write(f"**Accuracy:** {selected_model['accuracy']*100:.2f}%")
                st.write(f"**ROC-AUC:** {selected_model['roc_auc']:.4f}")
                st.write(f"**Created:** {selected_model['created_at'][:10]}")
                if selected_model['description']:
                    st.write(f"**Description:** {selected_model['description']}")
        else:
            if model_info:
                st.write(f"**Type:** {model_info['type']}")
                st.write(f"**Features:** {model_info['features']}")
                st.write(f"**Classes:** {model_info['classes']}")
                st.write("**Status:** ✅ Ready")
    
    # Sidebar navigation - URL Routing Sistemi
    st.sidebar.markdown("## 🚀 NASA Exoplanet Detection")
    
    # Get current page from URL parameters
    current_page = st.query_params.get("page", "home")
    
    # Navigation options with their corresponding page values
    nav_options = [
        ("🏠 Home", "home"),
        ("🔮 Single Prediction", "prediction"), 
        ("📁 Batch Upload", "batch_upload"),
        ("📊 Analytics Dashboard", "analytics"),
        ("🤖 Model Management", "models"),
        ("📊 Model Performance", "performance"),
        ("🎯 Model Training", "training"),
        ("ℹ️ About", "about")
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
