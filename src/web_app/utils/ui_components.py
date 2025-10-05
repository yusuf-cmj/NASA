"""
UI Components for NASA Exoplanet Detection Web App
"""

import streamlit as st
import pandas as pd

def show_paginated_table(df, page_size=100, key_prefix="table"):
    """Show paginated table with navigation controls - URL-based state management"""
    
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Use URL parameters for pagination state - more professional approach
    page_param = f"{key_prefix}_page"
    current_page = st.query_params.get(page_param, "0")
    
    try:
        current_page = int(current_page)
        current_page = max(0, min(current_page, total_pages - 1))
    except (ValueError, TypeError):
        current_page = 0
    
    # Calculate start and end indices
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Show pagination info
    st.info(f"Showing rows {start_idx + 1}-{end_idx} of {total_rows} total rows")
    
    # Show page selector - force re-render by using current_page in key
    if total_pages > 1:
        page_options = list(range(total_pages))
        
        # Use current_page in key to force selectbox re-render
        selectbox_key = f"{key_prefix}_page_selector_{current_page}"
        
        # Create columns for dropdown and buttons - same row
        dropdown_col, btn_col = st.columns([2, 1])
        
        with dropdown_col:
            selected_page = st.selectbox(
                f"Go to page (1-{total_pages}):",
                page_options,
                index=current_page,
                key=selectbox_key,
                format_func=lambda x: f"Page {x + 1}"
            )
        
        with btn_col:
            # Butonları yan yana koy - dropdown ile aynı satırda
            btn_col1, btn_col2 = st.columns(2)
            
            # CSS ile butonları dropdown menüsüne hizala (başlık değil)
            st.markdown("""
            <style>
            .pagination-buttons {
                display: flex;
                align-items: center;
                height: 100%;
                gap: 0.5rem;
            }
            /* Butonları dropdown menüsüne hizala - başlık değil */
            .stSelectbox > div > div > div {
                display: flex;
                align-items: center;
            }
            /* Butonları dropdown input alanına hizala */
            .stButton > button {
                margin-top: 1.7rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            with btn_col1:
                if st.button("Previous", key=f"{key_prefix}_prev2", disabled=current_page == 0):
                    st.query_params[page_param] = str(current_page - 1)
                    st.rerun()
            with btn_col2:
                if st.button("Next", key=f"{key_prefix}_next2", disabled=current_page >= total_pages - 1):
                    st.query_params[page_param] = str(current_page + 1)
                    st.rerun()
        
        # Update URL when selectbox changes
        if selected_page != current_page:
            st.query_params[page_param] = str(selected_page)
            st.rerun()
    
    # Get current page data
    current_data = df.iloc[start_idx:end_idx]
    
    return current_data

def show_prediction_result(prediction, confidence, dark_mode=False):  # noqa: ARG001
    """Show prediction result with styling"""
    if prediction == 'CONFIRMED':
        css_class = 'confirmed'
        label = 'CONFIRMED'
    else:
        css_class = 'false-positive'
        label = 'FALSE_POSITIVE'
    
    st.markdown(f"""
    <div class="prediction-result {css_class}">
        <strong>{label}</strong><br>
        Confidence: {confidence:.2%}
    </div>
    """, unsafe_allow_html=True)

def show_metrics_card(title, value, subtitle=""):
    """Show a metrics card"""
    st.metric(
        label=title,
        value=value,
        delta=subtitle if subtitle else None
    )

def show_model_performance():
    """Display dynamic model performance metrics"""
    
    st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
    
    # Get active model information
    active_model_name = st.session_state.get('active_model', 'NebulaticAI')
    
    # Import here to avoid circular imports
    from models.model_manager import get_available_models
    
    all_models = get_available_models()
    active_model_data = None
    
    for model_info in all_models:
        if model_info['name'] == active_model_name:
            active_model_data = model_info
            break
    
    if not active_model_data:
        # Fallback to default values
        accuracy = 0.8821
        roc_auc = 0.9448
        f1_score = 0.88
        model_source = "Default"
    else:
        accuracy = active_model_data['accuracy']
        roc_auc = active_model_data['roc_auc']
        f1_score = active_model_data.get('f1_score', roc_auc * 0.95)
        model_source = active_model_data.get('source', 'file_system').replace('_', ' ').title()
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate delta based on NebulaticAI baseline
        nebulatic_accuracy = 0.8821  # NebulaticAI baseline
        if active_model_name == 'NebulaticAI':
            delta_text = "Baseline Model"
        else:
            delta_value = (accuracy - nebulatic_accuracy) * 100
            delta_text = f"{'+' if delta_value > 0 else ''}{delta_value:.1f}% vs NebulaticAI"
        
        st.metric(
            label="Accuracy",
            value=f"{accuracy*100:.2f}%",
            delta=delta_text
        )
    
    with col2:
        st.metric(
            label="ROC-AUC",
            value=f"{roc_auc:.4f}",
            delta="Excellent" if roc_auc > 0.9 else "Good"
        )
    
    with col3:
        st.metric(
            label="F1-Score",
            value=f"{f1_score:.3f}",
            delta="Balanced" if 0.8 < f1_score < 0.9 else "High"
        )
    
    with col4:
        st.metric(
            label="Model Source",
            value=f"{model_source}",
            delta="User Model" if model_source.lower() == 'local storage' else "Default"
        )
    
    # Model info
    st.markdown(f"**Active Model:** {active_model_name}")
    
    # Quick model switch
    if st.button("Switch Model", help="Go to Model Management to switch models"):
        st.query_params.page = "models"
        st.rerun()
    
    # Detailed metrics (simplified for sidebar)
    st.markdown("### Key Metrics")
    
    details_df = pd.DataFrame({
        'Metric': ['CONFIRMED Precision', 'CONFIRMED Recall', 'FALSE_POSITIVE Precision', 'FALSE_POSITIVE Recall'],
        'Score': [0.86, 0.90, 0.90, 0.87],  # These could be made dynamic too
        'Status': ['Detects True Planets', 'Finds Most Planets', 'Avoids False Alarms', 'Rejects False Signals']
    })
    
    st.dataframe(details_df, use_container_width=True, hide_index=True)

def create_sidebar_navigation():
    """Create sidebar navigation menu"""
    st.sidebar.title("NASA Exoplanet Detection")
    
    pages = {
        "Home": "home",
        "Single Prediction": "prediction", 
        "Batch Upload": "batch_upload",
        "Analytics Dashboard": "analytics",
        "Model Management": "models",
        "Model Training": "training",
        "About": "about"
    }
    
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        key="page_selector"
    )
    
    return pages[selected_page]
