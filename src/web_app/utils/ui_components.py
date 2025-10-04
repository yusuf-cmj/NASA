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
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info(f"📊 Showing rows {start_idx + 1}-{end_idx} of {total_rows} total rows")
    
    with col2:
        if st.button("⬅️ Previous", key=f"{key_prefix}_prev", disabled=current_page == 0):
            st.query_params[page_param] = str(current_page - 1)
            st.rerun()
    
    with col3:
        if st.button("Next ➡️", key=f"{key_prefix}_next", disabled=current_page >= total_pages - 1):
            st.query_params[page_param] = str(current_page + 1)
            st.rerun()
    
    # Show page selector
    if total_pages > 1:
        page_options = list(range(total_pages))
        selected_page = st.selectbox(
            f"Go to page (1-{total_pages}):",
            page_options,
            index=current_page,
            key=f"{key_prefix}_page_selector",
            format_func=lambda x: f"Page {x + 1}"
        )
        
        if selected_page != current_page:
            st.query_params[page_param] = str(selected_page)
            st.rerun()
    
    # Get current page data
    current_data = df.iloc[start_idx:end_idx]
    
    return current_data

def show_prediction_result(prediction, confidence, dark_mode=False):
    """Show prediction result with styling"""
    if prediction == 'CONFIRMED':
        css_class = 'confirmed'
        icon = '✅'
        label = 'CONFIRMED'
    else:
        css_class = 'false-positive'
        icon = '❌'
        label = 'FALSE_POSITIVE'
    
    st.markdown(f"""
    <div class="prediction-result {css_class}">
        {icon} <strong>{label}</strong><br>
        Confidence: {confidence:.2%}
    </div>
    """, unsafe_allow_html=True)

def show_metrics_card(title, value, subtitle="", icon="📊"):
    """Show a metrics card"""
    st.metric(
        label=f"{icon} {title}",
        value=value,
        delta=subtitle if subtitle else None
    )

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

def create_sidebar_navigation():
    """Create sidebar navigation menu"""
    st.sidebar.title("🚀 NASA Exoplanet Detection")
    
    pages = {
        "🏠 Home": "home",
        "🔮 Single Prediction": "prediction", 
        "📁 Batch Upload": "batch_upload",
        "📊 Analytics Dashboard": "analytics",
        "🤖 Model Management": "models",
        "🎯 Model Training": "training",
        "ℹ️ About": "about"
    }
    
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        key="page_selector"
    )
    
    return pages[selected_page]
