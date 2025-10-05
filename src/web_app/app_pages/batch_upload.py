"""
Batch upload page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processing import detect_nasa_format, get_auto_mapping, show_mapping_interface, validate_and_process_mapping, process_predictions
from utils.validators import validate_file_upload, validate_dataframe
from utils.ui_components import show_paginated_table

def highlight_predictions(row):
    """Highlight prediction rows with appropriate colors based on theme"""
    # Check if dark mode is enabled
    dark_mode = st.session_state.get('dark_mode', False)
    
    if row['prediction'] == 'CONFIRMED':
        if dark_mode:
            return ['background-color: #1DB954; color: #191414'] * len(row)
        else:
            return ['background-color: #bbf7d0; color: #166534'] * len(row)
    elif row['prediction'] == 'FALSE_POSITIVE':
        if dark_mode:
            return ['background-color: #e22134; color: #ffffff'] * len(row)
        else:
            return ['background-color: #fecaca; color: #991b1b'] * len(row)
    return [''] * len(row)

def _add_table_css():
    """Add CSS for table borders"""
    st.markdown("""
    <style>
    .dataframe {
        border-collapse: collapse !important;
    }
    .dataframe th, .dataframe td {
        border: 1px solid #d1d5db !important;
    }
    .dataframe th {
        background-color: #f9fafb !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

def _load_uploaded_file(uploaded_file):
    """Load and validate uploaded file"""
    # Validate file
    is_valid, message = validate_file_upload(uploaded_file)
    
    if not is_valid:
        st.error(f"{message}")
        return None
    
    try:
        # Load data with robust error handling
        if uploaded_file.name.endswith('.csv'):
            try:
                # Try UTF-8 first, skip bad lines
                df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', on_bad_lines='skip', 
                               low_memory=False, comment='#')
            except UnicodeDecodeError:
                # Try different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin-1', sep=',', on_bad_lines='skip', 
                                   low_memory=False, comment='#')
                except Exception:
                    df = pd.read_csv(uploaded_file, encoding='cp1252', sep=',', on_bad_lines='skip', 
                                   low_memory=False, comment='#')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"File loaded successfully! Shape: {df.shape}")
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("**Tips:**\n- Ensure CSV file is properly formatted\n- Check for special characters or encoding issues\n- Try saving as UTF-8 CSV format")
        return None

def _process_predictions_and_redirect(df, model, scaler, label_encoder):
    """Process predictions and redirect to results view"""
    # Detect format and get auto mapping
    format_type = detect_nasa_format(df)
    auto_mapping = get_auto_mapping(format_type)
    
    if not auto_mapping:
        st.warning("No mapping available for this format. Please check your data.")
        return
    
    st.markdown("## Column Mapping")
    
    # Show mapping interface (no disposition for batch prediction)
    final_mapping = show_mapping_interface(df, format_type, auto_mapping, include_disposition=False)
    
    if st.button("Process Predictions", type="primary"):
        # Validate mapping
        processed_df = validate_and_process_mapping(df, final_mapping)
        
        if processed_df is not None:
            # Process predictions
            results_df = process_predictions(processed_df, model, scaler, label_encoder)
            
            if results_df is not None:
                # Generate unique key for results
                import time
                results_key = f"batch_{int(time.time())}"
                
                # Save results to session state with unique key
                st.session_state[f'batch_results_{results_key}'] = results_df
                st.success("Predictions completed successfully!")
                
                # Redirect to results view with URL parameters
                st.query_params.show_results = "true"
                st.query_params.results_key = results_key
                st.rerun()

def _show_results_view(results_df, results_key):
    """Show results view with pagination and download options"""
    st.markdown("## Prediction Results")
    
    # Back to upload button
    if st.button("← Back to Upload", type="secondary"):
        st.query_params.show_results = "false"
        st.query_params.results_key = ""
        st.rerun()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", len(results_df))
    
    with col2:
        confirmed_count = (results_df['prediction'] == 'CONFIRMED').sum()
        st.metric("Confirmed", confirmed_count)
    
    with col3:
        false_positive_count = (results_df['prediction'] == 'FALSE_POSITIVE').sum()
        st.metric("False Positives", false_positive_count)
    
    with col4:
        avg_confidence = results_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Confidence analysis
    st.markdown("### Confidence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig_confidence = px.histogram(
            results_df, 
            x='confidence',
            title="Confidence Distribution",
            nbins=20,
            color_discrete_sequence=['#1f77b4']
        )
        fig_confidence.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig_confidence, use_container_width=True, key=f"confidence_hist_{results_key}")
    
    with col2:
        # Confidence by prediction type
        fig_box = px.box(
            results_df,
            x='prediction',
            y='confidence',
            title="Confidence by Prediction Type",
            color='prediction',
            color_discrete_map={'CONFIRMED': '#2E8B57', 'FALSE_POSITIVE': '#DC143C'}
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True, key=f"confidence_box_{results_key}")
    
    # Confidence statistics
    st.markdown("#### Confidence Statistics")
    confidence_stats = results_df.groupby('prediction')['confidence'].agg(['mean', 'std', 'min', 'max']).round(1)
    st.dataframe(confidence_stats)
    
    # Show paginated results
    st.markdown("### Detailed Results")
    paginated_results = show_paginated_table(results_df, page_size=50, key_prefix=f"batch_{results_key}")
    
    # Style the dataframe with prediction colors
    styled_df = paginated_results.style.apply(highlight_predictions, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"exoplanet_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"download_{results_key}"
    )
    
    # Clear results button
    if st.button("Clear Results", type="secondary", key=f"clear_{results_key}"):
        del st.session_state[f'batch_results_{results_key}']
        st.query_params.show_results = "false"
        st.query_params.results_key = ""
        st.rerun()

def batch_upload_page(model, scaler, label_encoder):
    """Batch upload and processing page"""
    
    # Add CSS for table borders
    _add_table_css()
    
    st.markdown("# Batch Upload & Processing")
    
    # Show active model indicator
    active_model = st.session_state.get('active_model', 'NebulaticAI')
    st.info(f"**Using Model:** {active_model} | [Switch Model](?page=models)")
    
    st.markdown("Upload CSV files for bulk exoplanet candidate analysis")
    st.markdown("---")
    
    # Check if we're showing results (URL parameter based)
    show_results = st.query_params.get("show_results", "false").lower() == "true"
    results_key = st.query_params.get("results_key", "")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload NASA exoplanet data for batch processing"
    )
    
    if uploaded_file is not None:
        # Load and validate file
        df = _load_uploaded_file(uploaded_file)
        
        if df is not None:
            # Show data preview
            st.markdown("## Data Preview")
            st.dataframe(df.head())
            
            # Process predictions
            _process_predictions_and_redirect(df, model, scaler, label_encoder)
    
    else:
        st.markdown("""
        ## Supported Formats
        
        ### NASA Raw Datasets:
        - **TESS TOI:** Raw TESS data with columns like `pl_orbper`, `st_teff`, etc.
        - **Kepler KOI:** Raw Kepler data with columns like `koi_period`, `koi_prad`, etc.
        - **K2 Candidates:** Raw K2 data with columns like `pl_name`, `disposition`, etc.
        
        ### Standard Format:
        - **Pre-processed data** with `orbital_period`, `transit_duration`, etc.
        
        ### Tips:
        - Upload CSV files with proper column headers
        - Ensure data is properly formatted
        - Missing values will be handled automatically
        """)
    
    # Show results if URL parameters indicate we should show them
    if show_results and results_key and f'batch_results_{results_key}' in st.session_state:
        results_df = st.session_state[f'batch_results_{results_key}']
        _show_results_view(results_df, results_key)
    
    # Show saved results if available (for pagination) - fallback for old results
    elif 'batch_results' in st.session_state and uploaded_file is None and not show_results:
        results_df = st.session_state['batch_results']
        
        st.markdown("## Previous Results")
        st.info("Showing results from previous upload. Upload a new file to process new data.")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candidates", len(results_df))
        
        with col2:
            confirmed_count = (results_df['prediction'] == 'CONFIRMED').sum()
            st.metric("Confirmed", confirmed_count)
        
        with col3:
            false_positive_count = (results_df['prediction'] == 'FALSE_POSITIVE').sum()
            st.metric("False Positives", false_positive_count)
        
        with col4:
            avg_confidence = results_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Show paginated results
        st.markdown("### Detailed Results")
        paginated_results = show_paginated_table(results_df, page_size=50, key_prefix="batch_saved")
        
        # Style the dataframe with prediction colors
        styled_df = paginated_results.style.apply(highlight_predictions, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"exoplanet_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear results button
        if st.button("Clear Results", type="secondary"):
            del st.session_state['batch_results']
            st.rerun()
