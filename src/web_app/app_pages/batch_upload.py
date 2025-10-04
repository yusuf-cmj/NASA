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
            return ['background-color: #1e4d2b; color: #4ade80'] * len(row)
        else:
            return ['background-color: #bbf7d0; color: #166534'] * len(row)
    elif row['prediction'] == 'FALSE_POSITIVE':
        if dark_mode:
            return ['background-color: #4c1d1d; color: #f87171'] * len(row)
        else:
            return ['background-color: #fecaca; color: #991b1b'] * len(row)
    return [''] * len(row)

def batch_upload_page(model, scaler, label_encoder):
    """Batch upload and processing page"""
    
    # Add CSS for table borders
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
    
    st.markdown("# Batch Upload & Processing")
    st.markdown("Upload CSV files for bulk exoplanet candidate analysis")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload NASA exoplanet data for batch processing"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, message = validate_file_upload(uploaded_file)
        
        if not is_valid:
            st.error(f"{message}")
            return
        
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
                    except:
                        df = pd.read_csv(uploaded_file, encoding='cp1252', sep=',', on_bad_lines='skip', 
                                       low_memory=False, comment='#')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            st.markdown("## Data Preview")
            st.dataframe(df.head())
            
            # Detect format and get auto mapping
            format_type = detect_nasa_format(df)
            auto_mapping = get_auto_mapping(format_type)
            
            if auto_mapping:
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
                            # Save results to session state to prevent loss on pagination
                            st.session_state['batch_results'] = results_df
                            st.success("Predictions completed successfully!")
                            
                            # Show results
                            st.markdown("## Prediction Results")
                            
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
                                st.plotly_chart(fig_confidence, use_container_width=True, key="confidence_hist_new")
                            
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
                                st.plotly_chart(fig_box, use_container_width=True, key="confidence_box_new")
                            
                            # Confidence statistics
                            st.markdown("#### Confidence Statistics")
                            confidence_stats = results_df.groupby('prediction')['confidence'].agg(['mean', 'std', 'min', 'max']).round(1)
                            st.dataframe(confidence_stats)
                            
                            # Show paginated results
                            st.markdown("### Detailed Results")
                            paginated_results = show_paginated_table(results_df, page_size=50, key_prefix="batch_new")
                            
                            # Style the dataframe with prediction colors
                            styled_df = paginated_results.style.apply(highlight_predictions, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="exoplanet_predictions.csv",
                                mime="text/csv"
                            )
            else:
                st.warning("No mapping available for this format. Please check your data.")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("**Tips:**\n- Ensure CSV file is properly formatted\n- Check for special characters or encoding issues\n- Try saving as UTF-8 CSV format")
    
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
    
    # Show saved results if available (for pagination)
    if 'batch_results' in st.session_state:
        results_df = st.session_state['batch_results']
        
        st.markdown("## Prediction Results")
        
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
            st.plotly_chart(fig_confidence, use_container_width=True, key="confidence_hist_saved")
        
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
            st.plotly_chart(fig_box, use_container_width=True, key="confidence_box_saved")
        
        # Confidence statistics
        st.markdown("#### Confidence Statistics")
        confidence_stats = results_df.groupby('prediction')['confidence'].agg(['mean', 'std', 'min', 'max']).round(1)
        st.dataframe(confidence_stats)
        
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
