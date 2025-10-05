"""
Model training page for NASA Exoplanet Detection Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PERFORMANCE_METRICS
from models.model_manager import save_model

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
            'disposition': 'pl_disposition'
        },
        'kepler': {
            'orbital_period': 'koi_period',
            'transit_duration': 'koi_duration',
            'planet_radius': 'koi_prad',
            'stellar_temp': 'koi_steff',
            'stellar_radius': 'koi_srad',
            'transit_depth': 'koi_depth',
            'disposition': 'koi_disposition'
        },
        'k2': {
            'orbital_period': 'pl_orbper',
            'transit_duration': 'pl_trandur',
            'planet_radius': 'pl_rade',
            'stellar_temp': 'st_teff',
            'stellar_radius': 'st_rad',
            'transit_depth': 'pl_trandep',
            'disposition': 'pl_disposition'
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
    
    # Common disposition column names for each format
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

def show_mapping_interface(df, format_type, auto_mapping):
    """Show interactive column mapping interface"""
    
    st.markdown("### Column Mapping Configuration")
    
    # Format detection result
    if format_type != 'unknown':
        st.success(f"**{format_type.upper()}** format detected!")
    else:
        st.warning("**Unknown format** - Manual mapping required")
    
    # Available columns
    available_columns = [''] + df.columns.tolist()
    
    # Required columns
    required_columns = [
        'orbital_period', 'transit_duration', 'planet_radius',
        'stellar_temp', 'stellar_radius', 'transit_depth'
    ]
    
    # Add disposition column if found
    disposition_col = find_disposition_column(df, format_type)
    if disposition_col:
        required_columns.append('disposition')
        # Update auto_mapping with found disposition column
        auto_mapping['disposition'] = disposition_col
        st.info(f"**Disposition column found:** `{disposition_col}`")
    else:
        st.warning("**Disposition column not found!** You may need to add labels manually.")
    
    # Mapping interface
    final_mapping = {}
    mapping_data = []
    
    for req_col in required_columns:
        # Get auto mapping if available
        auto_selected = auto_mapping.get(req_col, '')
        
        # Create selectbox
        selected_col = st.selectbox(
            f"**{req_col.replace('_', ' ').title()}**",
            available_columns,
            index=available_columns.index(auto_selected) if auto_selected in available_columns else 0,
            key=f"map_{req_col}",
            help=f"Select which column contains {req_col.replace('_', ' ')} data"
        )
        
        final_mapping[req_col] = selected_col
        
        # Status indicator
        if selected_col:
            if auto_selected == selected_col and auto_selected:
                status = "Auto"
            else:
                status = "Manual"
        else:
            status = "Missing"
        
        # Sample value
        sample_value = 'N/A'
        if selected_col and selected_col in df.columns:
            try:
                sample_value = f"{df[selected_col].iloc[0]:.2f}" if pd.api.types.is_numeric_dtype(df[selected_col]) else str(df[selected_col].iloc[0])
            except:
                sample_value = str(df[selected_col].iloc[0])
        
        mapping_data.append({
            'Required Column': req_col.replace('_', ' ').title(),
            'Mapped To': selected_col if selected_col else 'Not mapped',
            'Status': status,
            'Sample Value': sample_value
        })
    
    # Show mapping preview table
    st.markdown("#### Mapping Preview:")
    mapping_df = pd.DataFrame(mapping_data)
    st.dataframe(mapping_df, use_container_width=True)
    
    return final_mapping

def validate_and_process_mapping(df, final_mapping):
    """Validate mapping and create processed dataframe"""
    
    # Check missing columns
    missing_columns = [col for col, mapped_col in final_mapping.items() if not mapped_col]
    
    if missing_columns:
        st.error(f"**Missing mappings:** {', '.join(missing_columns)}")
        st.markdown("Please map all required columns before processing.")
        return None
    
    # Create mapped dataframe
    mapped_df = df.copy()
    for req_col, source_col in final_mapping.items():
        if source_col and source_col in df.columns:
            mapped_df[req_col] = df[source_col]
    
    # Show success message
    st.success("**All columns mapped successfully!**")
    
    # Show preview of mapped data
    st.markdown("#### Mapped Data Preview:")
    preview_cols = ['orbital_period', 'transit_duration', 'planet_radius', 
                   'stellar_temp', 'stellar_radius', 'transit_depth']
    st.dataframe(mapped_df[preview_cols].head(), use_container_width=True)
    
    return mapped_df

def training_page(model, scaler, label_encoder):
    """Model training page"""
    
    st.markdown("# Model Training & Optimization")
    st.markdown("Train models and optimize hyperparameters for improved performance")
    st.markdown("---")
    
    st.markdown("## Train Existing Model")
        
    st.markdown("""
    ### Train with New Data
        
    Upload new NASA exoplanet data to train the existing model:""")
        
    uploaded_data = st.file_uploader(
            "Upload new training data (CSV format)",
            type=['csv'],
        help="New NASA exoplanet data for model training. Supports NASA raw datasets (Kepler, TESS, K2) or standard format."
        )
        
    if uploaded_data is not None:
            try:
                # Use robust CSV reading like batch upload
                try:
                    df = pd.read_csv(uploaded_data, encoding='utf-8', sep=',', on_bad_lines='skip', 
                                   low_memory=False, comment='#')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(uploaded_data, encoding='latin-1', sep=',', on_bad_lines='skip', 
                                       low_memory=False, comment='#')
                    except:
                        df = pd.read_csv(uploaded_data, encoding='cp1252', sep=',', on_bad_lines='skip', 
                                       low_memory=False, comment='#')
                
                st.success(f"**File loaded successfully!** {len(df)} rows, {len(df.columns)} columns")
                
                # Show data preview
                st.markdown("### Uploaded Data Preview")
                st.dataframe(df.head())
                
                # Show column info
                with st.expander("Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': [str(df[col].dtype) for col in df.columns],
                        'Non-Null Count': [df[col].count() for col in df.columns],
                        'Null Count': [df[col].isnull().sum() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                # Smart mapping like batch upload
                format_type = detect_nasa_format(df)
                auto_mapping = get_auto_mapping(format_type)
                
                # Show mapping interface
                final_mapping = show_mapping_interface(df, format_type, auto_mapping)
                
                # Auto-process mapping
                mapped_df = validate_and_process_mapping(df, final_mapping)
                
                if mapped_df is not None:
                    # Check for disposition column
                    if 'disposition' not in mapped_df.columns:
                        st.warning("**No disposition column found!**")
                        st.markdown("""
                        **For training, you need labeled data with a 'disposition' column.**
                            
                            **Options:**
                            1. **Add disposition column** to your CSV with values: 'CONFIRMED' or 'FALSE_POSITIVE'
                            2. **Use Batch Upload** for unlabeled data prediction
                            """)
                    else:
                        # Show disposition analysis
                        st.markdown("#### Disposition Analysis")
                        disposition_counts = mapped_df['disposition'].value_counts()
                        st.write("**Disposition Value Counts:**")
                        st.dataframe(disposition_counts.reset_index().rename(columns={'index': 'Value', 'disposition': 'Count'}))
                        
                    # Check for classification compatibility
                        unique_values = mapped_df['disposition'].unique()
                        
                        # Handle TESS disposition values
                        tess_confirmed_values = ['CP', 'KP']  # Confirmed Planet, Kepler Planet
                        tess_candidate_values = ['PC', 'APC']  # Planetary Candidate, Additional Planetary Candidate
                        tess_false_values = ['FP', 'FA']  # False Positive, False Alarm
                        
                        has_confirmed = any(val in unique_values for val in tess_confirmed_values)
                        has_candidates = any(val in unique_values for val in tess_candidate_values)
                        has_false = any(val in unique_values for val in tess_false_values)
                        
                        if has_confirmed or has_candidates or has_false:
                            # Show detected categories
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if has_confirmed:
                                    confirmed_vals = [val for val in tess_confirmed_values if val in unique_values]
                                st.success(f"**Confirmed:** {confirmed_vals}")
                            
                            with col2:
                                if has_candidates:
                                    candidate_vals = [val for val in tess_candidate_values if val in unique_values]
                                st.warning(f"**Candidates:** {candidate_vals}")
                            
                            with col3:
                                if has_false:
                                    false_vals = [val for val in tess_false_values if val in unique_values]
                                st.error(f"**False:** {false_vals}")
                            
                            st.markdown("""
                        **For Classification:**
                        - **Confirmed** (`CP`, `KP`) → `CONFIRMED`
                            - **Candidates** (`PC`, `APC`) → **REMOVED** (not used for training)
                        - **False** (`FP`, `FA`) → `FALSE_POSITIVE`
                            
                            **Note:** Only confirmed planets and false positives will be used for training.
                            """)
                            
                    # NASA data combination option
                    use_nasa_data = st.checkbox(
                        "Combine with NASA dataset",
                        value=True,
                        help="Combine with NASA's 21,271 exoplanet records to create a more reliable model. If unchecked, training will use only your uploaded data."
                    )
                    
                    # Show previous results if available
                    if 'preparation_results' in st.session_state and st.session_state.get('data_prepared', False):
                        st.success("**Data Prepared Successfully!**")
                        
                        # Results summary
                        st.markdown("#### Preparation Summary")
                        
                        results = st.session_state['preparation_results']
                        
                        # Main processing metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Records", f"{results['original_count']:,}")
                        with col2:
                            st.metric("Removed Candidates", f"{results['removed_count']:,}")
                        with col3:
                            st.metric("Final Records", f"{results['final_count']:,}")
                        
                        st.markdown("---")
                        
                        # Classification breakdown
                        st.markdown("#### Classification Breakdown")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confirmed Records", f"{results['confirmed_count']:,}")
                        with col2:
                            st.metric("False Positive Records", f"{results['false_positive_count']:,}")
                        with col3:
                            st.metric("Confirmed Rate", f"{results['confirmed_rate']:.1f}%")
                        
                        st.info("**Ready for Training**")
                    else:
                            # Auto-mapping option
                        if st.button("Prepare Data", type="primary"):
                            # Create containers for visual feedback
                            progress_container = st.container()
                            status_container = st.container()
                            results_container = st.container()
                            
                            with progress_container:
                                st.markdown("#### Data Preparation")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                    
                                # Step indicators
                                step_container = st.container()
                                with step_container:
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        step1_status = st.empty()
                                    with col2:
                                        step2_status = st.empty()
                                    with col3:
                                        step3_status = st.empty()
                                    with col4:
                                        step4_status = st.empty()
                                    
                                    # Step 1: Filter data
                                step1_status.markdown("**Step 1:** Filtering data...")
                                status_text.text("Analyzing disposition values...")
                                progress_bar.progress(10)
                                import time
                                time.sleep(0.5)
                                    
                                keep_values = tess_confirmed_values + tess_false_values
                                original_count = len(mapped_df)
                                mapped_df = mapped_df[mapped_df['disposition'].isin(keep_values)]
                                removed_count = original_count - len(mapped_df)
                                
                                step1_status.markdown("**Step 1:** Completed")
                                status_text.text("Filtering confirmed and false positive records...")
                                progress_bar.progress(25)
                                time.sleep(0.5)
                                    
                                    # Step 2: Map values
                                step2_status.markdown("**Step 2:** Mapping values...")
                                status_text.text("Mapping TESS disposition values...")
                                progress_bar.progress(40)
                                time.sleep(0.5)
                                
                                mapping_dict = {}
                                for val in tess_confirmed_values:
                                    if val in unique_values:
                                        mapping_dict[val] = 'CONFIRMED'
                                for val in tess_false_values:
                                    if val in unique_values:
                                        mapping_dict[val] = 'FALSE_POSITIVE'
                                
                                mapped_df['disposition'] = mapped_df['disposition'].map(mapping_dict)
                                    
                                step2_status.markdown("**Step 2:** Completed")
                                status_text.text("Converting to classification format...")
                                progress_bar.progress(60)
                                time.sleep(0.5)
                                
                                # Step 3: Validate
                                step3_status.markdown("**Step 3:** Validating...")
                                status_text.text("Validating data...")
                                progress_bar.progress(80)
                                time.sleep(0.5)
                                
                                step3_status.markdown("**Step 3:** Completed")
                                    
                                    # Step 4: Complete
                                step4_status.markdown("**Step 4:** Finalizing...")
                                status_text.text("Finalizing data preparation...")
                                progress_bar.progress(100)
                                time.sleep(0.5)
                                
                                step4_status.markdown("**Step 4:** Completed")
                                status_text.text("Data preparation ready!")
                                
                                # Keep progress indicators visible - do not clear
                                
                                # Show results
                                with results_container:
                                    st.success("**Data Prepared Successfully!**")
                                    
                                    # Results summary
                                    st.markdown("#### Preparation Summary")
                                    
                                    # Main processing metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Original Records", f"{original_count:,}")
                                    with col2:
                                        st.metric("Removed Candidates", f"{removed_count:,}")
                                    with col3:
                                        st.metric("Final Records", f"{len(mapped_df):,}")
                                    
                                    st.markdown("---")
                                    
                                    # Classification breakdown
                                    confirmed_count = (mapped_df['disposition'] == 'CONFIRMED').sum()
                                    false_count = (mapped_df['disposition'] == 'FALSE_POSITIVE').sum()
                                    confirmed_rate = (confirmed_count / len(mapped_df)) * 100 if len(mapped_df) > 0 else 0
                                    
                                    # Store results in session state for persistence
                                    st.session_state['preparation_results'] = {
                                        'original_count': original_count,
                                        'removed_count': removed_count,
                                        'final_count': len(mapped_df),
                                        'confirmed_count': confirmed_count,
                                        'false_positive_count': false_count,
                                        'confirmed_rate': confirmed_rate
                                    }
                                    
                                    st.markdown("#### Classification Breakdown")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Confirmed Records", f"{confirmed_count:,}")
                                    with col2:
                                        st.metric("False Positive Records", f"{false_count:,}")
                                    with col3:
                                        st.metric("Confirmed Rate", f"{confirmed_rate:.1f}%")
                                    
                                    st.info("**Ready for Training**")
                                    
                                    # Mark as prepared
                                    st.session_state['data_prepared'] = True
                                    st.session_state['prepared_data'] = mapped_df
                                    st.session_state['use_nasa_data'] = use_nasa_data
                                    
                                    # Don't rerun - keep progress visible
                        
                        elif 'CONFIRMED' in unique_values and 'FALSE_POSITIVE' in unique_values:
                            st.success("**Perfect!** Data already has CONFIRMED and FALSE_POSITIVE values.")
                        else:
                            st.warning("**Unknown disposition values.** Please check the data format.")
                    
                    # Model Configuration
                    st.markdown("### Model Configuration")
                    
                    # Model name and training method
                    col1, col2 = st.columns([2, 1])
                        
                    with col1:
                            # Generate default model name with timestamp
                        default_name = f"trained_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
                        new_model_name = st.text_input(
                            "Model Name:",
                            value=st.session_state.get('model_name', default_name),
                            help="Enter a unique name for your trained model"
                        )
                        
                        with col2:
                            training_method = st.selectbox(
                                "Training Method:",
                                ["Binary Stacking", "Random Forest", "XGBoost", "Extra Trees"],
                                index=["Binary Stacking", "Random Forest", "XGBoost", "Extra Trees"].index(st.session_state.get('training_method', 'Binary Stacking')),
                                help="Choose the machine learning algorithm for training"
                        )
                        
                        # Model description
                        model_description = st.text_area(
                            "Model Description (Optional):",
                        value=st.session_state.get('model_description', ''),
                        placeholder="Describe what makes this trained model special...",
                        help="Optional description for your trained model"
                    )
                    
                    # Update session state when values change
                    st.session_state['model_name'] = new_model_name
                    st.session_state['training_method'] = training_method
                    st.session_state['model_description'] = model_description
                    
                    # Use session state values for display and validation
                    display_model_name = st.session_state.get('model_name', new_model_name)
                    display_training_method = st.session_state.get('training_method', training_method)
                    display_model_description = st.session_state.get('model_description', model_description)
                    
                    # Hyperparameter Tweaking Section
                    st.markdown("### Hyperparameter Tweaking")
                    st.markdown("Fine-tune model parameters for optimal performance:")
                    
                    # Dynamic hyperparameters based on selected method
                    if training_method == "Random Forest":
                        st.markdown("#### Random Forest Parameters")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            rf_n_estimators = st.slider(
                                "Number of Trees",
                                min_value=50,
                                max_value=500,
                                value=st.session_state.get('rf_n_estimators', 200),
                                help="More trees = better performance but slower training"
                            )
                            rf_max_depth = st.slider(
                                "Max Depth",
                                min_value=5,
                                max_value=30,
                                value=st.session_state.get('rf_max_depth', 15),
                                help="Maximum depth of each tree"
                            )
                        
                        with col2:
                            rf_min_samples_split = st.slider(
                                "Min Samples Split",
                                min_value=2,
                                max_value=10,
                                value=st.session_state.get('rf_min_samples_split', 3),
                                help="Minimum samples required to split a node"
                            )
                            rf_min_samples_leaf = st.slider(
                                "Min Samples Leaf",
                                min_value=1,
                                max_value=5,
                                value=st.session_state.get('rf_min_samples_leaf', 1),
                                help="Minimum samples required in a leaf node"
                            )
                        
                        # Store RF parameters
                        hyperparams = {
                            'n_estimators': rf_n_estimators,
                            'max_depth': rf_max_depth,
                            'min_samples_split': rf_min_samples_split,
                            'min_samples_leaf': rf_min_samples_leaf
                        }
                    
                    elif training_method == "XGBoost":
                        st.markdown("#### XGBoost Parameters")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            xgb_n_estimators = st.slider(
                                "Boosting Rounds",
                                min_value=50,
                                max_value=500,
                                value=st.session_state.get('xgb_n_estimators', 200),
                                help="Number of boosting rounds"
                            )
                            xgb_max_depth = st.slider(
                                "Max Depth",
                                min_value=3,
                                max_value=15,
                                value=st.session_state.get('xgb_max_depth', 8),
                                help="Maximum depth of each tree"
                            )
                        
                        with col2:
                            xgb_learning_rate = st.slider(
                                "Learning Rate",
                                min_value=0.01,
                                max_value=0.3,
                                value=st.session_state.get('xgb_learning_rate', 0.1),
                                step=0.01,
                                help="Step size shrinkage to prevent overfitting"
                            )
                            xgb_subsample = st.slider(
                                "Subsample",
                                min_value=0.6,
                                max_value=1.0,
                                value=st.session_state.get('xgb_subsample', 0.9),
                                step=0.05,
                                help="Fraction of samples used for training"
                            )
                        
                        # Store XGB parameters
                        hyperparams = {
                            'n_estimators': xgb_n_estimators,
                            'max_depth': xgb_max_depth,
                            'learning_rate': xgb_learning_rate,
                            'subsample': xgb_subsample
                        }
                    
                    elif training_method == "Extra Trees":
                        st.markdown("#### Extra Trees Parameters")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            et_n_estimators = st.slider(
                                "Number of Trees",
                                min_value=50,
                                max_value=500,
                                value=st.session_state.get('et_n_estimators', 200),
                                help="More trees = better performance but slower training"
                            )
                            et_max_depth = st.slider(
                                "Max Depth",
                                min_value=5,
                                max_value=30,
                                value=st.session_state.get('et_max_depth', 15),
                                help="Maximum depth of each tree"
                            )
                        
                        with col2:
                            et_min_samples_split = st.slider(
                                "Min Samples Split",
                                min_value=2,
                                max_value=10,
                                value=st.session_state.get('et_min_samples_split', 3),
                                help="Minimum samples required to split a node"
                            )
                            et_min_samples_leaf = st.slider(
                                "Min Samples Leaf",
                                min_value=1,
                                max_value=5,
                                value=st.session_state.get('et_min_samples_leaf', 1),
                                help="Minimum samples required in a leaf node"
                            )
                        
                        # Store ET parameters
                        hyperparams = {
                            'n_estimators': et_n_estimators,
                            'max_depth': et_max_depth,
                            'min_samples_split': et_min_samples_split,
                            'min_samples_leaf': et_min_samples_leaf
                        }
                    
                    elif training_method == "Binary Stacking":
                        st.markdown("#### Binary Stacking Parameters")
                        st.info("Stacking uses multiple base models. Configure individual models below:")
                        
                        # Random Forest for Stacking
                        st.markdown("**Random Forest Base Model:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            stack_rf_n_estimators = st.slider(
                                "RF Trees",
                                min_value=50,
                                max_value=300,
                                value=st.session_state.get('stack_rf_n_estimators', 200),
                                help="Number of Random Forest trees"
                            )
                        with col2:
                            stack_rf_max_depth = st.slider(
                                "RF Max Depth",
                                min_value=5,
                                max_value=25,
                                value=st.session_state.get('stack_rf_max_depth', 15),
                                help="Random Forest max depth"
                            )
                        
                        # XGBoost for Stacking
                        st.markdown("**XGBoost Base Model:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            stack_xgb_n_estimators = st.slider(
                                "XGB Trees",
                                min_value=50,
                                max_value=300,
                                value=st.session_state.get('stack_xgb_n_estimators', 200),
                                help="Number of XGBoost trees"
                            )
                        with col2:
                            stack_xgb_max_depth = st.slider(
                                "XGB Max Depth",
                                min_value=3,
                                max_value=12,
                                value=st.session_state.get('stack_xgb_max_depth', 8),
                                help="XGBoost max depth"
                            )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            stack_xgb_learning_rate = st.slider(
                                "XGB Learning Rate",
                                min_value=0.01,
                                max_value=0.2,
                                value=st.session_state.get('stack_xgb_learning_rate', 0.1),
                                step=0.01,
                                help="XGBoost learning rate"
                            )
                        with col2:
                            # Empty column for balance
                            pass
                        
                        # Extra Trees for Stacking
                        st.markdown("**Extra Trees Base Model:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            stack_et_n_estimators = st.slider(
                                "ET Trees",
                                min_value=50,
                                max_value=300,
                                value=st.session_state.get('stack_et_n_estimators', 200),
                                help="Number of Extra Trees"
                            )
                        with col2:
                            stack_et_max_depth = st.slider(
                                "ET Max Depth",
                                min_value=5,
                                max_value=25,
                                value=st.session_state.get('stack_et_max_depth', 15),
                                help="Extra Trees max depth"
                            )
                        
                        # Store Stacking parameters
                        hyperparams = {
                            'rf': {
                                'n_estimators': stack_rf_n_estimators,
                                'max_depth': stack_rf_max_depth
                            },
                            'xgb': {
                                'n_estimators': stack_xgb_n_estimators,
                                'max_depth': stack_xgb_max_depth,
                                'learning_rate': stack_xgb_learning_rate
                            },
                            'et': {
                                'n_estimators': stack_et_n_estimators,
                                'max_depth': stack_et_max_depth
                            }
                        }
                    
                    # Update session state with hyperparameters
                    st.session_state['hyperparameters'] = hyperparams
                    
                    # Validate model name using session state
                    if display_model_name:
                            # Check for invalid characters
                            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
                            has_invalid = any(char in display_model_name for char in invalid_chars)
                            
                            if has_invalid:
                                st.error(f"**Invalid characters in model name!** Please avoid: {', '.join(invalid_chars)}")
                            elif len(display_model_name.strip()) == 0:
                                st.error("**Model name cannot be empty!**")
                            else:
                                st.success(f"**Model name valid:** `{display_model_name}`")
                        
                    # Training button
                    data_prepared = st.session_state.get('data_prepared', False)
                    model_name_valid = display_model_name and not has_invalid and len(display_model_name.strip()) > 0
                        
                    if not data_prepared:
                        st.warning("**Please prepare data first!**")
                        
                    if not model_name_valid:
                        st.warning("**Please enter a valid model name!**")
                    
                    if st.button("Start Training", type="primary", disabled=not (data_prepared and model_name_valid)):
                        # Get model configuration from session state
                        final_model_name = st.session_state['model_name']
                        final_model_description = st.session_state['model_description']
                        final_training_method = display_training_method
                        
                        with st.spinner("Training model..."):
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                try:
                                    # Step 1: Load NASA dataset
                                    status_text.text("Loading NASA dataset...")
                                    progress_bar.progress(10)

                                    current_dir = os.path.dirname(os.path.abspath(__file__))
                                    project_root = os.path.join(current_dir, '..', '..', '..')
                                    nasa_data_path = os.path.join(project_root, 'data', 'processed', 'ml_ready_data.csv')

                                    if not os.path.exists(nasa_data_path):
                                        st.error("NASA dataset not found!")
                                        return

                                    nasa_data = pd.read_csv(nasa_data_path)

                                    # Step 2: Use prepared data and filter NASA data for classification
                                    status_text.text("Preparing data...")
                                    progress_bar.progress(20)

                                    prepared_data = st.session_state.get('prepared_data', mapped_df)
                                    use_nasa_data = st.session_state.get('use_nasa_data', True)

                                    # Combine datasets based on checkbox selection
                                    if use_nasa_data:
                                        # Filter NASA data to remove CANDIDATE entries for classification
                                        nasa_binary_mask = nasa_data['disposition'].isin(['CONFIRMED', 'FALSE_POSITIVE'])
                                        nasa_binary_data = nasa_data[nasa_binary_mask]

                                        # Combine datasets
                                        combined_data = pd.concat([nasa_binary_data, prepared_data], ignore_index=True)
                                    else:
                                        # Use only user data
                                        combined_data = prepared_data

                                    if len(prepared_data) < 100:
                                        st.warning("Training with small dataset may be risky. NASA data is recommended.")
                                    st.info(f"Using only user data: {len(combined_data):,} records")

                                    # Step 3: Preprocessing
                                    status_text.text("Preprocessing data...")
                                    progress_bar.progress(30)

                                    # Feature extraction
                                    required_features = [
                                        'orbital_period', 'transit_duration', 'planet_radius',
                                        'stellar_temp', 'stellar_radius', 'transit_depth'
                                    ]

                                    X = combined_data[required_features].copy()
                                    y = combined_data['disposition'].copy()

                                    # Handle missing values
                                    X = X.fillna(X.median())

                                    # Remove outliers (IQR method)
                                    outlier_removed = 0
                                    for col in X.columns:
                                        Q1 = X[col].quantile(0.25)
                                        Q3 = X[col].quantile(0.75)
                                        IQR = Q3 - Q1
                                        mask = (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
                                        before_count = len(X)
                                        X = X[mask]
                                        y = y[mask]
                                        outlier_removed += before_count - len(X)

                                    # Step 4: Encoding and Scaling
                                    status_text.text("Encoding and scaling features...")
                                    progress_bar.progress(40)

                                    from sklearn.preprocessing import StandardScaler, LabelEncoder
                                    from sklearn.model_selection import train_test_split
                                    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
                                    from sklearn.linear_model import LogisticRegression
                                    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
                                    import xgboost as xgb

                                    # Label encoding for classification
                                    label_encoder_new = LabelEncoder()
                                    y_encoded = label_encoder_new.fit_transform(y)

                                    # Feature scaling
                                    scaler_new = StandardScaler()
                                    X_scaled = scaler_new.fit_transform(X)

                                    # Step 5: Train-test split
                                    status_text.text("Splitting data...")
                                    progress_bar.progress(50)

                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                                    )

                                    # Get training method from session state
                                    training_method = st.session_state.get('training_method', 'Binary Stacking')

                                    # Step 6-9: Train selected model
                                    if training_method == "Binary Stacking":
                                        # Train all base models first
                                        status_text.text("Training base models for stacking...")
                                        progress_bar.progress(60)

                                        rf_model = RandomForestClassifier(
                                            n_estimators=200,
                                            max_depth=15,
                                            min_samples_split=3,
                                            min_samples_leaf=1,
                                            random_state=42,
                                            n_jobs=-1,
                                            class_weight='balanced'
                                        )
                                        rf_model.fit(X_train, y_train)

                                        xgb_model = xgb.XGBClassifier(
                                            n_estimators=200,
                                            max_depth=8,
                                            learning_rate=0.1,
                                            subsample=0.9,
                                            colsample_bytree=0.9,
                                            random_state=42,
                                            eval_metric='logloss',
                                            scale_pos_weight=1
                                        )
                                        xgb_model.fit(X_train, y_train)

                                        et_model = ExtraTreesClassifier(
                                            n_estimators=200,
                                            max_depth=15,
                                            min_samples_split=3,
                                            min_samples_leaf=1,
                                            random_state=42,
                                            n_jobs=-1,
                                            class_weight='balanced'
                                        )
                                        et_model.fit(X_train, y_train)

                                        # Train stacking ensemble
                                        status_text.text("Training Binary Stacking Ensemble...")
                                        progress_bar.progress(90)

                                        final_model = StackingClassifier(
                                            estimators=[
                                                ('rf', rf_model),
                                                ('xgb', xgb_model),
                                                ('et', et_model)
                                            ],
                                            final_estimator=LogisticRegression(
                                                random_state=42,
                                                max_iter=1000,
                                                class_weight='balanced'
                                            ),
                                            cv=5,
                                            n_jobs=-1
                                        )
                                        final_model.fit(X_train, y_train)

                                    elif training_method == "Random Forest":
                                        status_text.text("Training Random Forest...")
                                        progress_bar.progress(90)

                                        final_model = RandomForestClassifier(
                                            n_estimators=200,
                                            max_depth=15,
                                            min_samples_split=3,
                                            min_samples_leaf=1,
                                            random_state=42,
                                            n_jobs=-1,
                                            class_weight='balanced'
                                        )
                                        final_model.fit(X_train, y_train)

                                    elif training_method == "XGBoost":
                                        status_text.text("Training XGBoost...")
                                        progress_bar.progress(90)

                                        final_model = xgb.XGBClassifier(
                                            n_estimators=200,
                                            max_depth=8,
                                            learning_rate=0.1,
                                            subsample=0.9,
                                            colsample_bytree=0.9,
                                            random_state=42,
                                            eval_metric='logloss',
                                            scale_pos_weight=1
                                        )
                                        final_model.fit(X_train, y_train)

                                    elif training_method == "Extra Trees":
                                        status_text.text("Training Extra Trees...")
                                        progress_bar.progress(90)

                                        final_model = ExtraTreesClassifier(
                                            n_estimators=200,
                                            max_depth=15,
                                            min_samples_split=3,
                                            min_samples_leaf=1,
                                            random_state=42,
                                            n_jobs=-1,
                                            class_weight='balanced'
                                        )
                                        final_model.fit(X_train, y_train)

                                    # Step 10: Evaluation
                                    status_text.text("Evaluating models...")
                                    progress_bar.progress(95)

                                    # Evaluate selected model
                                    final_pred = final_model.predict(X_test)
                                    final_pred_proba = final_model.predict_proba(X_test)[:, 1]
                                    accuracy_new = accuracy_score(y_test, final_pred)
                                    roc_auc_new = roc_auc_score(y_test, final_pred_proba)

                                    # Step 11: Save model
                                    status_text.text("Saving trained model...")
                                    progress_bar.progress(98)

                                    # Create metadata
                                    metadata = {
                                        'name': final_model_name,
                                        'description': final_model_description or f"Trained model using {training_method}",
                                        'created_at': datetime.now().isoformat(),
                                        'accuracy': accuracy_new,
                                        'roc_auc': roc_auc_new,
                                        'training_data_size': len(combined_data),
                                        'user_data_size': len(prepared_data),
                                        'base_model': training_method,
                                        'features': required_features,
                                        'classes': ['CONFIRMED', 'FALSE_POSITIVE']
                                    }

                                    # Save model using model_manager
                                    success = save_model(final_model, scaler_new, label_encoder_new, metadata, final_model_name)

                                    if success:
                                        # Step 12: Complete
                                        progress_bar.progress(100)
                                        status_text.text("Training completed!")

                                        # Show results
                                        st.success("**Model trained and saved successfully!**")

                                        # Show training results
                                        st.markdown("#### Training Results")

                                        if final_model_description:
                                            st.info(f"**Description:** {final_model_description}")

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric("New Accuracy", f"{accuracy_new*100:.2f}%")

                                        with col2:
                                            st.metric("ROC-AUC", f"{roc_auc_new:.4f}")

                                        with col3:
                                            st.metric("Training Data", f"{len(combined_data):,}", delta=f"+{len(prepared_data):,} New Records")

                                        # Update session state
                                        st.session_state['training_completed'] = True
                                        st.session_state['new_model_name'] = final_model_name

                                        st.success(f"**Model '{final_model_name}' is now available in the Model Management section!**")

                                    else:
                                        st.error("**Failed to save model!** Please check file permissions.")

                                except Exception as e:
                                    st.error(f"**Training failed:** {str(e)}")
                                    st.markdown("Please check your data and try again.")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.markdown("Please check your CSV format and try again.")
    
    st.markdown("---")
    
    # Show training notes
    st.markdown("## Training Notes")
        
    st.markdown("""
    ### Training Considerations:
        
        1. **Data Quality:** Ensure new data is properly cleaned and validated
        2. **Feature Consistency:** New data must have the same 6 required features
        3. **Computational Resources:** Training may take several minutes
    4. **Model Validation:** Always validate trained models before deployment
        
        ### Best Practices:
        - Use cross-validation for robust performance estimates
        - Monitor for overfitting with validation curves
        - Save model metadata for reproducibility
        - Test on held-out NASA validation data
        """)