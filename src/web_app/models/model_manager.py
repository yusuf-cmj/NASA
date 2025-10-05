"""
Model Management for NASA Exoplanet Detection Web App
"""

import os
import joblib
import streamlit as st
import pickle
import base64
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_DIR, MODEL_FILES

def get_available_models():
    """Get list of available models (hybrid: Local Storage + File System)"""
    models = []
    
    # Add default NebulaticAI model first
    models.append({
        'filename': 'default_nebulaticai',
        'name': 'NebulaticAI',
        'description': 'Default NASA Exoplanet Detection Model - Binary Stacking Ensemble',
        'created_at': '2024-01-01T00:00:00',
        'accuracy': 0.8821,
        'roc_auc': 0.9448,
        'is_default': True,
        'source': 'file_system'
    })
    
    # Add models from Local Storage (session state approach)
    if 'local_storage_models' in st.session_state:
        for model_data in st.session_state['local_storage_models']:
            models.append({
                'filename': model_data['filename'],
                'name': model_data['name'],
                'description': model_data['description'],
                'created_at': model_data['created_at'],
                'accuracy': model_data['accuracy'],
                'roc_auc': model_data['roc_auc'],
                'is_default': False,
                'source': 'local_storage'
            })
    
    # Add custom trained models from File System (legacy support)
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if file.endswith('_metadata.pkl'):
                try:
                    metadata = joblib.load(os.path.join(MODEL_DIR, file))
                    filename = file.replace('_metadata.pkl', '')
                    
                    # Skip if already in Local Storage
                    if filename not in [m['filename'] for m in models if m['source'] == 'local_storage']:
                        models.append({
                            'filename': filename,
                            'name': metadata.get('name', filename),
                            'description': metadata.get('description', ''),
                            'created_at': metadata.get('created_at', ''),
                            'accuracy': metadata.get('accuracy', 0),
                            'roc_auc': metadata.get('roc_auc', 0),
                            'is_default': False,
                            'source': 'file_system'
                        })
                except:
                    continue
    
    # Sort by creation date (newest first), but keep default first
    models.sort(key=lambda x: (not x.get('is_default', False), x['created_at']), reverse=True)
    return models

def load_models():
    """Load trained models and preprocessing objects with enhanced caching"""
    try:
        # Define paths
        model_path = os.path.join(MODEL_DIR, MODEL_FILES['model'])
        scaler_path = os.path.join(MODEL_DIR, MODEL_FILES['scaler'])
        encoder_path = os.path.join(MODEL_DIR, MODEL_FILES['encoder'])
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
            st.error("❌ Model files not found. Please run training first.")
            return None, None, None, None
        
        # Load with spinner
        with st.spinner("🔄 Loading models..."):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
        
        # Model info
        model_info = {
            'name': 'NebulaticAI',
            'type': 'Binary Classification',
            'features': 6,
            'classes': ['CONFIRMED', 'FALSE_POSITIVE'],
            'accuracy': 0.8821,
            'roc_auc': 0.9448,
            'description': 'Default NASA Exoplanet Detection Model - Binary Stacking Ensemble'
        }
        
        return model, scaler, label_encoder, model_info
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

def load_model_by_name(model_name):
    """Load model by name (hybrid: Local Storage + File System)"""
    # Handle default NebulaticAI model
    if model_name == 'default_nebulaticai':
        return load_models()
    
    # Try to load from Local Storage first
    if 'local_storage_models' in st.session_state:
        for model_data in st.session_state['local_storage_models']:
            if model_data['filename'] == model_name:
                try:
                    # Load from session state (already decoded)
                    model = model_data['model']
                    scaler = model_data['scaler']
                    encoder = model_data['encoder']
                    metadata = model_data['metadata']
                    return model, scaler, encoder, metadata
                except Exception as e:
                    st.error(f"❌ Error loading model from local storage: {e}")
                    break
    
    # Fallback to File System
    model_filename = model_name.replace(' ', '_').replace('/', '_')
    
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_scaler.pkl'))
        encoder = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_encoder.pkl'))
        metadata = joblib.load(os.path.join(MODEL_DIR, f'{model_filename}_metadata.pkl'))
        
        return model, scaler, encoder, metadata
    except:
        return None, None, None, None

def save_model(model, scaler, encoder, metadata, model_name):
    """Save model with metadata to Local Storage"""
    try:
        # Save to Local Storage using session state approach
        if 'local_storage_models' not in st.session_state:
            st.session_state['local_storage_models'] = []
        
        # Check if model already exists
        model_filename = model_name.replace(' ', '_').replace('/', '_')
        existing_models = st.session_state['local_storage_models']
        
        # Remove existing model with same name
        st.session_state['local_storage_models'] = [
            m for m in existing_models if m['filename'] != model_filename
        ]
        
        # Add new model
        model_data = {
            'filename': model_filename,
            'name': model_name,
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'metadata': metadata,
            'description': metadata.get('description', ''),
            'created_at': metadata.get('created_at', ''),
            'accuracy': metadata.get('accuracy', 0),
            'roc_auc': metadata.get('roc_auc', 0),
            'source': 'local_storage'
        }
        
        st.session_state['local_storage_models'].append(model_data)
        
        # Also save to browser's localStorage using JavaScript
        save_model_to_local_storage(model, scaler, encoder, metadata, model_filename)
        
        return True
    except Exception as e:
        st.error(f"❌ Error saving model: {e}")
        return False

def get_model_info(model_name):
    """Get model information"""
    models = get_available_models()
    for model in models:
        if model['filename'] == model_name:
            return model
    return None

# =============================================================================
# LOCAL STORAGE FUNCTIONS
# =============================================================================

def save_model_to_local_storage(model, scaler, encoder, metadata, model_name):
    """Save model to browser's local storage"""
    try:
        # Convert model objects to base64 strings
        model_data = {
            'model': base64.b64encode(pickle.dumps(model)).decode('utf-8'),
            'scaler': base64.b64encode(pickle.dumps(scaler)).decode('utf-8'),
            'encoder': base64.b64encode(pickle.dumps(encoder)).decode('utf-8'),
            'metadata': metadata,
            'created_at': metadata.get('created_at', ''),
            'source': 'local_storage'
        }
        
        # Use JavaScript to save to localStorage
        st.components.v1.html(f"""
            <script>
                try {{
                    const modelData = {json.dumps(model_data)};
                    localStorage.setItem('nasaUserModel_{model_name}', JSON.stringify(modelData));
                    console.log('Model saved to localStorage:', '{model_name}');
                }} catch (error) {{
                    console.error('Error saving to localStorage:', error);
                }}
            </script>
        """, height=0)
        
        return True
    except Exception as e:
        st.error(f"❌ Error saving model to local storage: {e}")
        return False

def load_model_from_local_storage(model_name):
    """Load model from browser's local storage"""
    try:
        # Use JavaScript to get from localStorage
        result = st.components.v1.html(f"""
            <script>
                try {{
                    const modelData = localStorage.getItem('nasaUserModel_{model_name}');
                    if (modelData) {{
                        const data = JSON.parse(modelData);
                        // Send data back to Python
                        window.parent.postMessage({{
                            type: 'model_data',
                            data: data
                        }}, '*');
                    }} else {{
                        window.parent.postMessage({{
                            type: 'model_data',
                            data: null
                        }}, '*');
                    }}
                }} catch (error) {{
                    console.error('Error loading from localStorage:', error);
                    window.parent.postMessage({{
                        type: 'model_data',
                        data: null
                    }}, '*');
                }}
            </script>
        """, height=0)
        
        # Note: This approach has limitations with Streamlit
        # We'll use a different approach with session state
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model from local storage: {e}")
        return None, None, None, None

def get_local_storage_models():
    """Get list of models from local storage"""
    try:
        # Use JavaScript to get all model keys
        result = st.components.v1.html("""
            <script>
                try {
                    const models = [];
                    for (let i = 0; i < localStorage.length; i++) {
                        const key = localStorage.key(i);
                        if (key && key.startsWith('nasaUserModel_')) {
                            const modelData = JSON.parse(localStorage.getItem(key));
                            models.push({
                                filename: key.replace('nasaUserModel_', ''),
                                name: modelData.metadata.name || key.replace('nasaUserModel_', ''),
                                description: modelData.metadata.description || '',
                                created_at: modelData.created_at || '',
                                accuracy: modelData.metadata.accuracy || 0,
                                roc_auc: modelData.metadata.roc_auc || 0,
                                source: 'local_storage'
                            });
                        }
                    }
                    window.parent.postMessage({
                        type: 'models_list',
                        data: models
                    }, '*');
                } catch (error) {
                    console.error('Error getting models from localStorage:', error);
                    window.parent.postMessage({
                        type: 'models_list',
                        data: []
                    }, '*');
                }
            </script>
        """, height=0)
        
        # Note: This approach has limitations with Streamlit
        # We'll use session state approach instead
        return []
    except Exception as e:
        st.error(f"❌ Error getting models from local storage: {e}")
        return []

def delete_model_from_local_storage(model_name):
    """Delete model from local storage"""
    try:
        st.components.v1.html(f"""
            <script>
                try {{
                    localStorage.removeItem('nasaUserModel_{model_name}');
                    console.log('Model deleted from localStorage:', '{model_name}');
                }} catch (error) {{
                    console.error('Error deleting from localStorage:', error);
                }}
            </script>
        """, height=0)
        return True
    except Exception as e:
        st.error(f"❌ Error deleting model from local storage: {e}")
        return False
