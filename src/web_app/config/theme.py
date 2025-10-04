"""
Theme and CSS configuration for NASA Exoplanet Detection Web App
"""

import streamlit as st

def toggle_theme():
    """Toggle between dark and light theme"""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    st.session_state.dark_mode = not st.session_state.dark_mode
    return st.session_state.dark_mode

def get_theme_css(dark_mode=False):
    """Get CSS based on theme mode"""
    if dark_mode:
        return """
        <style>
        /* Override Streamlit's default theme */
        .stApp {
            background-color: #191414 !important;
        }
        
        .main .block-container {
            background-color: #191414 !important;
        }
        
        .stSidebar {
            background-color: #191414 !important;
        }
        
        .sub-header {
            font-size: 1.5rem;
            color: #1DB954 !important;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .metrics-container {
            background-color: #191414 !important;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #282828 !important;
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
            background-color: #1DB954 !important;
            color: #191414 !important;
            border: 2px solid #1ed760 !important;
        }

        .false-positive {
            background-color: #e22134 !important;
            color: #ffffff !important;
            border: 2px solid #ff6b6b !important;
        }

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        /* Sidebar button text alignment */
        .stSidebar .stButton > button {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        
        .stSidebar .stButton > button > div {
            text-align: left !important;
        }
        </style>
        """
    else:
        return """
        <style>
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

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        /* Sidebar button text alignment */
        .stSidebar .stButton > button {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        
        .stSidebar .stButton > button > div {
            text-align: left !important;
        }
        </style>
        """

def apply_theme():
    """Apply theme CSS to the page"""
    dark_mode = st.session_state.get('dark_mode', False)
    st.markdown(get_theme_css(dark_mode), unsafe_allow_html=True)
    
    # Theme toggle button
    if st.button("Dark" if not dark_mode else "Light", key="theme_toggle", help="Toggle theme"):
        toggle_theme()
        st.rerun()
