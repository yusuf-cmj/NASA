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
        .sub-header {
            font-size: 1.5rem;
            color: #90EE90;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .metrics-container {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #4a5568;
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
            background-color: #2d5016;
            color: #90EE90;
            border: 2px solid #4ade80;
        }

        .false-positive {
            background-color: #5c1a1a;
            color: #fca5a5;
            border: 2px solid #f87171;
        }

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
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
        </style>
        """

def apply_theme():
    """Apply theme CSS to the page"""
    dark_mode = st.session_state.get('dark_mode', False)
    st.markdown(get_theme_css(dark_mode), unsafe_allow_html=True)
    
    # Theme toggle button
    if st.button("🌙" if not dark_mode else "☀️", key="theme_toggle", help="Toggle theme"):
        toggle_theme()
        st.rerun()
