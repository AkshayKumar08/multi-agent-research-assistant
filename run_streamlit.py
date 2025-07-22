#!/usr/bin/env python3
"""
Streamlit App Launcher with Proper Context Management

This script launches the Streamlit app with proper warning suppression
and context management.
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress Streamlit warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Set environment variables to suppress warnings
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'

# Import and run the app
if __name__ == "__main__":
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import streamlit after setting environment
    import streamlit as st
    from streamlit import runtime
    from streamlit.web import cli as stcli
    
    # Set up proper runtime context
    if not runtime.exists():
        # Force streamlit to run in script mode
        sys.argv = ["streamlit", "run", "ui_streamlit.py", "--server.headless", "true"]
        stcli.main()
    else:
        # Import and run our app directly
        from ui_streamlit import main
        main()
