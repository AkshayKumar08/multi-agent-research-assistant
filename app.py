#!/usr/bin/env python3
"""
Multi-Agent Research Assistant - Cloud Deployment Version

This is the cloud-optimized version for Streamlit Cloud deployment
using Hugging Face free inference API.
"""

import os
import warnings

# Force cloud deployment settings
os.environ['LLM_PROVIDER'] = 'huggingface'
os.environ['IS_CLOUD_DEPLOYMENT'] = 'true'
os.environ['STREAMLIT_SHARING'] = 'true'

# Suppress warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import streamlit as st

# Import the main UI class
from ui_streamlit import StreamlitResearchUI


def main():
    """Main function for cloud deployment."""
    
    # Add cloud deployment notice
    st.info("""
    🚀 **Cloud Version**: This deployment uses Hugging Face's free inference API for LLM processing.
    
    **Features Available:**
    - ✅ Multi-source paper retrieval (ArXiv + Web)
    - ✅ AI-powered summarization
    - ✅ Interactive Q&A
    - ✅ Citation generation
    - ✅ Session management
    
    **Note**: Responses may take a few seconds as the model loads on first use.
    """)
    
    # Initialize and run the UI
    ui = StreamlitResearchUI()
    ui.main_interface()


if __name__ == "__main__":
    main()
