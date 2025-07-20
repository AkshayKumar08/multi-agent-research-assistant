"""
Main entry point for the Multi-Agent Research Assistant.
"""
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import config
from src.utils.logger import setup_logger


def main():
    """Main function to run the research assistant."""
    # Setup logging
    logger = setup_logger()
    
    logger.info("🚀 Starting Multi-Agent Research Assistant")
    logger.info(f"Configuration loaded: {config.to_dict()}")
    
    # For now, just verify the setup is working
    print("✅ Multi-Agent Research Assistant initialized successfully!")
    print(f"📊 Ollama URL: {config.OLLAMA_BASE_URL}")
    print(f"🤖 Model: {config.OLLAMA_MODEL}")
    print(f"💾 Data directory: {config.DATA_DIR}")
    print(f"📁 Vector store: {config.VECTOR_STORE_PATH}")
    
    logger.info("Setup verification complete")


if __name__ == "__main__":
    main()
