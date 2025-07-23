"""
Configuration mana    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")ent for the Multi-Agent Research Assistant.
"""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # LLM Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")  # "ollama" or "huggingface"
    
    # Ollama Configuration (for local development)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/qwen2.5:0.5b")
    
    # Hugging Face Configuration (for cloud deployment)
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Optional for better rate limits
    HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
    
    # Cloud deployment settings
    IS_CLOUD_DEPLOYMENT = os.getenv("STREAMLIT_SHARING", "false").lower() == "true" or \
                         os.getenv("IS_CLOUD_DEPLOYMENT", "false").lower() == "true"
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(DATA_DIR / "vector_store"))
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/research_assistant.db")
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))
    
    # Search Configuration
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    MAX_PAPERS_PER_QUERY = int(os.getenv("MAX_PAPERS_PER_QUERY", "5"))
    
    # Agent Configuration
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2048"))
    
    # CrewAI Configuration
    CREW_VERBOSE = os.getenv("CREW_VERBOSE", "true").lower() == "true"
    CREW_MEMORY = os.getenv("CREW_MEMORY", "true").lower() == "true"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and not callable(getattr(cls, key))
        }

# Global configuration instance
config = Config()
