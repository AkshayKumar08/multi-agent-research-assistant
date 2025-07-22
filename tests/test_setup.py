"""
Test basic setup and configuration.
"""
import sys
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_config_import():
    """Test that configuration can be imported."""
    from config.settings import config
    assert config is not None
    assert hasattr(config, 'OLLAMA_BASE_URL')
    assert hasattr(config, 'OLLAMA_MODEL')


def test_models_import():
    """Test that models can be imported."""
    from src.models import ResearchPaper, ResearchQuery, Summary
    
    # Test basic model creation
    paper = ResearchPaper(
        id="test-001",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract"
    )
    assert paper.id == "test-001"
    assert paper.title == "Test Paper"


def test_logger_setup():
    """Test that logger can be set up."""
    from src.utils.logger import setup_logger, get_logger
    
    logger = setup_logger()
    assert logger is not None
    
    named_logger = get_logger("test")
    assert named_logger is not None


def test_directory_structure():
    """Test that required directories exist."""
    from config.settings import config
    
    assert config.DATA_DIR.exists()
    assert config.LOGS_DIR.exists()
    
    src_dir = Path(__file__).parent.parent / "src"
    assert src_dir.exists()
    assert (src_dir / "agents").exists()
    assert (src_dir / "tools").exists()
    assert (src_dir / "models").exists()
    assert (src_dir / "storage").exists()
    assert (src_dir / "utils").exists()


if __name__ == "__main__":
    # Run tests directly
    test_config_import()
    test_models_import()
    test_logger_setup()
    test_directory_structure()
    print("✅ All basic setup tests passed!")
