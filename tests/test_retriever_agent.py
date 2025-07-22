"""
Tests for the Retriever Agent and search tools.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models import ResearchQuery, ResearchPaper, AgentTask
from src.agents.retriever_agent import RetrieverAgent
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.duckduckgo_search import DuckDuckGoSearchTool


class TestArxivSearchTool:
    """Test cases for ArXiv search tool."""
    
    def test_init(self):
        """Test ArXiv tool initialization."""
        tool = ArxivSearchTool(max_results=5)
        assert tool.max_results == 5
        assert tool.client is not None
    
    def test_search_query_formatting(self):
        """Test query formatting for ArXiv search."""
        tool = ArxivSearchTool()
        
        # This is a basic test - in a real scenario, you might mock the arxiv client
        # For now, we'll test that the method exists and can be called
        assert hasattr(tool, 'search')
        assert hasattr(tool, 'search_by_category')
        assert hasattr(tool, 'search_by_author')
    
    def test_convert_to_research_paper(self):
        """Test conversion of ArXiv result to ResearchPaper."""
        tool = ArxivSearchTool()
        
        # Mock ArXiv result
        mock_result = Mock()
        mock_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
        mock_result.title = "Test Paper Title"
        mock_result.summary = "This is a test abstract."
        mock_result.published = datetime(2023, 1, 15)
        mock_result.categories = ["cs.AI", "cs.LG"]
        mock_result.doi = "10.1234/test.doi"
        
        # Mock authors
        mock_author = Mock()
        mock_author.name = "John Doe"
        mock_result.authors = [mock_author]
        
        paper = tool._convert_to_research_paper(mock_result)
        
        assert paper.id == "arxiv:2301.12345v1"
        assert paper.title == "Test Paper Title"
        assert paper.authors == ["John Doe"]
        assert paper.abstract == "This is a test abstract."
        assert paper.source == "arxiv"
        assert "cs.AI" in paper.categories


class TestDuckDuckGoSearchTool:
    """Test cases for DuckDuckGo search tool."""
    
    def test_init(self):
        """Test DuckDuckGo tool initialization."""
        tool = DuckDuckGoSearchTool(max_results=5)
        assert tool.max_results == 5
        assert tool.ddgs is not None
    
    def test_is_research_content(self):
        """Test research content detection."""
        tool = DuckDuckGoSearchTool()
        
        # Research content
        assert tool._is_research_content(
            "Machine Learning Research Paper",
            "This paper presents a novel algorithm for deep learning",
            "https://arxiv.org/abs/test"
        )
        
        # Non-research content
        assert not tool._is_research_content(
            "Best Pizza in Town",
            "Find the most delicious pizza recipes",
            "https://foodblog.com/pizza"
        )
    
    def test_extract_authors(self):
        """Test author extraction from text."""
        tool = DuckDuckGoSearchTool()
        
        # Test different author patterns
        body1 = "by John Doe and Jane Smith"
        authors1 = tool._extract_authors(body1, "")
        assert "John Doe" in authors1
        
        body2 = "Smith et al. proposed this method"
        authors2 = tool._extract_authors(body2, "")
        assert len(authors2) > 0
    
    def test_determine_source(self):
        """Test source determination from URL."""
        tool = DuckDuckGoSearchTool()
        
        assert tool._determine_source("https://arxiv.org/abs/test") == "arxiv"
        assert tool._determine_source("https://scholar.google.com/test") == "google_scholar"
        assert tool._determine_source("https://ieee.org/test") == "ieee"
        assert tool._determine_source("https://example.com/test") == "web"
    
    def test_extract_doi(self):
        """Test DOI extraction from text."""
        tool = DuckDuckGoSearchTool()
        
        text_with_doi = "The paper has DOI: 10.1234/example.doi.12345"
        doi = tool._extract_doi(text_with_doi)
        assert doi == "10.1234/example.doi.12345"
        
        text_without_doi = "This text has no DOI"
        doi = tool._extract_doi(text_without_doi)
        assert doi is None


class TestRetrieverAgent:
    """Test cases for Retriever Agent."""
    
    def test_init(self):
        """Test Retriever Agent initialization."""
        agent = RetrieverAgent()
        assert agent.agent_id == "retriever_agent"
        assert agent.arxiv_tool is not None
        assert agent.ddg_tool is not None
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        agent = RetrieverAgent()
        
        # Create test papers with duplicates
        paper1 = ResearchPaper(
            id="test1",
            title="Machine Learning Methods",
            authors=["Author 1"],
            abstract="Test abstract",
            url="http://example1.com"
        )
        
        paper2 = ResearchPaper(
            id="test2",
            title="Machine Learning Methods",  # Same title
            authors=["Author 2"],
            abstract="Different abstract",
            url="http://example2.com"
        )
        
        paper3 = ResearchPaper(
            id="test3",
            title="Different Paper",
            authors=["Author 3"],
            abstract="Another abstract",
            url="http://example3.com"
        )
        
        papers = [paper1, paper2, paper3]
        unique_papers = agent._remove_duplicates(papers)
        
        # Should remove the duplicate title
        assert len(unique_papers) == 2
    
    def test_titles_similar(self):
        """Test title similarity detection."""
        agent = RetrieverAgent()
        
        title1 = "Machine Learning for Natural Language Processing"
        title2 = "Natural Language Processing with Machine Learning"
        title3 = "Computer Vision and Deep Learning"
        
        # Similar titles
        assert agent._titles_similar(title1, title2)
        
        # Different titles
        assert not agent._titles_similar(title1, title3)
    
    def test_get_supported_sources(self):
        """Test getting supported sources."""
        agent = RetrieverAgent()
        sources = agent.get_supported_sources()
        
        assert "arxiv" in sources
        assert "duckduckgo" in sources
    
    def test_get_agent_info(self):
        """Test getting agent information."""
        agent = RetrieverAgent()
        info = agent.get_agent_info()
        
        assert info["agent_id"] == "retriever_agent"
        assert info["agent_type"] == "retriever"
        assert "supported_sources" in info
    
    def test_execute_task(self):
        """Test task execution."""
        agent = RetrieverAgent()
        
        # Create test task
        task = AgentTask(
            task_id="test_task_001",
            agent_type="retriever",
            input_data={
                "query": "machine learning",
                "sources": ["arxiv"],
                "max_papers": 2
            }
        )
        
        # Mock the retrieve_papers method to avoid actual API calls
        with patch.object(agent, 'retrieve_papers') as mock_retrieve:
            mock_retrieve.return_value = [
                ResearchPaper(
                    id="test1",
                    title="Test Paper 1",
                    authors=["Author 1"],
                    abstract="Test abstract 1",
                    url="http://example1.com"
                )
            ]
            
            result_task = agent.execute_task(task)
            
            assert result_task.status == "completed"
            assert "papers" in result_task.output_data
            assert result_task.completed_at is not None


def test_integration_basic():
    """Basic integration test."""
    # Test that we can create all components without errors
    query = ResearchQuery(query="test query")
    agent = RetrieverAgent()
    
    assert query.query == "test query"
    assert agent.agent_id == "retriever_agent"
    
    # Test agent info
    info = agent.get_agent_info()
    assert isinstance(info, dict)
    assert "agent_id" in info


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
