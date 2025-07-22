"""
Test suite for Citation Agent.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from src.agents.citation_agent import CitationAgent
from src.models import (
    ResearchPaper, Citation, CitationRequest, Bibliography,
    AgentTask
)
from src.tools.ollama_client import OllamaClient


class TestCitationAgent:
    """Test cases for CitationAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Ollama client
        self.mock_ollama = Mock(spec=OllamaClient)
        self.mock_ollama.is_available.return_value = True
        self.mock_ollama.generate.return_value = "Mock LLM-generated citation text"
        
        # Create Citation agent with mocked client
        self.citation_agent = CitationAgent(ollama_client=self.mock_ollama)
        
        # Sample test data
        self.sample_paper = ResearchPaper(
            id="test_paper_1",
            title="Machine Learning in Healthcare: A Comprehensive Review",
            authors=["John Doe", "Jane Smith", "Bob Johnson"],
            abstract="This paper provides a comprehensive review of machine learning applications in healthcare.",
            url="https://arxiv.org/abs/1234.5678",
            published_date=datetime(2023, 5, 15),
            source="arxiv",
            categories=["cs.LG", "cs.AI"],
            doi="10.1000/182"
        )
        
        self.sample_paper_minimal = ResearchPaper(
            id="test_paper_2",
            title="Minimal Paper Example",
            authors=[],
            abstract="",
            url="",
            source="unknown"
        )
    
    def test_agent_initialization(self):
        """Test Citation agent initialization."""
        agent = CitationAgent()
        assert agent.agent_id == "citation_agent"
        assert agent.ollama_client is not None
        assert len(agent.SUPPORTED_FORMATS) >= 4
    
    def test_agent_initialization_with_custom_client(self):
        """Test Citation agent initialization with custom Ollama client."""
        custom_client = Mock(spec=OllamaClient)
        custom_client.is_available.return_value = True
        
        agent = CitationAgent(ollama_client=custom_client)
        assert agent.ollama_client == custom_client
    
    def test_agent_initialization_ollama_unavailable(self):
        """Test Citation agent initialization when Ollama is unavailable."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = False
        
        with patch('src.agents.citation_agent.OllamaClient', return_value=mock_client):
            agent = CitationAgent()
            assert agent.ollama_client == mock_client
    
    def test_supported_formats(self):
        """Test that all expected citation formats are supported."""
        expected_formats = ["bibtex", "apa", "mla", "ieee"]
        for fmt in expected_formats:
            assert fmt in self.citation_agent.SUPPORTED_FORMATS
    
    def test_generate_bibtex_citation(self):
        """Test BibTeX citation generation."""
        citation = self.citation_agent.generate_citation(self.sample_paper, "bibtex")
        
        assert isinstance(citation, Citation)
        assert citation.paper_id == self.sample_paper.id
        assert citation.citation_format == "bibtex"
        assert citation.citation_text.startswith("@")
        assert "Machine Learning in Healthcare" in citation.citation_text
        assert "John Doe" in citation.citation_text or "Doe" in citation.citation_text
        assert "2023" in citation.citation_text
    
    def test_generate_apa_citation(self):
        """Test APA citation generation."""
        citation = self.citation_agent.generate_citation(self.sample_paper, "apa")
        
        assert isinstance(citation, Citation)
        assert citation.citation_format == "apa"
        assert "Doe, J." in citation.citation_text
        assert "(2023)" in citation.citation_text
        assert "Machine Learning in Healthcare" in citation.citation_text
    
    def test_generate_mla_citation(self):
        """Test MLA citation generation."""
        citation = self.citation_agent.generate_citation(self.sample_paper, "mla")
        
        assert isinstance(citation, Citation)
        assert citation.citation_format == "mla"
        assert "Doe, John" in citation.citation_text
        assert '"Machine Learning in Healthcare' in citation.citation_text
        assert "2023" in citation.citation_text
    
    def test_generate_ieee_citation(self):
        """Test IEEE citation generation."""
        citation = self.citation_agent.generate_citation(self.sample_paper, "ieee")
        
        assert isinstance(citation, Citation)
        assert citation.citation_format == "ieee"
        assert "J. Doe" in citation.citation_text
        assert '"Machine Learning in Healthcare' in citation.citation_text
        assert "2023" in citation.citation_text
    
    def test_generate_citation_with_llm(self):
        """Test citation generation using LLM for unsupported formats."""
        citation = self.citation_agent.generate_citation(self.sample_paper, "chicago")
        
        assert isinstance(citation, Citation)
        assert citation.citation_format == "chicago"
        assert citation.citation_text == "Mock LLM-generated citation text"
        self.mock_ollama.generate.assert_called_once()
    
    def test_generate_citation_unsupported_format(self):
        """Test citation generation with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported citation format"):
            self.citation_agent.generate_citation(self.sample_paper, "invalid_format")
    
    def test_generate_citation_minimal_paper(self):
        """Test citation generation with minimal paper data."""
        citation = self.citation_agent.generate_citation(self.sample_paper_minimal, "apa")
        
        assert isinstance(citation, Citation)
        assert "Minimal Paper Example" in citation.citation_text
        assert "Unknown Author" in citation.citation_text or "Unknown" in citation.citation_text
    
    def test_generate_multiple_citations(self):
        """Test generating citations for multiple papers."""
        papers = [self.sample_paper, self.sample_paper_minimal]
        citations = self.citation_agent.generate_multiple_citations(papers, "bibtex")
        
        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)
        assert all(c.citation_format == "bibtex" for c in citations)
        assert citations[0].paper_id == self.sample_paper.id
        assert citations[1].paper_id == self.sample_paper_minimal.id
    
    def test_create_bibliography(self):
        """Test bibliography creation."""
        papers = [self.sample_paper, self.sample_paper_minimal]
        bibliography = self.citation_agent.create_bibliography(
            papers, "Test Bibliography", "apa"
        )
        
        assert isinstance(bibliography, Bibliography)
        assert bibliography.title == "Test Bibliography"
        assert bibliography.format_style == "apa"
        assert len(bibliography.citations) == 2
        assert bibliography.metadata["total_papers"] == 2
    
    def test_extract_citation_data(self):
        """Test citation data extraction."""
        raw_data = self.citation_agent._extract_citation_data(self.sample_paper)
        
        assert raw_data["title"] == self.sample_paper.title
        assert raw_data["authors"] == self.sample_paper.authors
        assert raw_data["year"] == 2023
        assert raw_data["url"] == self.sample_paper.url
        assert raw_data["doi"] == self.sample_paper.doi
    
    def test_extract_venue_arxiv(self):
        """Test venue extraction for ArXiv papers."""
        venue = self.citation_agent._extract_venue(self.sample_paper)
        assert "arXiv" in venue
        assert self.sample_paper.id in venue
    
    def test_determine_publication_type(self):
        """Test publication type determination."""
        # ArXiv paper
        pub_type = self.citation_agent._determine_publication_type(self.sample_paper)
        assert pub_type == "preprint"
        
        # Conference paper
        conference_paper = ResearchPaper(
            id="conf_paper",
            title="Conference Paper",
            authors=["Author Name"],
            abstract="This paper was published in a conference proceedings.",
            source="duckduckgo"
        )
        pub_type = self.citation_agent._determine_publication_type(conference_paper)
        assert pub_type == "conference"
    
    def test_format_apa_author(self):
        """Test APA author formatting."""
        formatted = self.citation_agent._format_apa_author("John Michael Doe")
        assert formatted == "Doe, J. M."
        
        formatted = self.citation_agent._format_apa_author("Madonna")
        assert formatted == "Madonna"
    
    def test_validate_citation_success(self):
        """Test successful citation validation."""
        citation = Citation(
            citation_id="test_citation",
            paper_id=self.sample_paper.id,
            citation_format="apa",
            citation_text="Doe, J. (2023). Machine Learning in Healthcare: A Comprehensive Review.",
            raw_data={},
            agent_id="test_agent"
        )
        
        self.citation_agent._validate_citation(citation, self.sample_paper)
        assert citation.validation_status == "validated"
    
    def test_validate_citation_warnings(self):
        """Test citation validation with warnings."""
        citation = Citation(
            citation_id="test_citation",
            paper_id=self.sample_paper.id,
            citation_format="apa",
            citation_text="Some random citation text without title or authors.",
            raw_data={},
            agent_id="test_agent"
        )
        
        self.citation_agent._validate_citation(citation, self.sample_paper)
        assert "warning" in citation.validation_status
    
    def test_create_fallback_citation(self):
        """Test fallback citation creation."""
        fallback = self.citation_agent._create_fallback_citation(
            self.sample_paper, "apa", "Test error"
        )
        
        assert isinstance(fallback, Citation)
        assert "fallback" in fallback.validation_status
        assert "Test error" in fallback.validation_status
        assert self.sample_paper.title in fallback.citation_text
    
    def test_get_sort_key(self):
        """Test bibliography sorting key extraction."""
        citation = Citation(
            citation_id="test",
            paper_id="test",
            citation_format="apa",
            citation_text="Smith, J. (2023). Test Paper.",
            raw_data={},
            agent_id="test"
        )
        
        sort_key = self.citation_agent._get_sort_key(citation)
        assert "smith" in sort_key.lower()
    
    def test_execute_task_single_citations(self):
        """Test task execution for single citations."""
        task = AgentTask(
            task_id="test_task",
            agent_type="citation_agent",
            input_data={
                "papers": [self.sample_paper.model_dump()],
                "citation_format": "bibtex"
            }
        )
        
        result = self.citation_agent.execute_task(task)
        
        assert result.status == "completed"
        assert result.completed_at is not None
        assert "citations" in result.output_data
        assert result.output_data["total_citations"] == 1
        assert result.output_data["format"] == "bibtex"
    
    def test_execute_task_bibliography(self):
        """Test task execution for bibliography creation."""
        task = AgentTask(
            task_id="test_task",
            agent_type="citation_agent",
            input_data={
                "papers": [self.sample_paper.model_dump(), self.sample_paper_minimal.model_dump()],
                "citation_format": "apa",
                "create_bibliography": True,
                "bibliography_title": "Test Bibliography"
            }
        )
        
        result = self.citation_agent.execute_task(task)
        
        assert result.status == "completed"
        assert "bibliography" in result.output_data
        assert result.output_data["total_citations"] == 2
    
    def test_execute_task_no_papers(self):
        """Test task execution with no papers."""
        task = AgentTask(
            task_id="test_task",
            agent_type="citation_agent",
            input_data={}
        )
        
        result = self.citation_agent.execute_task(task)
        
        assert result.status == "failed"
        assert "No papers provided" in result.error_message
    
    def test_execute_task_with_error(self):
        """Test task execution with generation error."""
        # Mock an error in citation generation
        with patch.object(self.citation_agent, 'generate_multiple_citations', side_effect=Exception("Test error")):
            task = AgentTask(
                task_id="test_task",
                agent_type="citation_agent",
                input_data={
                    "papers": [self.sample_paper.model_dump()]
                }
            )
            
            result = self.citation_agent.execute_task(task)
            
            assert result.status == "failed"
            assert "Test error" in result.error_message
    
    def test_llm_fallback_when_unavailable(self):
        """Test fallback behavior when LLM is unavailable."""
        self.mock_ollama.is_available.return_value = False
        
        citation = self.citation_agent.generate_citation(self.sample_paper, "chicago")
        
        assert isinstance(citation, Citation)
        assert citation.citation_format == "chicago"
        # Should use fallback text instead of LLM
        assert "John Doe" in citation.citation_text
        assert self.sample_paper.title in citation.citation_text
    
    def test_llm_generation_error_handling(self):
        """Test error handling in LLM generation."""
        self.mock_ollama.generate.side_effect = Exception("LLM error")
        
        citation = self.citation_agent.generate_citation(self.sample_paper, "chicago")
        
        assert isinstance(citation, Citation)
        # Should fall back to basic citation format
        assert self.sample_paper.title in citation.citation_text


@pytest.fixture
def citation_agent():
    """Fixture for Citation agent with mocked Ollama client."""
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.is_available.return_value = True
    mock_ollama.generate.return_value = "Test citation"
    return CitationAgent(ollama_client=mock_ollama)


@pytest.fixture
def sample_papers():
    """Fixture for sample research papers."""
    paper1 = ResearchPaper(
        id="paper1",
        title="First Test Paper",
        authors=["Alice Smith", "Bob Jones"],
        abstract="First test paper abstract",
        url="https://example.com/paper1",
        published_date=datetime(2023, 1, 1),
        source="test"
    )
    
    paper2 = ResearchPaper(
        id="paper2",
        title="Second Test Paper",
        authors=["Charlie Brown"],
        abstract="Second test paper abstract",
        url="https://example.com/paper2",
        published_date=datetime(2023, 6, 15),
        source="test"
    )
    
    return [paper1, paper2]


def test_integration_with_fixtures(citation_agent, sample_papers):
    """Test integration using fixtures."""
    citations = citation_agent.generate_multiple_citations(sample_papers, "apa")
    
    assert len(citations) == 2
    assert all(isinstance(c, Citation) for c in citations)
    assert all(c.citation_format == "apa" for c in citations)


def test_bibliography_creation_with_fixtures(citation_agent, sample_papers):
    """Test bibliography creation using fixtures."""
    bibliography = citation_agent.create_bibliography(
        sample_papers, "Test Bibliography", "ieee"
    )
    
    assert isinstance(bibliography, Bibliography)
    assert len(bibliography.citations) == 2
    assert bibliography.format_style == "ieee"
    assert all(c.citation_format == "ieee" for c in bibliography.citations)
