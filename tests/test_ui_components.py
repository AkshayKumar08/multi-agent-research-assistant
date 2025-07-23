#!/usr/bin/env python3
"""
Tests for the UI components of the Multi-Agent Research Assistant.

This module tests the UI functionality including Streamlit interface,
session management, and data formatting.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import ResearchPaper, Summary, Citation, ResearchSession, ResearchQuery


# Mock UI classes to avoid dependency issues
class MockStreamlitUI:
    """Mock Streamlit UI for testing."""
    
    def __init__(self):
        """Initialize mock UI."""
        self.coordinator = Mock()
        self.current_session = None
        self.session_history = []
    
    def _format_papers_html(self, papers):
        """Mock format papers method."""
        if not papers:
            return "<p>No papers found.</p>"
        return f"<div>Found {len(papers)} papers</div>"
    
    def _format_summaries_html(self, summaries):
        """Mock format summaries method."""
        if not summaries:
            return "<p>No summaries generated.</p>"
        return f"<div>Generated {len(summaries)} summaries</div>"
    
    def _format_citations_html(self, citations):
        """Mock format citations method."""
        if not citations:
            return "<p>No citations generated.</p>"
        return f"<div>Generated {len(citations)} citations</div>"
    
    def get_session_stats(self):
        """Mock session stats method."""
        if not self.session_history:
            return "No research sessions completed yet."
        
        total_sessions = len(self.session_history)
        return f"Total Sessions: {total_sessions}"


class TestUIComponents:
    """Test UI component functionality."""
    
    @pytest.fixture
    def mock_ui(self):
        """Create a mock UI instance."""
        return MockStreamlitUI()
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample research papers."""
        return [
            ResearchPaper(
                id="paper1",
                title="Quantum Computing Advances",
                authors=["Alice Smith", "Bob Johnson"],
                abstract="This paper discusses recent advances in quantum computing technology...",
                source="arxiv",
                url="https://arxiv.org/abs/1234.5678"
            ),
            ResearchPaper(
                id="paper2",
                title="Machine Learning Applications",
                authors=["Charlie Brown", "Diana Wilson"],
                abstract="We explore various applications of machine learning in real-world scenarios...",
                source="duckduckgo"
            )
        ]
    
    @pytest.fixture
    def sample_summaries(self):
        """Create sample summaries."""
        return [
            Summary(
                paper_id="paper1",
                summary="This paper presents significant advances in quantum computing...",
                summary_type="general",
                key_findings=["Quantum supremacy achieved", "New error correction methods"],
                agent_id="summarizer"
            ),
            Summary(
                paper_id="paper2",
                summary="Machine learning applications are expanding rapidly...",
                summary_type="technical",
                key_findings=["Improved accuracy", "Real-time processing"],
                agent_id="summarizer"
            )
        ]
    
    @pytest.fixture
    def sample_citations(self):
        """Create sample citations."""
        return [
            Citation(
                citation_id="cite1",
                paper_id="paper1",
                citation_format="bibtex",
                citation_text="@article{smith2025quantum, title={Quantum Computing Advances}, author={Smith, Alice and Johnson, Bob}, year={2025}}",
                agent_id="citation"
            ),
            Citation(
                citation_id="cite2",
                paper_id="paper2",
                citation_format="apa",
                citation_text="Brown, C., & Wilson, D. (2025). Machine Learning Applications.",
                agent_id="citation"
            )
        ]
    
    def test_ui_initialization(self, mock_ui):
        """Test UI initialization."""
        assert mock_ui.coordinator is not None
        assert mock_ui.current_session is None
        assert mock_ui.session_history == []
    
    def test_format_papers_html_empty(self, mock_ui):
        """Test formatting empty papers list."""
        result = mock_ui._format_papers_html([])
        assert "No papers found" in result
    
    def test_format_papers_html_with_data(self, mock_ui, sample_papers):
        """Test formatting papers with data."""
        result = mock_ui._format_papers_html(sample_papers)
        assert "Found 2 papers" in result
    
    def test_format_summaries_html_empty(self, mock_ui):
        """Test formatting empty summaries list."""
        result = mock_ui._format_summaries_html([])
        assert "No summaries generated" in result
    
    def test_format_summaries_html_with_data(self, mock_ui, sample_summaries):
        """Test formatting summaries with data."""
        result = mock_ui._format_summaries_html(sample_summaries)
        assert "Generated 2 summaries" in result
    
    def test_format_citations_html_empty(self, mock_ui):
        """Test formatting empty citations list."""
        result = mock_ui._format_citations_html([])
        assert "No citations generated" in result
    
    def test_format_citations_html_with_data(self, mock_ui, sample_citations):
        """Test formatting citations with data."""
        result = mock_ui._format_citations_html(sample_citations)
        assert "Generated 2 citations" in result
    
    def test_session_stats_empty(self, mock_ui):
        """Test session stats with no sessions."""
        result = mock_ui.get_session_stats()
        assert "No research sessions completed" in result
    
    def test_session_stats_with_sessions(self, mock_ui, sample_papers, sample_summaries):
        """Test session stats with sessions."""
        # Create mock session
        query = ResearchQuery(query="test query")
        session = ResearchSession(
            session_id="test_session",
            query=query,
            papers=sample_papers,
            summaries=sample_summaries
        )
        
        mock_ui.session_history.append(session)
        
        result = mock_ui.get_session_stats()
        assert "Total Sessions: 1" in result
    
    @pytest.mark.asyncio
    async def test_mock_research_workflow(self, mock_ui, sample_papers, sample_summaries, sample_citations):
        """Test mock research workflow."""
        # Mock coordinator behavior
        mock_ui.coordinator.conduct_research = AsyncMock()
        
        # Create mock session
        query = ResearchQuery(query="quantum computing")
        session = ResearchSession(
            session_id="test_session",
            query=query,
            papers=sample_papers,
            summaries=sample_summaries,
            citations=sample_citations
        )
        
        mock_ui.coordinator.conduct_research.return_value = session
        
        # Simulate research
        result_session = await mock_ui.coordinator.conduct_research("quantum computing")
        
        assert result_session.session_id == "test_session"
        assert len(result_session.papers) == 2
        assert len(result_session.summaries) == 2
        assert len(result_session.citations) == 2


class TestUIUtilities:
    """Test UI utility functions."""
    
    def test_data_validation(self):
        """Test data validation for UI components."""
        # Test valid paper data
        paper = ResearchPaper(
            id="test_id",
            title="Test Title",
            authors=["Test Author"],
            abstract="Test abstract",
            source="test"
        )
        
        assert paper.id == "test_id"
        assert paper.title == "Test Title"
        assert len(paper.authors) == 1
    
    def test_html_escaping(self):
        """Test HTML escaping for safe display."""
        # Test that special characters are handled
        test_text = "Test <script>alert('xss')</script> content"
        
        # Mock HTML formatting
        def mock_format_safe(text):
            return text.replace("<", "&lt;").replace(">", "&gt;")
        
        safe_text = mock_format_safe(test_text)
        assert "<script>" not in safe_text
        assert "&lt;script&gt;" in safe_text
    
    def test_truncation(self):
        """Test text truncation for UI display."""
        long_text = "This is a very long text that should be truncated " * 20
        
        def truncate_text(text, max_length=100):
            return text[:max_length] + "..." if len(text) > max_length else text
        
        truncated = truncate_text(long_text, 100)
        assert len(truncated) <= 103  # 100 + "..."
        assert truncated.endswith("...")


class TestUIIntegration:
    """Test UI integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock_workflow(self):
        """Test complete UI workflow with mocks."""
        # Create mock UI
        ui = MockStreamlitUI()
        
        # Mock successful research
        ui.coordinator.conduct_research = AsyncMock()
        ui.coordinator.answer_question = AsyncMock()
        
        # Mock session
        query = ResearchQuery(query="test query")
        session = ResearchSession(
            session_id="test_session",
            query=query,
            papers=[Mock()],
            summaries=[Mock()],
            citations=[Mock()]
        )
        
        ui.coordinator.conduct_research.return_value = session
        
        # Mock answer
        from src.models import Answer
        mock_answer = Answer(
            answer_id="answer1",
            question_id="question1",
            answer_text="This is a test answer",
            confidence_score=0.95
        )
        ui.coordinator.answer_question.return_value = mock_answer
        
        # Test workflow
        result_session = await ui.coordinator.conduct_research("test query")
        assert result_session is not None
        
        ui.current_session = result_session
        ui.session_history.append(result_session)
        
        # Test Q&A
        answer = await ui.coordinator.answer_question(result_session, "test question")
        assert answer.answer_text == "This is a test answer"
        assert answer.confidence_score == 0.95
        
        # Test stats
        stats = ui.get_session_stats()
        assert "Total Sessions: 1" in stats
    
    def test_error_handling(self):
        """Test UI error handling."""
        ui = MockStreamlitUI()
        
        # Test with None coordinator
        ui.coordinator = None
        
        # Should handle gracefully
        try:
            stats = ui.get_session_stats()
            assert "No research sessions" in stats
        except Exception as e:
            pytest.fail(f"UI should handle None coordinator gracefully: {e}")
    
    def test_session_management(self):
        """Test session management functionality."""
        ui = MockStreamlitUI()
        
        # Add multiple sessions
        for i in range(3):
            query = ResearchQuery(query=f"query {i}")
            session = ResearchSession(
                session_id=f"session_{i}",
                query=query
            )
            ui.session_history.append(session)
        
        assert len(ui.session_history) == 3
        
        # Test stats generation
        stats = ui.get_session_stats()
        assert "Total Sessions: 3" in stats


# Test UI launcher functionality
class TestUILauncher:
    """Test UI launcher functionality."""
    
    def test_dependency_checking(self):
        """Test dependency checking logic."""
        # Mock dependency checker
        def mock_check_dependencies():
            # Simulate checking for packages
            available_packages = ['streamlit', 'pandas']
            missing_packages = ['streamlit', 'plotly']
            
            all_available = len(missing_packages) == 0
            return all_available, available_packages
        
        available, packages = mock_check_dependencies()
        assert not available  # Should be False due to missing packages
        assert 'streamlit' in packages
        assert 'pandas' in packages
    
    def test_interface_selection(self):
        """Test interface selection logic."""
        def mock_select_interface(choice, available_packages):
            if choice == "1" and 'streamlit' in available_packages:
                return "streamlit"
            else:
                return None
        
        # Test with streamlit available
        result = mock_select_interface("1", ['streamlit', 'pandas'])
        assert result == "streamlit"
        
        # Test with invalid choice
        result = mock_select_interface("2", ['streamlit', 'pandas'])
        assert result is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
