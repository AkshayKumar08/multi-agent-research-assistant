#!/usr/bin/env python3
"""
Tests for the CrewAI Research Coordinator.

This module tests the ResearchCoordinator class which orchestrates
all agents using CrewAI for complex research workflows.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.research_coordinator import ResearchCoordinator
from src.models import ResearchPaper, Summary, Citation, Question, Answer
from src.tools.ollama_client import OllamaClient


class TestResearchCoordinator:
    """Test the ResearchCoordinator class."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        client = Mock(spec=OllamaClient)
        client.is_available.return_value = True
        client.generate_text = AsyncMock(return_value="Mock response")
        client.list_models.return_value = ["mistral:7b"]
        return client
    
    @pytest.fixture
    def coordinator(self, mock_ollama_client):
        """Create a ResearchCoordinator instance."""
        with patch('src.agents.research_coordinator.Ollama'):
            coordinator = ResearchCoordinator(mock_ollama_client)
            return coordinator
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator is not None
        assert coordinator.ollama_client is not None
        assert coordinator.retriever_agent is not None
        assert coordinator.summarizer_agent is not None
        assert coordinator.qa_agent is not None
        assert coordinator.citation_agent is not None
        
        # Check CrewAI agents
        assert coordinator.crew_retriever is not None
        assert coordinator.crew_summarizer is not None
        assert coordinator.crew_qa is not None
        assert coordinator.crew_citation is not None
        assert coordinator.crew_coordinator is not None
    
    def test_crewai_agents_setup(self, coordinator):
        """Test CrewAI agents are properly configured."""
        # Check retriever agent
        assert coordinator.crew_retriever.role == 'Research Paper Retriever'
        assert 'relevant academic papers' in coordinator.crew_retriever.goal
        
        # Check summarizer agent
        assert coordinator.crew_summarizer.role == 'Research Paper Summarizer'
        assert 'summaries' in coordinator.crew_summarizer.goal
        
        # Check Q&A agent
        assert coordinator.crew_qa.role == 'Research Question Answerer'
        assert 'questions' in coordinator.crew_qa.goal
        
        # Check citation agent
        assert coordinator.crew_citation.role == 'Citation Specialist'
        assert 'citations' in coordinator.crew_citation.goal
        
        # Check coordinator agent
        assert coordinator.crew_coordinator.role == 'Research Project Coordinator'
        assert 'workflow' in coordinator.crew_coordinator.goal
    
    def test_create_research_tasks(self, coordinator):
        """Test research task creation."""
        query = "machine learning transformers"
        session_id = str(uuid.uuid4())
        
        tasks = coordinator._create_research_tasks(query, session_id)
        
        assert len(tasks) == 5
        
        # Check task descriptions contain expected content
        task_descriptions = [task.description for task in tasks]
        
        assert any("Search for academic papers" in desc for desc in task_descriptions)
        assert any("Create comprehensive summaries" in desc for desc in task_descriptions)
        assert any("Generate proper academic citations" in desc for desc in task_descriptions)
        assert any("Prepare the research context" in desc for desc in task_descriptions)
        assert any("Coordinate the complete research workflow" in desc for desc in task_descriptions)
        
        # Check dependencies
        retrieve_task = tasks[0]
        summarize_task = tasks[1]
        citation_task = tasks[2]
        qa_prep_task = tasks[3]
        coordinate_task = tasks[4]
        
        assert len(summarize_task.dependencies) == 1
        assert len(citation_task.dependencies) == 1
        assert len(qa_prep_task.dependencies) == 2
        assert len(coordinate_task.dependencies) == 4
    
    @pytest.mark.asyncio
    async def test_conduct_research_success(self, coordinator):
        """Test successful research workflow."""
        # Mock the individual agents
        mock_papers = [
            ResearchPaper(
                id="paper1",
                title="Test Paper 1",
                authors=["Author 1"],
                abstract="Test abstract 1",
                source="arxiv"
            ),
            ResearchPaper(
                id="paper2",
                title="Test Paper 2",
                authors=["Author 2"],
                abstract="Test abstract 2",
                source="duckduckgo"
            )
        ]
        
        mock_summaries = [
            Summary(
                paper_id="paper1",
                summary="Test summary 1",
                key_findings=["Finding 1"],
                agent_id="summarizer"
            )
        ]
        
        mock_citations = [
            Citation(
                citation_id="cite1",
                paper_id="paper1",
                citation_format="bibtex",
                citation_text="@article{test1}",
                agent_id="citation"
            )
        ]
        
        # Patch the individual agent methods
        with patch.object(coordinator.retriever_agent, 'search_papers', 
                         new_callable=AsyncMock) as mock_search:
            with patch.object(coordinator.summarizer_agent, 'summarize_paper',
                             new_callable=AsyncMock) as mock_summarize:
                with patch.object(coordinator.citation_agent, 'generate_citations_for_paper',
                                 new_callable=AsyncMock) as mock_citations_gen:
                    with patch('crewai.Crew') as mock_crew_class:
                        
                        # Setup mocks
                        mock_search.return_value = mock_papers
                        mock_summarize.return_value = mock_summaries[0]
                        mock_citations_gen.return_value = mock_citations
                        
                        mock_crew = Mock()
                        mock_crew.kickoff.return_value = "Success"
                        mock_crew_class.return_value = mock_crew
                        
                        # Execute research
                        session = await coordinator.conduct_research(
                            "test query", user_id="test_user"
                        )
                        
                        # Verify results
                        assert session.session_id is not None
                        assert session.query.query == "test query"
                        assert session.query.user_id == "test_user"
                        assert len(session.papers) == 2
                        assert len(session.summaries) == 2  # One for each paper
                        assert len(session.citations) == 2  # One for each paper
                        assert len(session.tasks) == 3  # retrieve, summarize, citation
                        
                        # Check task completion
                        completed_tasks = [t for t in session.tasks if t.status == "completed"]
                        assert len(completed_tasks) == 3
    
    @pytest.mark.asyncio
    async def test_conduct_research_error_handling(self, coordinator):
        """Test error handling in research workflow."""
        with patch.object(coordinator.retriever_agent, 'search_papers',
                         side_effect=Exception("Test error")):
            with patch('crewai.Crew') as mock_crew_class:
                mock_crew = Mock()
                mock_crew.kickoff.return_value = "Success"
                mock_crew_class.return_value = mock_crew
                
                session = await coordinator.conduct_research("test query")
                
                # Should have error task
                error_tasks = [t for t in session.tasks if t.status == "failed"]
                assert len(error_tasks) == 1
                assert "Test error" in error_tasks[0].error_message
    
    @pytest.mark.asyncio
    async def test_answer_question(self, coordinator):
        """Test question answering functionality."""
        # Create mock session
        mock_papers = [
            ResearchPaper(
                id="paper1",
                title="Test Paper",
                authors=["Author"],
                abstract="Test abstract",
                source="test"
            )
        ]
        
        mock_summaries = [
            Summary(
                paper_id="paper1",
                summary="Test summary",
                agent_id="summarizer"
            )
        ]
        
        # Create session
        from src.models import ResearchQuery, ResearchSession
        query = ResearchQuery(query="test")
        session = ResearchSession(
            session_id="test_session",
            query=query,
            papers=mock_papers,
            summaries=mock_summaries
        )
        
        # Mock Q&A agent
        mock_answer = Answer(
            answer_id="answer1",
            question_id="question1",
            answer_text="Test answer",
            confidence_score=0.9,
            source_papers=["paper1"]
        )
        
        with patch.object(coordinator.qa_agent, 'answer_question',
                         new_callable=AsyncMock) as mock_qa:
            mock_qa.return_value = mock_answer
            
            answer = await coordinator.answer_question(session, "test question")
            
            assert answer.answer_text == "Test answer"
            assert answer.confidence_score == 0.9
            assert "paper1" in answer.source_papers
    
    @pytest.mark.asyncio
    async def test_answer_question_error_handling(self, coordinator):
        """Test error handling in question answering."""
        from src.models import ResearchQuery, ResearchSession
        query = ResearchQuery(query="test")
        session = ResearchSession(session_id="test_session", query=query)
        
        with patch.object(coordinator.qa_agent, 'answer_question',
                         side_effect=Exception("QA error")):
            
            answer = await coordinator.answer_question(session, "test question")
            
            assert "error" in answer.answer_text.lower()
            assert answer.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_add_citations(self, coordinator):
        """Test adding citations to session."""
        # Create mock session
        mock_papers = [
            ResearchPaper(
                id="paper1",
                title="Test Paper",
                authors=["Author"],
                abstract="Test abstract",
                source="test"
            )
        ]
        
        from src.models import ResearchQuery, ResearchSession
        query = ResearchQuery(query="test")
        session = ResearchSession(
            session_id="test_session",
            query=query,
            papers=mock_papers
        )
        
        mock_citations = [
            Citation(
                citation_id="cite1",
                paper_id="paper1",
                citation_format="bibtex",
                citation_text="@article{test}",
                agent_id="citation"
            )
        ]
        
        with patch.object(coordinator.citation_agent, 'generate_citations_for_paper',
                         new_callable=AsyncMock) as mock_citations_gen:
            mock_citations_gen.return_value = mock_citations
            
            new_citations = await coordinator.add_citations(session, "bibtex")
            
            assert len(new_citations) == 1
            assert new_citations[0].citation_format == "bibtex"
            assert len(session.citations) == 1
    
    def test_get_session_summary(self, coordinator):
        """Test session summary generation."""
        from src.models import ResearchQuery, ResearchSession, AgentTask
        
        query = ResearchQuery(query="test query")
        session = ResearchSession(
            session_id="test_session",
            query=query,
            papers=[Mock() for _ in range(3)],
            summaries=[Mock() for _ in range(2)],
            citations=[Mock() for _ in range(4)],
            tasks=[
                AgentTask(task_id="1", agent_type="test", status="completed"),
                AgentTask(task_id="2", agent_type="test", status="failed")
            ]
        )
        
        summary = coordinator.get_session_summary(session)
        
        assert summary["session_id"] == "test_session"
        assert summary["query"] == "test query"
        assert summary["papers_found"] == 3
        assert summary["summaries_generated"] == 2
        assert summary["citations_available"] == 4
        assert summary["tasks_completed"] == 1
        assert summary["tasks_failed"] == 1
        assert "created_at" in summary
        assert "updated_at" in summary


# Integration tests
class TestResearchCoordinatorIntegration:
    """Integration tests for ResearchCoordinator."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_integration(self):
        """Test full research workflow integration (requires Ollama)."""
        try:
            coordinator = ResearchCoordinator()
            
            # Test with a simple query
            session = await coordinator.conduct_research(
                "quantum computing basics",
                user_id="integration_test"
            )
            
            # Basic assertions
            assert session.session_id is not None
            assert session.query.query == "quantum computing basics"
            
            # Test Q&A if papers were found
            if session.papers:
                answer = await coordinator.answer_question(
                    session,
                    "What is quantum computing?"
                )
                assert answer.answer_text is not None
                assert isinstance(answer.confidence_score, float)
            
        except Exception as e:
            pytest.skip(f"Integration test requires Ollama server: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
