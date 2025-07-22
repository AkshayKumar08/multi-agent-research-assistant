"""
Tests for the Summarizer Agent and Ollama client.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models import ResearchPaper, Summary, AgentTask
from src.agents.summarizer_agent import SummarizerAgent
from src.tools.ollama_client import OllamaClient


class TestOllamaClient:
    """Test cases for Ollama client."""
    
    def test_init(self):
        """Test Ollama client initialization."""
        client = OllamaClient()
        assert client.base_url is not None
        assert client.model is not None
        assert client.session is not None
    
    def test_init_with_params(self):
        """Test Ollama client initialization with custom parameters."""
        client = OllamaClient(
            base_url="http://custom:11434",
            model="custom-model"
        )
        assert client.base_url == "http://custom:11434"
        assert client.model == "custom-model"
    
    @patch('requests.Session.post')
    def test_generate_success(self, mock_post):
        """Test successful text generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate("Test prompt")
        
        assert result == "Generated text"
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_generate_connection_error(self, mock_post):
        """Test connection error handling."""
        mock_post.side_effect = ConnectionError("Connection failed")
        
        client = OllamaClient()
        
        with pytest.raises(ConnectionError):
            client.generate("Test prompt")
    
    @patch('requests.Session.post')
    def test_chat_success(self, mock_post):
        """Test successful chat interaction."""
        # Mock successful chat response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Chat response"}
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat(messages)
        
        assert result == "Chat response"
        mock_post.assert_called_once()
    
    @patch('requests.Session.get')
    def test_is_available_success(self, mock_get):
        """Test server availability check success."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        assert client.is_available() is True
    
    @patch('requests.Session.get')
    def test_is_available_failure(self, mock_get):
        """Test server availability check failure."""
        mock_get.side_effect = Exception("Connection failed")
        
        client = OllamaClient()
        assert client.is_available() is False
    
    @patch('requests.Session.get')
    def test_list_models(self, mock_get):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:7b"},
                {"name": "llama3:8b"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        models = client.list_models()
        
        assert "mistral:7b" in models
        assert "llama3:8b" in models


class TestSummarizerAgent:
    """Test cases for Summarizer Agent."""
    
    def test_init(self):
        """Test Summarizer Agent initialization."""
        with patch.object(OllamaClient, 'is_available', return_value=True):
            agent = SummarizerAgent()
            assert agent.agent_id == "summarizer_agent"
            assert agent.ollama_client is not None
    
    def test_init_with_client(self):
        """Test initialization with custom Ollama client."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        
        agent = SummarizerAgent(ollama_client=mock_client)
        assert agent.ollama_client == mock_client
    
    def test_create_summary_prompt(self):
        """Test summary prompt creation."""
        with patch.object(OllamaClient, 'is_available', return_value=True):
            agent = SummarizerAgent()
            
            paper = ResearchPaper(
                id="test-001",
                title="Test Paper",
                authors=["Author 1", "Author 2"],
                abstract="This is a test abstract about machine learning.",
                url="http://example.com",
                source="test"
            )
            
            prompt = agent._create_summary_prompt(paper, "general", 200)
            
            assert "Test Paper" in prompt
            assert "Author 1" in prompt
            assert "machine learning" in prompt
            assert "200 words" in prompt
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        with patch.object(OllamaClient, 'is_available', return_value=True):
            agent = SummarizerAgent()
            
            general_prompt = agent._get_system_prompt("general")
            technical_prompt = agent._get_system_prompt("technical")
            
            assert "research assistant" in general_prompt.lower()
            assert "technical" in technical_prompt.lower()
    
    def test_summarize_paper_success(self):
        """Test successful paper summarization."""
        # Mock Ollama client
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "This is a generated summary of the research paper."
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        paper = ResearchPaper(
            id="test-001",
            title="Test Paper",
            authors=["Author 1"],
            abstract="Test abstract",
            url="http://example.com",
            source="test"
        )
        
        summary = agent.summarize_paper(paper)
        
        assert isinstance(summary, Summary)
        assert summary.paper_id == "test-001"
        assert "generated summary" in summary.summary
        assert summary.agent_id == "summarizer_agent"
    
    def test_summarize_paper_failure(self):
        """Test paper summarization failure handling."""
        # Mock Ollama client that fails
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = Exception("Generation failed")
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        paper = ResearchPaper(
            id="test-001",
            title="Test Paper",
            authors=["Author 1"],
            abstract="Test abstract",
            url="http://example.com",
            source="test"
        )
        
        summary = agent.summarize_paper(paper)
        
        assert isinstance(summary, Summary)
        assert summary.paper_id == "test-001"
        assert "failed" in summary.summary.lower()
    
    def test_summarize_papers(self):
        """Test summarizing multiple papers."""
        # Mock Ollama client
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Generated summary"
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        papers = [
            ResearchPaper(
                id="test-001",
                title="Paper 1",
                authors=["Author 1"],
                abstract="Abstract 1",
                url="http://example1.com",
                source="test"
            ),
            ResearchPaper(
                id="test-002",
                title="Paper 2",
                authors=["Author 2"],
                abstract="Abstract 2",
                url="http://example2.com",
                source="test"
            )
        ]
        
        summaries = agent.summarize_papers(papers)
        
        assert len(summaries) == 2
        assert all(isinstance(s, Summary) for s in summaries)
        assert summaries[0].paper_id == "test-001"
        assert summaries[1].paper_id == "test-002"
    
    def test_generate_comparative_summary(self):
        """Test comparative summary generation."""
        # Mock Ollama client
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Comparative analysis of the papers shows..."
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        papers = [
            ResearchPaper(
                id="test-001",
                title="Paper 1",
                authors=["Author 1"],
                abstract="Abstract about topic A",
                url="http://example1.com",
                source="test"
            ),
            ResearchPaper(
                id="test-002",
                title="Paper 2",
                authors=["Author 2"],
                abstract="Abstract about topic B",
                url="http://example2.com",
                source="test"
            )
        ]
        
        comparative = agent.generate_comparative_summary(papers, "machine learning")
        
        assert isinstance(comparative, str)
        assert "comparative" in comparative.lower()
    
    def test_execute_task_success(self):
        """Test successful task execution."""
        # Mock Ollama client
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Task summary result"
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        # Create task with paper data
        paper_data = {
            "id": "test-001",
            "title": "Test Paper",
            "authors": ["Author 1"],
            "abstract": "Test abstract",
            "url": "http://example.com",
            "source": "test"
        }
        
        task = AgentTask(
            task_id="test_task_001",
            agent_type="summarizer",
            input_data={
                "papers": [paper_data],
                "summary_type": "general",
                "max_length": 200
            }
        )
        
        result_task = agent.execute_task(task)
        
        assert result_task.status == "completed"
        assert "summaries" in result_task.output_data
        assert len(result_task.output_data["summaries"]) == 1
    
    def test_execute_task_with_comparative(self):
        """Test task execution with comparative summary."""
        # Mock Ollama client
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Summary or comparative result"
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        # Create task with multiple papers and comparative flag
        paper_data_1 = {
            "id": "test-001",
            "title": "Paper 1",
            "authors": ["Author 1"],
            "abstract": "Abstract 1",
            "url": "http://example1.com",
            "source": "test"
        }
        
        paper_data_2 = {
            "id": "test-002",
            "title": "Paper 2",
            "authors": ["Author 2"],
            "abstract": "Abstract 2",
            "url": "http://example2.com",
            "source": "test"
        }
        
        task = AgentTask(
            task_id="test_task_002",
            agent_type="summarizer",
            input_data={
                "papers": [paper_data_1, paper_data_2],
                "summary_type": "technical",
                "max_length": 300,
                "comparative": True,
                "focus_area": "methodology"
            }
        )
        
        result_task = agent.execute_task(task)
        
        assert result_task.status == "completed"
        assert "summaries" in result_task.output_data
        assert "comparative_summary" in result_task.output_data
        assert len(result_task.output_data["summaries"]) == 2
    
    def test_execute_task_failure(self):
        """Test task execution failure handling."""
        # Mock Ollama client that fails
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = Exception("Generation failed")
        
        agent = SummarizerAgent(ollama_client=mock_client)
        
        task = AgentTask(
            task_id="test_task_003",
            agent_type="summarizer",
            input_data={"papers": []}  # Empty papers to trigger error
        )
        
        result_task = agent.execute_task(task)
        
        assert result_task.status == "failed"
        assert result_task.error_message is not None
    
    def test_get_agent_info(self):
        """Test getting agent information."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = True
        mock_client.model = "mistral:7b"
        
        agent = SummarizerAgent(ollama_client=mock_client)
        info = agent.get_agent_info()
        
        assert info["agent_id"] == "summarizer_agent"
        assert info["agent_type"] == "summarizer"
        assert info["ollama_available"] is True
        assert info["model"] == "mistral:7b"
        assert "general" in info["supported_summary_types"]


def test_integration_basic():
    """Basic integration test."""
    # Test that we can create all components without errors
    with patch.object(OllamaClient, 'is_available', return_value=False):
        # Test with unavailable Ollama (should still initialize)
        agent = SummarizerAgent()
        assert agent.agent_id == "summarizer_agent"
        
        # Test agent info
        info = agent.get_agent_info()
        assert isinstance(info, dict)
        assert "agent_id" in info


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
