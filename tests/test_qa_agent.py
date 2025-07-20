"""
Test suite for Q&A Agent.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from src.agents.qa_agent import QAAgent
from src.models import (
    ResearchPaper, Summary, Question, Answer, ConversationContext,
    AgentTask, ResearchSession, ResearchQuery
)
from src.tools.ollama_client import OllamaClient


class TestQAAgent:
    """Test cases for QAAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Ollama client
        self.mock_ollama = Mock(spec=OllamaClient)
        self.mock_ollama.is_available.return_value = True
        self.mock_ollama.generate.return_value = "ANSWER: This is a test answer.\n\nREASONING: Based on the provided context.\n\nCONFIDENCE: High - Clear information available"
        
        # Create QA agent with mocked client
        self.qa_agent = QAAgent(ollama_client=self.mock_ollama)
        
        # Sample test data
        self.sample_paper = ResearchPaper(
            id="paper1",
            title="Machine Learning in Healthcare",
            authors=["John Doe", "Jane Smith"],
            abstract="This paper explores the application of machine learning techniques in healthcare diagnostics.",
            url="https://example.com/paper1",
            source="arxiv"
        )
        
        self.sample_summary = Summary(
            paper_id="paper1",
            summary="Machine learning shows promise in healthcare diagnostics, particularly in image analysis and pattern recognition.",
            key_findings=["ML improves diagnostic accuracy", "Reduces processing time", "Enables early detection"],
            agent_id="summarizer_agent"
        )
        
        self.sample_question = Question(
            question_id=str(uuid.uuid4()),
            question_text="What are the benefits of machine learning in healthcare?",
            context_type="general"
        )
    
    def test_agent_initialization(self):
        """Test QA agent initialization."""
        agent = QAAgent()
        assert agent.agent_id == "qa_agent"
        assert agent.ollama_client is not None
    
    def test_agent_initialization_with_custom_client(self):
        """Test QA agent initialization with custom Ollama client."""
        custom_client = Mock(spec=OllamaClient)
        custom_client.is_available.return_value = True
        
        agent = QAAgent(ollama_client=custom_client)
        assert agent.ollama_client == custom_client
    
    def test_agent_initialization_ollama_unavailable(self):
        """Test QA agent initialization when Ollama is unavailable."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.is_available.return_value = False
        
        with patch('src.agents.qa_agent.OllamaClient', return_value=mock_client):
            agent = QAAgent()
            assert agent.ollama_client == mock_client
    
    def test_create_conversation_context(self):
        """Test creating conversation context from research session."""
        session = ResearchSession(
            session_id="session1",
            query=ResearchQuery(query="test query"),
            papers=[self.sample_paper],
            summaries=[self.sample_summary]
        )
        
        context = self.qa_agent.create_conversation_context(session)
        
        assert context.session_id == "session1"
        assert len(context.papers_in_context) == 1
        assert len(context.summaries_in_context) == 1
        assert context.papers_in_context[0] == "paper1"
        assert context.summaries_in_context[0] == "paper1"
    
    def test_answer_question_success(self):
        """Test successful question answering."""
        context = ConversationContext(
            context_id="context1",
            session_id="session1"
        )
        
        answer = self.qa_agent.answer_question(
            self.sample_question,
            context,
            [self.sample_paper],
            [self.sample_summary]
        )
        
        assert isinstance(answer, Answer)
        assert answer.question_id == self.sample_question.question_id
        assert answer.answer_text == "This is a test answer."
        assert 0.0 <= answer.confidence_score <= 1.0
        assert len(answer.source_papers) >= 0
        assert len(answer.evidence) >= 0
    
    def test_answer_question_ollama_error(self):
        """Test question answering when Ollama fails."""
        self.mock_ollama.generate.side_effect = Exception("Ollama connection failed")
        
        context = ConversationContext(
            context_id="context1",
            session_id="session1"
        )
        
        answer = self.qa_agent.answer_question(
            self.sample_question,
            context,
            [self.sample_paper],
            [self.sample_summary]
        )
        
        assert isinstance(answer, Answer)
        assert answer.confidence_score == 0.0
        assert "error" in answer.answer_text.lower()
        assert "Ollama connection failed" in answer.agent_reasoning
    
    def test_ask_followup_question(self):
        """Test asking follow-up questions."""
        context = ConversationContext(
            context_id="context1",
            session_id="session1"
        )
        
        question, answer = self.qa_agent.ask_followup_question(
            "Can you elaborate on the diagnostic accuracy improvements?",
            context,
            [self.sample_paper],
            [self.sample_summary]
        )
        
        assert isinstance(question, Question)
        assert isinstance(answer, Answer)
        assert question.context_type == "conversation"
        assert len(context.questions) == 1
        assert len(context.answers) == 1
        assert context.questions[0] == question
        assert context.answers[0] == answer
    
    def test_build_llm_context(self):
        """Test building LLM context from papers and summaries."""
        context_str = self.qa_agent._build_llm_context(
            self.sample_question,
            [self.sample_paper],
            [self.sample_summary]
        )
        
        assert "RELEVANT RESEARCH PAPERS" in context_str
        assert "RESEARCH SUMMARIES" in context_str
        assert self.sample_paper.title in context_str
        assert self.sample_summary.summary in context_str
    
    def test_find_relevant_papers(self):
        """Test finding relevant papers."""
        papers = [
            self.sample_paper,
            ResearchPaper(
                id="paper2",
                title="Deep Learning Applications",
                authors=["Alice Johnson"],
                abstract="Exploring deep learning in various domains.",
                url="https://example.com/paper2",
                source="arxiv"
            )
        ]
        
        relevant = self.qa_agent._find_relevant_papers(
            "machine learning healthcare",
            papers
        )
        
        assert len(relevant) >= 1
        assert relevant[0].id == "paper1"  # Should rank higher due to keyword matches
    
    def test_find_relevant_summaries(self):
        """Test finding relevant summaries."""
        summaries = [
            self.sample_summary,
            Summary(
                paper_id="paper2",
                summary="Deep learning models show excellent performance in computer vision tasks.",
                key_findings=["High accuracy", "Fast inference"],
                agent_id="summarizer_agent"
            )
        ]
        
        relevant = self.qa_agent._find_relevant_summaries(
            "healthcare diagnostics",
            summaries
        )
        
        assert len(relevant) >= 1
        assert relevant[0].paper_id == "paper1"  # Should rank higher due to keyword matches
    
    def test_extract_evidence(self):
        """Test extracting evidence from answer."""
        answer_text = "According to Machine Learning in Healthcare research, ML improves diagnostic accuracy."
        
        evidence, source_papers = self.qa_agent._extract_evidence(
            answer_text,
            [self.sample_paper],
            [self.sample_summary]
        )
        
        assert len(evidence) >= 1
        assert any("Machine Learning in Healthcare" in ev for ev in evidence)
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # High confidence answer
        high_conf_answer = "Based on extensive research, machine learning significantly improves diagnostic accuracy in healthcare settings."
        evidence = ["Research from Paper 1", "Data from Study 2"]
        
        confidence = self.qa_agent._calculate_confidence(high_conf_answer, evidence)
        assert 0.5 <= confidence <= 1.0
        
        # Low confidence answer
        low_conf_answer = "This might be unclear and possibly insufficient information."
        minimal_evidence = []
        
        confidence = self.qa_agent._calculate_confidence(low_conf_answer, minimal_evidence)
        assert 0.0 <= confidence <= 0.5
    
    def test_execute_task_success(self):
        """Test successful task execution."""
        task = AgentTask(
            task_id="task1",
            agent_type="qa_agent",
            input_data={
                "question": "What are the benefits of machine learning?",
                "papers": [self.sample_paper.model_dump()],
                "summaries": [self.sample_summary.model_dump()],
                "session_id": "session1"
            }
        )
        
        result = self.qa_agent.execute_task(task)
        
        assert result.status == "completed"
        assert result.completed_at is not None
        assert "question" in result.output_data
        assert "answer" in result.output_data
        assert "context" in result.output_data
    
    def test_execute_task_missing_question(self):
        """Test task execution with missing question."""
        task = AgentTask(
            task_id="task1",
            agent_type="qa_agent",
            input_data={
                "papers": [self.sample_paper.model_dump()],
                "summaries": [self.sample_summary.model_dump()]
            }
        )
        
        result = self.qa_agent.execute_task(task)
        
        assert result.status == "failed"
        assert "No question provided" in result.error_message
    
    def test_execute_task_ollama_error(self):
        """Test task execution when Ollama fails."""
        self.mock_ollama.generate.side_effect = Exception("Ollama error")
        
        task = AgentTask(
            task_id="task1",
            agent_type="qa_agent",
            input_data={
                "question": "What are the benefits?",
                "papers": [self.sample_paper.model_dump()],
                "summaries": [self.sample_summary.model_dump()]
            }
        )
        
        result = self.qa_agent.execute_task(task)
        
        # Task should complete but with error answer
        assert result.status == "completed"
        assert "answer" in result.output_data
        answer_data = result.output_data["answer"]
        assert answer_data["confidence_score"] == 0.0


@pytest.fixture
def qa_agent():
    """Fixture for QA agent with mocked Ollama client."""
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.is_available.return_value = True
    mock_ollama.generate.return_value = "Test answer"
    return QAAgent(ollama_client=mock_ollama)


@pytest.fixture
def sample_data():
    """Fixture for sample test data."""
    paper = ResearchPaper(
        id="test_paper",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        url="https://example.com",
        source="test"
    )
    
    summary = Summary(
        paper_id="test_paper",
        summary="Test summary",
        key_findings=["Finding 1", "Finding 2"],
        agent_id="test_agent"
    )
    
    question = Question(
        question_id="test_question",
        question_text="What is the main finding?",
        context_type="test"
    )
    
    return paper, summary, question


def test_integration_with_fixtures(qa_agent, sample_data):
    """Test integration using fixtures."""
    paper, summary, question = sample_data
    
    context = ConversationContext(
        context_id="test_context",
        session_id="test_session"
    )
    
    answer = qa_agent.answer_question(question, context, [paper], [summary])
    
    assert isinstance(answer, Answer)
    assert answer.question_id == question.question_id
