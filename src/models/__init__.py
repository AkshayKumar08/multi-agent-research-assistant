"""
Data models for the Multi-Agent Research Assistant.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ResearchPaper(BaseModel):
    """Model for a research paper."""
    
    id: str = Field(..., description="Unique identifier for the paper")
    title: str = Field(..., description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(default="", description="Abstract of the paper")
    url: str = Field(default="", description="URL to the paper")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    source: str = Field(default="", description="Source (arxiv, duckduckgo, etc.)")
    categories: List[str] = Field(default_factory=list, description="Paper categories")
    doi: Optional[str] = Field(None, description="DOI of the paper")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ResearchQuery(BaseModel):
    """Model for a research query."""
    
    query: str = Field(..., description="The research query/topic")
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = Field(None, description="User identifier")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Summary(BaseModel):
    """Model for paper summaries."""
    
    paper_id: str = Field(..., description="ID of the summarized paper")
    summary: str = Field(..., description="Summary text")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    methodology: Optional[str] = Field(None, description="Methodology summary")
    created_at: datetime = Field(default_factory=datetime.now)
    agent_id: str = Field(..., description="ID of the agent that created the summary")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Citation(BaseModel):
    """Model for citations."""
    
    paper_id: str = Field(..., description="ID of the cited paper")
    citation_format: str = Field(..., description="Citation format (bibtex, apa, etc.)")
    citation_text: str = Field(..., description="Formatted citation text")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentTask(BaseModel):
    """Model for agent tasks."""
    
    task_id: str = Field(..., description="Unique task identifier")
    agent_type: str = Field(..., description="Type of agent (retriever, summarizer, etc.)")
    status: str = Field(default="pending", description="Task status")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the task")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from the task")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None, description="Task completion time")
    error_message: Optional[str] = Field(None, description="Error message if task failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ResearchSession(BaseModel):
    """Model for a research session."""
    
    session_id: str = Field(..., description="Unique session identifier")
    query: ResearchQuery = Field(..., description="Original research query")
    papers: List[ResearchPaper] = Field(default_factory=list, description="Retrieved papers")
    summaries: List[Summary] = Field(default_factory=list, description="Generated summaries")
    citations: List[Citation] = Field(default_factory=list, description="Generated citations")
    tasks: List[AgentTask] = Field(default_factory=list, description="Agent tasks")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Question(BaseModel):
    """Model for a research question."""
    
    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The question being asked")
    context_type: str = Field(default="general", description="Type of context (paper, summary, session)")
    context_ids: List[str] = Field(default_factory=list, description="IDs of relevant context items")
    user_id: Optional[str] = Field(None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Answer(BaseModel):
    """Model for a research answer."""
    
    answer_id: str = Field(..., description="Unique answer identifier")
    question_id: str = Field(..., description="ID of the associated question")
    answer_text: str = Field(..., description="The generated answer")
    confidence_score: float = Field(default=0.0, description="Confidence score (0-1)")
    source_papers: List[str] = Field(default_factory=list, description="IDs of papers used for the answer")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence from papers")
    agent_reasoning: Optional[str] = Field(None, description="Agent's reasoning process")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationContext(BaseModel):
    """Model for conversation context in Q&A sessions."""
    
    context_id: str = Field(..., description="Unique context identifier")
    session_id: str = Field(..., description="Associated research session")
    questions: List[Question] = Field(default_factory=list, description="Questions in this context")
    answers: List[Answer] = Field(default_factory=list, description="Answers in this context")
    papers_in_context: List[str] = Field(default_factory=list, description="Paper IDs available for Q&A")
    summaries_in_context: List[str] = Field(default_factory=list, description="Summary IDs available for Q&A")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
