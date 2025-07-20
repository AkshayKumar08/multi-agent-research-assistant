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
