"""
Multi-Agent Research Coordinator using CrewAI.

This module coordinates all individual agents (Retriever, Summarizer, Q&A, Citation)
using CrewAI's multi-agent framework for complex research workflows.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

from ..models import (
    ResearchSession, ResearchQuery, AgentTask,
    ResearchPaper, Summary, Citation, Question, Answer
)
from ..tools.ollama_client import OllamaClient
from ..utils.logger import logger
from .retriever_agent import RetrieverAgent
from .summarizer_agent import SummarizerAgent
from .qa_agent import QAAgent
from .citation_agent import CitationAgent


class ResearchCoordinator:
    """
    Coordinates multiple agents using CrewAI for comprehensive research workflows.
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """Initialize the research coordinator."""
        self.ollama_client = ollama_client or OllamaClient()
        
        # Initialize individual agents
        self.retriever_agent = RetrieverAgent()
        self.summarizer_agent = SummarizerAgent(self.ollama_client)
        self.qa_agent = QAAgent(self.ollama_client)
        self.citation_agent = CitationAgent(self.ollama_client)
        
        # Initialize CrewAI LLM
        self.llm = Ollama(model="mistral:7b", base_url="http://localhost:11434")
        
        # Define CrewAI agents
        self._setup_crewai_agents()
        
        logger.info("Research Coordinator initialized with CrewAI integration")
    
    def _setup_crewai_agents(self):
        """Set up CrewAI agents for coordination."""
        
        # Research Retriever Agent
        self.crew_retriever = Agent(
            role='Research Paper Retriever',
            goal='Find and retrieve relevant academic papers from multiple sources',
            backstory="""You are a specialized research assistant focused on finding 
            high-quality academic papers. You excel at searching ArXiv, academic 
            databases, and web sources to find papers most relevant to research queries.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Research Summarizer Agent
        self.crew_summarizer = Agent(
            role='Research Paper Summarizer',
            goal='Create comprehensive and insightful summaries of academic papers',
            backstory="""You are an expert at reading and understanding academic papers. 
            You can quickly identify key findings, methodologies, and important insights 
            from research papers and present them in clear, concise summaries.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Research Q&A Agent
        self.crew_qa = Agent(
            role='Research Question Answerer',
            goal='Answer questions about research papers and findings with high accuracy',
            backstory="""You are a research analyst who specializes in answering 
            questions about academic papers. You can analyze paper content, extract 
            relevant information, and provide well-reasoned answers with proper citations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Citation Specialist Agent
        self.crew_citation = Agent(
            role='Citation Specialist',
            goal='Generate properly formatted academic citations and bibliographies',
            backstory="""You are an academic writing expert who specializes in creating 
            proper citations and bibliographies. You know all major citation formats 
            and can ensure academic integrity in research documentation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Research Coordinator Agent
        self.crew_coordinator = Agent(
            role='Research Project Coordinator',
            goal='Orchestrate the entire research workflow and ensure quality outputs',
            backstory="""You are a senior research coordinator who oversees complex 
            research projects. You ensure that all aspects of research - from paper 
            retrieval to final citations - work together seamlessly to provide 
            comprehensive research assistance.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
    
    async def conduct_research(self, query: str, user_id: Optional[str] = None) -> ResearchSession:
        """
        Conduct a complete research workflow using CrewAI coordination.
        
        Args:
            query: Research query/topic
            user_id: Optional user identifier
            
        Returns:
            ResearchSession with complete research results
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Starting research session {session_id} for query: {query}")
        
        # Create research query and session
        research_query = ResearchQuery(query=query, user_id=user_id)
        session = ResearchSession(
            session_id=session_id,
            query=research_query
        )
        
        try:
            # Define research tasks
            tasks = self._create_research_tasks(query, session_id)
            
            # Create and run the crew
            crew = Crew(
                agents=[
                    self.crew_retriever,
                    self.crew_summarizer,
                    self.crew_qa,
                    self.crew_citation,
                    self.crew_coordinator
                ],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the research workflow
            result = crew.kickoff()
            
            # Process results and update session
            await self._process_crew_results(session, result)
            
            logger.info(f"Research session {session_id} completed successfully")
            return session
            
        except Exception as e:
            logger.error(f"Error in research session {session_id}: {str(e)}")
            # Add error task to session
            error_task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type="coordinator",
                status="failed",
                error_message=str(e)
            )
            session.tasks.append(error_task)
            return session
    
    def _create_research_tasks(self, query: str, session_id: str) -> List[Task]:
        """Create CrewAI tasks for the research workflow."""
        
        # Task 1: Retrieve papers
        retrieve_task = Task(
            description=f"""Search for academic papers related to: "{query}"
            
            Find relevant papers from ArXiv and other academic sources. Focus on:
            1. Recent and highly-cited papers
            2. Papers from reputable sources
            3. Papers that directly address the research topic
            4. Diverse perspectives on the topic
            
            Return a list of papers with titles, authors, abstracts, and URLs.""",
            agent=self.crew_retriever,
            expected_output="List of relevant academic papers with metadata"
        )
        
        # Task 2: Summarize papers
        summarize_task = Task(
            description="""Create comprehensive summaries of the retrieved papers.
            
            For each paper, provide:
            1. A general summary of the main contributions
            2. Key findings and results
            3. Methodology used
            4. Significance and implications
            
            Focus on extracting actionable insights and important discoveries.""",
            agent=self.crew_summarizer,
            expected_output="Detailed summaries of all retrieved papers",
            dependencies=[retrieve_task]
        )
        
        # Task 3: Generate citations
        citation_task = Task(
            description="""Generate proper academic citations for all retrieved papers.
            
            Create citations in multiple formats:
            1. BibTeX format for LaTeX documents
            2. APA format for academic writing
            3. MLA format if requested
            
            Ensure all citations are properly formatted and complete.""",
            agent=self.crew_citation,
            expected_output="Properly formatted citations for all papers",
            dependencies=[retrieve_task]
        )
        
        # Task 4: Prepare Q&A context
        qa_prep_task = Task(
            description="""Prepare the research context for question answering.
            
            Analyze all retrieved papers and summaries to:
            1. Identify key themes and topics
            2. Extract important facts and findings
            3. Note potential areas of contradiction or debate
            4. Prepare context for answering follow-up questions
            
            This will enable accurate answers to user questions about the research.""",
            agent=self.crew_qa,
            expected_output="Prepared Q&A context with key research insights",
            dependencies=[retrieve_task, summarize_task]
        )
        
        # Task 5: Coordinate and finalize
        coordinate_task = Task(
            description=f"""Coordinate the complete research workflow for: "{query}"
            
            Review all outputs from the team:
            1. Verify paper retrieval quality and relevance
            2. Ensure summaries capture key insights
            3. Validate citation accuracy and completeness
            4. Prepare final research package
            
            Provide a cohesive overview of the research findings and next steps.""",
            agent=self.crew_coordinator,
            expected_output="Final coordinated research report with all components",
            dependencies=[retrieve_task, summarize_task, citation_task, qa_prep_task]
        )
        
        return [retrieve_task, summarize_task, citation_task, qa_prep_task, coordinate_task]
    
    async def _process_crew_results(self, session: ResearchSession, result: Any):
        """Process CrewAI results and update the research session."""
        try:
            # Execute individual agent tasks and populate session
            
            # 1. Retrieve papers
            retrieve_task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type="retriever",
                status="running"
            )
            session.tasks.append(retrieve_task)
            
            papers = await self.retriever_agent.search_papers(session.query.query)
            session.papers.extend(papers)
            
            retrieve_task.status = "completed"
            retrieve_task.completed_at = datetime.now()
            retrieve_task.output_data = {"papers_count": len(papers)}
            
            # 2. Generate summaries
            summarize_task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type="summarizer",
                status="running"
            )
            session.tasks.append(summarize_task)
            
            for paper in papers:
                if paper.abstract:
                    summary = await self.summarizer_agent.summarize_paper(paper)
                    if summary:
                        session.summaries.append(summary)
            
            summarize_task.status = "completed"
            summarize_task.completed_at = datetime.now()
            summarize_task.output_data = {"summaries_count": len(session.summaries)}
            
            # 3. Generate citations
            citation_task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type="citation",
                status="running"
            )
            session.tasks.append(citation_task)
            
            for paper in papers:
                citations = await self.citation_agent.generate_citations_for_paper(paper)
                session.citations.extend(citations)
            
            citation_task.status = "completed"
            citation_task.completed_at = datetime.now()
            citation_task.output_data = {"citations_count": len(session.citations)}
            
            # Update session timestamp
            session.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing crew results: {str(e)}")
            raise
    
    async def answer_question(self, session: ResearchSession, question: str) -> Answer:
        """
        Answer a question about the research using the Q&A agent.
        
        Args:
            session: Research session with context
            question: Question to answer
            
        Returns:
            Answer object with response and metadata
        """
        try:
            # Create question object
            question_obj = Question(
                question_id=str(uuid.uuid4()),
                question_text=question,
                context_ids=[p.id for p in session.papers]
            )
            
            # Prepare context for Q&A
            papers_context = {p.id: p for p in session.papers}
            summaries_context = {s.paper_id: s for s in session.summaries}
            
            # Get answer from Q&A agent
            answer = await self.qa_agent.answer_question(
                question_obj, papers_context, summaries_context
            )
            
            logger.info(f"Answered question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            # Return error answer
            return Answer(
                answer_id=str(uuid.uuid4()),
                question_id=str(uuid.uuid4()),
                answer_text=f"I encountered an error while processing your question: {str(e)}",
                confidence_score=0.0
            )
    
    async def add_citations(self, session: ResearchSession, format_type: str = "bibtex") -> List[Citation]:
        """
        Add citations for all papers in the session.
        
        Args:
            session: Research session
            format_type: Citation format (bibtex, apa, mla, ieee)
            
        Returns:
            List of generated citations
        """
        try:
            new_citations = []
            
            for paper in session.papers:
                citations = await self.citation_agent.generate_citations_for_paper(
                    paper, [format_type]
                )
                new_citations.extend(citations)
            
            session.citations.extend(new_citations)
            session.updated_at = datetime.now()
            
            logger.info(f"Added {len(new_citations)} citations in {format_type} format")
            return new_citations
            
        except Exception as e:
            logger.error(f"Error adding citations: {str(e)}")
            return []
    
    def get_session_summary(self, session: ResearchSession) -> Dict[str, Any]:
        """
        Get a summary of the research session.
        
        Args:
            session: Research session
            
        Returns:
            Dictionary with session summary
        """
        return {
            "session_id": session.session_id,
            "query": session.query.query,
            "papers_found": len(session.papers),
            "summaries_generated": len(session.summaries),
            "citations_available": len(session.citations),
            "tasks_completed": len([t for t in session.tasks if t.status == "completed"]),
            "tasks_failed": len([t for t in session.tasks if t.status == "failed"]),
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }
