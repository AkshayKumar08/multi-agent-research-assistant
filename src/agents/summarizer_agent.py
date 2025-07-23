"""
Summarizer Agent for generating paper summaries using LLM.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models import ResearchPaper, Summary, AgentTask
from src.utils.logger import get_logger
from config.settings import config

# Import appropriate LLM client based on configuration
import os
if os.getenv("LLM_PROVIDER", "huggingface") == "huggingface" or \
   os.getenv("STREAMLIT_SHARING", "false").lower() == "true":
    from src.tools.huggingface_client import HuggingFaceClient as LLMClient
else:
    from src.tools.ollama_client import OllamaClient as LLMClient

logger = get_logger("summarizer_agent")


class SummarizerAgent:
    """Agent responsible for summarizing research papers using LLM."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the Summarizer Agent.
        
        Args:
            llm_client: Optional LLM client (creates new one if None)
        """
        self.llm_client = llm_client or LLMClient()
        self.agent_id = "summarizer_agent"
        
        # Verify LLM availability
        provider = os.getenv("LLM_PROVIDER", "huggingface")
        if hasattr(self.llm_client, 'is_available') and not self.llm_client.is_available():
            logger.warning(f"{provider} client not available. Summarization will use fallback responses.")
        else:
            logger.info(f"Summarizer Agent initialized with {provider} provider")
        
        logger.info("Summarizer Agent initialized")
    
    def summarize_paper(
        self,
        paper: ResearchPaper,
        summary_type: str = "general",
        max_length: int = 300
    ) -> Summary:
        """Summarize a single research paper.
        
        Args:
            paper: ResearchPaper to summarize
            summary_type: Type of summary ("general", "technical", "methodology", "findings")
            max_length: Maximum length of summary in words
            
        Returns:
            Summary object
        """
        logger.info(f"Summarizing paper: {paper.title[:50]}...")
        
        try:
            # Create appropriate prompt based on summary type
            prompt = self._create_summary_prompt(paper, summary_type, max_length)
            
            # Generate summary using LLM
            summary_text = self.llm_client.generate(
                prompt, max_tokens=max_length * 2, task_type="summarize"
            )
            
            # Extract key findings
            key_findings = self._extract_key_findings(paper, summary_text)
            
            # Extract methodology if available
            methodology = self._extract_methodology(paper, summary_text)
            
            # Create Summary object
            summary = Summary(
                paper_id=paper.id,
                summary=summary_text.strip(),
                key_findings=key_findings,
                methodology=methodology,
                agent_id=self.agent_id
            )
            
            logger.info(f"Generated summary ({len(summary_text)} chars) for paper {paper.id}")
            return summary
            
        except Exception as e:
            error_msg = f"Failed to summarize paper {paper.id}: {str(e)}"
            logger.error(error_msg)
            
            # Return empty summary with error indication
            return Summary(
                paper_id=paper.id,
                summary=f"Summary generation failed: {str(e)}",
                key_findings=[],
                methodology=None,
                agent_id=self.agent_id
            )
    
    def summarize_papers(
        self,
        papers: List[ResearchPaper],
        summary_type: str = "general",
        max_length: int = 300
    ) -> List[Summary]:
        """Summarize multiple research papers.
        
        Args:
            papers: List of ResearchPaper objects
            summary_type: Type of summary for all papers
            max_length: Maximum length per summary
            
        Returns:
            List of Summary objects
        """
        logger.info(f"Summarizing {len(papers)} papers")
        
        summaries = []
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}")
            
            try:
                summary = self.summarize_paper(paper, summary_type, max_length)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to summarize paper {i+1}: {str(e)}")
                continue
        
        logger.info(f"Generated {len(summaries)} summaries out of {len(papers)} papers")
        return summaries
    
    def generate_comparative_summary(
        self,
        papers: List[ResearchPaper],
        focus_area: Optional[str] = None
    ) -> str:
        """Generate a comparative summary across multiple papers.
        
        Args:
            papers: List of papers to compare
            focus_area: Specific area to focus comparison on
            
        Returns:
            Comparative summary text
        """
        logger.info(f"Generating comparative summary for {len(papers)} papers")
        
        if not papers:
            return "No papers provided for comparison."
        
        try:
            # Create comparative prompt
            prompt = self._create_comparative_prompt(papers, focus_area)
            
            # Generate comparative summary
            comparative_summary = self.llm_client.generate(
                prompt, max_tokens=800, task_type="summarize"
            )
            
            logger.info(f"Generated comparative summary ({len(comparative_summary)} chars)")
            return comparative_summary.strip()
            
        except Exception as e:
            error_msg = f"Failed to generate comparative summary: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def execute_task(self, task: AgentTask) -> AgentTask:
        """Execute a summarization task.
        
        Args:
            task: Agent task with summarization parameters
            
        Returns:
            Updated task with results
        """
        logger.info(f"Executing summarization task: {task.task_id}")
        
        try:
            task.status = "running"
            
            # Extract parameters from task input
            papers_data = task.input_data.get("papers", [])
            summary_type = task.input_data.get("summary_type", "general")
            max_length = task.input_data.get("max_length", 300)
            comparative = task.input_data.get("comparative", False)
            
            # Convert papers data to ResearchPaper objects
            papers = []
            for paper_data in papers_data:
                if isinstance(paper_data, dict):
                    paper = ResearchPaper(**paper_data)
                    papers.append(paper)
            
            if not papers:
                raise ValueError("No valid papers provided for summarization")
            
            results = {}
            
            # Generate individual summaries
            summaries = self.summarize_papers(papers, summary_type, max_length)
            results["summaries"] = [summary.model_dump() for summary in summaries]
            
            # Generate comparative summary if requested
            if comparative and len(papers) > 1:
                focus_area = task.input_data.get("focus_area")
                comp_summary = self.generate_comparative_summary(papers, focus_area)
                results["comparative_summary"] = comp_summary
            
            # Update task with results
            task.output_data = results
            task.status = "completed"
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)
            
            task.status = "failed"
            task.error_message = error_msg
            task.completed_at = datetime.now()
        
        return task
    
    def _create_summary_prompt(
        self, 
        paper: ResearchPaper, 
        summary_type: str, 
        max_length: int
    ) -> str:
        """Create prompt for paper summarization.
        
        Args:
            paper: Paper to summarize
            summary_type: Type of summary
            max_length: Maximum length in words
            
        Returns:
            Formatted prompt string
        """
        # Base paper information
        paper_info = f"""
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Abstract: {paper.abstract}
Source: {paper.source}
"""
        
        # Type-specific instructions
        type_instructions = {
            "general": "Provide a clear, accessible summary that explains the main contributions and significance.",
            "technical": "Focus on technical details, methodologies, algorithms, and implementation specifics.",
            "methodology": "Emphasize the research methods, experimental design, and analytical approaches used.",
            "findings": "Highlight the key results, conclusions, and implications of the research."
        }
        
        instruction = type_instructions.get(summary_type, type_instructions["general"])
        
        prompt = f"""Please summarize this research paper. {instruction}

{paper_info}

Requirements:
- Maximum {max_length} words
- Clear and concise language
- Focus on key contributions
- Maintain scientific accuracy

Summary:"""
        
        return prompt
    
    def _get_system_prompt(self, summary_type: str) -> str:
        """Get system prompt for summarization.
        
        Args:
            summary_type: Type of summary
            
        Returns:
            System prompt string
        """
        base_prompt = """You are an expert research assistant specializing in academic paper analysis and summarization. Your goal is to create clear, accurate, and insightful summaries that capture the essence of research papers."""
        
        type_specific = {
            "general": " Focus on making the research accessible to a broad academic audience.",
            "technical": " Emphasize technical depth and implementation details for expert readers.",
            "methodology": " Concentrate on research design, methods, and analytical approaches.",
            "findings": " Highlight results, conclusions, and their broader implications."
        }
        
        return base_prompt + type_specific.get(summary_type, "")
    
    def _create_comparative_prompt(
        self, 
        papers: List[ResearchPaper], 
        focus_area: Optional[str]
    ) -> str:
        """Create prompt for comparative analysis.
        
        Args:
            papers: Papers to compare
            focus_area: Specific focus area
            
        Returns:
            Comparative prompt string
        """
        papers_info = ""
        for i, paper in enumerate(papers[:5], 1):  # Limit to first 5 papers
            papers_info += f"""
Paper {i}:
Title: {paper.title}
Authors: {', '.join(paper.authors[:3])}  # Limit authors
Abstract: {paper.abstract[:500]}...
"""
        
        focus_instruction = f" Pay special attention to {focus_area}." if focus_area else ""
        
        prompt = f"""Compare and analyze these research papers.{focus_instruction}

{papers_info}

Please provide:
1. Common themes and approaches
2. Key differences in methodology or findings
3. Complementary insights
4. Research gaps or future directions
5. Overall synthesis of the field

Analysis:"""
        
        return prompt
    
    def _get_comparative_system_prompt(self) -> str:
        """Get system prompt for comparative analysis."""
        return """You are an expert research analyst specializing in comparative analysis of academic papers. Your task is to identify patterns, relationships, and insights across multiple research works, providing a synthesized view of the research landscape."""
    
    def _extract_key_findings(
        self, 
        paper: ResearchPaper, 
        summary_text: str
    ) -> List[str]:
        """Extract key findings from paper and summary.
        
        Args:
            paper: Original paper
            summary_text: Generated summary
            
        Returns:
            List of key findings
        """
        try:
            # Use LLM to extract key findings
            prompt = f"""From this paper summary, extract 3-5 key findings as bullet points:

Summary: {summary_text}

Original Abstract: {paper.abstract}

Key findings (one per line, start each with '- '):"""
            
            findings_text = self.llm_client.generate(
                prompt, max_tokens=200, task_type="summarize"
            )
            
            # Parse findings into list
            findings = []
            for line in findings_text.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    findings.append(line[1:].strip())
                elif line and not line.startswith('Key') and not line.startswith('Finding'):
                    findings.append(line)
            
            return findings[:5]  # Limit to 5 findings
            
        except Exception as e:
            logger.warning(f"Failed to extract key findings: {str(e)}")
            return []
    
    def _extract_methodology(
        self, 
        paper: ResearchPaper, 
        summary_text: str
    ) -> Optional[str]:
        """Extract methodology information.
        
        Args:
            paper: Original paper
            summary_text: Generated summary
            
        Returns:
            Methodology description or None
        """
        try:
            # Simple methodology extraction
            if any(word in paper.abstract.lower() for word in ['method', 'approach', 'algorithm', 'technique']):
                prompt = f"""Extract the methodology/approach used in this research in 1-2 sentences:

Summary: {summary_text}
Abstract: {paper.abstract}

Methodology:"""
                
                methodology = self.llm_client.generate(
                    prompt, max_tokens=100, task_type="summarize"
                )
                
                return methodology.strip() if methodology.strip() else None
                
        except Exception as e:
            logger.warning(f"Failed to extract methodology: {str(e)}")
            
        return None
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information.
        
        Returns:
            Dictionary with agent details
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": "summarizer",
            "llm_provider": os.getenv("LLM_PROVIDER", "huggingface"),
            "llm_available": hasattr(self.llm_client, 'is_available') and self.llm_client.is_available(),
            "supported_summary_types": ["general", "technical", "methodology", "findings"],
            "features": ["individual_summaries", "comparative_analysis", "key_findings_extraction"]
        }
