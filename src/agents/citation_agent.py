"""
Citation Agent for generating academic citations in various formats.
"""
import uuid
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from src.models import (
    ResearchPaper, Citation, CitationRequest, Bibliography, 
    AgentTask, ResearchSession
)
from src.utils.logger import get_logger
from config.settings import config

# Import appropriate LLM client based on configuration
import os
if os.getenv("LLM_PROVIDER", "huggingface") == "huggingface" or \
   os.getenv("STREAMLIT_SHARING", "false").lower() == "true":
    from src.tools.huggingface_client import HuggingFaceClient as LLMClient
else:
    from src.tools.ollama_client import OllamaClient as LLMClient

logger = get_logger("citation_agent")


class CitationAgent:
    """Agent responsible for generating academic citations in various formats."""
    
    # Supported citation formats
    SUPPORTED_FORMATS = [
        "bibtex", "apa", "mla", "ieee", "chicago", "harvard", 
        "vancouver", "nature", "science", "cell"
    ]
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the Citation Agent.
        
        Args:
            llm_client: Optional LLM client (creates new one if None)
        """
        self.llm_client = llm_client or LLMClient()
        self.agent_id = "citation_agent"
        
        # Verify LLM availability
        provider = os.getenv("LLM_PROVIDER", "huggingface")
        if hasattr(self.llm_client, 'is_available') and not self.llm_client.is_available():
            logger.warning(f"{provider} client not available. Citations will use fallback formatting.")
        else:
            logger.info(f"Citation Agent initialized with {provider} provider")
        
        logger.info("Citation Agent initialized")
    
    def generate_citation(
        self,
        paper: ResearchPaper,
        citation_format: str = "bibtex",
        style_options: Dict[str, Any] = None
    ) -> Citation:
        """Generate a citation for a single research paper.
        
        Args:
            paper: ResearchPaper to cite
            citation_format: Citation format (bibtex, apa, mla, etc.)
            style_options: Additional formatting options
            
        Returns:
            Citation object with formatted citation
        """
        logger.info(f"Generating {citation_format} citation for paper: {paper.title[:50]}...")
        
        try:
            citation_format = citation_format.lower()
            if citation_format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported citation format: {citation_format}")
            
            style_options = style_options or {}
            
            # Extract raw citation data
            raw_data = self._extract_citation_data(paper)
            
            # Generate formatted citation
            if citation_format == "bibtex":
                citation_text = self._generate_bibtex(paper, raw_data, style_options)
            elif citation_format == "apa":
                citation_text = self._generate_apa(paper, raw_data, style_options)
            elif citation_format == "mla":
                citation_text = self._generate_mla(paper, raw_data, style_options)
            elif citation_format == "ieee":
                citation_text = self._generate_ieee(paper, raw_data, style_options)
            else:
                # Use LLM for other formats
                citation_text = self._generate_with_llm(paper, citation_format, raw_data, style_options)
            
            # Create citation object
            citation = Citation(
                citation_id=str(uuid.uuid4()),
                paper_id=paper.id,
                citation_format=citation_format,
                citation_text=citation_text,
                raw_data=raw_data,
                agent_id=self.agent_id,
                validation_status="generated"
            )
            
            # Validate citation
            self._validate_citation(citation, paper)
            
            logger.info(f"Citation generated successfully: {citation_format}")
            return citation
            
        except Exception as e:
            logger.error(f"Error generating citation: {str(e)}")
            # Return fallback citation
            return self._create_fallback_citation(paper, citation_format, str(e))
    
    def generate_multiple_citations(
        self,
        papers: List[ResearchPaper],
        citation_format: str = "bibtex",
        style_options: Dict[str, Any] = None
    ) -> List[Citation]:
        """Generate citations for multiple papers.
        
        Args:
            papers: List of ResearchPaper objects to cite
            citation_format: Citation format for all papers
            style_options: Additional formatting options
            
        Returns:
            List of Citation objects
        """
        logger.info(f"Generating {len(papers)} citations in {citation_format} format")
        
        citations = []
        for paper in papers:
            try:
                citation = self.generate_citation(paper, citation_format, style_options)
                citations.append(citation)
            except Exception as e:
                logger.error(f"Failed to generate citation for paper {paper.id}: {str(e)}")
                fallback = self._create_fallback_citation(paper, citation_format, str(e))
                citations.append(fallback)
        
        logger.info(f"Generated {len(citations)} citations successfully")
        return citations
    
    def create_bibliography(
        self,
        papers: List[ResearchPaper],
        title: str,
        format_style: str = "apa",
        style_options: Dict[str, Any] = None
    ) -> Bibliography:
        """Create a bibliography from multiple papers.
        
        Args:
            papers: List of papers to include
            title: Bibliography title
            format_style: Citation format for the bibliography
            style_options: Additional formatting options
            
        Returns:
            Bibliography object with all citations
        """
        logger.info(f"Creating bibliography '{title}' with {len(papers)} papers")
        
        # Generate all citations
        citations = self.generate_multiple_citations(papers, format_style, style_options)
        
        # Sort citations alphabetically by first author
        citations.sort(key=lambda c: self._get_sort_key(c))
        
        # Create bibliography
        bibliography = Bibliography(
            bibliography_id=str(uuid.uuid4()),
            title=title,
            citations=citations,
            format_style=format_style,
            metadata={
                "total_papers": len(papers),
                "generation_date": datetime.now().isoformat(),
                "style_options": style_options or {}
            }
        )
        
        logger.info(f"Bibliography created: {title}")
        return bibliography
    
    def _extract_citation_data(self, paper: ResearchPaper) -> Dict[str, Any]:
        """Extract structured data for citation generation.
        
        Args:
            paper: ResearchPaper object
            
        Returns:
            Dictionary with citation data
        """
        # Parse publication year from various sources
        pub_year = None
        if paper.published_date:
            pub_year = paper.published_date.year
        else:
            # Try to extract year from title or abstract
            year_match = re.search(r'\b(19|20)\d{2}\b', paper.title + " " + paper.abstract)
            if year_match:
                pub_year = int(year_match.group())
        
        # Clean and format authors
        authors = [author.strip() for author in paper.authors if author.strip()]
        
        # Extract venue/journal from source or abstract
        venue = self._extract_venue(paper)
        
        # Determine publication type
        pub_type = self._determine_publication_type(paper)
        
        return {
            "title": paper.title.strip(),
            "authors": authors,
            "year": pub_year,
            "venue": venue,
            "url": paper.url,
            "doi": paper.doi,
            "source": paper.source,
            "categories": paper.categories,
            "publication_type": pub_type,
            "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
        }
    
    def _extract_venue(self, paper: ResearchPaper) -> Optional[str]:
        """Extract publication venue from paper metadata."""
        # For ArXiv papers, use the category
        if paper.source == "arxiv" and paper.categories:
            return f"arXiv preprint arXiv:{paper.id}"
        
        # Try to extract from abstract
        venue_patterns = [
            r"published in ([^.]+)",
            r"appeared in ([^.]+)",
            r"in proceedings of ([^.]+)",
            r"journal of ([^.]+)"
        ]
        
        for pattern in venue_patterns:
            match = re.search(pattern, paper.abstract.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _determine_publication_type(self, paper: ResearchPaper) -> str:
        """Determine the type of publication."""
        if paper.source == "arxiv":
            return "preprint"
        elif "conference" in paper.abstract.lower():
            return "conference"
        elif "journal" in paper.abstract.lower():
            return "journal"
        else:
            return "article"
    
    def _generate_bibtex(
        self, 
        paper: ResearchPaper, 
        raw_data: Dict[str, Any], 
        style_options: Dict[str, Any]
    ) -> str:
        """Generate BibTeX citation."""
        # Create BibTeX key
        first_author = raw_data["authors"][0].split()[-1] if raw_data["authors"] else "Unknown"
        year = raw_data["year"] or "n.d."
        key = f"{first_author.lower()}{year}"
        
        # Remove special characters from key
        key = re.sub(r'[^a-zA-Z0-9]', '', key)
        
        # Determine entry type
        entry_type = "article"
        if raw_data["publication_type"] == "preprint":
            entry_type = "misc"
        elif raw_data["publication_type"] == "conference":
            entry_type = "inproceedings"
        
        # Format authors
        author_str = " and ".join(raw_data["authors"]) if raw_data["authors"] else "Unknown"
        
        # Build BibTeX entry
        bibtex_lines = [f"@{entry_type}{{{key},"]
        bibtex_lines.append(f'  title={{{raw_data["title"]}}},')
        bibtex_lines.append(f'  author={{{author_str}}},')
        
        if raw_data["year"]:
            bibtex_lines.append(f'  year={{{raw_data["year"]}}},')
        
        if raw_data["venue"]:
            if entry_type == "article":
                bibtex_lines.append(f'  journal={{{raw_data["venue"]}}},')
            elif entry_type == "inproceedings":
                bibtex_lines.append(f'  booktitle={{{raw_data["venue"]}}},')
            else:
                bibtex_lines.append(f'  howpublished={{{raw_data["venue"]}}},')
        
        if raw_data["url"]:
            bibtex_lines.append(f'  url={{{raw_data["url"]}}},')
        
        if raw_data["doi"]:
            bibtex_lines.append(f'  doi={{{raw_data["doi"]}}},')
        
        bibtex_lines.append("}")
        
        return "\n".join(bibtex_lines)
    
    def _generate_apa(
        self, 
        paper: ResearchPaper, 
        raw_data: Dict[str, Any], 
        style_options: Dict[str, Any]
    ) -> str:
        """Generate APA format citation."""
        parts = []
        
        # Authors
        if raw_data["authors"]:
            if len(raw_data["authors"]) == 1:
                parts.append(self._format_apa_author(raw_data["authors"][0]))
            elif len(raw_data["authors"]) <= 7:
                formatted_authors = [self._format_apa_author(author) for author in raw_data["authors"]]
                if len(formatted_authors) > 1:
                    parts.append(", ".join(formatted_authors[:-1]) + ", & " + formatted_authors[-1])
                else:
                    parts.append(formatted_authors[0])
            else:
                # More than 7 authors
                formatted_authors = [self._format_apa_author(author) for author in raw_data["authors"][:6]]
                parts.append(", ".join(formatted_authors) + ", ... " + self._format_apa_author(raw_data["authors"][-1]))
        else:
            parts.append("Unknown Author")
        
        # Year
        year_part = f"({raw_data['year']})" if raw_data["year"] else "(n.d.)"
        parts.append(year_part)
        
        # Title
        title = raw_data["title"]
        if not title.endswith('.'):
            title += "."
        parts.append(title)
        
        # Venue/Journal
        if raw_data["venue"]:
            parts.append(f"*{raw_data['venue']}*")
        
        # URL
        if raw_data["url"]:
            parts.append(f"Retrieved from {raw_data['url']}")
        
        return " ".join(parts)
    
    def _format_apa_author(self, author: str) -> str:
        """Format author name for APA style."""
        name_parts = author.strip().split()
        if len(name_parts) >= 2:
            last_name = name_parts[-1]
            initials = ". ".join([name[0] for name in name_parts[:-1] if name]) + "."
            return f"{last_name}, {initials}"
        else:
            return author.strip()
    
    def _generate_mla(
        self, 
        paper: ResearchPaper, 
        raw_data: Dict[str, Any], 
        style_options: Dict[str, Any]
    ) -> str:
        """Generate MLA format citation."""
        parts = []
        
        # First author (Last, First)
        if raw_data["authors"]:
            first_author = raw_data["authors"][0]
            name_parts = first_author.strip().split()
            if len(name_parts) >= 2:
                first_author_formatted = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
            else:
                first_author_formatted = first_author
            parts.append(first_author_formatted)
            
            # Additional authors
            if len(raw_data["authors"]) > 1:
                other_authors = raw_data["authors"][1:]
                if len(other_authors) == 1:
                    parts.append(f"and {other_authors[0]}")
                else:
                    parts.append("et al.")
        
        # Title in quotes
        title = f'"{raw_data["title"]}"'
        if not title.endswith('."'):
            title = title[:-1] + '."'
        parts.append(title)
        
        # Venue in italics
        if raw_data["venue"]:
            parts.append(f"*{raw_data['venue']}*,")
        
        # Date
        if raw_data["year"]:
            parts.append(f"{raw_data['year']},")
        
        # URL
        if raw_data["url"]:
            parts.append(f"{raw_data['url']}.")
        
        return " ".join(parts)
    
    def _generate_ieee(
        self, 
        paper: ResearchPaper, 
        raw_data: Dict[str, Any], 
        style_options: Dict[str, Any]
    ) -> str:
        """Generate IEEE format citation."""
        parts = []
        
        # Authors
        if raw_data["authors"]:
            if len(raw_data["authors"]) <= 3:
                formatted_authors = []
                for author in raw_data["authors"]:
                    name_parts = author.strip().split()
                    if len(name_parts) >= 2:
                        # IEEE format: First Initial. Last Name
                        initials = ". ".join([name[0] for name in name_parts[:-1]]) + "."
                        formatted_authors.append(f"{initials} {name_parts[-1]}")
                    else:
                        formatted_authors.append(author)
                parts.append(", ".join(formatted_authors))
            else:
                # More than 3 authors, use et al.
                first_author = raw_data["authors"][0]
                name_parts = first_author.strip().split()
                if len(name_parts) >= 2:
                    initials = ". ".join([name[0] for name in name_parts[:-1]]) + "."
                    formatted_first = f"{initials} {name_parts[-1]}"
                else:
                    formatted_first = first_author
                parts.append(f"{formatted_first} et al.")
        
        # Title in quotes
        title = f'"{raw_data["title"]},"'
        parts.append(title)
        
        # Venue in italics
        if raw_data["venue"]:
            parts.append(f"*{raw_data['venue']}*,")
        
        # Year
        if raw_data["year"]:
            parts.append(f"{raw_data['year']}.")
        
        return " ".join(parts)
    
    def _generate_with_llm(
        self, 
        paper: ResearchPaper, 
        citation_format: str, 
        raw_data: Dict[str, Any], 
        style_options: Dict[str, Any]
    ) -> str:
        """Generate citation using LLM for unsupported formats."""
        if not self.llm_client.is_available():
            return self._create_fallback_citation_text(paper, citation_format)
        
        # Create prompt for LLM
        prompt = f"""Generate a proper academic citation in {citation_format.upper()} format for the following research paper:

Title: {raw_data['title']}
Authors: {', '.join(raw_data['authors']) if raw_data['authors'] else 'Unknown'}
Year: {raw_data['year'] or 'Unknown'}
Venue/Journal: {raw_data['venue'] or 'Unknown'}
URL: {raw_data['url'] or 'Unknown'}
DOI: {raw_data['doi'] or 'Unknown'}
Publication Type: {raw_data['publication_type']}

Please provide only the formatted citation text according to {citation_format.upper()} style guidelines. Do not include any explanations or additional text."""

        try:
            # Check if using HuggingFace client (doesn't support temperature)
            if hasattr(self.llm_client, 'api_url'):  # HuggingFace client
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=300,
                    task_type="general"
                )
            else:  # Ollama client
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.1  # Low temperature for consistent formatting
                )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM citation generation failed: {str(e)}")
            return self._create_fallback_citation_text(paper, citation_format)
    
    def _validate_citation(self, citation: Citation, paper: ResearchPaper):
        """Validate generated citation."""
        issues = []
        
        # Check if title is included
        if paper.title.lower() not in citation.citation_text.lower():
            issues.append("Title not found in citation")
        
        # Check if at least one author is included (if authors exist)
        if paper.authors:
            author_found = any(
                author.split()[-1].lower() in citation.citation_text.lower() 
                for author in paper.authors if author.strip()
            )
            if not author_found:
                issues.append("No authors found in citation")
        
        # Update validation status
        if issues:
            citation.validation_status = f"warning: {'; '.join(issues)}"
            logger.warning(f"Citation validation issues: {issues}")
        else:
            citation.validation_status = "validated"
    
    def _create_fallback_citation(
        self, 
        paper: ResearchPaper, 
        citation_format: str, 
        error_msg: str
    ) -> Citation:
        """Create a fallback citation when generation fails."""
        fallback_text = self._create_fallback_citation_text(paper, citation_format)
        
        return Citation(
            citation_id=str(uuid.uuid4()),
            paper_id=paper.id,
            citation_format=citation_format,
            citation_text=fallback_text,
            raw_data={"error": error_msg},
            agent_id=self.agent_id,
            validation_status=f"fallback: {error_msg}"
        )
    
    def _create_fallback_citation_text(self, paper: ResearchPaper, citation_format: str) -> str:
        """Create basic fallback citation text."""
        authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown Author"
        if len(paper.authors) > 3:
            authors += " et al."
        
        year = ""
        if paper.published_date:
            year = f" ({paper.published_date.year})"
        
        return f"{authors}{year}. {paper.title}. Retrieved from {paper.url or 'Unknown source'}"
    
    def _get_sort_key(self, citation: Citation) -> str:
        """Get sorting key for bibliography ordering."""
        # Extract first author's last name for sorting
        text = citation.citation_text.lower()
        
        # Try to find author pattern at the beginning
        author_match = re.match(r'^([^,]+)', text)
        if author_match:
            return author_match.group(1).strip()
        
        # Fallback to citation text
        return text
    
    def execute_task(self, task: AgentTask) -> AgentTask:
        """Execute a citation generation task.
        
        Args:
            task: AgentTask with citation parameters
            
        Returns:
            Updated AgentTask with results
        """
        try:
            task.status = "running"
            logger.info(f"Executing citation task: {task.task_id}")
            
            # Extract parameters from task
            papers_data = task.input_data.get("papers", [])
            citation_format = task.input_data.get("citation_format", "bibtex")
            style_options = task.input_data.get("style_options", {})
            create_bibliography = task.input_data.get("create_bibliography", False)
            bibliography_title = task.input_data.get("bibliography_title", "Research Bibliography")
            
            if not papers_data:
                raise ValueError("No papers provided in task input")
            
            # Convert paper data to ResearchPaper objects
            papers = [ResearchPaper.model_validate(p) for p in papers_data]
            
            if create_bibliography:
                # Create bibliography
                bibliography = self.create_bibliography(
                    papers, bibliography_title, citation_format, style_options
                )
                
                task.output_data = {
                    "bibliography": bibliography.model_dump(),
                    "citations": [citation.model_dump() for citation in bibliography.citations],
                    "total_citations": len(bibliography.citations)
                }
            else:
                # Generate individual citations
                citations = self.generate_multiple_citations(papers, citation_format, style_options)
                
                task.output_data = {
                    "citations": [citation.model_dump() for citation in citations],
                    "total_citations": len(citations),
                    "format": citation_format
                }
            
            task.status = "completed"
            task.completed_at = datetime.now()
            
            logger.info(f"Citation task completed: {task.task_id}")
            return task
            
        except Exception as e:
            error_msg = f"Citation task failed: {str(e)}"
            logger.error(error_msg)
            task.status = "failed"
            task.error_message = error_msg
            task.completed_at = datetime.now()
            return task
