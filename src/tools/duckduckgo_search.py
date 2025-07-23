"""
DuckDuckGo search tool for retrieving research-related content.
"""
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup
from src.models import ResearchPaper
from src.utils.logger import get_logger

# Try to import DuckDuckGo search with fallback
try:
    from ddgs import DDGS  # New package name
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # Old package name
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False
        logger = get_logger("duckduckgo_tool")
        logger.warning("DuckDuckGo search not available. Install 'ddgs' or 'duckduckgo-search' package.")



class DuckDuckGoSearchTool:
    """Tool for searching DuckDuckGo for research papers and content."""
    
    def __init__(self, max_results: int = 10):
        """Initialize DuckDuckGo search tool.
        
        Args:
            max_results: Maximum number of results to return per search
        """
        self.max_results = max_results
        if DDGS_AVAILABLE:
            self.ddgs = DDGS()
        else:
            self.ddgs = None
            logger.warning("DuckDuckGo search unavailable - will return empty results")
    
    def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> List[ResearchPaper]:
        """Search DuckDuckGo for research papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (overrides default)
            region: Search region
            safesearch: Safe search setting
            
        Returns:
            List of ResearchPaper objects
        """
        max_results = max_results or self.max_results
        
        logger.info(f"Searching DuckDuckGo for: '{query}' (max_results={max_results})")
        
        # Return empty list if DDGS is not available
        if not DDGS_AVAILABLE or self.ddgs is None:
            logger.warning("DuckDuckGo search not available - returning empty results")
            return []
        
        try:
            # Add research-specific terms to improve results
            research_query = f"{query} research paper OR academic OR journal OR arxiv OR doi"
            
            results = self.ddgs.text(
                keywords=research_query,
                region=region,
                safesearch=safesearch,
                max_results=max_results
            )
            
            papers = []
            for i, result in enumerate(results):
                if i >= max_results:
                    break
                    
                paper = self._convert_to_research_paper(result, i)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers from DuckDuckGo")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {str(e)}")
            return []
    
    def search_academic_sites(
        self, 
        query: str, 
        max_results: Optional[int] = None
    ) -> List[ResearchPaper]:
        """Search specific academic sites through DuckDuckGo.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of ResearchPaper objects
        """
        academic_sites = [
            "site:arxiv.org",
            "site:scholar.google.com",
            "site:pubmed.ncbi.nlm.nih.gov",
            "site:ieee.org",
            "site:acm.org",
            "site:springer.com",
            "site:sciencedirect.com"
        ]
        
        all_papers = []
        results_per_site = max(1, (max_results or self.max_results) // len(academic_sites))
        
        for site in academic_sites:
            site_query = f"{query} {site}"
            papers = self.search(site_query, results_per_site)
            all_papers.extend(papers)
            
            if len(all_papers) >= (max_results or self.max_results):
                break
        
        return all_papers[:max_results or self.max_results]
    
    def _convert_to_research_paper(self, ddg_result: Dict[str, Any], index: int) -> Optional[ResearchPaper]:
        """Convert DuckDuckGo result to ResearchPaper model.
        
        Args:
            ddg_result: DuckDuckGo search result dictionary
            index: Result index for ID generation
            
        Returns:
            ResearchPaper object or None if conversion fails
        """
        try:
            url = ddg_result.get('href', '')
            title = ddg_result.get('title', 'Untitled')
            body = ddg_result.get('body', '')
            
            # Skip if not research-related
            if not self._is_research_content(title, body, url):
                return None
            
            # Extract potential authors from body/title
            authors = self._extract_authors(body, title)
            
            # Determine source from URL
            source = self._determine_source(url)
            
            # Generate ID
            paper_id = f"ddg:{source}:{index}"
            
            # Try to extract abstract/summary
            abstract = self._extract_abstract(body)
            
            paper = ResearchPaper(
                id=paper_id,
                title=title.strip(),
                authors=authors,
                abstract=abstract,
                url=url,
                published_date=None,  # DuckDuckGo doesn't provide dates directly
                source=f"duckduckgo_{source}",
                categories=[],
                doi=self._extract_doi(body)
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to convert DuckDuckGo result: {str(e)}")
            return None
    
    def _is_research_content(self, title: str, body: str, url: str) -> bool:
        """Check if content appears to be research-related.
        
        Args:
            title: Result title
            body: Result body text
            url: Result URL
            
        Returns:
            True if appears to be research content
        """
        research_indicators = [
            'paper', 'research', 'study', 'journal', 'conference',
            'arxiv', 'doi', 'abstract', 'publication', 'academic',
            'proceedings', 'ieee', 'acm', 'springer', 'elsevier',
            'analysis', 'algorithm', 'method', 'approach', 'framework'
        ]
        
        text_to_check = f"{title} {body} {url}".lower()
        
        # Count research indicators
        indicator_count = sum(1 for indicator in research_indicators if indicator in text_to_check)
        
        # Require at least 2 indicators
        return indicator_count >= 2
    
    def _extract_authors(self, body: str, title: str) -> List[str]:
        """Extract potential authors from text.
        
        Args:
            body: Body text
            title: Title text
            
        Returns:
            List of author names
        """
        authors = []
        
        # Look for common author patterns
        import re
        
        # Pattern: "by Author Name" or "Author Name et al."
        author_patterns = [
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+et\s+al',
            r'Authors?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)'
        ]
        
        text_to_search = f"{title} {body}"
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text_to_search)
            for match in matches:
                if isinstance(match, str):
                    # Split multiple authors if comma-separated
                    author_names = [name.strip() for name in match.split(',')]
                    authors.extend(author_names)
        
        # Remove duplicates and limit to reasonable number
        authors = list(dict.fromkeys(authors))[:5]
        
        return authors if authors else ["Unknown Author"]
    
    def _determine_source(self, url: str) -> str:
        """Determine source from URL.
        
        Args:
            url: Result URL
            
        Returns:
            Source name
        """
        if 'arxiv.org' in url:
            return 'arxiv'
        elif 'scholar.google' in url:
            return 'google_scholar'
        elif 'pubmed' in url:
            return 'pubmed'
        elif 'ieee.org' in url:
            return 'ieee'
        elif 'acm.org' in url:
            return 'acm'
        elif 'springer' in url:
            return 'springer'
        elif 'sciencedirect' in url:
            return 'sciencedirect'
        else:
            return 'web'
    
    def _extract_abstract(self, body: str) -> str:
        """Extract abstract or summary from body text.
        
        Args:
            body: Body text
            
        Returns:
            Abstract text
        """
        # Take first 500 characters as abstract approximation
        abstract = body.strip()
        if len(abstract) > 500:
            # Try to cut at sentence boundary
            sentences = abstract.split('. ')
            truncated = ''
            for sentence in sentences:
                if len(truncated + sentence) <= 500:
                    truncated += sentence + '. '
                else:
                    break
            abstract = truncated.strip()
        
        return abstract if abstract else "No abstract available."
    
    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text if present.
        
        Args:
            text: Text to search for DOI
            
        Returns:
            DOI string or None
        """
        import re
        
        doi_pattern = r'10\.\d{4,}\/[^\s]+'
        match = re.search(doi_pattern, text)
        
        return match.group(0) if match else None
