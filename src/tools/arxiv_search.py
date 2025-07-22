"""
ArXiv search tool for retrieving academic papers.
"""
import arxiv
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models import ResearchPaper
from src.utils.logger import get_logger

logger = get_logger("arxiv_tool")


class ArxivSearchTool:
    """Tool for searching ArXiv papers."""
    
    def __init__(self, max_results: int = 10):
        """Initialize ArXiv search tool.
        
        Args:
            max_results: Maximum number of results to return per search
        """
        self.max_results = max_results
        self.client = arxiv.Client()
    
    def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> List[ResearchPaper]:
        """Search ArXiv for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (overrides default)
            sort_by: Sort criteria ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending", "descending")
            
        Returns:
            List of ResearchPaper objects
        """
        max_results = max_results or self.max_results
        
        logger.info(f"Searching ArXiv for: '{query}' (max_results={max_results})")
        
        try:
            # Map sort criteria
            sort_by_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }
            
            sort_order_map = {
                "ascending": arxiv.SortOrder.Ascending,
                "descending": arxiv.SortOrder.Descending
            }
            
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by_map.get(sort_by, arxiv.SortCriterion.Relevance),
                sort_order=sort_order_map.get(sort_order, arxiv.SortOrder.Descending)
            )
            
            papers = []
            for result in self.client.results(search):
                paper = self._convert_to_research_paper(result)
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            return []
    
    def _convert_to_research_paper(self, arxiv_result: arxiv.Result) -> ResearchPaper:
        """Convert ArXiv result to ResearchPaper model.
        
        Args:
            arxiv_result: ArXiv search result
            
        Returns:
            ResearchPaper object
        """
        # Extract categories
        categories = [cat for cat in arxiv_result.categories]
        
        # Extract authors
        authors = [author.name for author in arxiv_result.authors]
        
        # Create paper object
        paper = ResearchPaper(
            id=f"arxiv:{arxiv_result.entry_id.split('/')[-1]}",
            title=arxiv_result.title.strip(),
            authors=authors,
            abstract=arxiv_result.summary.strip(),
            url=arxiv_result.entry_id,
            published_date=arxiv_result.published,
            source="arxiv",
            categories=categories,
            doi=arxiv_result.doi
        )
        
        return paper
    
    def search_by_category(
        self, 
        category: str, 
        max_results: Optional[int] = None
    ) -> List[ResearchPaper]:
        """Search ArXiv by category.
        
        Args:
            category: ArXiv category (e.g., "cs.AI", "cs.LG", "stat.ML")
            max_results: Maximum number of results
            
        Returns:
            List of ResearchPaper objects
        """
        query = f"cat:{category}"
        return self.search(query, max_results)
    
    def search_by_author(
        self, 
        author: str, 
        max_results: Optional[int] = None
    ) -> List[ResearchPaper]:
        """Search ArXiv by author.
        
        Args:
            author: Author name
            max_results: Maximum number of results
            
        Returns:
            List of ResearchPaper objects
        """
        query = f"au:{author}"
        return self.search(query, max_results)
