"""
Retriever Agent for coordinating paper search across multiple sources.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models import ResearchPaper, ResearchQuery, AgentTask
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.duckduckgo_search import DuckDuckGoSearchTool
from src.utils.logger import get_logger
from config.settings import config

logger = get_logger("retriever_agent")


class RetrieverAgent:
    """Agent responsible for retrieving research papers from multiple sources."""
    
    def __init__(self):
        """Initialize the Retriever Agent with search tools."""
        self.arxiv_tool = ArxivSearchTool(max_results=config.MAX_PAPERS_PER_QUERY)
        self.ddg_tool = DuckDuckGoSearchTool(max_results=config.MAX_PAPERS_PER_QUERY)
        self.agent_id = "retriever_agent"
        
        logger.info("Retriever Agent initialized")
    
    def retrieve_papers(
        self, 
        query: ResearchQuery,
        sources: List[str] = None,
        max_papers_total: Optional[int] = None
    ) -> List[ResearchPaper]:
        """Retrieve papers from multiple sources for a research query.
        
        Args:
            query: Research query object
            sources: List of sources to search ("arxiv", "duckduckgo", "all")
            max_papers_total: Maximum total papers to return
            
        Returns:
            List of unique research papers
        """
        if sources is None:
            sources = ["arxiv", "duckduckgo"]
        
        if "all" in sources:
            sources = ["arxiv", "duckduckgo"]
        
        max_papers_total = max_papers_total or config.MAX_SEARCH_RESULTS
        max_per_source = max(1, max_papers_total // len(sources))
        
        logger.info(f"Starting paper retrieval for query: '{query.query}'")
        logger.info(f"Sources: {sources}, Max papers: {max_papers_total}")
        
        all_papers = []
        
        # Search ArXiv
        if "arxiv" in sources:
            try:
                arxiv_papers = self.arxiv_tool.search(
                    query.query, 
                    max_results=max_per_source
                )
                all_papers.extend(arxiv_papers)
                logger.info(f"Retrieved {len(arxiv_papers)} papers from ArXiv")
            except Exception as e:
                logger.error(f"ArXiv search failed: {str(e)}")
        
        # Search DuckDuckGo
        if "duckduckgo" in sources:
            try:
                ddg_papers = self.ddg_tool.search(
                    query.query, 
                    max_results=max_per_source
                )
                all_papers.extend(ddg_papers)
                logger.info(f"Retrieved {len(ddg_papers)} papers from DuckDuckGo")
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {str(e)}")
        
        # Remove duplicates and limit results
        unique_papers = self._remove_duplicates(all_papers)
        final_papers = unique_papers[:max_papers_total]
        
        logger.info(f"Total unique papers retrieved: {len(final_papers)}")
        return final_papers
    
    def retrieve_papers_by_category(
        self, 
        category: str, 
        max_papers: Optional[int] = None
    ) -> List[ResearchPaper]:
        """Retrieve papers by ArXiv category.
        
        Args:
            category: ArXiv category (e.g., "cs.AI", "cs.LG")
            max_papers: Maximum papers to return
            
        Returns:
            List of research papers
        """
        max_papers = max_papers or config.MAX_SEARCH_RESULTS
        
        logger.info(f"Retrieving papers for category: {category}")
        
        try:
            papers = self.arxiv_tool.search_by_category(category, max_papers)
            logger.info(f"Retrieved {len(papers)} papers for category {category}")
            return papers
        except Exception as e:
            logger.error(f"Category search failed: {str(e)}")
            return []
    
    def retrieve_papers_by_author(
        self, 
        author: str, 
        max_papers: Optional[int] = None
    ) -> List[ResearchPaper]:
        """Retrieve papers by author.
        
        Args:
            author: Author name
            max_papers: Maximum papers to return
            
        Returns:
            List of research papers
        """
        max_papers = max_papers or config.MAX_SEARCH_RESULTS
        
        logger.info(f"Retrieving papers for author: {author}")
        
        all_papers = []
        
        # Search ArXiv
        try:
            arxiv_papers = self.arxiv_tool.search_by_author(author, max_papers // 2)
            all_papers.extend(arxiv_papers)
        except Exception as e:
            logger.error(f"ArXiv author search failed: {str(e)}")
        
        # Search DuckDuckGo with author name
        try:
            author_query = f'author:"{author}" OR "{author}"'
            ddg_papers = self.ddg_tool.search(author_query, max_papers // 2)
            all_papers.extend(ddg_papers)
        except Exception as e:
            logger.error(f"DuckDuckGo author search failed: {str(e)}")
        
        unique_papers = self._remove_duplicates(all_papers)
        final_papers = unique_papers[:max_papers]
        
        logger.info(f"Retrieved {len(final_papers)} papers for author {author}")
        return final_papers
    
    def execute_task(self, task: AgentTask) -> AgentTask:
        """Execute a retrieval task.
        
        Args:
            task: Agent task with retrieval parameters
            
        Returns:
            Updated task with results
        """
        logger.info(f"Executing retrieval task: {task.task_id}")
        
        try:
            task.status = "running"
            
            # Extract parameters from task input
            query_text = task.input_data.get("query", "")
            sources = task.input_data.get("sources", ["arxiv", "duckduckgo"])
            max_papers = task.input_data.get("max_papers", config.MAX_SEARCH_RESULTS)
            
            # Create research query
            query = ResearchQuery(query=query_text)
            
            # Retrieve papers
            papers = self.retrieve_papers(query, sources, max_papers)
              # Update task with results
            task.output_data = {
                "papers": [paper.model_dump() for paper in papers],
                "total_found": len(papers),
                "sources_searched": sources
            }
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
    
    def _remove_duplicates(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity and URL.
        
        Args:
            papers: List of research papers
            
        Returns:
            List of unique papers
        """
        if not papers:
            return []
        
        unique_papers = []
        seen_titles = set()
        seen_urls = set()
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            normalized_url = paper.url.lower().strip()
            
            # Check for duplicates
            is_duplicate = False
            
            # Check exact title match
            if normalized_title in seen_titles:
                is_duplicate = True
            
            # Check exact URL match
            if normalized_url in seen_urls:
                is_duplicate = True
            
            # Check title similarity (simple approach)
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)
                seen_urls.add(normalized_url)
        
        logger.info(f"Removed {len(papers) - len(unique_papers)} duplicate papers")
        return unique_papers
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar using simple word overlap.
        
        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if titles are similar
        """
        if not title1 or not title2:
            return False
        
        # Simple word-based similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported search sources.
        
        Returns:
            List of source names
        """
        return ["arxiv", "duckduckgo"]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information.
        
        Returns:
            Dictionary with agent details
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": "retriever",
            "supported_sources": self.get_supported_sources(),
            "max_papers_per_source": config.MAX_PAPERS_PER_QUERY,
            "max_total_papers": config.MAX_SEARCH_RESULTS
        }
