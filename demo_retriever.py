"""
Demo script for testing the Retriever Agent.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.agents.retriever_agent import RetrieverAgent
from src.models import ResearchQuery
from src.utils.logger import setup_logger

def main():
    """Demo the Retriever Agent functionality."""
    # Setup logging
    logger = setup_logger()
    
    print("🔍 Multi-Agent Research Assistant - Retriever Agent Demo")
    print("=" * 60)
    
    # Initialize the retriever agent
    print("🤖 Initializing Retriever Agent...")
    agent = RetrieverAgent()
    
    # Display agent info
    info = agent.get_agent_info()
    print(f"✅ Agent ID: {info['agent_id']}")
    print(f"📊 Supported Sources: {info['supported_sources']}")
    print(f"📄 Max Papers per Source: {info['max_papers_per_source']}")
    print()
    
    # Test query
    test_query = "machine learning natural language processing"
    print(f"🔎 Testing search for: '{test_query}'")
    print()
    
    # Create research query
    query = ResearchQuery(query=test_query)
    
    # Test different search scenarios
    scenarios = [
        {
            "name": "ArXiv Only",
            "sources": ["arxiv"],
            "max_papers": 3
        },
        {
            "name": "DuckDuckGo Only", 
            "sources": ["duckduckgo"],
            "max_papers": 3
        },
        {
            "name": "Both Sources",
            "sources": ["arxiv", "duckduckgo"],
            "max_papers": 5
        }
    ]
    
    for scenario in scenarios:
        print(f"📋 Scenario: {scenario['name']}")
        print(f"   Sources: {scenario['sources']}")
        print(f"   Max Papers: {scenario['max_papers']}")
        
        try:
            papers = agent.retrieve_papers(
                query=query,
                sources=scenario['sources'],
                max_papers_total=scenario['max_papers']
            )
            
            print(f"   ✅ Found {len(papers)} papers")
            
            # Display first few papers
            for i, paper in enumerate(papers[:2]):
                print(f"   📄 Paper {i+1}:")
                print(f"      Title: {paper.title[:80]}...")
                print(f"      Authors: {', '.join(paper.authors[:3])}")
                print(f"      Source: {paper.source}")
                print(f"      URL: {paper.url[:60]}...")
                print()
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Test category search (ArXiv only)
    print("🏷️  Testing Category Search...")
    try:
        category_papers = agent.retrieve_papers_by_category("cs.AI", max_papers=2)
        print(f"   ✅ Found {len(category_papers)} papers in cs.AI category")
        
        for i, paper in enumerate(category_papers[:1]):
            print(f"   📄 Paper {i+1}:")
            print(f"      Title: {paper.title[:80]}...")
            print(f"      Categories: {', '.join(paper.categories)}")
            print()
            
    except Exception as e:
        print(f"   ❌ Category search error: {str(e)}")
    
    print("-" * 40)
    
    # Test task execution
    print("⚙️  Testing Task Execution...")
    from src.models import AgentTask
    from datetime import datetime
    
    task = AgentTask(
        task_id="demo_task_001",
        agent_type="retriever",
        input_data={
            "query": "artificial intelligence",
            "sources": ["arxiv"],
            "max_papers": 2
        }
    )
    
    try:
        result_task = agent.execute_task(task)
        print(f"   ✅ Task Status: {result_task.status}")
        if result_task.status == "completed":
            print(f"   📊 Papers Found: {result_task.output_data.get('total_found', 0)}")
        elif result_task.status == "failed":
            print(f"   ❌ Error: {result_task.error_message}")
            
    except Exception as e:
        print(f"   ❌ Task execution error: {str(e)}")
    
    print()
    print("🎉 Retriever Agent Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
