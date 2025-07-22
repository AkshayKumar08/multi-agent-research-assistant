#!/usr/bin/env python3
"""
Demo script for the CrewAI-integrated Multi-Agent Research Assistant.

This script demonstrates the complete research workflow using the ResearchCoordinator
which orchestrates all agents using CrewAI.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.research_coordinator import ResearchCoordinator
from src.tools.ollama_client import OllamaClient
from src.utils.logger import logger


async def demo_crewai_research():
    """Demonstrate the complete CrewAI research workflow."""
    
    print("🚀 Multi-Agent Research Assistant - CrewAI Integration Demo")
    print("=" * 60)
    
    # Initialize the coordinator
    print("\n📋 Initializing ResearchCoordinator with CrewAI...")
    try:
        coordinator = ResearchCoordinator()
        print("✅ Coordinator initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize coordinator: {e}")
        return
    
    # Get research query from user
    print("\n🔍 Enter your research query:")
    query = input("Query: ").strip()
    
    if not query:
        query = "machine learning transformers attention mechanisms"
        print(f"Using default query: {query}")
    
    print(f"\n🎯 Starting research for: '{query}'")
    print("-" * 50)
    
    # Conduct research using CrewAI workflow
    try:
        session = await coordinator.conduct_research(query, user_id="demo_user")
        
        # Display results
        print("\n📊 Research Session Summary:")
        summary = coordinator.get_session_summary(session)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\n📚 Found {len(session.papers)} papers:")
        for i, paper in enumerate(session.papers[:5], 1):  # Show first 5
            print(f"  {i}. {paper.title[:80]}...")
            print(f"     Authors: {', '.join(paper.authors[:3])}...")
            print(f"     Source: {paper.source}")
            print()
        
        if len(session.papers) > 5:
            print(f"     ... and {len(session.papers) - 5} more papers")
        
        print(f"\n📝 Generated {len(session.summaries)} summaries:")
        for i, summary in enumerate(session.summaries[:3], 1):  # Show first 3
            print(f"  {i}. Paper ID: {summary.paper_id}")
            print(f"     Summary: {summary.summary[:100]}...")
            if summary.key_findings:
                print(f"     Key findings: {len(summary.key_findings)} identified")
            print()
        
        print(f"\n📖 Generated {len(session.citations)} citations:")
        for i, citation in enumerate(session.citations[:3], 1):  # Show first 3
            print(f"  {i}. Format: {citation.citation_format}")
            print(f"     Citation: {citation.citation_text[:100]}...")
            print()
        
        # Interactive Q&A demo
        print("\n❓ Interactive Q&A Session:")
        print("Ask questions about the research (type 'quit' to exit)")
        
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                print("🤔 Thinking...")
                answer = await coordinator.answer_question(session, question)
                print(f"\n💡 Answer: {answer.answer_text}")
                print(f"🎯 Confidence: {answer.confidence_score:.2f}")
                if answer.evidence:
                    print("📋 Evidence:")
                    for evidence in answer.evidence[:2]:  # Show first 2
                        print(f"  • {evidence[:100]}...")
        
        print("\n✨ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


async def demo_quick_research():
    """Quick demo with predefined query."""
    
    print("🚀 Quick Research Demo - Transformers in AI")
    print("=" * 50)
    
    coordinator = ResearchCoordinator()
    
    # Conduct research
    session = await coordinator.conduct_research(
        "transformer neural networks attention mechanism",
        user_id="quick_demo"
    )
    
    # Show results
    summary = coordinator.get_session_summary(session)
    print(f"\nResults: {summary['papers_found']} papers, "
          f"{summary['summaries_generated']} summaries, "
          f"{summary['citations_available']} citations")
    
    # Ask a sample question
    if session.papers:
        answer = await coordinator.answer_question(
            session, 
            "What are the main advantages of transformer architectures?"
        )
        print(f"\nSample Q&A:")
        print(f"Question: What are the main advantages of transformer architectures?")
        print(f"Answer: {answer.answer_text[:200]}...")
        print(f"Confidence: {answer.confidence_score:.2f}")


def check_dependencies():
    """Check if all required dependencies are available."""
    
    print("🔍 Checking dependencies...")
    
    # Check Ollama client
    try:
        client = OllamaClient()
        models = client.list_models()
        print(f"✅ Ollama client: {len(models)} models available")
    except Exception as e:
        print(f"⚠️  Ollama client: {e}")
        print("   Make sure Ollama is running with: ollama serve")
        return False
    
    # Check CrewAI
    try:
        import crewai
        print(f"✅ CrewAI: {crewai.__version__}")
    except ImportError:
        print("❌ CrewAI not found. Install with: pip install crewai")
        return False
    
    print("✅ All dependencies available!")
    return True


def main():
    """Main demo function."""
    
    if not check_dependencies():
        print("\n❌ Dependencies missing. Please install and configure required components.")
        return
    
    print("\nChoose demo mode:")
    print("1. Full interactive demo")
    print("2. Quick demo with predefined query")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(demo_crewai_research())
    elif choice == "2":
        asyncio.run(demo_quick_research())
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
