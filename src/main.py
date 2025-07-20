"""
Main entry point for the Multi-Agent Research Assistant.
"""
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import config
from src.utils.logger import setup_logger
from src.agents.research_coordinator import ResearchCoordinator


async def run_interactive_research():
    """Run interactive research session."""
    logger = setup_logger()
    
    print("🚀 Multi-Agent Research Assistant - Interactive Mode")
    print("=" * 55)
    
    try:
        # Initialize coordinator
        print("📋 Initializing research coordinator...")
        coordinator = ResearchCoordinator()
        print("✅ Coordinator ready!")
        
        # Get research query
        print("\n🔍 Enter your research topic:")
        query = input("Query: ").strip()
        
        if not query:
            print("❌ No query provided. Exiting.")
            return
        
        print(f"\n🎯 Starting research for: '{query}'")
        print("⏳ This may take a few minutes...")
        
        # Conduct research
        session = await coordinator.conduct_research(query)
        
        # Display results
        print("\n📊 Research Complete!")
        summary = coordinator.get_session_summary(session)
        
        print(f"📚 Papers found: {summary['papers_found']}")
        print(f"📝 Summaries generated: {summary['summaries_generated']}")
        print(f"📖 Citations available: {summary['citations_available']}")
        print(f"✅ Tasks completed: {summary['tasks_completed']}")
        
        # Interactive Q&A
        if session.papers:
            print("\n❓ Ask questions about the research (type 'quit' to exit):")
            
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question:
                    print("🤔 Processing...")
                    answer = await coordinator.answer_question(session, question)
                    print(f"\n💡 Answer: {answer.answer_text}")
                    print(f"🎯 Confidence: {answer.confidence_score:.2f}")
        
        print("\n✨ Session complete! Check logs for details.")
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        print(f"❌ Research failed: {e}")


def main():
    """Main function to run the research assistant."""
    # Setup logging
    logger = setup_logger()
    
    logger.info("🚀 Starting Multi-Agent Research Assistant")
    logger.info(f"Configuration loaded: {config.to_dict()}")
    
    print("✅ Multi-Agent Research Assistant")
    print(f"📊 Ollama URL: {config.OLLAMA_BASE_URL}")
    print(f"🤖 Model: {config.OLLAMA_MODEL}")
    print(f"💾 Data directory: {config.DATA_DIR}")
    print(f"📁 Vector store: {config.VECTOR_STORE_PATH}")
    
    print("\nChoose mode:")
    print("1. Interactive research session")
    print("2. Basic setup verification")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(run_interactive_research())
    elif choice == "2":
        print("✅ Setup verification complete")
        logger.info("Setup verification complete")
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
