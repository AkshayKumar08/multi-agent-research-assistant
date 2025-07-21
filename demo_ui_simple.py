#!/usr/bin/env python3
"""
Simple UI Demo for Multi-Agent Research Assistant.

This demo shows the UI functionality without requiring external
UI frameworks by using a simple command-line interface.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.research_coordinator import ResearchCoordinator
from src.utils.logger import logger


class SimpleUIDemo:
    """Simple command-line UI demo."""
    
    def __init__(self):
        """Initialize the demo."""
        self.coordinator = None
        self.current_session = None
        self.initialize_coordinator()
    
    def initialize_coordinator(self):
        """Initialize the research coordinator."""
        try:
            print("📋 Initializing research coordinator...")
            self.coordinator = ResearchCoordinator()
            print("✅ Coordinator initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize coordinator: {e}")
            self.coordinator = None
    
    def display_header(self):
        """Display the application header."""
        print("\n" + "=" * 60)
        print("🤖 Multi-Agent Research Assistant - Simple UI Demo")
        print("Powered by CrewAI • Ollama • LangChain")
        print("=" * 60)
    
    def display_papers(self, papers):
        """Display papers in a formatted way."""
        if not papers:
            print("📚 No papers found.")
            return
        
        print(f"\n📚 Found {len(papers)} papers:")
        print("-" * 50)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}")
            if len(paper.authors) > 3:
                print(f"   ... and {len(paper.authors) - 3} more")
            print(f"   Source: {paper.source.upper()}")
            if paper.url:
                print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:200]}...")
    
    def display_summaries(self, summaries):
        """Display summaries in a formatted way."""
        if not summaries:
            print("📝 No summaries generated.")
            return
        
        print(f"\n📝 Generated {len(summaries)} summaries:")
        print("-" * 50)
        
        for i, summary in enumerate(summaries, 1):
            print(f"\n{i}. Summary for Paper {summary.paper_id[:8]}...")
            print(f"   Type: {summary.summary_type}")
            print(f"   Summary: {summary.summary[:300]}...")
            
            if summary.key_findings:
                print("   Key Findings:")
                for finding in summary.key_findings[:3]:
                    print(f"   • {finding}")
    
    def display_citations(self, citations):
        """Display citations in a formatted way."""
        if not citations:
            print("📖 No citations generated.")
            return
        
        # Group by format
        citation_formats = {}
        for citation in citations:
            format_type = citation.citation_format
            if format_type not in citation_formats:
                citation_formats[format_type] = []
            citation_formats[format_type].append(citation)
        
        print(f"\n📖 Generated {len(citations)} citations:")
        print("-" * 50)
        
        for format_type, format_citations in citation_formats.items():
            print(f"\n{format_type.upper()} Format:")
            for citation in format_citations:
                print(f"   {citation.citation_text}")
    
    async def conduct_research(self, query):
        """Conduct research and display results."""
        if not self.coordinator:
            print("❌ Research coordinator not available")
            return None
        
        print(f"\n🔍 Starting research for: '{query}'")
        print("⏳ This may take a few minutes...")
        
        try:
            # Conduct research
            session = await self.coordinator.conduct_research(query, user_id="demo_user")
            self.current_session = session
            
            print(f"\n✅ Research completed!")
            print(f"📊 Session ID: {session.session_id}")
            print(f"⏰ Completed: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display results
            self.display_papers(session.papers)
            self.display_summaries(session.summaries)
            self.display_citations(session.citations)
            
            return session
            
        except Exception as e:
            print(f"❌ Research failed: {str(e)}")
            return None
    
    async def interactive_qa(self):
        """Interactive Q&A session."""
        if not self.current_session:
            print("⚠️  Please conduct research first to enable Q&A.")
            return
        
        print("\n❓ Interactive Q&A Session")
        print("Ask questions about your research (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting Q&A session")
                break
            
            if not question:
                continue
            
            try:
                print("🤔 Processing...")
                answer = await self.coordinator.answer_question(self.current_session, question)
                
                print(f"\n💡 Answer:")
                print(f"   {answer.answer_text}")
                print(f"\n🎯 Confidence: {answer.confidence_score:.2f}")
                print(f"📚 Sources: {len(answer.source_papers)} papers")
                
                if answer.evidence:
                    print(f"\n📋 Evidence:")
                    for i, evidence in enumerate(answer.evidence[:2], 1):
                        print(f"   {i}. {evidence[:150]}...")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def display_stats(self):
        """Display session statistics."""
        if not self.current_session:
            print("📊 No active session")
            return
        
        session = self.current_session
        print(f"\n📊 Current Session Statistics:")
        print("-" * 50)
        print(f"Query: {session.query.query}")
        print(f"Papers Found: {len(session.papers)}")
        print(f"Summaries Generated: {len(session.summaries)}")
        print(f"Citations Available: {len(session.citations)}")
        print(f"Tasks Completed: {len([t for t in session.tasks if t.status == 'completed'])}")
        print(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Updated: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def main_menu(self):
        """Main menu interface."""
        while True:
            print("\n" + "=" * 40)
            print("📋 Main Menu:")
            print("1. 🔍 Conduct Research")
            print("2. ❓ Ask Questions (Q&A)")
            print("3. 📊 View Statistics")
            print("4. ❌ Exit")
            print("=" * 40)
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                query = input("\n🔍 Enter research query: ").strip()
                if query:
                    await self.conduct_research(query)
                else:
                    print("❌ Please enter a valid query")
            
            elif choice == "2":
                await self.interactive_qa()
            
            elif choice == "3":
                self.display_stats()
            
            elif choice == "4":
                print("\n👋 Thank you for using Multi-Agent Research Assistant!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
    
    async def run(self):
        """Run the demo."""
        self.display_header()
        
        if not self.coordinator:
            print("\n❌ Cannot start demo without research coordinator")
            return
        
        print("\n🎯 Welcome to the Multi-Agent Research Assistant!")
        print("This demo showcases the core functionality using a simple interface.")
        
        await self.main_menu()


def main():
    """Main function."""
    demo = SimpleUIDemo()
    asyncio.run(demo.run())


if __name__ == "__main__":
    main()
