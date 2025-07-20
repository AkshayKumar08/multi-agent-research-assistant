#!/usr/bin/env python3
"""
Demo script for Citation Agent functionality.
"""
import uuid
from datetime import datetime

from src.agents.citation_agent import CitationAgent
from src.agents.retriever_agent import RetrieverAgent
from src.models import ResearchQuery, ResearchPaper
from src.utils.logger import get_logger

logger = get_logger("citation_demo")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def demo_citation_agent():
    """Demonstrate Citation Agent functionality."""
    print_section("CITATION AGENT DEMO")
    print("This demo showcases the Citation Agent's ability to generate")
    print("academic citations in various formats from research papers.")
    
    try:
        # Initialize agents
        print_subsection("Initializing Agents")
        retriever = RetrieverAgent()
        citation_agent = CitationAgent()
        
        print("✅ Retriever Agent initialized")
        print("✅ Citation Agent initialized")
        print(f"📋 Supported formats: {', '.join(citation_agent.SUPPORTED_FORMATS)}")
        
        # Step 1: Get sample papers (retrieve or create mock data)
        print_subsection("Step 1: Obtaining Research Papers")
        
        try:
            # Try to retrieve real papers
            query = ResearchQuery(query="machine learning healthcare")
            papers = retriever.retrieve_papers(query, max_papers_total=3)
            print(f"📄 Retrieved {len(papers)} real papers from search")
        except Exception as e:
            print(f"⚠️  Could not retrieve papers ({str(e)}), using mock data")
            # Create mock papers for demo
            papers = [
                ResearchPaper(
                    id="demo_paper_1",
                    title="Deep Learning for Medical Image Analysis: A Comprehensive Survey",
                    authors=["Alice Johnson", "Bob Smith", "Carol Davis"],
                    abstract="This survey provides a comprehensive overview of deep learning techniques applied to medical image analysis, covering recent advances and future directions.",
                    url="https://arxiv.org/abs/2101.00001",
                    published_date=datetime(2023, 3, 15),
                    source="arxiv",
                    categories=["cs.CV", "cs.LG"],
                    doi="10.1000/demo1"
                ),
                ResearchPaper(
                    id="demo_paper_2",
                    title="Artificial Intelligence in Healthcare: Past, Present and Future",
                    authors=["David Wilson", "Emma Brown"],
                    abstract="An analysis of AI applications in healthcare from historical perspective to current implementations and future possibilities.",
                    url="https://example.com/ai-healthcare",
                    published_date=datetime(2023, 7, 22),
                    source="journal",
                    doi="10.1000/demo2"
                ),
                ResearchPaper(
                    id="demo_paper_3",
                    title="Federated Learning for Privacy-Preserving Healthcare Analytics",
                    authors=["Frank Miller"],
                    abstract="This paper explores federated learning approaches for healthcare data analysis while preserving patient privacy.",
                    url="https://example.com/federated-learning",
                    published_date=datetime(2023, 9, 10),
                    source="conference"
                )
            ]
            print(f"📄 Using {len(papers)} mock papers for demonstration")
        
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper.title[:60]}{'...' if len(paper.title) > 60 else ''}")
            print(f"     Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
            print(f"     Source: {paper.source}")
        
        # Step 2: Demonstrate different citation formats
        print_subsection("Step 2: Generating Citations in Different Formats")
        
        # Select first paper for detailed format demonstration
        demo_paper = papers[0]
        print(f"📝 Generating citations for: {demo_paper.title[:50]}...")
        
        citation_formats = ["bibtex", "apa", "mla", "ieee"]
        citations_by_format = {}
        
        for fmt in citation_formats:
            print(f"\n🔄 Generating {fmt.upper()} citation...")
            try:
                citation = citation_agent.generate_citation(demo_paper, fmt)
                citations_by_format[fmt] = citation
                
                print(f"✅ {fmt.upper()} Citation:")
                print(f"   {citation.citation_text}")
                print(f"   Status: {citation.validation_status}")
                print(f"   Confidence: {'High' if 'validated' in citation.validation_status else 'Medium'}")
                
            except Exception as e:
                print(f"❌ Error generating {fmt} citation: {str(e)}")
        
        # Step 3: Demonstrate advanced citation formats using LLM
        print_subsection("Step 3: Advanced Citation Formats (LLM-Generated)")
        
        advanced_formats = ["chicago", "harvard", "nature"]
        
        for fmt in advanced_formats:
            print(f"\n🤖 Generating {fmt.upper()} citation using LLM...")
            try:
                if citation_agent.ollama_client.is_available():
                    citation = citation_agent.generate_citation(demo_paper, fmt)
                    print(f"✅ {fmt.upper()} Citation:")
                    print(f"   {citation.citation_text}")
                    print(f"   Status: {citation.validation_status}")
                else:
                    print(f"⚠️  Ollama not available, showing fallback citation:")
                    fallback_citation = citation_agent._create_fallback_citation_text(demo_paper, fmt)
                    print(f"   {fallback_citation}")
            
            except Exception as e:
                print(f"❌ Error generating {fmt} citation: {str(e)}")
        
        # Step 4: Generate multiple citations
        print_subsection("Step 4: Bulk Citation Generation")
        
        print(f"📚 Generating APA citations for all {len(papers)} papers...")
        try:
            all_citations = citation_agent.generate_multiple_citations(papers, "apa")
            
            print(f"✅ Generated {len(all_citations)} citations")
            for i, citation in enumerate(all_citations, 1):
                print(f"\n{i}. {citation.citation_text}")
                if "warning" in citation.validation_status:
                    print(f"   ⚠️  Warning: {citation.validation_status}")
        
        except Exception as e:
            print(f"❌ Error in bulk generation: {str(e)}")
        
        # Step 5: Create a bibliography
        print_subsection("Step 5: Bibliography Creation")
        
        print("📖 Creating a comprehensive bibliography...")
        try:
            bibliography = citation_agent.create_bibliography(
                papers, 
                "Machine Learning in Healthcare: Research Bibliography",
                "apa"
            )
            
            print(f"✅ Bibliography created: {bibliography.title}")
            print(f"📊 Statistics:")
            print(f"   • Total papers: {bibliography.metadata['total_papers']}")
            print(f"   • Format: {bibliography.format_style.upper()}")
            print(f"   • Generated: {bibliography.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • Bibliography ID: {bibliography.bibliography_id}")
            
            print(f"\n📋 Complete Bibliography:")
            print(f"{'='*50}")
            print(f"{bibliography.title}")
            print(f"{'='*50}")
            
            for i, citation in enumerate(bibliography.citations, 1):
                print(f"\n{i}. {citation.citation_text}")
        
        except Exception as e:
            print(f"❌ Error creating bibliography: {str(e)}")
        
        # Step 6: Demonstrate task execution
        print_subsection("Step 6: Task-Based Citation Generation")
        
        from src.models import AgentTask
        
        print("🔧 Testing task-based citation generation...")
        try:
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type="citation_agent",
                input_data={
                    "papers": [paper.model_dump() for paper in papers[:2]],
                    "citation_format": "bibtex",
                    "create_bibliography": False
                }
            )
            
            result = citation_agent.execute_task(task)
            
            print(f"✅ Task completed: {result.status}")
            print(f"📊 Results:")
            print(f"   • Citations generated: {result.output_data.get('total_citations', 0)}")
            print(f"   • Format: {result.output_data.get('format', 'unknown')}")
            print(f"   • Task ID: {result.task_id}")
            
        except Exception as e:
            print(f"❌ Error in task execution: {str(e)}")
        
        # Step 7: Show citation validation
        print_subsection("Step 7: Citation Quality and Validation")
        
        print("🔍 Citation validation statistics:")
        if 'all_citations' in locals():
            validated_count = sum(1 for c in all_citations if 'validated' in c.validation_status)
            warning_count = sum(1 for c in all_citations if 'warning' in c.validation_status)
            fallback_count = sum(1 for c in all_citations if 'fallback' in c.validation_status)
            
            print(f"   ✅ Validated: {validated_count}/{len(all_citations)}")
            print(f"   ⚠️  Warnings: {warning_count}/{len(all_citations)}")
            print(f"   🔄 Fallbacks: {fallback_count}/{len(all_citations)}")
            
            quality_score = (validated_count / len(all_citations)) * 100
            print(f"   📈 Quality Score: {quality_score:.1f}%")
        
        print_section("Demo Complete!")
        print("The Citation Agent successfully demonstrated:")
        print("✅ Multiple citation format support (BibTeX, APA, MLA, IEEE)")
        print("✅ LLM-powered advanced format generation")
        print("✅ Bulk citation processing")
        print("✅ Bibliography creation and management")
        print("✅ Citation validation and quality assessment")
        print("✅ Task-based execution")
        print("✅ Error handling and fallback mechanisms")
        
        if not citation_agent.ollama_client.is_available():
            print("\n💡 Note: Ollama server was not available for advanced formats.")
            print("   Start Ollama server to enable LLM-powered citation generation.")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection for paper retrieval")
        print("2. Ensure all dependencies are installed")
        print("3. Verify Ollama server for advanced citation formats")


if __name__ == "__main__":
    demo_citation_agent()
