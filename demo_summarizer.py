"""
Demo script for testing the Summarizer Agent.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.agents.summarizer_agent import SummarizerAgent
from src.agents.retriever_agent import RetrieverAgent
from src.models import ResearchQuery, ResearchPaper
from src.tools.ollama_client import OllamaClient
from src.utils.logger import setup_logger

def create_sample_papers():
    """Create sample papers for testing when Ollama is not available."""
    return [
        ResearchPaper(
            id="sample-001",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            url="https://arxiv.org/abs/1706.03762",
            source="arxiv",
            categories=["cs.CL", "cs.AI"]
        ),
        ResearchPaper(
            id="sample-002", 
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            url="https://arxiv.org/abs/1810.04805",
            source="arxiv",
            categories=["cs.CL"]
        ),
        ResearchPaper(
            id="sample-003",
            title="GPT-3: Language Models are Few-Shot Learners", 
            authors=["Tom B. Brown", "Benjamin Mann", "Nick Ryder"],
            abstract="Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions.",
            url="https://arxiv.org/abs/2005.14165",
            source="arxiv",
            categories=["cs.CL", "cs.AI"]
        )
    ]

def main():
    """Demo the Summarizer Agent functionality."""
    # Setup logging
    logger = setup_logger()
    
    print("📝 Multi-Agent Research Assistant - Summarizer Agent Demo")
    print("=" * 60)
    
    # Check Ollama availability
    print("🔍 Checking Ollama availability...")
    ollama_client = OllamaClient()
    ollama_available = ollama_client.is_available()
    
    if ollama_available:
        print("✅ Ollama server is available")
        models = ollama_client.list_models()
        print(f"📊 Available models: {models}")
        
        if ollama_client.model not in [m.split(':')[0] for m in models]:
            print(f"⚠️  Warning: Configured model '{ollama_client.model}' not found")
            print("   You may need to run: ollama pull mistral:7b")
    else:
        print("❌ Ollama server not available")
        print("   Please start Ollama server: https://ollama.ai")
        print("   Demo will continue with mock functionality")
    
    print()
    
    # Initialize the summarizer agent
    print("🤖 Initializing Summarizer Agent...")
    agent = SummarizerAgent()
    
    # Display agent info
    info = agent.get_agent_info()
    print(f"✅ Agent ID: {info['agent_id']}")
    print(f"📊 Ollama Available: {info['ollama_available']}")
    print(f"🔧 Model: {info['model']}")
    print(f"📝 Summary Types: {info['supported_summary_types']}")
    print()
    
    # Get papers for testing
    print("📄 Preparing test papers...")
    
    if ollama_available:
        # Try to get real papers using retriever
        print("   Attempting to retrieve real papers...")
        try:
            retriever = RetrieverAgent()
            query = ResearchQuery(query="transformer neural networks")
            papers = retriever.retrieve_papers(query, sources=["arxiv"], max_papers_total=2)
            
            if papers:
                print(f"   ✅ Retrieved {len(papers)} real papers")
            else:
                print("   ⚠️  No papers retrieved, using samples")
                papers = create_sample_papers()[:2]
        except Exception as e:
            print(f"   ⚠️  Retrieval failed: {str(e)}")
            print("   Using sample papers")
            papers = create_sample_papers()[:2]
    else:
        papers = create_sample_papers()[:2]
        print(f"   Using {len(papers)} sample papers")
    
    print()
    
    # Test different summary types
    summary_types = ["general", "technical", "methodology", "findings"]
    
    for i, paper in enumerate(papers):
        print(f"📑 Paper {i+1}: {paper.title[:60]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Abstract: {paper.abstract[:100]}...")
        print()
        
        # Test one summary type per paper
        summary_type = summary_types[i % len(summary_types)]
        print(f"   📝 Testing {summary_type} summary...")
        
        try:
            if ollama_available:
                # Real summarization
                summary = agent.summarize_paper(paper, summary_type, max_length=150)
                print(f"   ✅ Summary generated ({len(summary.summary)} chars)")
                print(f"   📄 Summary: {summary.summary[:200]}...")
                
                if summary.key_findings:
                    print(f"   🔍 Key findings: {len(summary.key_findings)} found")
                    for j, finding in enumerate(summary.key_findings[:2]):
                        print(f"      {j+1}. {finding[:80]}...")
                
                if summary.methodology:
                    print(f"   ⚙️  Methodology: {summary.methodology[:80]}...")
                    
            else:
                # Mock summarization
                print("   ⚠️  Ollama not available - showing mock summary structure")
                mock_summary = f"[Mock {summary_type} summary of '{paper.title[:30]}...']"
                print(f"   📄 Mock Summary: {mock_summary}")
                
        except Exception as e:
            print(f"   ❌ Summarization failed: {str(e)}")
        
        print("-" * 40)
    
    # Test comparative summary
    if len(papers) > 1:
        print("🔍 Testing Comparative Summary...")
        
        try:
            if ollama_available:
                comparative = agent.generate_comparative_summary(
                    papers, 
                    focus_area="transformer architecture"
                )
                print(f"   ✅ Comparative summary generated ({len(comparative)} chars)")
                print(f"   📊 Analysis: {comparative[:200]}...")
            else:
                print("   ⚠️  Ollama not available - showing mock comparative summary")
                print("   📊 Mock Analysis: [Comparative analysis would compare transformer approaches...]")
                
        except Exception as e:
            print(f"   ❌ Comparative summary failed: {str(e)}")
    
    print("-" * 40)
    
    # Test task execution
    print("⚙️  Testing Task Execution...")
    from src.models import AgentTask
    from datetime import datetime
    
    task = AgentTask(
        task_id="demo_summarizer_task_001",
        agent_type="summarizer",
        input_data={
            "papers": [paper.model_dump() for paper in papers[:1]],
            "summary_type": "general",
            "max_length": 200,
            "comparative": False
        }
    )
    
    try:
        result_task = agent.execute_task(task)
        print(f"   ✅ Task Status: {result_task.status}")
        
        if result_task.status == "completed":
            summaries = result_task.output_data.get("summaries", [])
            print(f"   📊 Summaries Generated: {len(summaries)}")
            
            if summaries and ollama_available:
                first_summary = summaries[0]
                print(f"   📄 First Summary: {first_summary.get('summary', '')[:100]}...")
                
        elif result_task.status == "failed":
            print(f"   ❌ Error: {result_task.error_message}")
            
    except Exception as e:
        print(f"   ❌ Task execution error: {str(e)}")
    
    print()
    print("🎉 Summarizer Agent Demo Complete!")
    
    if ollama_available:
        print("✅ All features tested with real Ollama integration")
    else:
        print("⚠️  Demo completed in mock mode - install Ollama for full functionality")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Run: ollama pull mistral:7b")
        print("   3. Start Ollama service")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
