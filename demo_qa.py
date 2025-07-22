#!/usr/bin/env python3
"""
Demo script for Q&A Agent functionality.
"""
import uuid
from datetime import datetime

from src.agents.qa_agent import QAAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.models import ResearchQuery, Question, ConversationContext, ResearchSession
from src.utils.logger import get_logger

logger = get_logger("qa_demo")


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


def demo_qa_agent():
    """Demonstrate Q&A Agent functionality."""
    print_section("Q&A AGENT DEMO")
    print("This demo showcases the Q&A Agent's ability to answer research questions")
    print("based on retrieved papers and generated summaries.")
    
    try:
        # Initialize agents
        print_subsection("Initializing Agents")
        retriever = RetrieverAgent()
        summarizer = SummarizerAgent()
        qa_agent = QAAgent()
        
        print("✅ Retriever Agent initialized")
        print("✅ Summarizer Agent initialized") 
        print("✅ Q&A Agent initialized")
          # Step 1: Retrieve papers
        print_subsection("Step 1: Retrieving Research Papers")
        query = ResearchQuery(query="machine learning healthcare diagnostics")
        print(f"Research Query: {query.query}")
        
        papers = retriever.retrieve_papers(query)
        print(f"📄 Retrieved {len(papers)} papers")
        
        if papers:
            for i, paper in enumerate(papers[:3], 1):
                print(f"  {i}. {paper.title}")
                print(f"     Authors: {', '.join(paper.authors[:2])}...")
                print(f"     Source: {paper.source}")
        
        # Step 2: Generate summaries
        print_subsection("Step 2: Generating Summaries")
        summaries = []
        
        if not summarizer.ollama_client.is_available():
            print("⚠️  Ollama server not available - using mock summaries")
            # Create mock summaries for demo
            from src.models import Summary
            for paper in papers[:2]:
                summary = Summary(
                    paper_id=paper.id,
                    summary=f"Mock summary for {paper.title[:50]}... This paper explores important aspects of machine learning in healthcare.",
                    key_findings=[
                        "Improved diagnostic accuracy",
                        "Reduced processing time", 
                        "Better patient outcomes"
                    ],
                    agent_id="summarizer_agent"
                )
                summaries.append(summary)
        else:
            print("🤖 Generating summaries using Ollama...")
            for paper in papers[:2]:  # Limit to 2 papers for demo
                summary = summarizer.summarize_paper(paper)
                summaries.append(summary)
                print(f"  ✅ Summarized: {paper.title[:40]}...")
        
        print(f"📝 Generated {len(summaries)} summaries")
        
        # Step 3: Create research session and conversation context
        print_subsection("Step 3: Setting Up Q&A Context")
        session = ResearchSession(
            session_id=str(uuid.uuid4()),
            query=query,
            papers=papers,
            summaries=summaries
        )
        
        context = qa_agent.create_conversation_context(session)
        print(f"🗨️  Created conversation context with {len(papers)} papers and {len(summaries)} summaries")
        
        # Step 4: Demonstrate Q&A functionality
        print_subsection("Step 4: Interactive Q&A Demo")
        
        # List of demo questions
        demo_questions = [
            "What are the main benefits of machine learning in healthcare?",
            "How does machine learning improve diagnostic accuracy?",
            "What are the key challenges mentioned in the research?",
            "Can you compare the different approaches discussed?"
        ]
        
        for i, question_text in enumerate(demo_questions, 1):
            print(f"\n🤔 Question {i}: {question_text}")
            
            question = Question(
                question_id=str(uuid.uuid4()),
                question_text=question_text,
                context_type="demo"
            )
            
            try:
                if not qa_agent.ollama_client.is_available():
                    print("⚠️  Ollama server not available - showing mock answer")
                    print("🤖 Mock Answer: Based on the available research, machine learning")
                    print("   shows significant promise in healthcare applications, particularly")
                    print("   in improving diagnostic accuracy and reducing processing times.")
                    print("   Confidence: Medium (mock response)")
                else:
                    answer = qa_agent.answer_question(question, context, papers, summaries)
                    
                    print(f"🤖 Answer: {answer.answer_text}")
                    print(f"📊 Confidence: {answer.confidence_score:.2f}")
                    
                    if answer.evidence:
                        print("📚 Evidence:")
                        for evidence in answer.evidence[:2]:
                            print(f"   • {evidence}")
                    
                    if answer.source_papers:
                        print(f"📄 Based on {len(answer.source_papers)} source paper(s)")
            
            except Exception as e:
                print(f"❌ Error answering question: {str(e)}")
            
            if i < len(demo_questions):
                print("\n" + "."*50)
        
        # Step 5: Demonstrate follow-up questions
        print_subsection("Step 5: Follow-up Question Demo")
        
        try:
            followup_text = "Can you provide more specific examples of these benefits?"
            print(f"🤔 Follow-up: {followup_text}")
            
            if qa_agent.ollama_client.is_available():
                question, answer = qa_agent.ask_followup_question(
                    followup_text, context, papers, summaries
                )
                
                print(f"🤖 Follow-up Answer: {answer.answer_text}")
                print(f"📊 Confidence: {answer.confidence_score:.2f}")
                print(f"🗨️  Context now contains {len(context.questions)} questions and {len(context.answers)} answers")
            else:
                print("⚠️  Ollama server not available - would generate follow-up answer")
                print("🤖 Mock Follow-up: Specific examples include early cancer detection,")
                print("   automated radiology analysis, and personalized treatment recommendations.")
        
        except Exception as e:
            print(f"❌ Error with follow-up question: {str(e)}")
        
        # Step 6: Show context information
        print_subsection("Step 6: Context Information")
        print(f"📊 Session Statistics:")
        print(f"   • Papers retrieved: {len(papers)}")
        print(f"   • Summaries generated: {len(summaries)}")
        print(f"   • Questions asked: {len(context.questions)}")
        print(f"   • Answers provided: {len(context.answers)}")
        print(f"   • Session ID: {session.session_id}")
        print(f"   • Context ID: {context.context_id}")
        
        print_section("Demo Complete!")
        print("The Q&A Agent successfully demonstrated:")
        print("✅ Question answering based on research papers")
        print("✅ Evidence extraction and citation")
        print("✅ Confidence scoring")
        print("✅ Follow-up question handling")
        print("✅ Conversation context management")
        
        if not qa_agent.ollama_client.is_available():
            print("\n💡 Note: Ollama server was not available, so mock responses were used.")
            print("   Start Ollama server to see actual LLM-generated answers.")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection for paper retrieval")
        print("2. Ensure Ollama server is running for LLM features")
        print("3. Verify all dependencies are installed")


if __name__ == "__main__":
    demo_qa_agent()
