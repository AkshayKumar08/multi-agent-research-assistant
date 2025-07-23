#!/usr/bin/env python3
"""
Multi-Agent Research Assistant - Streamlit Web Interface

This module provides an alternative Streamlit-based interface for the
Multi-Agent Research Assistant.
"""

import os
import warnings

# Suppress Streamlit warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page (must be first Streamlit command)
try:
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except st.errors.StreamlitAPIException:
    # Page config already set, ignore
    pass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.research_coordinator import ResearchCoordinator
from src.models import ResearchSession, ResearchPaper, Summary, Citation
from src.utils.logger import logger

# Import appropriate LLM client based on configuration
import os
if os.getenv("LLM_PROVIDER", "huggingface") == "huggingface" or \
   os.getenv("STREAMLIT_SHARING", "false").lower() == "true":
    from src.tools.huggingface_client import HuggingFaceClient as LLMClient
else:
    from src.tools.ollama_client import OllamaClient as LLMClient


class StreamlitResearchUI:
    """Streamlit-based interface for the Multi-Agent Research Assistant."""
    
    def __init__(self):
        """Initialize the Streamlit UI."""
        self.initialize_session_state()
        self.initialize_coordinator()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'coordinator' not in st.session_state:
            st.session_state.coordinator = None
        if 'current_session' not in st.session_state:
            st.session_state.current_session = None
        if 'session_history' not in st.session_state:
            st.session_state.session_history = []
        if 'research_results' not in st.session_state:
            st.session_state.research_results = None
    
    def initialize_coordinator(self):
        """Initialize the research coordinator."""
        if st.session_state.coordinator is None:
            try:
                # Use a placeholder for initialization status
                placeholder = st.empty()
                placeholder.info("🔄 Initializing research coordinator...")
                
                st.session_state.coordinator = ResearchCoordinator()
                placeholder.success("✅ Research coordinator initialized!")
                
                # Clear the placeholder after a short delay
                import time
                time.sleep(1)
                placeholder.empty()
                
            except Exception as e:
                st.error(f"❌ Failed to initialize coordinator: {e}")
                st.session_state.coordinator = None
    
    async def conduct_research(self, query: str):
        """Conduct research and update session state."""
        if not st.session_state.coordinator:
            st.error("❌ Research coordinator not available")
            return None
        
        if not query.strip():
            st.error("❌ Please enter a research query")
            return None
        
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Starting research session...")
            progress_bar.progress(10)
            
            # Conduct research
            status_text.text("📚 Retrieving papers...")
            progress_bar.progress(30)
            
            session = await st.session_state.coordinator.conduct_research(
                query, user_id="streamlit_user"
            )
            
            progress_bar.progress(80)
            status_text.text("📝 Processing results...")
            
            # Update session state
            st.session_state.current_session = session
            st.session_state.session_history.append(session)
            st.session_state.research_results = {
                'papers': session.papers,
                'summaries': session.summaries,
                'citations': session.citations
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Research completed!")
            
            return session
            
        except Exception as e:
            st.error(f"❌ Research failed: {str(e)}")
            logger.error(f"Research failed: {e}")
            return None
    
    def display_papers(self, papers: List[ResearchPaper]):
        """Display papers in Streamlit format."""
        if not papers:
            st.info("No papers found.")
            return
        
        for i, paper in enumerate(papers, 1):
            with st.expander(f"📄 Paper {i}: {paper.title}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Authors:**", ", ".join(paper.authors[:5]))
                    if len(paper.authors) > 5:
                        st.write(f"*...and {len(paper.authors) - 5} more*")
                    
                    st.write("**Abstract:**")
                    st.write(paper.abstract[:800] + "..." if len(paper.abstract) > 800 else paper.abstract)
                
                with col2:
                    st.write("**Source:**", paper.source.upper())
                    if paper.url:
                        st.link_button("🔗 View Paper", paper.url)
                    
                    # Add metrics
                    st.metric("Paper ID", paper.id[:8] + "...")
    
    def display_summaries(self, summaries: List[Summary]):
        """Display summaries in Streamlit format."""
        if not summaries:
            st.info("No summaries generated.")
            return
        
        for i, summary in enumerate(summaries, 1):
            with st.expander(f"📝 Summary {i} - Paper {summary.paper_id[:8]}", expanded=False):
                st.write("**Paper ID:**", summary.paper_id[:8] + "...")
                st.write("**Summary:**")
                st.info(summary.summary)
                
                if summary.key_findings:
                    st.write("**Key Findings:**")
                    for finding in summary.key_findings[:5]:
                        st.write(f"• {finding}")
    
    def display_citations(self, citations: List[Citation]):
        """Display citations in Streamlit format."""
        if not citations:
            st.info("No citations generated.")
            return
        
        # Group by format
        citation_formats = {}
        for citation in citations:
            format_type = citation.citation_format
            if format_type not in citation_formats:
                citation_formats[format_type] = []
            citation_formats[format_type].append(citation)
        
        for format_type, format_citations in citation_formats.items():
            with st.expander(f"📖 {format_type.upper()} Citations", expanded=False):
                for citation in format_citations:
                    st.code(citation.citation_text, language='text')
    
    def display_statistics(self):
        """Display session statistics."""
        if not st.session_state.session_history:
            st.info("No research sessions completed yet.")
            return
        
        # Summary metrics
        total_sessions = len(st.session_state.session_history)
        total_papers = sum(len(session.papers) for session in st.session_state.session_history)
        total_summaries = sum(len(session.summaries) for session in st.session_state.session_history)
        total_citations = sum(len(session.citations) for session in st.session_state.session_history)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sessions", total_sessions)
        col2.metric("Papers", total_papers)
        col3.metric("Summaries", total_summaries)
        col4.metric("Citations", total_citations)
        
        # Create visualization
        session_data = []
        for i, session in enumerate(st.session_state.session_history, 1):
            session_data.append({
                'Session': f'Session {i}',
                'Papers': len(session.papers),
                'Summaries': len(session.summaries),
                'Citations': len(session.citations)
            })
        
        if session_data:
            df = pd.DataFrame(session_data)
            fig = px.bar(df, x='Session', y=['Papers', 'Summaries', 'Citations'],
                        title='Research Session Results', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    def qa_interface(self):
        """Interactive Q&A interface."""
        if not st.session_state.current_session:
            st.warning("⚠️ Please conduct research first to enable Q&A.")
            return
        
        st.write("### Ask Questions About Your Research")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="What are the main findings? How do these papers relate to...?",
            height=100
        )
        
        if st.button("💡 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        answer = loop.run_until_complete(
                            st.session_state.coordinator.answer_question(
                                st.session_state.current_session, question
                            )
                        )
                        
                        # Display answer
                        st.success("💡 **Answer:**")
                        st.write(answer.answer_text)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Confidence", f"{answer.confidence_score:.2f}")
                        col2.metric("Sources", len(answer.source_papers))
                        
                        if answer.evidence:
                            with st.expander("📋 Supporting Evidence"):
                                for i, evidence in enumerate(answer.evidence[:3], 1):
                                    st.write(f"**{i}.** {evidence}")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("Please enter a question.")
    
    def main_interface(self):
        """Create the main Streamlit interface."""
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stTab [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTab [data-baseweb="tab"] {
            padding-left: 10px;
            padding-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.title("🔧 Controls")
            
            # System status
            st.subheader("System Status")
            if st.session_state.coordinator:
                st.success("✅ Coordinator Ready")
            else:
                st.error("❌ Coordinator Unavailable")
            
            # Quick stats
            if st.session_state.session_history:
                st.subheader("Quick Stats")
                st.metric("Sessions", len(st.session_state.session_history))
                if st.session_state.current_session:
                    current = st.session_state.current_session
                    st.metric("Current Papers", len(current.papers))
                    st.metric("Current Summaries", len(current.summaries))
            
            # Settings
            st.subheader("Settings")
            if st.button("🔄 Reset Session"):
                st.session_state.current_session = None
                st.session_state.research_results = None
                st.rerun()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Research", "❓ Q&A", "📊 Statistics", "ℹ️ About"])
        
        with tab1:
            st.header("Conduct Research")
            
            # Research input
            query = st.text_input(
                "Research Query:",
                placeholder="Enter your research topic (e.g., 'quantum computing', 'machine learning transformers')",
                help="Enter a specific research topic or question"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                research_button = st.button("🚀 Start Research", type="primary")
            
            if research_button and query:
                # Run research
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                session = loop.run_until_complete(self.conduct_research(query))
                
                if session:
                    st.success(f"✅ Research completed! Found {len(session.papers)} papers.")
                    st.rerun()
            
            # Display results
            if st.session_state.research_results:
                results = st.session_state.research_results
                
                # Results tabs
                result_tab1, result_tab2, result_tab3 = st.tabs(["📚 Papers", "📝 Summaries", "📖 Citations"])
                
                with result_tab1:
                    self.display_papers(results['papers'])
                
                with result_tab2:
                    self.display_summaries(results['summaries'])
                
                with result_tab3:
                    self.display_citations(results['citations'])
        
        with tab2:
            st.header("Interactive Q&A")
            self.qa_interface()
        
        with tab3:
            st.header("Research Statistics")
            self.display_statistics()
        
        with tab4:
            st.header("About Multi-Agent Research Assistant")
            
            st.markdown("""
            ### 🎯 Purpose
            This system helps researchers by automatically discovering, summarizing, and analyzing academic papers using multiple AI agents working in coordination.
            
            ### 🤖 How It Works
            1. **Paper Retrieval:** Searches ArXiv and web sources for relevant papers
            2. **Summarization:** Uses LLM to generate comprehensive summaries  
            3. **Q&A:** Answers questions based on paper content
            4. **Citations:** Generates properly formatted academic citations
            5. **Coordination:** CrewAI orchestrates all agents for optimal workflow
            
            ### 🛠️ Technology Stack
            - **Multi-Agent Framework:** CrewAI
            - **LLM:** Hugging Face Inference API (Cloud) / Ollama (Local)
            - **Search:** ArXiv API, DuckDuckGo
            - **UI:** Streamlit
            - **Language:** Python
            - **Deployment:** Streamlit Cloud
            
            ### 📈 Features
            - ✅ Multi-source paper discovery
            - ✅ Intelligent summarization
            - ✅ Context-aware Q&A
            - ✅ Multiple citation formats
            - ✅ Session management
            - ✅ Real-time progress tracking
            
            *Built with GitHub Copilot assistance*
            """)


def main():
    """Main function to run the Streamlit app."""
    ui = StreamlitResearchUI()
    ui.main_interface()


if __name__ == "__main__":
    main()
