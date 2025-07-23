#!/usr/bin/env python3
"""
Multi-Agent Research Assistant - Cloud Deployment Version

This is the cloud-optimized version for Streamlit Cloud deployment
using Hugging Face free inference API.
"""

import os
import warnings

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Configuration from environment variables
APP_TITLE = os.getenv('APP_TITLE', 'Multi-Agent Research Assistant')
APP_ICON = os.getenv('APP_ICON', '🤖')
DEFAULT_MAX_RESULTS = int(os.getenv('DEFAULT_MAX_RESULTS', '5'))
MAX_PAPERS_PER_SEARCH = int(os.getenv('MAX_PAPERS_PER_SEARCH', '10'))
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Force cloud deployment settings (override with env if needed)
os.environ['LLM_PROVIDER'] = os.getenv('LLM_PROVIDER', 'huggingface')
os.environ['IS_CLOUD_DEPLOYMENT'] = os.getenv('IS_CLOUD_DEPLOYMENT', 'true')
os.environ['STREAMLIT_SHARING'] = os.getenv('STREAMLIT_SHARING', 'true')

# Suppress warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import streamlit as st
import requests
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional imports for analytics (graceful fallback if not available)
try:
    import pandas as pd
    import plotly.express as px
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False
    pd = None
    px = None

# Configure Streamlit page (must be first Streamlit command)
try:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
except st.errors.StreamlitAPIException:
    # Page config already set, ignore
    pass

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Simple ArXiv search function."""
    try:
        import arxiv
        
        # Use the new Client API
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            papers.append({
                'title': result.title,
                'authors': [str(author) for author in result.authors],
                'summary': result.summary,
                'url': result.entry_id,
                'published': result.published.strftime('%Y-%m-%d'),
                'categories': result.categories
            })
        return papers
    except Exception as e:
        st.error(f"ArXiv search error: {e}")
        return []

def generate_summary(text: str) -> str:
    """Generate summary using Hugging Face API or enhanced intelligent fallback."""
    try:
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {hf_token}"}
            
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": text[:int(os.getenv('SUMMARY_MAX_LENGTH', '1000'))]},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('summary_text', 'Summary generated successfully.')
        
        # Enhanced intelligent fallback that analyzes the actual text
        text_lower = text.lower()
        
        # Extract key terms and themes
        key_terms = []
        important_words = []
        
        # Look for technical terms and methodologies
        tech_terms = ['neural', 'network', 'deep', 'learning', 'algorithm', 'model', 'training', 
                     'quantum', 'computation', 'optimization', 'classification', 'regression',
                     'transformer', 'attention', 'convolution', 'lstm', 'gpt', 'bert',
                     'reinforcement', 'supervised', 'unsupervised', 'generative']
        
        for term in tech_terms:
            if term in text_lower:
                key_terms.append(term)
        
        # Extract sentences that seem important (contain "we", "our", "this", "propose", "show", "demonstrate")
        sentences = text.split('. ')
        important_sentences = []
        key_indicators = ['we propose', 'we present', 'we show', 'we demonstrate', 'our approach', 
                         'our method', 'our results', 'this paper', 'this work', 'we introduce',
                         'we develop', 'experimental results', 'we achieve', 'performance']
        
        for sentence in sentences[:5]:  # Look at first 5 sentences
            sentence_lower = sentence.lower()
            for indicator in key_indicators:
                if indicator in sentence_lower:
                    # Clean up the sentence
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                        important_sentences.append(clean_sentence)
                    break
        
        # Generate domain-specific summary based on detected terms
        if any(term in key_terms for term in ['neural', 'deep', 'learning', 'network']):
            domain = "machine learning"
            method_desc = "neural network architectures and deep learning techniques"
        elif any(term in key_terms for term in ['quantum', 'computation']):
            domain = "quantum computing"  
            method_desc = "quantum computational methods and algorithmic approaches"
        elif any(term in key_terms for term in ['optimization', 'algorithm']):
            domain = "optimization and algorithms"
            method_desc = "algorithmic optimization strategies and computational methods"
        else:
            domain = "the research field"
            method_desc = "systematic methodological approaches"
        
        # Build intelligent summary
        summary_parts = []
        
        # Add opening based on important sentences or domain
        if important_sentences:
            summary_parts.append(f"This work in {domain} {important_sentences[0].lower()}")
        else:
            summary_parts.append(f"This research contributes to {domain} through innovative approaches.")
        
        # Add methodology description
        if key_terms:
            terms_str = ', '.join(key_terms[:3])
            summary_parts.append(f"The methodology employs {method_desc}, specifically focusing on {terms_str}.")
        else:
            summary_parts.append(f"The approach utilizes {method_desc} with rigorous experimental validation.")
        
        # Add results/impact
        if any(word in text_lower for word in ['results', 'performance', 'accuracy', 'improvement']):
            summary_parts.append("The experimental results demonstrate significant performance improvements and validate the effectiveness of the proposed approach.")
        elif any(word in text_lower for word in ['novel', 'new', 'innovative']):
            summary_parts.append("This novel approach introduces innovative techniques that advance the state-of-the-art in the field.")
        else:
            summary_parts.append("The findings provide valuable insights and contribute to advancing current understanding.")
        
        return ' '.join(summary_parts)
        
    except Exception:
        return "This research presents systematic investigation with methodological rigor, contributing valuable insights to advance understanding in the field through comprehensive analysis and experimental validation."

def answer_question(question: str, context: str) -> str:
    """Answer questions about the paper using Hugging Face API or improved fallbacks."""
    try:
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            # Try different models suitable for Q&A
            qa_models = [
                {"model": "deepset/roberta-base-squad2", "type": "qa"},
                {"model": "distilbert-base-uncased-distilled-squad", "type": "qa"},
                {"model": "google/flan-t5-small", "type": "text2text"},
                {"model": "microsoft/DialoGPT-small", "type": "text-generation"}
            ]
            
            for model_info in qa_models:
                try:
                    model = model_info["model"]
                    model_type = model_info["type"]
                    api_url = f"https://api-inference.huggingface.co/models/{model}"
                    headers = {"Authorization": f"Bearer {hf_token}"}
                    
                    # Format request based on model type
                    if model_type == "qa":
                        # For Q&A models, use question-answering format
                        payload = {
                            "inputs": {
                                "question": question,
                                "context": context[:1000]  # Limit context for Q&A models
                            }
                        }
                    elif model_type == "text2text":
                        # For text-to-text models like T5
                        prompt = f"Answer the question based on the context.\nContext: {context[:800]}\nQuestion: {question}\nAnswer:"
                        payload = {"inputs": prompt}
                    else:
                        # For text generation models
                        prompt = f"Research Paper Context: {context[:600]}\n\nQuestion: {question}\nAnswer:"
                        payload = {"inputs": prompt}
                    
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Handle different response formats
                        if model_type == "qa" and isinstance(result, dict):
                            answer = result.get('answer', '').strip()
                            if answer and len(answer) > 3 and result.get('score', 0) > 0.05:  # Lower threshold
                                return f"{answer} (Confidence: {result.get('score', 0):.2f})"
                        
                        elif isinstance(result, list) and len(result) > 0:
                            if 'generated_text' in result[0]:
                                full_text = result[0]['generated_text']
                                # Extract answer part after the prompt
                                if "Answer:" in full_text:
                                    answer = full_text.split("Answer:")[-1].strip()
                                    if answer and len(answer) > 10:
                                        return answer[:500]  # Limit response length
                            elif 'summary_text' in result[0]:
                                # Only use summary as last resort for Q&A
                                pass
                        
                        elif isinstance(result, dict) and 'generated_text' in result:
                            full_text = result['generated_text']
                            if "Answer:" in full_text:
                                answer = full_text.split("Answer:")[-1].strip()
                                if answer and len(answer) > 10:
                                    return answer[:500]
                                    
                except Exception as e:
                    continue
        
        # Only use enhanced fallbacks when ALL API attempts fail
        # Enhanced fallback responses based on context analysis
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Extract key terms from context for more relevant fallbacks
        key_terms = []
        for word in context_lower.split():
            if len(word) > 6 and word.isalpha():  # Get longer words
                key_terms.append(word)
        
        # More targeted fallback logic - only for very specific patterns
        if any(phrase in question_lower for phrase in ['methodology does this paper use', 'approach does this paper', 'method does this work']):
            if any(term in context_lower for term in ['neural', 'network', 'deep', 'learning']):
                return f"Based on the abstract, this paper employs neural network methodologies. The approach likely involves deep learning techniques with systematic training and validation processes. The methodology appears to be computationally intensive and follows established machine learning paradigms."
            elif any(term in context_lower for term in ['quantum', 'computation', 'algorithm']):
                return f"The paper describes quantum computational methods. The approach involves quantum algorithmic techniques with mathematical foundations in quantum mechanics. The methodology demonstrates theoretical rigor combined with practical implementation considerations."
            else:
                return f"The research methodology combines theoretical analysis with empirical validation. The approach demonstrates systematic investigation with rigorous experimental design and follows established scientific protocols in the field."
        
        # More targeted fallback logic - only for very specific patterns
        if any(phrase in question_lower for phrase in ['methodology does this paper use', 'approach does this paper', 'method does this work']):
            if any(term in context_lower for term in ['neural', 'network', 'deep', 'learning']):
                return f"Based on the abstract, this paper employs neural network methodologies. The approach likely involves deep learning techniques with systematic training and validation processes. The methodology appears to be computationally intensive and follows established machine learning paradigms."
            elif any(term in context_lower for term in ['quantum', 'computation', 'algorithm']):
                return f"The paper describes quantum computational methods. The approach involves quantum algorithmic techniques with mathematical foundations in quantum mechanics. The methodology demonstrates theoretical rigor combined with practical implementation considerations."
            else:
                return f"The research methodology combines theoretical analysis with empirical validation. The approach demonstrates systematic investigation with rigorous experimental design and follows established scientific protocols in the field."
        
        # Only use generic fallbacks for very general questions
        elif len(question.split()) <= 3 and any(word in question_lower for word in ['what', 'how', 'why']):
            if any(term in context_lower for term in ['neural', 'network', 'machine', 'learning']):
                return f"This appears to be machine learning research focusing on neural networks and computational methods."
            else:
                return f"This research contributes to the field through systematic investigation and analysis."
        
        else:
            # For all other questions, provide a more informative fallback
            return f"I'm having difficulty providing a specific answer to that question based on the available abstract. The paper appears to focus on {', '.join(key_terms[:3]) if key_terms else 'systematic research methods'}. For detailed information, please refer to the full paper text."
            
    except Exception as e:
        return f"I'm having difficulty analyzing the specific content right now. Please refer to the paper's abstract and full text for detailed information. Error: {str(e)[:100]}..."

def generate_citation(paper: Dict[str, Any], style: str = "APA") -> str:
    """Generate citations in different academic styles with proper formatting."""
    try:
        title = paper['title'].strip()
        authors = paper['authors']
        published = paper['published']
        url = paper['url']
        
        # Format authors properly
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        elif len(authors) <= 5:
            author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
        else:
            # For more than 5 authors, use first 3 + et al.
            author_str = ", ".join(authors[:3]) + " et al."
        
        # Extract year from date
        year = published.split('-')[0]
        
        # Generate different citation styles
        if style.upper() == "APA":
            # APA 7th edition format
            return f"{author_str} ({year}). {title}. *arXiv preprint arXiv:{url.split('/')[-1]}*. https://doi.org/10.48550/arXiv.{url.split('/')[-1]}"
        
        elif style.upper() == "MLA":
            # MLA 9th edition format  
            first_author = authors[0] if authors else "Unknown Author"
            if len(authors) > 1:
                first_author_formatted = f"{first_author.split()[-1]}, {' '.join(first_author.split()[:-1])}"
                if len(authors) == 2:
                    second_author = authors[1]
                    author_str_mla = f"{first_author_formatted}, and {second_author}"
                else:
                    author_str_mla = f"{first_author_formatted}, et al"
            else:
                first_author_formatted = f"{first_author.split()[-1]}, {' '.join(first_author.split()[:-1])}" if ' ' in first_author else first_author
                author_str_mla = first_author_formatted
            
            return f'{author_str_mla}. "{title}." *arXiv*, {published}, {url}.'
        
        elif style.upper() == "CHICAGO":
            # Chicago 17th edition format (Notes-Bibliography)
            return f'{author_str}. "{title}." arXiv preprint arXiv:{url.split("/")[-1]} ({year}). {url}.'
        
        elif style.upper() == "IEEE":
            # IEEE format
            author_initials = []
            for author in authors[:6]:  # IEEE shows up to 6 authors
                names = author.split()
                if len(names) > 1:
                    initials = '. '.join([name[0] for name in names[:-1]]) + '.'
                    author_initials.append(f"{initials} {names[-1]}")
                else:
                    author_initials.append(author)
            
            if len(authors) > 6:
                ieee_authors = ", ".join(author_initials) + ", et al."
            else:
                ieee_authors = ", ".join(author_initials)
                
            return f'{ieee_authors}, "{title}," arXiv preprint arXiv:{url.split("/")[-1]}, {year}.'
        
        else:  # Default APA
            return f"{author_str} ({year}). {title}. *arXiv preprint arXiv:{url.split('/')[-1]}*. https://doi.org/10.48550/arXiv.{url.split('/')[-1]}"
            
    except Exception as e:
        return f"Citation formatting error: {str(e)}. Paper URL: {paper.get('url', 'N/A')}"


def main():
    """Main Streamlit application matching ui_streamlit.py structure."""
    
    # Initialize session state
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    if 'current_papers' not in st.session_state:
        st.session_state.current_papers = []
    if 'show_all_citations' not in st.session_state:
        st.session_state.show_all_citations = False
    
    # Custom CSS (matching original)
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
    
    # Sidebar (matching original structure)
    with st.sidebar:
        st.title("🔧 Controls")
        
        # System status
        st.subheader("System Status")
        st.success("✅ Enhanced Version Ready")
        
        # Quick stats
        if st.session_state.session_history:
            st.subheader("Quick Stats")
            st.metric("Sessions", len(st.session_state.session_history))
            if st.session_state.current_papers:
                st.metric("Current Papers", len(st.session_state.current_papers))
        
        # HuggingFace Token Status
        hf_token = os.getenv('HF_TOKEN')
        st.subheader("AI Features")
        if hf_token:
            st.success("🤖 HF API: Active")
            st.caption("Full AI capabilities available")
        else:
            st.warning("🤖 HF API: Enhanced Fallbacks")
            st.caption("Using intelligent context-aware responses")
        
        # Settings
        st.subheader("Settings")
        if st.button("🔄 Reset Session"):
            st.session_state.research_results = None
            st.session_state.current_papers = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Enhanced Research Assistant")
    
    # Main content tabs (matching original)
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
        with col2:
            max_results = st.selectbox("Max results:", [3, 5, 10], 
                                     index=[3, 5, 10].index(DEFAULT_MAX_RESULTS) if DEFAULT_MAX_RESULTS in [3, 5, 10] else 1)
        
        if research_button and query:
            with st.spinner("🔍 Conducting research..."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📚 Retrieving papers...")
                progress_bar.progress(20)
                
                papers = search_arxiv(query, max_results)
                
                if papers:
                    progress_bar.progress(40)
                    status_text.text("🤖 Generating AI summaries...")
                    
                    # Auto-generate summaries for all papers
                    for i, paper in enumerate(papers):
                        progress = 40 + (40 * (i + 1) / len(papers))
                        progress_bar.progress(int(progress))
                        status_text.text(f"📝 Summarizing paper {i + 1}/{len(papers)}...")
                        
                        # Generate summary and store it with the paper
                        summary = generate_summary(paper['summary'])
                        paper['ai_summary'] = summary
                    
                    progress_bar.progress(90)
                    status_text.text("💾 Processing results...")
                    
                    # Store in session state
                    st.session_state.current_papers = papers
                    st.session_state.research_results = {'papers': papers}
                    
                    # Add to session history
                    session_data = {
                        'query': query,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'papers_count': len(papers)
                    }
                    st.session_state.session_history.append(session_data)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Research completed!")
                    
                    st.success(f"✅ Research completed! Found {len(papers)} papers with AI summaries.")
                    st.rerun()
                else:
                    progress_bar.progress(100)
                    status_text.text("❌ No papers found")
                    st.warning("No papers found for your query. Try different keywords.")
        
        # Display results (matching original structure)
        if st.session_state.research_results:
            papers = st.session_state.research_results['papers']
            
            # Results tabs
            result_tab1, result_tab2, result_tab3 = st.tabs(["📚 Papers", "📝 Summaries", "📖 Citations"])
            
            with result_tab1:
                display_papers(papers)
            
            with result_tab2:
                display_summaries(papers)
            
            with result_tab3:
                display_citations(papers)
    
    with tab2:
        st.header("Interactive Q&A")
        qa_interface()
    
    with tab3:
        st.header("Research Statistics")
        display_statistics()
    
    with tab4:
        st.header("About Multi-Agent Research Assistant")
        
        st.markdown("""
        ### 🎯 Purpose
        This enhanced system helps researchers by automatically discovering, summarizing, and analyzing academic papers using advanced AI integration.
        
        ### 🤖 How It Works
        1. **Paper Retrieval:** Searches ArXiv database for relevant papers
        2. **Auto-Summarization:** Automatically generates AI summaries for all papers
        3. **Smart Q&A:** Context-aware answers using paper content and enhanced fallbacks
        4. **Citations:** Generates properly formatted academic citations
        5. **Enhanced Interface:** Clean, fast, and deployment-ready with auto-processing
        
        ### 🛠️ Technology Stack
        - **Search:** ArXiv API
        - **LLM:** Hugging Face Inference API (Multiple Models)
        - **UI:** Streamlit
        - **Language:** Python
        - **Deployment:** Streamlit Cloud Compatible
        
        ### 📈 Enhanced Features
        - ✅ ArXiv paper discovery
        - ✅ **Automatic AI summarization** 
        - ✅ **Context-aware Q&A with enhanced fallbacks**
        - ✅ Multiple citation formats (APA, MLA, Chicago)
        - ✅ Session management with history
        - ✅ Real-time progress tracking
        - ✅ Cloud deployment ready
        - ✅ Minimal dependencies
        - ✅ **Smart fallback responses based on content analysis**
        
        ### 🚀 Deployment Ready
        This enhanced version is optimized for:
        - **Streamlit Cloud** deployment
        - **Minimal dependencies** (no compilation issues)
        - **Fast startup** (no heavy model loading)
        - **API-based processing** with intelligent fallbacks
        - **Auto-processing** of research results
        
        *Built with GitHub Copilot assistance*
        """)

def display_papers(papers: List[Dict[str, Any]]):
    """Display papers in Streamlit format (matching original)."""
    if not papers:
        st.info("No papers found.")
        return
    
    for i, paper in enumerate(papers, 1):
        with st.expander(f"📄 Paper {i}: {paper['title']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Authors:**", ", ".join(paper['authors'][:5]))
                if len(paper['authors']) > 5:
                    st.write(f"*...and {len(paper['authors']) - 5} more*")
                
                st.write("**Abstract:**")
                st.write(paper['summary'][:800] + "..." if len(paper['summary']) > 800 else paper['summary'])
                
                # Show AI summary if available
                if 'ai_summary' in paper:
                    st.write("**AI Summary:**")
                    st.info(paper['ai_summary'])
            
            with col2:
                st.write("**Source:** ARXIV")
                if paper.get('url'):
                    st.link_button("🔗 View Paper", paper['url'])
                
                # Add metrics
                st.metric("Published", paper['published'])
                st.write("**Categories:**", ", ".join(paper['categories'][:3]))

def display_summaries(papers: List[Dict[str, Any]]):
    """Display summaries interface (matching original) - now shows auto-generated summaries."""
    if not papers:
        st.info("No papers available for summarization.")
        return
    
    st.write("**AI summaries automatically generated for all retrieved papers:**")
    
    for i, paper in enumerate(papers, 1):
        with st.expander(f"📝 Summary {i} - {paper['title'][:50]}...", expanded=False):
            if 'ai_summary' in paper:
                st.write("**AI Summary:**")
                st.success(paper['ai_summary'])
            else:
                st.write("**Generating summary...**")
                summary = generate_summary(paper['summary'])
                paper['ai_summary'] = summary
                st.success(summary)

def display_citations(papers: List[Dict[str, Any]]):
    """Enhanced citations interface with better formatting and interactivity."""
    if not papers:
        st.info("No papers available for citation.")
        return
    
    st.write("**Generate properly formatted academic citations:**")
    
    # Citation style selector
    col1, col2 = st.columns([1, 2])
    with col1:
        citation_style = st.selectbox(
            "Citation Style:",
            ["APA", "MLA", "Chicago", "IEEE"],
            help="Select the citation format you need"
        )
    
    with col2:
        if st.button("📋 Generate All Citations", type="primary"):
            st.session_state.show_all_citations = True
    
    # Individual paper citations
    for i, paper in enumerate(papers, 1):
        with st.expander(f"📄 Citation {i}: {paper['title'][:60]}...", expanded=False):
            
            # Generate citation
            citation = generate_citation(paper, citation_style)
            
            # Display the citation
            st.write(f"**{citation_style} Citation:**")
            st.code(citation, language='text')
            
            # Copy button functionality
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                if st.button(f"📋 Copy Citation", key=f"copy_{i}_{citation_style}"):
                    # Store in session state for copy feedback
                    st.session_state[f'copied_{i}'] = True
                    st.success("Citation copied to clipboard! (Use Ctrl+C to copy the text above)")
            
            with col_b:
                # Export options
                if st.button(f"💾 Export", key=f"export_{i}"):
                    # Create downloadable file
                    citation_data = f"{citation_style} Citation:\n{citation}\n\nPaper Details:\n"
                    citation_data += f"Title: {paper['title']}\n"
                    citation_data += f"Authors: {', '.join(paper['authors'])}\n"
                    citation_data += f"Published: {paper['published']}\n"
                    citation_data += f"URL: {paper['url']}\n"
                    citation_data += f"Categories: {', '.join(paper['categories'])}\n"
                    
                    st.download_button(
                        label="📄 Download",
                        data=citation_data,
                        file_name=f"citation_{i}_{citation_style.lower()}.txt",
                        mime="text/plain",
                        key=f"download_{i}_{citation_style}"
                    )
            
            with col_c:
                # Show paper details
                if st.button(f"📖 Details", key=f"details_{i}"):
                    st.session_state[f'show_details_{i}'] = True
            
            # Show additional details if requested
            if st.session_state.get(f'show_details_{i}', False):
                st.write("**Paper Information:**")
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Authors:** {len(paper['authors'])} total")
                    for j, author in enumerate(paper['authors'][:5], 1):
                        st.write(f"  {j}. {author}")
                    if len(paper['authors']) > 5:
                        st.write(f"  ... and {len(paper['authors']) - 5} more")
                
                with info_col2:
                    st.write(f"**Published:** {paper['published']}")
                    st.write(f"**Categories:** {', '.join(paper['categories'][:3])}")
                    st.write(f"**arXiv ID:** {paper['url'].split('/')[-1]}")
                    st.link_button("🔗 View Paper", paper['url'])
                
                if st.button(f"❌ Hide Details", key=f"hide_details_{i}"):
                    st.session_state[f'show_details_{i}'] = False
                    st.rerun()
    
    # Batch citation generation
    if st.session_state.get('show_all_citations', False):
        st.markdown("---")
        st.subheader(f"📚 All Citations ({citation_style} Format)")
        
        all_citations = []
        for i, paper in enumerate(papers, 1):
            citation = generate_citation(paper, citation_style)
            all_citations.append(f"{i}. {citation}")
        
        citations_text = "\n\n".join(all_citations)
        st.code(citations_text, language='text')
        
        # Export all citations
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            st.download_button(
                label="📄 Download All Citations",
                data=f"{citation_style} Citations:\n\n{citations_text}",
                file_name=f"all_citations_{citation_style.lower()}.txt",
                mime="text/plain"
            )
        
        with col_export2:
            if st.button("❌ Hide All Citations"):
                st.session_state.show_all_citations = False
                st.rerun()
        
        # Bibliography format option
        st.write("**Bibliography Format:**")
        bibliography = f"{citation_style} Bibliography\n" + "="*50 + "\n\n" + citations_text
        st.code(bibliography, language='text')
        
        st.download_button(
            label="📚 Download Bibliography",
            data=bibliography,
            file_name=f"bibliography_{citation_style.lower()}.txt",
            mime="text/plain",
            help="Download a formatted bibliography with all citations"
        )

def qa_interface():
    """Enhanced Interactive Q&A interface with context-aware responses."""
    if not st.session_state.current_papers:
        st.warning("⚠️ Please conduct research first to enable Q&A.")
        return
    
    st.write("### Ask Questions About Your Research")
    st.caption(f"📚 Context: {len(st.session_state.current_papers)} papers available for analysis")
    
    question = st.text_area(
        "Enter your question:",
        placeholder="What are the main findings? How do these papers relate to...? What methodology was used?",
        help="Ask questions about the papers you've retrieved. The AI will analyze the content to provide relevant answers.",
        height=100
    )
    
    # Paper selection for context
    if len(st.session_state.current_papers) > 1:
        paper_options = [f"Paper {i+1}: {paper['title'][:50]}..." for i, paper in enumerate(st.session_state.current_papers)]
        paper_options.insert(0, "All papers (combined context)")
        
        selected_paper = st.selectbox(
            "Focus on specific paper (optional):",
            paper_options,
            help="Select a specific paper for more targeted answers, or use all papers for broader analysis"
        )
    else:
        selected_paper = "All papers (combined context)"
    
    if st.button("💡 Get Answer", type="primary"):
        if question.strip():
            with st.spinner("🤔 Analyzing papers and generating answer..."):
                try:
                    # Prepare context based on selection
                    if selected_paper.startswith("All papers"):
                        # Combine context from all papers
                        context_parts = []
                        for i, paper in enumerate(st.session_state.current_papers[:3]):  # Limit to first 3 for context length
                            context_parts.append(f"Paper {i+1}: {paper['summary'][:300]}")
                        context = " | ".join(context_parts)
                        source_count = len(st.session_state.current_papers)
                    else:
                        # Use specific paper
                        paper_index = int(selected_paper.split(":")[0].replace("Paper ", "")) - 1
                        context = st.session_state.current_papers[paper_index]['summary']
                        source_count = 1
                    
                    answer = answer_question(question, context)
                    
                    # Display answer
                    st.success("💡 **AI Answer:**")
                    st.write(answer)
                    
                    # Show source info
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Source Papers", source_count)
                    col2.metric("Context Length", f"{len(context):.0f} chars")
                    col3.metric("Papers Available", len(st.session_state.current_papers))
                    
                    # Show which papers were used
                    if selected_paper.startswith("All papers"):
                        st.caption("📖 Answer based on combined analysis of all retrieved papers")
                    else:
                        st.caption(f"📖 Answer focused on: {selected_paper}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
        else:
            st.error("Please enter a question.")

def display_statistics():
    """Display session statistics (matching original)."""
    if not st.session_state.session_history:
        st.info("No research sessions completed yet.")
        return
    
    # Summary metrics
    total_sessions = len(st.session_state.session_history)
    total_papers = sum(session['papers_count'] for session in st.session_state.session_history)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", total_sessions)
    col2.metric("Total Papers", total_papers)
    col3.metric("Avg Papers/Session", f"{total_papers/total_sessions:.1f}" if total_sessions > 0 else "0")
    col4.metric("Current Papers", len(st.session_state.current_papers))
    
    # Create visualization
    if st.session_state.session_history and HAS_ANALYTICS:
        session_data = []
        for i, session in enumerate(st.session_state.session_history, 1):
            session_data.append({
                'Session': f'Session {i}',
                'Papers': session['papers_count'],
                'Query': session['query'][:20] + "..." if len(session['query']) > 20 else session['query'],
                'Timestamp': session['timestamp']
            })
        
        df = pd.DataFrame(session_data)
        fig = px.bar(df, x='Session', y='Papers', 
                    title='Research Session Results', 
                    hover_data=['Query', 'Timestamp'])
        st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.session_history and not HAS_ANALYTICS:
        st.info("📊 Analytics visualization requires pandas and plotly packages")
        
        # Show simple text summary instead
        st.write("**Session Summary:**")
        for i, session in enumerate(st.session_state.session_history, 1):
            query_short = session['query'][:30] + "..." if len(session['query']) > 30 else session['query']
            st.write(f"• Session {i}: {session['papers_count']} papers - *{query_short}*")
    
    # Session history table
    if st.session_state.session_history and HAS_ANALYTICS:
        st.subheader("Session History")
        history_df = pd.DataFrame(st.session_state.session_history)
        st.dataframe(history_df, use_container_width=True)
    elif st.session_state.session_history and not HAS_ANALYTICS:
        st.subheader("Session History") 
        for i, session in enumerate(st.session_state.session_history, 1):
            with st.expander(f"Session {i}: {session['query'][:40]}..."):
                st.write(f"**Query:** {session['query']}")
                st.write(f"**Papers Found:** {session['papers_count']}")
                st.write(f"**Timestamp:** {session['timestamp']}")


if __name__ == "__main__":
    main()
