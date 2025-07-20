# Multi-Agent Research Assistant

A specialized multi-agent system that helps researchers by coordinating LLM agents to retrieve, summarize, and analyze academic papers.

## Project Overview

This system allows users to input a research topic and get:
- Relevant academic papers from ArXiv and other sources
- Summarized key findings
- Answers to follow-up questions
- Suggestions for future research directions

## Tech Stack

- **LLM Agents**: Ollama + Mistral 7B / LLaMA 3
- **Multi-agent coordination**: CrewAI + LangChain
- **Search & Retrieval**: ArXiv API, DuckDuckGo
- **Vector Store**: FAISS
- **Memory/Storage**: SQLite / JSON files / Chroma DB
- **UI**: Gradio or Streamlit (optional)

## Agent Roles

- **Retriever**: Fetches relevant papers using ArXiv or DuckDuckGo
- **Summarizer**: Summarizes paper abstracts or full text
- **Q&A Agent**: Handles user queries about the research
- **Citation Agent**: Extracts and formats academic citations
- **Research Coordinator**: Orchestrates multi-agent workflows using CrewAI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and setup Ollama:
```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
# Pull required models
ollama pull mistral:7b
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Quick Start with CrewAI Integration
```bash
# Run the main application with interactive research
python src/main.py

# Run CrewAI demo
python demo_crewai.py

# Validate CrewAI integration
python validate_crewai.py
```

### Individual Agent Demos
```bash
# Test individual agents
python demo_retriever.py    # Paper retrieval
python demo_summarizer.py   # Paper summarization
python demo_qa.py          # Q&A functionality
python demo_citation.py    # Citation generation
```

### Research Workflow
1. **Input**: Research query/topic
2. **Retrieval**: Agents search ArXiv and web sources
3. **Summarization**: Key findings extracted using LLM
4. **Q&A**: Interactive questions about the research
5. **Citations**: Formatted academic citations generated
6. **Coordination**: CrewAI orchestrates the entire workflow

## 🚀 Current Status

### ✅ Completed
- **Step 1**: Project setup and basic structure
- **Step 2**: Retriever Agent (ArXiv + DuckDuckGo search)
- **Step 3**: Summarizer Agent (Ollama LLM integration)
- **Step 4**: Q&A Agent (Interactive research questions)
- **Step 5**: Citation Agent (Academic citations & bibliography)
- **Step 6**: CrewAI Integration (Multi-agent coordination)

### 🔄 In Progress
- **Step 7**: UI Implementation (planned next)

### Upcoming
- **Step 7**: UI Implementation (Gradio/Streamlit)
## License

MIT License - see LICENSE file for details

<sub><sup>I built it copilot. 90% of code is written by copilot</sup></sub>