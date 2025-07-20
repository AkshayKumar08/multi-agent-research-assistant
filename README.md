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
- **Citation Agent**: Extracts BibTeX or citations
- **Planner**: Manages workflow between agents (handled by CrewAI)

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

## Current Status

### Completed
- **Step 1**: Project setup and basic structure
- **Step 2**: Retriever Agent (ArXiv + DuckDuckGo search)

### In Progress
- **Step 3**: Summarizer Agent (planned next)

### Upcoming
- **Step 4**: Q&A Agent
- **Step 5**: Citation Agent
- **Step 6**: CrewAI Integration
- **Step 7**: UI Implementation

## License

MIT License - see LICENSE file for details

<sub><sup>I built it copilot. 90% of code is written by copilot</sup></sub>