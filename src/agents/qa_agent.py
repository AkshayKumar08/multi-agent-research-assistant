"""
Q&A Agent for answering research questions based on retrieved papers and summaries.
"""
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from src.models import (
    ResearchPaper, Summary, Question, Answer, ConversationContext, 
    AgentTask, ResearchSession, ResearchQuery
)
from src.utils.logger import get_logger
from config.settings import config

# Import appropriate LLM client based on configuration
import os
if os.getenv("LLM_PROVIDER", "huggingface") == "huggingface" or \
   os.getenv("STREAMLIT_SHARING", "false").lower() == "true":
    from src.tools.huggingface_client import HuggingFaceClient as LLMClient
else:
    from src.tools.ollama_client import OllamaClient as LLMClient

logger = get_logger("qa_agent")


class QAAgent:
    """Agent responsible for answering research questions using LLM and research context."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the Q&A Agent.
        
        Args:
            llm_client: Optional LLM client (creates new one if None)
        """
        self.llm_client = llm_client or LLMClient()
        self.agent_id = "qa_agent"
        
        # Verify LLM availability
        provider = os.getenv("LLM_PROVIDER", "huggingface")
        if hasattr(self.llm_client, 'is_available') and not self.llm_client.is_available():
            logger.warning(f"{provider} client not available. Q&A will use fallback responses.")
        else:
            logger.info(f"Q&A Agent initialized with {provider} provider")
        
        logger.info("Q&A Agent initialized")
    
    def answer_question(
        self,
        question: Question,
        context: ConversationContext,
        papers: List[ResearchPaper],
        summaries: List[Summary]
    ) -> Answer:
        """Answer a research question using available context.
        
        Args:
            question: Question to answer
            context: Conversation context
            papers: Available research papers
            summaries: Available summaries
            
        Returns:
            Answer object with response and metadata
        """
        logger.info(f"Answering question: {question.question_text[:50]}...")
        
        try:
            # Build context for the LLM
            llm_context = self._build_llm_context(question, papers, summaries)
            
            # Generate answer using LLM
            answer_text, reasoning = self._generate_answer(question.question_text, llm_context)
            
            # Extract evidence and calculate confidence
            evidence, source_papers = self._extract_evidence(answer_text, papers, summaries)
            confidence_score = self._calculate_confidence(answer_text, evidence)
            
            # Create answer object
            answer = Answer(
                answer_id=str(uuid.uuid4()),
                question_id=question.question_id,
                answer_text=answer_text,
                confidence_score=confidence_score,
                source_papers=source_papers,
                evidence=evidence,
                agent_reasoning=reasoning
            )
            
            logger.info(f"Answer generated with confidence: {confidence_score:.2f}")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return Answer(
                answer_id=str(uuid.uuid4()),
                question_id=question.question_id,
                answer_text=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                confidence_score=0.0,
                agent_reasoning=f"Error occurred during processing: {str(e)}"
            )
    
    def create_conversation_context(
        self,
        session: ResearchSession
    ) -> ConversationContext:
        """Create a conversation context from a research session.
        
        Args:
            session: Research session containing papers and summaries
            
        Returns:
            ConversationContext for Q&A
        """
        context = ConversationContext(
            context_id=str(uuid.uuid4()),
            session_id=session.session_id,
            papers_in_context=[paper.id for paper in session.papers],
            summaries_in_context=[summary.paper_id for summary in session.summaries]
        )
        
        logger.info(f"Created conversation context with {len(session.papers)} papers")
        return context
    
    def ask_followup_question(
        self,
        question_text: str,
        context: ConversationContext,
        papers: List[ResearchPaper],
        summaries: List[Summary]
    ) -> Tuple[Question, Answer]:
        """Ask a follow-up question in an existing conversation.
        
        Args:
            question_text: The follow-up question
            context: Existing conversation context
            papers: Available research papers
            summaries: Available summaries
            
        Returns:
            Tuple of (Question, Answer)
        """
        # Create question object
        question = Question(
            question_id=str(uuid.uuid4()),
            question_text=question_text,
            context_type="conversation",
            context_ids=[context.context_id]
        )
        
        # Generate answer with conversation history
        answer = self.answer_question(question, context, papers, summaries)
        
        # Update context
        context.questions.append(question)
        context.answers.append(answer)
        context.updated_at = datetime.now()
        
        return question, answer
    
    def _build_llm_context(
        self,
        question: Question,
        papers: List[ResearchPaper],
        summaries: List[Summary]
    ) -> str:
        """Build context string for LLM from papers and summaries.
        
        Args:
            question: Question being asked
            papers: Available papers
            summaries: Available summaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add relevant papers
        relevant_papers = self._find_relevant_papers(question.question_text, papers)
        if relevant_papers:
            context_parts.append("=== RELEVANT RESEARCH PAPERS ===")
            for paper in relevant_papers[:5]:  # Limit to top 5 papers
                context_parts.append(f"Title: {paper.title}")
                context_parts.append(f"Authors: {', '.join(paper.authors)}")
                context_parts.append(f"Abstract: {paper.abstract}")
                context_parts.append("")
        
        # Add relevant summaries
        relevant_summaries = self._find_relevant_summaries(question.question_text, summaries)
        if relevant_summaries:
            context_parts.append("=== RESEARCH SUMMARIES ===")
            for summary in relevant_summaries[:5]:  # Limit to top 5 summaries
                context_parts.append(f"Summary: {summary.summary}")
                if summary.key_findings:
                    context_parts.append(f"Key Findings: {'; '.join(summary.key_findings)}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question_text: str, context: str) -> Tuple[str, str]:
        """Generate answer using LLM.
        
        Args:
            question_text: The question to answer
            context: Research context
            
        Returns:
            Tuple of (answer_text, reasoning)
        """
        # Create a detailed prompt for Q&A
        prompt = f"""You are a research assistant AI helping to answer questions about academic research papers.

RESEARCH CONTEXT:
{context}

QUESTION: {question_text}

Please provide a comprehensive answer based on the research context provided. Follow these guidelines:

1. Base your answer primarily on the information provided in the research papers and summaries
2. If the context doesn't contain sufficient information, clearly state this limitation
3. Cite specific papers or findings when relevant
4. Provide a confidence assessment of your answer
5. Include reasoning for your conclusions

Answer in this format:
ANSWER: [Your detailed answer here]

REASONING: [Explain your reasoning and which sources you used]

CONFIDENCE: [High/Medium/Low] - [Brief explanation of confidence level]"""

        try:
            # Check if using HuggingFace client (doesn't support temperature)
            if hasattr(self.llm_client, 'api_url'):  # HuggingFace client
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=1000,
                    task_type="qa"
                )
            else:  # Ollama client
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.3  # Lower temperature for more factual responses
                )
            
            # Parse the response to extract answer and reasoning
            if "ANSWER:" in response and "REASONING:" in response:
                parts = response.split("REASONING:")
                answer_part = parts[0].replace("ANSWER:", "").strip()
                reasoning_part = parts[1].split("CONFIDENCE:")[0].strip() if "CONFIDENCE:" in parts[1] else parts[1].strip()
                return answer_part, reasoning_part
            else:
                return response, "Generated using LLM without structured reasoning"
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def _find_relevant_papers(self, question: str, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Find papers relevant to the question using simple keyword matching.
        
        Args:
            question: Question text
            papers: Available papers
            
        Returns:
            List of relevant papers sorted by relevance
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_papers = []
        for paper in papers:
            score = 0
            
            # Score based on title matches
            title_words = set(paper.title.lower().split())
            title_matches = len(question_words.intersection(title_words))
            score += title_matches * 3
            
            # Score based on abstract matches
            abstract_words = set(paper.abstract.lower().split())
            abstract_matches = len(question_words.intersection(abstract_words))
            score += abstract_matches
            
            if score > 0:
                scored_papers.append((score, paper))
        
        # Sort by score and return papers
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        return [paper for score, paper in scored_papers]
    
    def _find_relevant_summaries(self, question: str, summaries: List[Summary]) -> List[Summary]:
        """Find summaries relevant to the question using simple keyword matching.
        
        Args:
            question: Question text
            summaries: Available summaries
            
        Returns:
            List of relevant summaries sorted by relevance
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_summaries = []
        for summary in summaries:
            score = 0
            
            # Score based on summary text matches
            summary_words = set(summary.summary.lower().split())
            summary_matches = len(question_words.intersection(summary_words))
            score += summary_matches * 2
            
            # Score based on key findings matches
            if summary.key_findings:
                findings_text = " ".join(summary.key_findings).lower()
                findings_words = set(findings_text.split())
                findings_matches = len(question_words.intersection(findings_words))
                score += findings_matches * 3
            
            if score > 0:
                scored_summaries.append((score, summary))
        
        # Sort by score and return summaries
        scored_summaries.sort(key=lambda x: x[0], reverse=True)
        return [summary for score, summary in scored_summaries]
    
    def _extract_evidence(
        self,
        answer_text: str,
        papers: List[ResearchPaper],
        summaries: List[Summary]
    ) -> Tuple[List[str], List[str]]:
        """Extract evidence and source papers from the answer.
        
        Args:
            answer_text: Generated answer
            papers: Available papers
            summaries: Available summaries
            
        Returns:
            Tuple of (evidence_list, source_paper_ids)
        """
        evidence = []
        source_papers = []
        
        # Simple extraction based on paper titles mentioned in answer
        answer_lower = answer_text.lower()
        
        for paper in papers:
            # Check if paper title or key terms are mentioned
            title_words = paper.title.lower().split()
            if any(word in answer_lower for word in title_words if len(word) > 4):
                evidence.append(f"Based on research from: {paper.title}")
                source_papers.append(paper.id)
        
        # If no specific papers found, add general evidence
        if not evidence:
            evidence.append("Based on the available research literature")
        
        return evidence, source_papers
    
    def _calculate_confidence(self, answer_text: str, evidence: List[str]) -> float:
        """Calculate confidence score for the answer.
        
        Args:
            answer_text: Generated answer
            evidence: Supporting evidence
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.5
        
        # Increase confidence based on evidence
        evidence_bonus = min(0.3, len(evidence) * 0.1)
        
        # Increase confidence based on answer length and specificity
        answer_length_bonus = min(0.2, len(answer_text.split()) / 500)
        
        # Check for uncertainty indicators
        uncertainty_words = ["might", "could", "perhaps", "possibly", "unclear", "insufficient"]
        uncertainty_penalty = sum(0.05 for word in uncertainty_words if word in answer_text.lower())
        
        confidence = base_confidence + evidence_bonus + answer_length_bonus - uncertainty_penalty
        return max(0.0, min(1.0, confidence))
    
    def execute_task(self, task: AgentTask) -> AgentTask:
        """Execute a Q&A task.
        
        Args:
            task: AgentTask with Q&A parameters
            
        Returns:
            Updated AgentTask with results
        """
        try:
            task.status = "running"
            logger.info(f"Executing Q&A task: {task.task_id}")
            
            # Extract parameters from task
            question_text = task.input_data.get("question")
            papers = [ResearchPaper.model_validate(p) for p in task.input_data.get("papers", [])]
            summaries = [Summary.model_validate(s) for s in task.input_data.get("summaries", [])]
            session_id = task.input_data.get("session_id", "default")
            
            if not question_text:
                raise ValueError("No question provided in task input")
              # Create conversation context
            dummy_query = ResearchQuery(query=question_text, user_id="task_user")
            dummy_session = ResearchSession(
                session_id=session_id,
                query=dummy_query,
                papers=papers,
                summaries=summaries
            )
            context = self.create_conversation_context(dummy_session)
            
            # Create question and answer
            question = Question(
                question_id=str(uuid.uuid4()),
                question_text=question_text,
                context_type="task"
            )
            
            answer = self.answer_question(question, context, papers, summaries)
            
            # Store results
            task.output_data = {
                "question": question.model_dump(),
                "answer": answer.model_dump(),
                "context": context.model_dump()
            }
            task.status = "completed"
            task.completed_at = datetime.now()
            
            logger.info(f"Q&A task completed: {task.task_id}")
            return task
            
        except Exception as e:
            error_msg = f"Q&A task failed: {str(e)}"
            logger.error(error_msg)
            task.status = "failed"
            task.error_message = error_msg
            task.completed_at = datetime.now()
            return task
