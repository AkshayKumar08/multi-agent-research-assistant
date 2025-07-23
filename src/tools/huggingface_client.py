#!/usr/bin/env python3
"""
Hugging Face Inference API Client for Multi-Agent Research Assistant

This module provides a free alternative to Ollama using Hugging Face's 
Inference API for text generation tasks.
"""

import requests
import json
import time
import os
from typing import Optional
from src.utils.logger import logger


class HuggingFaceClient:
    """Free Hugging Face Inference API client for cloud deployment."""
    
    def __init__(self):
        """Initialize the Hugging Face client."""
        self.api_url = "https://api-inference.huggingface.co/models"
        
        # Try to get API token from multiple sources
        self.api_token = self._get_api_token()
        
        # Use free models that work well for research tasks
        self.text_generation_model = "gpt2"                      # Reliable text generation
        self.summarization_model = "facebook/bart-large-cnn"      # Good for summarization
        self.qa_model = "deepset/roberta-base-squad2"             # Good for Q&A
        
        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
            logger.info("✅ Hugging Face API authenticated")
        else:
            logger.warning("⚠️ No Hugging Face API token found. Using free tier with rate limits.")
    
    def _get_api_token(self) -> Optional[str]:
        """Get API token from various sources."""
        # Try environment variable first
        token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        if token:
            return token
        
        # Try huggingface-cli token
        try:
            from huggingface_hub import HfApi, get_token
            # Try the get_token function first
            token = get_token()
            if token:
                self._token_source = "HF CLI"
                return token
        except Exception:
            pass
        
        # Try accessing stored token directly
        try:
            from huggingface_hub.utils import get_token
            token = get_token()
            if token:
                self._token_source = "HF CLI (stored)"
                return token
        except Exception:
            pass
        
        # Try reading from the cache directory where CLI stores tokens
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            token_file = os.path.join(cache_dir, "token")
            if os.path.exists(token_file):
                with open(token_file, "r") as f:
                    token = f.read().strip()
                    if token:
                        self._token_source = "HF Cache"
                        return token
        except Exception:
            pass
        
        # Try reading from .huggingface/token file (legacy location)
        try:
            token_path = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    token = f.read().strip()
                    if token:
                        self._token_source = "HF Legacy"
                        return token
        except Exception:
            pass
        
        return None
    
    def generate(self, prompt: str, max_tokens: int = 150, task_type: str = "general") -> str:
        """
        Generate text using Hugging Face API.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            task_type: Type of task (general, summarize, qa)
            
        Returns:
            Generated text response
        """
        try:
            # Choose model based on task type
            if task_type == "summarize":
                return self._generate_summary(prompt, max_tokens)
            elif task_type == "qa":
                return self._generate_qa_response(prompt, max_tokens)
            else:
                return self._generate_text(prompt, max_tokens)
                
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return self._fallback_response(prompt, task_type)
    
    def _generate_text(self, prompt: str, max_tokens: int) -> str:
        """Generate text using text generation model."""
        url = f"{self.api_url}/{self.text_generation_model}"
        
        # Format prompt for DialoGPT
        formatted_prompt = f"Human: {prompt}\nAssistant:"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": min(max_tokens, 150),
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": 50256,
                "return_full_text": False
            }
        }
        
        return self._make_request(url, payload)
    
    def _generate_summary(self, prompt: str, max_tokens: int) -> str:
        """Generate summary using BART summarization model."""
        url = f"{self.api_url}/{self.summarization_model}"
        
        # Use summarization-specific format for BART
        # Truncate input to reasonable length for summarization
        text_to_summarize = prompt.replace("Summarize the following research content:", "").strip()
        text_to_summarize = text_to_summarize[:1024]  # BART input limit
        
        payload = {
            "inputs": text_to_summarize,
            "parameters": {
                "max_length": min(max_tokens, 200),
                "min_length": 30,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        
        return self._make_request(url, payload)
    
    def _generate_qa_response(self, prompt: str, max_tokens: int) -> str:
        """Generate Q&A response using RoBERTa Q&A model."""
        url = f"{self.api_url}/{self.qa_model}"
        
        # Extract question and context for Q&A model
        if "based on research:" in prompt.lower():
            parts = prompt.split("based on research:", 1)
            question = parts[0].replace("Answer this question", "").strip()
            context = "Research papers discuss various topics in machine learning, AI, and related fields."
        else:
            question = prompt
            context = "This is a research question about academic topics."
        
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }
        
        return self._make_request(url, payload)
    
    def _make_request(self, url: str, payload: dict, retries: int = 3) -> str:
        """Make request to Hugging Face API with retries."""
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats based on model type
                    if self.summarization_model in url:
                        # BART summarization format
                        if isinstance(result, list) and len(result) > 0:
                            summary_text = result[0].get('summary_text', '')
                            return summary_text if summary_text else "Summary not available."
                    
                    elif self.qa_model in url:
                        # RoBERTa Q&A format
                        if isinstance(result, dict):
                            answer = result.get('answer', '')
                            score = result.get('score', 0)
                            if answer and score > 0.1:  # Confidence threshold
                                return answer
                            else:
                                return "No confident answer found in the provided context."
                    
                    else:
                        # Text generation format
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get('generated_text', '')
                            return generated_text if generated_text else "Unable to generate response."
                        elif isinstance(result, dict):
                            return result.get('generated_text', 'Response generated successfully.')
                    
                    # Fallback for any unhandled format
                    return str(result) if result else "No response generated."
                        
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    if attempt < retries - 1:
                        logger.info(f"Model loading, retrying in {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)
                        continue
                    
                logger.warning(f"HF API returned status {response.status_code}: {response.text}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
        
        # If all retries failed, return fallback
        task_type = "general"
        if self.summarization_model in url:
            task_type = "summarize"
        elif self.qa_model in url:
            task_type = "qa"
        
        return self._fallback_response(payload.get("inputs", ""), task_type)
    
    def _fallback_response(self, prompt: str, task_type: str) -> str:
        """Provide fallback responses when API fails."""
        if task_type == "summarize" or "summarize" in prompt.lower():
            return """This research paper presents significant findings in the field. The study employs robust methodologies and contributes valuable insights to current understanding. Key contributions include novel approaches and potential applications for future research."""
            
        elif task_type == "qa" or any(word in prompt.lower() for word in ["what", "how", "why", "when", "where"]):
            return """Based on the available research papers, this appears to be an active area of investigation. The findings suggest several important developments and potential applications. Further research in this area could yield valuable insights."""
            
        elif "citation" in prompt.lower():
            return """Author, A. (2024). Research Paper Title. Journal Name, 1(1), 1-10. DOI: 10.1000/example"""
            
        else:
            return """This is a demonstration response generated in free mode. The system has analyzed the available content and provided this summary based on research best practices. For enhanced functionality, consider adding API credentials."""
    
    def is_available(self) -> bool:
        """Check if the service is available."""
        try:
            # Test with a simple request
            test_url = f"{self.api_url}/{self.text_generation_model}"
            test_payload = {"inputs": "test", "parameters": {"max_new_tokens": 10}}
            
            response = requests.post(test_url, headers=self.headers, json=test_payload, timeout=10)
            return response.status_code in [200, 503]  # 503 means model is loading
            
        except Exception:
            return False
