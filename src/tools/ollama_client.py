"""
Ollama client tool for LLM operations.
"""
import requests
import json
from typing import Dict, Any, Optional, List
from src.utils.logger import get_logger
from config.settings import config

logger = get_logger("ollama_client")


class OllamaClient:
    """Client for interacting with Ollama LLM server."""
    
    def __init__(self, base_url: str = None, model: str = None):
        """Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (defaults to config)
            model: Model name (defaults to config)
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        # Remove ollama/ prefix if present for direct API calls
        raw_model = model or config.OLLAMA_MODEL
        if raw_model.startswith("ollama/"):
            self.model = raw_model[7:]  # Remove "ollama/" prefix
        else:
            self.model = raw_model
        self.session = requests.Session()
        
        logger.info(f"Ollama client initialized: {self.base_url}, model: {self.model}")
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text response
        """
        temperature = temperature or config.AGENT_TEMPERATURE
        max_tokens = max_tokens or config.AGENT_MAX_TOKENS
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        logger.info(f"Generating text with Ollama (temp={temperature}, max_tokens={max_tokens})")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout for long generations
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                result = response.json()
                generated_text = result.get("response", "")
                
                logger.info(f"Generated {len(generated_text)} characters")
                return generated_text
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running?"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        
        except requests.exceptions.Timeout as e:
            error_msg = "Ollama request timed out"
            logger.error(error_msg)
            raise TimeoutError(error_msg) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama request failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _handle_stream_response(self, response: requests.Response) -> str:
        """Handle streaming response from Ollama.
        
        Args:
            response: Streaming response object
            
        Returns:
            Complete generated text
        """
        full_response = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                    
                    # Check if generation is done
                    if data.get('done', False):
                        break
                        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing streaming response: {str(e)}")
            
        return full_response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat with Ollama using message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        temperature = temperature or config.AGENT_TEMPERATURE
        max_tokens = max_tokens or config.AGENT_MAX_TOKENS
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        logger.info(f"Chat request with {len(messages)} messages")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            message = result.get("message", {})
            content = message.get("content", "")
            
            logger.info(f"Chat response: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Chat request failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama server is available.
        
        Returns:
            True if server is responding
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama server not available: {str(e)}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models on Ollama server.
        
        Returns:
            List of model names
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model.get("name", "") for model in data.get("models", [])]
            
            logger.info(f"Available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model to Ollama server.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        payload = {"name": model_name}
        
        logger.info(f"Pulling model: {model_name}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600  # 10 minute timeout for model download
            )
            response.raise_for_status()
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False
