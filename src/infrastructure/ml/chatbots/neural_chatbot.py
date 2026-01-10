"""
Gemini 2.5 Flash-based conversational chatbot.

This module implements a neural conversational chatbot using Google's Gemini 2.5 Flash API.
Gemini is a state-of-the-art model with excellent instruction-following and conversational capabilities.
"""

import logging
import time
import os
from typing import Optional, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


class NeuralChatbot:
    """
    Neural conversational chatbot using Google Gemini 2.5 Flash API.
    
    This chatbot uses Google's Gemini API for generating high-quality
    conversational responses without requiring local model downloads.
    
    Attributes:
        model_name: Gemini model identifier
        conversation_history: List of conversation turns
        max_history_turns: Maximum number of conversation turns to keep
    """
    
    def __init__(
        self,
        settings: Optional[NeuralChatbotSettings] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None  # Kept for backward compatibility, not used
    ):
        """
        Initialize Gemini chatbot.
        
        Args:
            settings: Neural chatbot configuration settings (12-Factor App compliant)
            model_name: Gemini model identifier (overrides settings)
            device: Deprecated parameter, kept for backward compatibility
        """
        # Load settings from config or use defaults
        self.settings = settings or NeuralChatbotSettings()
        
        # Allow parameter overrides
        self.model_name = model_name or self.settings.model_name
        
        # Device not needed for API calls but kept for compatibility
        self.device = "api"
        
        self.client: Optional[genai.Client] = None
        self.chat_history: List[types.Content] = []
        self.conversation_history: List[dict] = []
        self._initialized = False
        self.last_response_time: float = 0.0
        self.last_tokens_generated: int = 0
        self.last_tokens_per_sec: float = 0.0
        
        logger.info(f"Gemini chatbot initialized with model: {self.model_name}")
        logger.info(f"Settings: temp={self.settings.temperature}, max_tokens={self.settings.max_new_tokens}")
    
    def load_model(self) -> None:
        """
        Initialize the Gemini API client.
        
        Configures the API key and creates the client instance.
        No download required - uses API calls.
        
        Raises:
            RuntimeError: If API initialization fails or API key is missing
        """
        try:
            logger.info(f"Initializing Gemini API with model: {self.model_name}")
            
            # Get API key from settings or environment
            api_key = (
                self.settings.api_key or 
                os.environ.get("GEMINI_API_KEY") or 
                os.environ.get("GOOGLE_API_KEY") or
                os.environ.get("NEURAL_API_KEY")
            )
            
            if not api_key:
                raise RuntimeError(
                    "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
                )
            
            # Create the client
            self.client = genai.Client(api_key=api_key)
            
            self._initialized = True
            logger.info("Gemini API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise RuntimeError(f"Gemini API initialization failed: {e}")
    
    def get_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None  # Not used for Gemini but kept for compatibility
    ) -> str:
        """
        Generate a response to user input using Gemini API.
        
        Args:
            user_input: User's message
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            top_k: Top-k sampling parameter (uses config default if None)
            repetition_penalty: Not used for Gemini API, kept for compatibility
            
        Returns:
            Generated response string
            
        Raises:
            RuntimeError: If model is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you say that again?"
        
        try:
            # Start timing
            start_time = time.time()
            
            # Add user input to local history
            self.conversation_history.append({'user': user_input})
            
            # Keep only recent history
            if len(self.conversation_history) > self.settings.max_history_turns:
                self.conversation_history = self.conversation_history[-self.settings.max_history_turns:]
            
            # Add user message to chat history
            self.chat_history.append(
                types.Content(role="user", parts=[types.Part(text=user_input)])
            )
            
            # Create generation config
            gen_config = types.GenerateContentConfig(
                temperature=temperature if temperature is not None else self.settings.temperature,
                top_p=top_p if top_p is not None else self.settings.top_p,
                top_k=top_k if top_k is not None else self.settings.top_k,
                max_output_tokens=self.settings.max_new_tokens,
            )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.chat_history,
                config=gen_config,
            )
            
            # Extract response text
            response_text = response.text.strip()
            
            # Add assistant response to chat history
            self.chat_history.append(
                types.Content(role="model", parts=[types.Part(text=response_text)])
            )
            
            # Calculate timing metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            # Estimate tokens
            tokens_generated = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_generated = getattr(response.usage_metadata, 'candidates_token_count', 0)
            if tokens_generated == 0:
                # Rough estimate: ~4 characters per token
                tokens_generated = len(response_text) // 4
            
            tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0
            
            # Store benchmark data
            self.last_response_time = response_time
            self.last_tokens_generated = tokens_generated
            self.last_tokens_per_sec = tokens_per_sec
            
            logger.info(f"Response generated in {response_time:.2f}s ({tokens_generated} tokens, {tokens_per_sec:.2f} tok/s)")
            
            # Update history with assistant response
            self.conversation_history[-1]['assistant'] = response_text
            
            return response_text or "I'm not sure how to respond to that."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Remove the failed turn from history
            if self.conversation_history and 'assistant' not in self.conversation_history[-1]:
                self.conversation_history.pop()
            if self.chat_history and self.chat_history[-1].role == "user":
                self.chat_history.pop()
            return f"Sorry, I encountered an error: {str(e)}"
    
    def chat(self, user_input: str, **kwargs) -> str:
        """
        Alias for get_response() for consistency with other chatbot interfaces.
        
        Args:
            user_input: User's message
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated response string
        """
        return self.get_response(user_input, **kwargs)
    
    def reset(self) -> None:
        """
        Reset conversation history.
        
        Clears the chat history to start a fresh conversation.
        """
        self.conversation_history = []
        self.chat_history = []
        logger.debug("Conversation history reset")
    
    def is_ready(self) -> bool:
        """
        Check if the chatbot is ready to use.
        
        Returns:
            True if API is initialized and ready
        """
        return self._initialized and self.client is not None
    
    def get_conversation_length(self) -> int:
        """
        Get the current conversation history length in turns.
        
        Returns:
            Number of conversation turns
        """
        return len(self.conversation_history)
    
    def get_benchmark_stats(self) -> Tuple[float, int, float]:
        """
        Get performance statistics from last response generation.
        
        Returns:
            Tuple of (response_time_seconds, tokens_generated, tokens_per_second)
        """
        return (self.last_response_time, self.last_tokens_generated, self.last_tokens_per_sec)
