"""
Neural Chatbot using DDD Architecture with Conversation Engine.

Refactored to separate domain logic (conversation management) from
infrastructure (model loading, tokenization). Follows 12-Factor App principles.
"""

import logging
from typing import Optional

from src.domain.services.conversation_engine import (
    ConversationEngine, 
    SimpleConversationFormatter,
    ChatMLFormatter
)
from src.infrastructure.ml.models.neural_language_model import NeuralLanguageModel
from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings


logger = logging.getLogger(__name__)


class NeuralChatbot:
    """
    Neural conversational chatbot using modern language models.
    
    This chatbot follows DDD architecture:
    - Domain Layer: ConversationEngine (conversation logic, history, formatting)
    - Infrastructure Layer: NeuralLanguageModel (model loading, tokenization, generation)
    - Configuration: Settings-based, not hardcoded (12-Factor App)
    
    Attributes:
        language_model: Infrastructure component for text generation
        conversation_engine: Domain component for conversation management
        settings: Configuration settings
    """
    
    def __init__(
        self,
        settings: Optional[NeuralChatbotSettings] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        formatter_type: str = "simple",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize neural chatbot.
        
        Args:
            settings: Neural chatbot configuration settings (12-Factor App compliant)
            model_name: HuggingFace model identifier (overrides settings)
            device: Device to run model on (overrides settings)
            formatter_type: Conversation format ('simple' or 'chatml')
            system_prompt: Optional system instruction
        """
        # Load settings from config or use defaults
        self.settings = settings or NeuralChatbotSettings()
        
        # Initialize infrastructure layer (model)
        self.language_model = NeuralLanguageModel(
            settings=self.settings,
            model_name=model_name,
            device=device
        )
        
        # Choose formatter
        if formatter_type == "chatml":
            formatter = ChatMLFormatter()
        else:
            formatter = SimpleConversationFormatter()
        
        # Default system prompt for Phi-2 to reduce hallucinations
        if system_prompt is None and "phi-2" in self.language_model.model_name.lower():
            system_prompt = (
                "You are a helpful, respectful assistant. "
                "Provide concise, direct answers to questions. "
                "Do not create fictional scenarios or examples."
            )
        
        # Initialize domain layer (conversation engine)
        self.conversation_engine = ConversationEngine(
            model=self.language_model,
            formatter=formatter,
            max_history_turns=self.settings.max_history_turns,
            system_prompt=system_prompt
        )
        
        logger.info(f"Neural chatbot initialized with {formatter_type} formatter")
        logger.info(f"Model: {self.language_model.model_name}")
        logger.info(f"Max history: {self.settings.max_history_turns} turns")
    
    def load_model(self) -> None:
        """
        Load the neural language model.
        
        Delegates to infrastructure layer.
        """
        self.language_model.load_model()
        logger.info("Neural chatbot model loaded and ready")
    
    def get_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's message
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            top_k: Top-k sampling parameter (uses config default if None)
            repetition_penalty: Penalty for repeating tokens (uses config default if None)
            
        Returns:
            Generated response string
            
        Raises:
            RuntimeError: If model is not initialized
        """
        # Use settings defaults if parameters not provided
        temperature = temperature if temperature is not None else self.settings.temperature
        top_p = top_p if top_p is not None else self.settings.top_p
        top_k = top_k if top_k is not None else self.settings.top_k
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.settings.repetition_penalty
        
        # Add Phi-2 specific stop strings
        stop_strings = None
        if "phi-2" in self.language_model.model_name.lower():
            stop_strings = ["\n\n\n", "User:", "Let's", "Now let's", "Scenario:"]
        
        # Delegate to conversation engine
        return self.conversation_engine.generate_response(
            user_input=user_input,
            temperature=temperature,
            max_tokens=self.settings.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
    
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
        
        Delegates to conversation engine.
        """
        self.conversation_engine.reset_conversation()
        logger.debug("Chatbot conversation history reset")
    
    def is_ready(self) -> bool:
        """
        Check if the chatbot is ready to use.
        
        Returns:
            True if model is loaded and ready
        """
        return self.language_model.is_ready()
    
    def get_conversation_length(self) -> int:
        """
        Get the current conversation history length in turns.
        
        Returns:
            Number of conversation turns
        """
        return len(self.conversation_engine.get_history())
    
    def get_benchmark_stats(self) -> dict:
        """
        Get performance statistics from last response generation.
        
        Returns:
            Dictionary with performance metrics
        """
        model_info = self.language_model.get_model_info()
        conversation_summary = self.conversation_engine.get_conversation_summary()
        
        return {
            **model_info,
            **conversation_summary,
            'response_time': model_info['last_generation_time'],
            'tokens_generated': model_info['last_tokens_generated']
        }
    
    def get_history(self) -> list:
        """
        Get conversation history.
        
        Returns:
            List of ConversationTurn objects
        """
        return self.conversation_engine.get_history()
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt.
        
        Args:
            prompt: New system prompt
        """
        self.conversation_engine.system_prompt = prompt
        logger.info(f"System prompt updated: {prompt[:50]}...")
