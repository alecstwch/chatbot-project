"""
Core Conversation Engine - Domain Layer.

Provides the fundamental conversation logic abstracted from specific model implementations.
Follows Domain-Driven Design principles with business logic separated from infrastructure.
"""

import logging
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """
    Represents a single turn in a conversation.
    
    Attributes:
        user_input: User's message
        assistant_response: Assistant's response (None if not yet generated)
        metadata: Additional metadata (intent, confidence, etc.)
        timestamp: When this turn occurred
    """
    user_input: str
    assistant_response: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LanguageModelProtocol(Protocol):
    """
    Protocol defining the interface for language models.
    
    This allows any language model implementation to work with the conversation engine
    as long as it implements these methods.
    """
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        ...
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        ...


class ConversationFormatter(ABC):
    """
    Abstract base for conversation formatting strategies.
    
    Different models may require different conversation formats
    (e.g., ChatML, Alpaca, simple User/Assistant format).
    """
    
    @abstractmethod
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format conversation history into a prompt string."""
        pass
    
    @abstractmethod
    def extract_response(self, generated_text: str, prompt: str) -> str:
        """Extract the assistant's response from generated text."""
        pass


class SimpleConversationFormatter(ConversationFormatter):
    """
    Simple User/Assistant conversation format.
    
    Format:
        User: message
        Assistant: response
    """
    
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format conversation as simple User/Assistant exchange."""
        formatted = ""
        for turn in turns:
            formatted += f"User: {turn.user_input}\n"
            if turn.assistant_response:
                formatted += f"Assistant: {turn.assistant_response}\n"
        return formatted
    
    def extract_response(self, generated_text: str, prompt: str) -> str:
        """Extract response by removing the prompt."""
        response = generated_text[len(prompt):].strip()
        
        # Stop at next "User:" if model continues
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        return response


class ChatMLFormatter(ConversationFormatter):
    """
    ChatML format used by some modern models.
    
    Format:
        <|im_start|>user
        message<|im_end|>
        <|im_start|>assistant
        response<|im_end|>
    """
    
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format conversation in ChatML format."""
        formatted = ""
        for turn in turns:
            formatted += f"<|im_start|>user\n{turn.user_input}<|im_end|>\n"
            if turn.assistant_response:
                formatted += f"<|im_start|>assistant\n{turn.assistant_response}<|im_end|>\n"
        return formatted
    
    def extract_response(self, generated_text: str, prompt: str) -> str:
        """Extract response from ChatML format."""
        response = generated_text[len(prompt):].strip()
        
        # Remove end markers
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        return response


class ConversationEngine:
    """
    Core conversation engine following DDD principles.
    
    This engine manages conversation state and orchestrates the interaction
    between user input, conversation history, and language model generation.
    
    Domain responsibilities:
    - Maintain conversation history
    - Format conversations appropriately
    - Manage conversation context
    - Apply conversation rules (max history, etc.)
    
    Infrastructure (delegated):
    - Model loading and inference
    - Tokenization
    - Device management
    """
    
    def __init__(
        self,
        model: LanguageModelProtocol,
        formatter: Optional[ConversationFormatter] = None,
        max_history_turns: int = 10,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize conversation engine.
        
        Args:
            model: Language model implementing LanguageModelProtocol
            formatter: Conversation formatter (defaults to SimpleConversationFormatter)
            max_history_turns: Maximum conversation turns to keep in history
            system_prompt: Optional system instruction to prepend to conversations
        """
        self.model = model
        self.formatter = formatter or SimpleConversationFormatter()
        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt
        
        self.conversation_history: List[ConversationTurn] = []
        self.metadata: Dict[str, Any] = {}
        
        logger.info("Conversation engine initialized")
        logger.info(f"Formatter: {type(self.formatter).__name__}")
        logger.info(f"Max history turns: {max_history_turns}")
    
    def generate_response(
        self,
        user_input: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        **kwargs
    ) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for model generation
            
        Returns:
            Generated response string
            
        Raises:
            RuntimeError: If model is not ready
        """
        if not self.model.is_ready():
            raise RuntimeError("Language model is not ready")
        
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you say that again?"
        
        # Create new conversation turn
        turn = ConversationTurn(user_input=user_input.strip())
        self.conversation_history.append(turn)
        
        # Trim history if needed
        self._trim_history()
        
        # Format conversation into prompt
        conversation_text = self.formatter.format_conversation(self.conversation_history)
        
        # Add system prompt if provided
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{conversation_text}Assistant:"
        else:
            prompt = f"{conversation_text}Assistant:"
        
        # Generate response
        try:
            generated = self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract clean response
            response = self.formatter.extract_response(generated, prompt)
            
            # Update turn with response
            turn.assistant_response = response
            
            logger.debug(f"Generated response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Remove failed turn
            self.conversation_history.pop()
            return "I'm sorry, I had trouble processing that. Could you try again?"
    
    def _trim_history(self) -> None:
        """Trim conversation history to max_history_turns."""
        if len(self.conversation_history) > self.max_history_turns:
            # Keep most recent turns
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
            logger.debug(f"Trimmed history to {self.max_history_turns} turns")
    
    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        self.metadata = {}
        logger.info("Conversation reset")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        return {
            'total_turns': len(self.conversation_history),
            'has_system_prompt': self.system_prompt is not None,
            'formatter_type': type(self.formatter).__name__,
            'metadata': self.metadata
        }
    
    def get_history(self) -> List[ConversationTurn]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set conversation metadata."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get conversation metadata."""
        return self.metadata.get(key, default)
