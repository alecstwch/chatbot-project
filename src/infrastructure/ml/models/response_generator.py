"""
Response Generation Service using GPT-2.

This service generates contextual responses for chatbot conversations
using GPT-2 transformer model with various generation strategies.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Configuration for response generation.
    
    Attributes:
        max_length: Maximum length of generated response
        min_length: Minimum length of generated response
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        num_return_sequences: Number of responses to generate
        repetition_penalty: Penalty for repeating tokens
        do_sample: Whether to use sampling (vs greedy)
    """
    max_length: int = 100
    min_length: int = 10
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 1
    repetition_penalty: float = 1.2
    do_sample: bool = True


class ResponseGenerationService:
    """
    GPT-2 based response generation service.
    
    Generates contextual, coherent responses for chatbot conversations
    using transformer-based language models. Supports therapy-focused
    and general conversational response generation.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize response generation service.
        
        Args:
            model_name: HuggingFace model identifier (default: gpt2)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            cache_dir: Directory for model caching
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models/cache")
        self.tokenizer = None
        self.model = None
        self._initialized = False
        
        logger.info(f"Response generator created with model: {model_name}")
    
    def load_model(self) -> None:
        """
        Load GPT-2 model and tokenizer.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch
            
            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Loading GPT-2 model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Cache directory: {self.cache_dir}")
            
            # Create cache directory if needed
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load tokenizer and model
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self._initialized = True
            logger.info("GPT-2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e
    
    def generate_response(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt/context
            config: Generation configuration (uses defaults if None)
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If model not initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not prompt or not prompt.strip():
            return ""
        
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()
        
        try:
            import torch
            
            # Encode the prompt
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=config.max_length,
                    min_length=config.min_length,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    num_return_sequences=config.num_return_sequences,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            logger.debug(f"Generated response for '{prompt[:50]}...'")
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            raise
    
    def generate_therapy_response(
        self,
        user_input: str,
        intent: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a therapy-focused response.
        
        Args:
            user_input: User's message
            intent: Detected intent (depression, anxiety, etc.)
            config: Generation configuration
            
        Returns:
            Empathetic, therapy-focused response
        """
        # Create therapy-focused prompt
        if intent:
            prompt = f"Patient: {user_input}\nTherapist (addressing {intent}):"
        else:
            prompt = f"Patient: {user_input}\nTherapist:"
        
        # Use more conservative settings for therapy
        if config is None:
            config = GenerationConfig(
                max_length=80,
                temperature=0.6,  # Less random for therapy
                top_p=0.85,
                repetition_penalty=1.3
            )
        
        return self.generate_response(prompt, config)
    
    def generate_multiple_responses(
        self,
        prompt: str,
        num_responses: int = 3,
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate multiple response candidates.
        
        Args:
            prompt: Input prompt
            num_responses: Number of responses to generate
            config: Generation configuration
            
        Returns:
            List of generated responses
        """
        if config is None:
            config = GenerationConfig()
        
        config.num_return_sequences = num_responses
        
        try:
            import torch
            
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=config.max_length,
                    min_length=config.min_length,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    num_return_sequences=num_responses,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=True,  # Must be True for multiple responses
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode all responses
            responses = []
            for output in outputs:
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                responses.append(response)
            
            logger.debug(f"Generated {len(responses)} responses")
            
            return responses
            
        except Exception as e:
            logger.error(f"Multiple response generation error: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized
