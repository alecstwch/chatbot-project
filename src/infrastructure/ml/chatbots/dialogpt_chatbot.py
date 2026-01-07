"""
DialoGPT-based conversational chatbot.

This module implements a neural conversational chatbot using Microsoft's DialoGPT model.
DialoGPT is a transformer-based model trained on Reddit conversations for multi-turn dialogue.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.infrastructure.config.chatbot_settings import DialoGPTSettings


logger = logging.getLogger(__name__)


class DialoGPTChatbot:
    """
    Neural conversational chatbot using DialoGPT.
    
    This chatbot uses Microsoft's DialoGPT model for generating contextual responses
    in multi-turn conversations. It maintains conversation history for context.
    
    Attributes:
        model_name: HuggingFace model identifier
        tokenizer: DialoGPT tokenizer
        model: DialoGPT model
        chat_history_ids: Tensor storing conversation history
        max_length: Maximum conversation history length
    """
    
    def __init__(
        self,
        settings: Optional[DialoGPTSettings] = None,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize DialoGPT chatbot.
        
        Args:
            settings: DialoGPT configuration settings (12-Factor App compliant)
            model_name: HuggingFace model identifier (overrides settings)
            max_length: Maximum length for conversation history (overrides settings)
            device: Device to run model on (overrides settings)
        """
        # Load settings from config or use defaults
        self.settings = settings or DialoGPTSettings()
        
        # Allow parameter overrides
        self.model_name = model_name or self.settings.model_name
        self.max_length = max_length or self.settings.max_history_length
        
        # Handle device selection
        if device:
            self.device = device
        elif self.settings.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.settings.device
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.chat_history_ids: Optional[torch.Tensor] = None
        self._initialized = False
        
        logger.info(f"DialoGPT chatbot initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Settings: temp={self.settings.temperature}, top_p={self.settings.top_p}, rep_penalty={self.settings.repetition_penalty}")
    
    def load_model(self) -> None:
        """
        Load the DialoGPT model and tokenizer from HuggingFace.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading DialoGPT model: {self.model_name}")
            logger.info(f"Cache directory: {self.settings.cache_dir}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.settings.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.settings.cache_dir
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info("DialoGPT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DialoGPT model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def get_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: int = 2
    ) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's message
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            top_k: Top-k sampling parameter (uses config default if None)
            repetition_penalty: Penalty for repeating tokens (uses config default if None)
            no_repeat_ngram_size: Size of n-grams that cannot be repeated
            
        Returns:
            Generated response string
            
        Raises:
            RuntimeError: If model is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you say that again?"
        
        # Use settings defaults if parameters not provided
        temperature = temperature if temperature is not None else self.settings.temperature
        top_p = top_p if top_p is not None else self.settings.top_p
        top_k = top_k if top_k is not None else self.settings.top_k
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.settings.repetition_penalty
        
        try:
            # Encode user input with attention mask
            encoded = self.tokenizer.encode_plus(
                user_input + self.tokenizer.eos_token,
                return_tensors='pt',
                add_special_tokens=True
            )
            new_input_ids = encoded['input_ids'].to(self.device)
            new_attention_mask = encoded['attention_mask'].to(self.device)
            
            # Append to conversation history
            if self.chat_history_ids is not None:
                bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
                
                # Truncate if history is too long (prevent gibberish)
                if bot_input_ids.shape[-1] > self.max_length:
                    bot_input_ids = bot_input_ids[:, -self.max_length:]
                    logger.debug(f"Truncated conversation history to {self.max_length} tokens")
                
                # Extend attention mask
                attention_mask = torch.cat([
                    torch.ones_like(bot_input_ids[:, :-new_input_ids.shape[-1]]),
                    new_attention_mask
                ], dim=-1).to(self.device)
            else:
                bot_input_ids = new_input_ids
                attention_mask = new_attention_mask
            
            # Generate response
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.settings.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=True
            )
            
            # Decode only the new tokens (bot's response)
            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            return response.strip() or "I'm not sure how to respond to that."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error. Could you rephrase that?"
    
    def chat(self, user_input: str, **kwargs) -> str:
        """
        Alias for get_response() for consistency with AIML chatbot interface.
        
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
        self.chat_history_ids = None
        logger.debug("Conversation history reset")
    
    def is_ready(self) -> bool:
        """
        Check if the chatbot is ready to use.
        
        Returns:
            True if model is loaded and ready
        """
        return self._initialized and self.model is not None and self.tokenizer is not None
    
    def get_conversation_length(self) -> int:
        """
        Get the current conversation history length in tokens.
        
        Returns:
            Number of tokens in conversation history
        """
        if self.chat_history_ids is None:
            return 0
        return self.chat_history_ids.shape[-1]
    
    def truncate_history(self, max_tokens: Optional[int] = None) -> None:
        """
        Truncate conversation history to prevent context overflow.
        
        Args:
            max_tokens: Maximum tokens to keep (defaults to half of max_length)
        """
        if self.chat_history_ids is None:
            return
        
        max_tokens = max_tokens or (self.max_length // 2)
        
        if self.chat_history_ids.shape[-1] > max_tokens:
            self.chat_history_ids = self.chat_history_ids[:, -max_tokens:]
            logger.debug(f"Conversation history truncated to {max_tokens} tokens")
