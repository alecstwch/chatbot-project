"""
Mistral-7B-based conversational chatbot.

This module implements a neural conversational chatbot using Mistral-7B-Instruct model.
Mistral-7B is a state-of-the-art open model with excellent instruction-following capabilities.
"""

import logging
import time
from typing import Optional, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings


logger = logging.getLogger(__name__)


class NeuralChatbot:
    """
    Neural conversational chatbot using modern language models.
    
    This chatbot supports various neural models (Phi-2, GPT-2, etc.) for generating
    high-quality conversational responses.
    
    Attributes:
        model_name: HuggingFace model identifier
        tokenizer: Model tokenizer
        model: Neural language model
        conversation_history: List of conversation turns
        max_history_turns: Maximum number of conversation turns to keep
    """
    
    def __init__(
        self,
        settings: Optional[NeuralChatbotSettings] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize neural chatbot.
        
        Args:
            settings: Neural chatbot configuration settings (12-Factor App compliant)
            model_name: HuggingFace model identifier (overrides settings)
            device: Device to run model on (overrides settings)
        """
        # Load settings from config or use defaults
        self.settings = settings or NeuralChatbotSettings()
        
        # Allow parameter overrides
        self.model_name = model_name or self.settings.model_name
        
        # Handle device selection
        if device:
            self.device = device
        elif self.settings.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.settings.device
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.conversation_history: List[dict] = []
        self._initialized = False
        self.last_response_time: float = 0.0
        self.last_tokens_generated: int = 0
        self.last_tokens_per_sec: float = 0.0
        
        logger.info(f"Mistral chatbot initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Settings: temp={self.settings.temperature}, max_tokens={self.settings.max_new_tokens}")
    
    def load_model(self) -> None:
        """
        Load the Phi-2 model and tokenizer from HuggingFace.
        
        Note: Supports both FP16 (5.4GB) and 8-bit quantization (2.7GB).
        First download ~5GB.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Cache directory: {self.settings.cache_dir}")
            logger.info(f"Device: {self.device}")
            
            use_8bit = getattr(self.settings, 'use_8bit_quantization', False)
            
            if use_8bit and self.device == "cuda":
                logger.info("Loading Phi-2 with 8-bit quantization (~2.7GB VRAM)...")
            elif self.device == "cuda":
                logger.info("Loading Phi-2 with FP16 (~5.4GB VRAM)...")
            else:
                logger.info("Loading Phi-2 on CPU (slow but works)...")
            
            logger.info("First download will be ~5GB. This may take several minutes...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.settings.cache_dir,
                trust_remote_code=True
            )
            
            # Load model based on device and quantization settings
            if self.device == "cuda":
                if use_8bit:
                    # 8-bit quantization - fits in 4GB VRAM
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    logger.info("Using 8-bit quantization for optimal 4GB VRAM fit...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.settings.cache_dir,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    # FP16 precision - may offload to CPU if > 4GB
                    logger.info("Using FP16 precision (may offload to CPU)...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.settings.cache_dir,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:
                # CPU mode - full precision (no device_map needed for CPU)
                logger.warning("Running on CPU - generation will be slow. GPU recommended.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.settings.cache_dir,
                    trust_remote_code=True
                )
                self.model = self.model.to('cpu')
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            
            self._initialized = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _format_conversation(self) -> str:
        """
        Format conversation history for Phi-2 (simple conversational format).
        
        Returns:
            Formatted conversation string
        """
        if not self.conversation_history:
            return ""
        
        # Phi-2 uses simple conversational format:
        # User: message\nAssistant: response\n
        formatted = ""
        for turn in self.conversation_history:
            formatted += f"User: {turn['user']}\n"
            if turn.get('assistant'):
                formatted += f"Assistant: {turn['assistant']}\n"
        
        return formatted
    
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
            # Start timing
            start_time = time.time()
            
            # Add user input to history
            self.conversation_history.append({'user': user_input})
            
            # Keep only recent history
            if len(self.conversation_history) > self.settings.max_history_turns:
                self.conversation_history = self.conversation_history[-self.settings.max_history_turns:]
            
            # Format conversation
            conversation = self._format_conversation()
            prompt = conversation + "Assistant:"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024  # Limit context to prevent memory issues
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_token_count = inputs['input_ids'].shape[1]
            
            # Generate response
            with torch.no_grad():
                # Phi-2 specific: Add stop strings to prevent educational scenario hallucinations
                if "phi-2" in self.model_name.lower():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.settings.max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        stop_strings=["\n\n\n", "User:", "Let's", "Now let's", "Scenario:"],
                        tokenizer=self.tokenizer
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.settings.max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            # Calculate tokens generated
            output_token_count = outputs.shape[1]
            tokens_generated = output_token_count - input_token_count
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new assistant response
            # Remove the prompt from the output
            response = full_response[len(prompt):].strip()
            
            # Clean up - stop at next "User:" if model continues
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            # Phi-2 specific: Stop at common hallucination patterns
            if "phi-2" in self.model_name.lower():
                stop_markers = ["\n\nLet's", "\n\nNow let's", "\n\nScenario:", "\n\nImagine", "\n\n1)"]
                for marker in stop_markers:
                    if marker in response:
                        response = response.split(marker)[0].strip()
                        break
            
            # End timing and calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0
            
            # Store benchmark data
            self.last_response_time = response_time
            self.last_tokens_generated = tokens_generated
            self.last_tokens_per_sec = tokens_per_sec
            
            logger.info(f"Response generated in {response_time:.2f}s ({tokens_generated} tokens, {tokens_per_sec:.2f} tok/s)")
            
            # Update history with assistant response
            self.conversation_history[-1]['assistant'] = response
            
            return response or "I'm not sure how to respond to that."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Remove the failed turn from history
            if self.conversation_history and 'assistant' not in self.conversation_history[-1]:
                self.conversation_history.pop()
            return "Sorry, I encountered an error. Could you rephrase that?"
    
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
