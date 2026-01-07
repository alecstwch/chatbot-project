"""
Neural Language Model - Infrastructure Layer.

Implements the LanguageModelProtocol for transformer-based models.
Separates model management from conversation logic (DDD).
"""

import logging
import time
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings

logger = logging.getLogger(__name__)


class NeuralLanguageModel:
    """
    Transformer-based language model implementation.
    
    This class handles all infrastructure concerns:
    - Model loading from HuggingFace
    - Tokenization
    - Device management
    - Quantization
    - Token generation
    
    Business logic (conversation management) is handled by ConversationEngine.
    """
    
    def __init__(
        self,
        settings: Optional[NeuralChatbotSettings] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize neural language model.
        
        Args:
            settings: Model configuration settings (12-Factor App compliant)
            model_name: HuggingFace model identifier (overrides settings)
            device: Device to run on (overrides settings)
        """
        self.settings = settings or NeuralChatbotSettings()
        self.model_name = model_name or self.settings.model_name
        
        # Device selection
        if device:
            self.device = device
        elif self.settings.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.settings.device
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._initialized = False
        
        # Performance metrics
        self.last_generation_time: float = 0.0
        self.last_tokens_generated: int = 0
        self.total_generations: int = 0
        
        logger.info(f"Neural language model initialized: {self.model_name}")
        logger.info(f"Target device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load model and tokenizer from HuggingFace.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Cache directory: {self.settings.cache_dir}")
            
            use_8bit = getattr(self.settings, 'use_8bit_quantization', False)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.settings.cache_dir,
                trust_remote_code=True
            )
            
            # Load model based on device and quantization
            if self.device == "cuda":
                if use_8bit:
                    logger.info("Loading with 8-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.settings.cache_dir,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    logger.info("Loading with FP16 precision...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.settings.cache_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:
                logger.warning("Loading on CPU - generation will be slow")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.settings.cache_dir,
                    trust_remote_code=True
                )
                self.model = self.model.to('cpu')
            
            # Configure padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            self._initialized = True
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[list] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop_strings: Strings that stop generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text including the prompt
            
        Raises:
            RuntimeError: If model not initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first")
        
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            # Generate
            with torch.no_grad():
                generation_kwargs = {
                    **inputs,
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'repetition_penalty': repetition_penalty,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    **kwargs
                }
                
                # Add stop strings if model supports it
                if stop_strings and 'phi-2' in self.model_name.lower():
                    generation_kwargs['stop_strings'] = stop_strings
                    generation_kwargs['tokenizer'] = self.tokenizer
                
                outputs = self.model.generate(**generation_kwargs)
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metrics
            output_length = outputs.shape[1]
            tokens_generated = output_length - input_length
            generation_time = time.time() - start_time
            
            self.last_generation_time = generation_time
            self.last_tokens_generated = tokens_generated
            self.total_generations += 1
            
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
            logger.debug(
                f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                f"({tokens_per_sec:.2f} tok/s)"
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized and self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'initialized': self._initialized,
            'total_generations': self.total_generations,
            'last_generation_time': self.last_generation_time,
            'last_tokens_generated': self.last_tokens_generated
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.last_generation_time = 0.0
        self.last_tokens_generated = 0
        self.total_generations = 0
