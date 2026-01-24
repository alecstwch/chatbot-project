"""
Transformer-Enhanced DialoGPT Chatbot.

This chatbot enhances the base DialoGPT with intent classification
for better context understanding and response appropriateness.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.infrastructure.config.chatbot_settings import DialoGPTSettings
from src.domain.services.intent_classifier import (
    IntentClassificationService,
    IntentPrediction
)

logger = logging.getLogger(__name__)


class TransformerEnhancedChatbot:
    """
    DialoGPT chatbot with intent classification enhancement.
    
    This chatbot combines:
    - DialoGPT for natural conversation generation
    - Intent classification for context understanding
    - Adaptive response generation based on detected intent
    
    Benefits:
    - Better contextual awareness
    - Intent-appropriate responses
    - Maintains conversation history like DialoGPT
    - Suitable for both therapy and general conversation
    """
    
    def __init__(
        self,
        settings: Optional[DialoGPTSettings] = None,
        intent_classifier: Optional[IntentClassificationService] = None,
        use_intent_classification: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize transformer-enhanced chatbot.
        
        Args:
            settings: DialoGPT configuration settings
            intent_classifier: Intent classification service (creates new if None)
            use_intent_classification: Enable intent classification
            device: Device to run on
        """
        # Load settings
        self.settings = settings or DialoGPTSettings()
        
        # Handle device selection
        if device:
            self.device = device
        elif self.settings.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.settings.device
        
        # DialoGPT components
        self.model_name = self.settings.model_name
        self.max_length = self.settings.max_history_length
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.chat_history_ids: Optional[torch.Tensor] = None
        
        # Intent classification
        self.use_intent_classification = use_intent_classification
        self.intent_classifier = intent_classifier
        
        if use_intent_classification and not self.intent_classifier:
            self.intent_classifier = IntentClassificationService(device=self.device)
        
        self._initialized = False
        
        # Conversation tracking
        self.conversation_history: List[Dict[str, Any]] = []
        
        logger.info(f"Transformer-enhanced chatbot initialized")
        logger.info(f"DialoGPT model: {self.model_name}")
        logger.info(f"Intent classification: {'enabled' if use_intent_classification else 'disabled'}")
        logger.info(f"Device: {self.device}")
    
    def load_models(self) -> None:
        """
        Load DialoGPT and intent classification models.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info("Loading transformer models...")
            
            # Load DialoGPT
            logger.info(f"Loading DialoGPT: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.settings.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.settings.cache_dir
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("DialoGPT loaded successfully")
            
            # Load intent classifier if enabled
            if self.use_intent_classification and self.intent_classifier:
                logger.info("Loading intent classifier...")
                self.intent_classifier.load_model()
                logger.info("Intent classifier loaded successfully")
            
            self._initialized = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def respond(
        self,
        user_input: str,
        return_metadata: bool = False
    ) -> str | Dict[str, Any]:
        """
        Generate a response with optional intent classification.
        
        Args:
            user_input: User's message
            return_metadata: If True, return dict with response and metadata
            
        Returns:
            Response string, or dict with response and metadata
            
        Raises:
            RuntimeError: If models not initialized
        """
        if not self._initialized:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        metadata = {
            'intent': None,
            'confidence': None,
            'tokens_generated': 0
        }
        
        try:
            # Classify intent if enabled
            intent_prediction: Optional[IntentPrediction] = None
            
            if self.use_intent_classification and self.intent_classifier:
                intent_prediction = self.intent_classifier.classify(user_input)
                metadata['intent'] = intent_prediction.intent
                metadata['confidence'] = intent_prediction.confidence
                
                logger.debug(
                    f"Intent: {intent_prediction.intent} "
                    f"({intent_prediction.confidence:.2f})"
                )
            
            # Encode user input
            new_input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token,
                return_tensors='pt'
            ).to(self.device)
            
            # Append to chat history
            if self.chat_history_ids is not None:
                bot_input_ids = torch.cat(
                    [self.chat_history_ids, new_input_ids],
                    dim=-1
                )
            else:
                bot_input_ids = new_input_ids
            
            # Adjust generation parameters based on intent if available
            temperature = self.settings.temperature
            top_p = self.settings.top_p
            
            # More conservative generation for therapy intents
            if intent_prediction and intent_prediction.intent in [
                'depression', 'anxiety', 'grief', 'stress'
            ]:
                temperature = max(0.5, temperature - 0.2)
                top_p = max(0.8, top_p - 0.1)
                logger.debug(f"Using therapy-adjusted parameters: temp={temperature}, top_p={top_p}")
            
            # Generate response
            with torch.no_grad():
                self.chat_history_ids = self.model.generate(
                    bot_input_ids,
                    max_length=self.max_length,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=self.settings.top_k,
                    repetition_penalty=self.settings.repetition_penalty,
                    do_sample=True
                )
            
            # Decode response
            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            metadata['tokens_generated'] = len(
                self.chat_history_ids[0]
            ) - len(bot_input_ids[0])
            
            # Track conversation
            self.conversation_history.append({
                'user': user_input,
                'bot': response,
                'intent': intent_prediction.intent if intent_prediction else None,
                'confidence': intent_prediction.confidence if intent_prediction else None
            })
            
            logger.debug(f"Generated response: {response[:50]}...")
            
            if return_metadata:
                return {
                    'response': response,
                    'metadata': metadata
                }
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.chat_history_ids = None
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def chat(self) -> None:
        """Interactive chat session."""
        if not self._initialized:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print("Transformer-Enhanced Chatbot")
        print("DialoGPT + Intent Classification")
        print("\nBot: Hi! I'm here to chat. How can I help you today?")
        print("(Type 'quit', 'exit', or 'bye' to end the conversation)")
        print("(Type 'reset' to start a new conversation)")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nBot: Goodbye! Take care!")
                    break
                
                if user_input.lower() == 'reset':
                    self.reset_conversation()
                    print("\nBot: Conversation reset. Let's start fresh!")
                    print()
                    continue
                
                # Generate response
                result = self.respond(user_input, return_metadata=True)
                response = result['response']
                metadata = result['metadata']
                
                # Display response
                print(f"\nBot: {response}")
                
                # Show intent in debug mode
                if metadata['intent']:
                    logger.debug(
                        f"Intent: {metadata['intent']} "
                        f"(confidence: {metadata['confidence']:.2f})"
                    )
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nBot: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print("\nBot: I'm sorry, I had trouble processing that. Could you try again?")
                print()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history with metadata.
        
        Returns:
            List of conversation turns with user input, bot response, and metadata
        """
        return self.conversation_history
    
    def is_initialized(self) -> bool:
        """Check if models are loaded."""
        return self._initialized
