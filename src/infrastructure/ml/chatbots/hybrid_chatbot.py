"""
Hybrid Chatbot combining AIML pattern matching with GPT-2 generation.

This chatbot uses AIML for structured patterns and falls back to GPT-2
for complex or unmatched queries, providing the best of both approaches.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.infrastructure.ml.chatbots.aiml_chatbot import AimlChatbot
from src.infrastructure.ml.models.response_generator import (
    ResponseGenerationService,
    GenerationConfig
)
from src.domain.services.intent_classifier import (
    IntentClassificationService,
    IntentPrediction
)

logger = logging.getLogger(__name__)


class HybridChatbot:
    """
    Hybrid chatbot combining AIML and transformer models.
    
    This chatbot uses a multi-strategy approach:
    1. Try AIML pattern matching first (fast, deterministic)
    2. If AIML response is weak, use intent classification
    3. Generate response using GPT-2 based on intent
    
    This provides:
    - Fast responses for known patterns (AIML)
    - Flexible handling of complex queries (GPT-2)
    - Intent-aware response generation
    - Therapy-focused capabilities
    """
    
    def __init__(
        self,
        aiml_dir: Optional[Path] = None,
        gpt2_model: str = "gpt2",
        intent_model: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        use_intent_classification: bool = True,
        aiml_confidence_threshold: int = 10
    ):
        """
        Initialize hybrid chatbot.
        
        Args:
            aiml_dir: Directory containing AIML files
            gpt2_model: GPT-2 model name
            intent_model: Intent classification model name
            device: Device for neural models
            use_intent_classification: Enable intent classification
            aiml_confidence_threshold: Min chars in AIML response to trust it
        """
        # Initialize AIML chatbot
        self.aiml_bot = AimlChatbot(aiml_dir=aiml_dir)
        
        # Initialize GPT-2 generator
        self.gpt2_generator = ResponseGenerationService(
            model_name=gpt2_model,
            device=device
        )
        
        # Initialize intent classifier (optional)
        self.use_intent_classification = use_intent_classification
        self.intent_classifier: Optional[IntentClassificationService] = None
        
        if use_intent_classification:
            self.intent_classifier = IntentClassificationService(
                model_name=intent_model,
                device=device
            )
        
        self.aiml_confidence_threshold = aiml_confidence_threshold
        self._initialized = False
        
        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'aiml_responses': 0,
            'gpt2_responses': 0,
            'fallback_responses': 0
        }
        
        logger.info("Hybrid chatbot created")
        logger.info(f"AIML directory: {aiml_dir}")
        logger.info(f"GPT-2 model: {gpt2_model}")
        logger.info(f"Intent classification: {'enabled' if use_intent_classification else 'disabled'}")
    
    def initialize(self) -> None:
        """
        Initialize all components (load AIML files and models).
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing hybrid chatbot...")
            
            # Load AIML files
            logger.info("Loading AIML knowledge base...")
            num_files = self.aiml_bot.load_aiml_files()
            logger.info(f"Loaded {num_files} AIML files")
            
            # Load GPT-2 model
            logger.info("Loading GPT-2 model...")
            self.gpt2_generator.load_model()
            
            # Load intent classifier if enabled
            if self.use_intent_classification and self.intent_classifier:
                logger.info("Loading intent classification model...")
                self.intent_classifier.load_model()
            
            self._initialized = True
            logger.info("Hybrid chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid chatbot: {e}")
            raise RuntimeError(f"Initialization failed: {e}") from e
    
    def respond(
        self,
        user_input: str,
        return_metadata: bool = False
    ) -> str | Dict[str, Any]:
        """
        Generate a response using hybrid approach.
        
        Args:
            user_input: User's message
            return_metadata: If True, return dict with response and metadata
            
        Returns:
            Response string, or dict with response and metadata
            
        Raises:
            RuntimeError: If chatbot not initialized
        """
        if not self._initialized:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")
        
        self.stats['total_queries'] += 1
        
        metadata = {
            'strategy': None,
            'intent': None,
            'confidence': None,
            'aiml_response': None,
            'gpt2_used': False
        }
        
        try:
            # Step 1: Try AIML pattern matching
            aiml_response = self.aiml_bot.get_response(user_input)
            metadata['aiml_response'] = aiml_response
            
            # Check if AIML response is good enough
            if aiml_response and len(aiml_response) >= self.aiml_confidence_threshold:
                logger.debug(f"Using AIML response: {aiml_response[:50]}...")
                self.stats['aiml_responses'] += 1
                metadata['strategy'] = 'aiml'
                
                if return_metadata:
                    return {
                        'response': aiml_response,
                        'metadata': metadata
                    }
                return aiml_response
            
            # Step 2: AIML failed, classify intent if enabled
            intent_prediction: Optional[IntentPrediction] = None
            
            if self.use_intent_classification and self.intent_classifier:
                intent_prediction = self.intent_classifier.classify(user_input)
                metadata['intent'] = intent_prediction.intent
                metadata['confidence'] = intent_prediction.confidence
                logger.debug(
                    f"Detected intent: {intent_prediction.intent} "
                    f"({intent_prediction.confidence:.2f})"
                )
            
            # Step 3: Generate response with GPT-2
            logger.debug("Using GPT-2 for response generation")
            self.stats['gpt2_responses'] += 1
            metadata['strategy'] = 'gpt2'
            metadata['gpt2_used'] = True
            
            # Generate therapy-focused response if intent detected
            if intent_prediction:
                response = self.gpt2_generator.generate_therapy_response(
                    user_input=user_input,
                    intent=intent_prediction.intent
                )
            else:
                # Generic response generation
                prompt = f"User: {user_input}\nAssistant:"
                response = self.gpt2_generator.generate_response(prompt)
            
            if return_metadata:
                return {
                    'response': response,
                    'metadata': metadata
                }
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.stats['fallback_responses'] += 1
            metadata['strategy'] = 'fallback'
            
            fallback = "I'm here to listen. Could you tell me more about how you're feeling?"
            
            if return_metadata:
                return {
                    'response': fallback,
                    'metadata': metadata,
                    'error': str(e)
                }
            return fallback
    
    def chat(self) -> None:
        """Interactive chat session."""
        if not self._initialized:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")
        
        print("=" * 60)
        print("Hybrid Therapy Chatbot")
        print("Combining AIML patterns with GPT-2 generation")
        print("=" * 60)
        print("\nTherapist: Hello! I'm here to listen. How are you feeling today?")
        print("(Type 'quit', 'exit', or 'bye' to end the conversation)")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nTherapist: Take care! Remember, it's okay to reach out for support.")
                    break
                
                # Generate response with metadata
                result = self.respond(user_input, return_metadata=True)
                response = result['response']
                metadata = result['metadata']
                
                # Display response
                print(f"\nTherapist: {response}")
                
                # Show strategy info in debug mode
                logger.debug(f"Strategy: {metadata['strategy']}")
                if metadata['intent']:
                    logger.debug(f"Intent: {metadata['intent']} ({metadata['confidence']:.2f})")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nTherapist: Goodbye! Take care.")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print("\nTherapist: I apologize, I'm having trouble processing that. Could you rephrase?")
                print()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with statistics about response strategies
        """
        total = self.stats['total_queries']
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'aiml_percentage': (self.stats['aiml_responses'] / total) * 100,
            'gpt2_percentage': (self.stats['gpt2_responses'] / total) * 100,
            'fallback_percentage': (self.stats['fallback_responses'] / total) * 100
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.stats = {
            'total_queries': 0,
            'aiml_responses': 0,
            'gpt2_responses': 0,
            'fallback_responses': 0
        }
        logger.info("Statistics reset")
    
    def is_initialized(self) -> bool:
        """Check if chatbot is initialized."""
        return self._initialized
