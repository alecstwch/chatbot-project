"""
Intent Classification Service for chatbot user inputs.

This service classifies user intents using both zero-shot and fine-tuned approaches.
Configuration-driven (12-Factor App) - loads intents and keywords from external YAML files.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


def load_intent_config(domain: str) -> Dict[str, Any]:
    """
    Load intent configuration from YAML files.
    
    Args:
        domain: Domain name (e.g., 'therapy_intents', 'chef_intents')
        
    Returns:
        Dictionary with intent and keyword configuration
        
    Raises:
        FileNotFoundError: If configuration files not found
    """
    config_dir = Path(__file__).parent.parent.parent.parent / 'config' / 'model_configs' / domain
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Intent config directory not found: {config_dir}")
    
    # Load intents
    intents_file = config_dir / 'intents.yaml'
    with open(intents_file, 'r', encoding='utf-8') as f:
        intents_config = yaml.safe_load(f)
    
    # Load keywords
    keywords_file = config_dir / 'keywords.yaml'
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords_config = yaml.safe_load(f)
    
    # Load ingredients if chef domain
    ingredients_config = None
    if domain == 'chef_intents':
        ingredients_file = config_dir / 'ingredients.yaml'
        if ingredients_file.exists():
            with open(ingredients_file, 'r', encoding='utf-8') as f:
                ingredients_config = yaml.safe_load(f)
    
    result = {
        'intents': intents_config,
        'keywords': keywords_config,
        'domain': domain
    }
    
    if ingredients_config:
        result['ingredients'] = ingredients_config
    
    return result


@dataclass
class IntentPrediction:
    """
    Intent prediction result.
    
    Attributes:
        intent: Predicted intent label
        confidence: Confidence score (0.0 to 1.0)
        all_scores: Scores for all candidate intents
    """
    intent: str
    confidence: float
    all_scores: Dict[str, float]


class IntentClassificationService:
    """
    Configuration-driven intent classification service.
    
    Loads intent definitions and keyword patterns from external YAML files,
    following 12-Factor App principles. Supports multiple domains (therapy, chef, etc.)
    with zero-shot classification and keyword fallback.
    """
    
    def __init__(
        self,
        domain: str = "therapy_intents",
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        use_keyword_fallback: bool = True
    ):
        """
        Initialize intent classifier.
        
        Args:
            domain: Intent domain to load (e.g., 'therapy_intents', 'chef_intents')
            model_name: HuggingFace model for zero-shot classification
            device: Device to run on ('cpu', 'cuda', or None for auto)
            use_keyword_fallback: Use keyword matching as fallback
        """
        self.domain = domain
        self.model_name = model_name
        self.device = device
        self.use_keyword_fallback = use_keyword_fallback
        self.pipeline = None
        self._initialized = False
        
        # Load configuration from YAML files
        logger.info(f"Loading intent configuration for domain: {domain}")
        self.config = load_intent_config(domain)
        
        # Extract intents and keywords from config
        self.intents_config = self.config['intents']
        self.keywords_config = self.config['keywords']
        
        # Build keyword patterns from config
        self.keyword_patterns = self._build_keyword_patterns()
        
        # Get list of intent labels
        self.intent_labels = list(self.intents_config['intents'].keys())
        
        logger.info(f"Loaded {len(self.intent_labels)} intents for {domain}")
        logger.info(f"Intent classifier using model: {model_name}")
    
    def _build_keyword_patterns(self) -> Dict[str, List[str]]:
        """
        Build keyword patterns from configuration.
        
        Returns:
            Dictionary mapping intent names to keyword lists
        """
        patterns = {}
        
        if 'keywords' not in self.keywords_config:
            logger.warning("No keywords found in configuration")
            return patterns
        
        for intent, keyword_data in self.keywords_config['keywords'].items():
            all_keywords = []
            
            # Collect all keyword types (primary, secondary, phrases)
            if isinstance(keyword_data, dict):
                for category, keywords in keyword_data.items():
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
            elif isinstance(keyword_data, list):
                all_keywords.extend(keyword_data)
            
            if all_keywords:
                patterns[intent] = all_keywords
        
        logger.debug(f"Built keyword patterns for {len(patterns)} intents")
        return patterns
    
    def load_model(self) -> None:
        """
        Load the transformer model for intent classification.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            from transformers import pipeline
            import torch
            
            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Loading intent classification model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load zero-shot classification pipeline
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == 'cuda' else -1
            )
            
            self._initialized = True
            logger.info("Intent classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load intent classification model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e
    
    def classify(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        multi_label: bool = False
    ) -> IntentPrediction:
        """
        Classify the intent of a text input.
        
        Args:
            text: Input text to classify
            candidate_labels: List of possible intent labels (uses domain intents if None)
            multi_label: Whether to treat as multi-label classification
            
        Returns:
            IntentPrediction with predicted intent and confidence
            
        Raises:
            RuntimeError: If model not initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not text or not text.strip():
            # Return default intent based on domain
            default_intent = 'general' if 'general' in self.intent_labels else self.intent_labels[0]
            return IntentPrediction(
                intent=default_intent,
                confidence=0.0,
                all_scores={}
            )
        
        # Use domain intents if not specified
        if candidate_labels is None:
            candidate_labels = self.intent_labels
        
        try:
            # Perform zero-shot classification
            result = self.pipeline(
                text,
                candidate_labels=candidate_labels,
                multi_label=multi_label
            )
            
            # Extract results
            intent = result['labels'][0]
            confidence = result['scores'][0]
            all_scores = dict(zip(result['labels'], result['scores']))
            
            logger.debug(f"Classified '{text[:50]}...' as {intent} ({confidence:.2f})")
            
            return IntentPrediction(
                intent=intent,
                confidence=confidence,
                all_scores=all_scores
            )
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            
            # Fallback to keyword matching
            if self.use_keyword_fallback:
                logger.info("Using keyword fallback for classification")
                return self._keyword_classify(text)
            else:
                raise
    
    def _keyword_classify(self, text: str) -> IntentPrediction:
        """
        Fallback keyword-based classification.
        
        Args:
            text: Input text
            
        Returns:
            IntentPrediction based on keyword matching
        """
        text_lower = text.lower()
        intent_scores = {}
        
        # Count keyword matches for each intent
        for intent, keywords in self.keyword_patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return best match or default intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            # Normalize score to 0-1 range (simple heuristic)
            confidence = min(intent_scores[best_intent] * 0.3, 1.0)
            
            return IntentPrediction(
                intent=best_intent,
                confidence=confidence,
                all_scores=intent_scores
            )
        else:
            # Use general or first intent as default
            default_intent = 'general' if 'general' in self.intent_labels else self.intent_labels[0]
            return IntentPrediction(
                intent=default_intent,
                confidence=0.5,
                all_scores={default_intent: 0.5}
            )
    
    def batch_classify(
        self,
        texts: List[str],
        candidate_labels: Optional[List[str]] = None
    ) -> List[IntentPrediction]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of input texts
            candidate_labels: List of possible intent labels
            
        Returns:
            List of IntentPrediction objects
        """
        return [self.classify(text, candidate_labels) for text in texts]
    
    def get_intent_keywords(self, intent: str) -> List[str]:
        """
        Get keywords associated with an intent.
        
        Args:
            intent: Intent label
            
        Returns:
            List of keywords for the intent
        """
        return self.keyword_patterns.get(intent, [])
    
    def is_initialized(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized
    
    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded domain configuration.
        
        Returns:
            Dictionary with domain metadata
        """
        return {
            'domain': self.domain,
            'num_intents': len(self.intent_labels),
            'intent_labels': self.intent_labels,
            'has_keyword_fallback': self.use_keyword_fallback,
            'num_keyword_patterns': len(self.keyword_patterns),
            'model_name': self.model_name,
            'initialized': self._initialized
        }
