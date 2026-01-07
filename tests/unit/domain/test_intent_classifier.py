"""
Unit tests for Intent Classification Service.

Tests the intent classifier functionality including keyword fallback,
therapy intent detection, and model initialization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.domain.services.intent_classifier import (
    IntentClassificationService,
    IntentPrediction,
    TherapyIntent
)


class TestIntentClassifierInitialization:
    """Test intent classifier initialization."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        classifier = IntentClassificationService()
        
        assert classifier.model_name == "facebook/bart-large-mnli"
        assert classifier.use_keyword_fallback is True
        assert classifier._initialized is False
        assert classifier.pipeline is None
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        classifier = IntentClassificationService(
            model_name="custom-model",
            device="cpu",
            use_keyword_fallback=False
        )
        
        assert classifier.model_name == "custom-model"
        assert classifier.device == "cpu"
        assert classifier.use_keyword_fallback is False
    
    def test_keyword_patterns_loaded(self):
        """Test that keyword patterns are properly loaded."""
        classifier = IntentClassificationService()
        
        # Check specific therapy intents have patterns (not all including 'general')
        key_intents = [
            TherapyIntent.DEPRESSION,
            TherapyIntent.ANXIETY,
            TherapyIntent.STRESS,
            TherapyIntent.GREETING,
            TherapyIntent.FAREWELL
        ]
        for intent in key_intents:
            assert intent.value in classifier.keyword_patterns
            assert len(classifier.keyword_patterns[intent.value]) > 0


class TestKeywordClassification:
    """Test keyword-based fallback classification."""
    
    def test_depression_keywords(self):
        """Test detection of depression keywords."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("I feel so depressed and hopeless")
        
        assert result.intent == TherapyIntent.DEPRESSION.value
        assert result.confidence > 0
        assert TherapyIntent.DEPRESSION.value in result.all_scores
    
    def test_anxiety_keywords(self):
        """Test detection of anxiety keywords."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("I'm feeling very anxious and worried")
        
        assert result.intent == TherapyIntent.ANXIETY.value
        assert result.confidence > 0
    
    def test_stress_keywords(self):
        """Test detection of stress keywords."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("I'm so stressed and overwhelmed")
        
        assert result.intent == TherapyIntent.STRESS.value
        assert result.confidence > 0
    
    def test_greeting_keywords(self):
        """Test detection of greeting keywords."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("Hello, how are you?")
        
        assert result.intent == TherapyIntent.GREETING.value
        assert result.confidence > 0
    
    def test_farewell_keywords(self):
        """Test detection of farewell keywords."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("Goodbye, thank you for your help")
        
        assert result.intent == TherapyIntent.FAREWELL.value
        assert result.confidence > 0
    
    def test_no_keywords_match(self):
        """Test classification when no keywords match."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("The weather is nice today")
        
        assert result.intent == TherapyIntent.GENERAL.value
        assert result.confidence == 0.5
    
    def test_multiple_keywords(self):
        """Test that multiple keywords increase confidence."""
        classifier = IntentClassificationService()
        
        result = classifier._keyword_classify("I feel depressed, sad, and hopeless")
        
        assert result.intent == TherapyIntent.DEPRESSION.value
        assert result.confidence > 0.3  # Multiple matches = higher confidence


class TestModelLoadingAndClassification:
    """Test model loading and classification with mocked transformer."""
    
    def test_load_model_cpu(self):
        """Test loading model on CPU (integration test - loads real model)."""
        # This is an integration test that loads the real model
        # Skip if transformers not available or too slow for unit tests
        pytest.skip("Integration test - loads real transformers model (slow)")
        
        classifier = IntentClassificationService()
        classifier.load_model()
        
        assert classifier._initialized is True
        assert classifier.device in ['cpu', 'cuda']
    
    @patch('transformers.pipeline')
    def test_load_model_cuda(self, mock_pipeline):
        """Test loading model on CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            mock_pipeline.return_value = Mock()
            
            classifier = IntentClassificationService()
            classifier.load_model()
            
            assert classifier._initialized is True
            assert classifier.device == 'cuda'
    
    @patch('transformers.pipeline')
    def test_classify_with_model(self, mock_pipeline):
        """Test classification with loaded model."""
        with patch('torch.cuda.is_available', return_value=False):
            # Mock pipeline results
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = {
                'labels': ['anxiety', 'depression', 'general'],
                'scores': [0.7, 0.2, 0.1]
            }
            mock_pipeline.return_value = mock_pipeline_instance
            
            classifier = IntentClassificationService()
            classifier.load_model()
            result = classifier.classify("I'm feeling anxious")
            
            assert result.intent == 'anxiety'
            assert result.confidence == 0.7
            assert 'anxiety' in result.all_scores
    
    def test_classify_without_loading_model(self):
        """Test that classification fails if model not loaded."""
        classifier = IntentClassificationService(use_keyword_fallback=False)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            classifier.classify("test input")
    
    def test_classify_empty_input(self):
        """Test classification with empty input."""
        classifier = IntentClassificationService()
        classifier._initialized = True  # Fake initialization
        
        result = classifier.classify("")
        
        assert result.intent == TherapyIntent.GENERAL.value
        assert result.confidence == 0.0


class TestBatchClassification:
    """Test batch classification functionality."""
    
    @patch('transformers.pipeline')
    def test_batch_classify(self, mock_pipeline):
        """Test classifying multiple texts."""
        with patch('torch.cuda.is_available', return_value=False):
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.side_effect = [
                {'labels': ['anxiety', 'depression'], 'scores': [0.8, 0.2]},
                {'labels': ['depression', 'anxiety'], 'scores': [0.9, 0.1]},
            ]
            mock_pipeline.return_value = mock_pipeline_instance
            
            classifier = IntentClassificationService()
            classifier.load_model()
            
            texts = ["I'm anxious", "I'm depressed"]
            results = classifier.batch_classify(texts)
            
            assert len(results) == 2
            assert all(isinstance(r, IntentPrediction) for r in results)


class TestIntentKeywordRetrieval:
    """Test keyword retrieval for intents."""
    
    def test_get_intent_keywords_depression(self):
        """Test retrieving depression keywords."""
        classifier = IntentClassificationService()
        
        keywords = classifier.get_intent_keywords('depression')
        
        assert len(keywords) > 0
        assert 'depressed' in keywords
        assert 'sad' in keywords
    
    def test_get_intent_keywords_anxiety(self):
        """Test retrieving anxiety keywords."""
        classifier = IntentClassificationService()
        
        keywords = classifier.get_intent_keywords('anxiety')
        
        assert len(keywords) > 0
        assert 'anxious' in keywords
        assert 'worried' in keywords
    
    def test_get_intent_keywords_nonexistent(self):
        """Test retrieving keywords for nonexistent intent."""
        classifier = IntentClassificationService()
        
        keywords = classifier.get_intent_keywords('nonexistent')
        
        assert keywords == []


class TestInitializationStatus:
    """Test initialization status checking."""
    
    def test_is_initialized_false(self):
        """Test is_initialized returns False initially."""
        classifier = IntentClassificationService()
        
        assert classifier.is_initialized() is False
    
    @patch('transformers.pipeline')
    def test_is_initialized_true(self, mock_pipeline):
        """Test is_initialized returns True after loading."""
        with patch('torch.cuda.is_available', return_value=False):
            mock_pipeline.return_value = Mock()
            
            classifier = IntentClassificationService()
            classifier.load_model()
            
            assert classifier.is_initialized() is True