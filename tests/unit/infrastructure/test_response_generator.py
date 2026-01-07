"""
Unit tests for Response Generation Service.

Tests GPT-2 based response generation functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.infrastructure.ml.models.response_generator import (
    ResponseGenerationService,
    GenerationConfig
)


class TestGenerationConfig:
    """Test generation configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.max_length == 100
        assert config.min_length == 10
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.num_return_sequences == 1
        assert config.repetition_penalty == 1.2
        assert config.do_sample is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_length=150,
            temperature=0.8,
            top_k=40
        )
        
        assert config.max_length == 150
        assert config.temperature == 0.8
        assert config.top_k == 40


class TestResponseGeneratorInitialization:
    """Test response generator initialization."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        generator = ResponseGenerationService()
        
        assert generator.model_name == "gpt2"
        assert generator.device is None
        assert generator._initialized is False
        assert generator.tokenizer is None
        assert generator.model is None
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        generator = ResponseGenerationService(
            model_name="gpt2-medium",
            device="cpu",
            cache_dir=Path("custom/cache")
        )
        
        assert generator.model_name == "gpt2-medium"
        assert generator.device == "cpu"
        assert generator.cache_dir == Path("custom/cache")


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_load_model_cpu(self, mock_torch, mock_model, mock_tokenizer):
        """Test loading model on CPU."""
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer.from_pretrained.return_value = Mock(
            pad_token=None,
            eos_token='<eos>'
        )
        mock_model.from_pretrained.return_value = Mock()
        
        generator = ResponseGenerationService()
        generator.load_model()
        
        assert generator._initialized is True
        assert generator.device == 'cpu'
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_load_model_cuda(self, mock_torch, mock_model, mock_tokenizer):
        """Test loading model on CUDA."""
        mock_torch.cuda.is_available.return_value = True
        mock_tokenizer.from_pretrained.return_value = Mock(
            pad_token=None,
            eos_token='<eos>'
        )
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        generator = ResponseGenerationService()
        generator.load_model()
        
        assert generator._initialized is True
        assert generator.device == 'cuda'
        mock_model_instance.to.assert_called_with('cuda')
        mock_model_instance.eval.assert_called_once()
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_load_model_sets_padding_token(self, mock_torch, mock_model, mock_tokenizer):
        """Test that padding token is set if not present."""
        mock_torch.cuda.is_available.return_value = False
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '<eos>'
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = Mock()
        
        generator = ResponseGenerationService()
        generator.load_model()
        
        assert generator.tokenizer.pad_token == '<eos>'


class TestResponseGeneration:
    """Test response generation functionality."""
    
    def test_generate_without_loading_model(self):
        """Test that generation fails if model not loaded."""
        generator = ResponseGenerationService()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            generator.generate_response("test prompt")
    
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_generate_empty_input(self, mock_torch):
        """Test generation with empty input."""
        generator = ResponseGenerationService()
        generator._initialized = True
        
        result = generator.generate_response("")
        
        assert result == ""
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_generate_response_basic(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test basic response generation."""
        mock_torch.cuda.is_available.return_value = False
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode.return_value = Mock(to=Mock(return_value=Mock()))
        mock_tokenizer.decode.return_value = "Test prompt Generated response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = [Mock()]
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create generator and load
        generator = ResponseGenerationService()
        generator.load_model()
        
        # Generate response
        result = generator.generate_response("Test prompt")
        
        assert "Generated response" in result or result != ""
        mock_model.generate.assert_called_once()


class TestTherapyResponseGeneration:
    """Test therapy-specific response generation."""
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_therapy_response_with_intent(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test therapy response generation with intent."""
        mock_torch.cuda.is_available.return_value = False
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode.return_value = Mock(to=Mock(return_value=Mock()))
        mock_tokenizer.decode.return_value = "Patient: I feel sad\nTherapist (addressing depression): Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = [Mock()]
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Generate
        generator = ResponseGenerationService()
        generator.load_model()
        result = generator.generate_therapy_response("I feel sad", intent="depression")
        
        # Verify generation was called
        mock_model.generate.assert_called_once()
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_therapy_response_without_intent(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test therapy response generation without intent."""
        mock_torch.cuda.is_available.return_value = False
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode.return_value = Mock(to=Mock(return_value=Mock()))
        mock_tokenizer.decode.return_value = "Patient: I feel sad\nTherapist: Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = [Mock()]
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Generate
        generator = ResponseGenerationService()
        generator.load_model()
        result = generator.generate_therapy_response("I feel sad")
        
        mock_model.generate.assert_called_once()


class TestMultipleResponseGeneration:
    """Test generating multiple response candidates."""
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_generate_multiple_responses(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test generating multiple response candidates."""
        mock_torch.cuda.is_available.return_value = False
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode.return_value = Mock(to=Mock(return_value=Mock()))
        mock_tokenizer.decode.side_effect = [
            "Prompt Response 1",
            "Prompt Response 2",
            "Prompt Response 3"
        ]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = [Mock(), Mock(), Mock()]
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Generate
        generator = ResponseGenerationService()
        generator.load_model()
        results = generator.generate_multiple_responses("Prompt", num_responses=3)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestInitializationStatus:
    """Test initialization status checking."""
    
    def test_is_initialized_false(self):
        """Test is_initialized returns False initially."""
        generator = ResponseGenerationService()
        
        assert generator.is_initialized() is False
    
    @patch('src.infrastructure.ml.models.response_generator.GPT2Tokenizer')
    @patch('src.infrastructure.ml.models.response_generator.GPT2LMHeadModel')
    @patch('src.infrastructure.ml.models.response_generator.torch')
    def test_is_initialized_true(self, mock_torch, mock_model, mock_tokenizer):
        """Test is_initialized returns True after loading."""
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer.from_pretrained.return_value = Mock(
            pad_token=None,
            eos_token='<eos>'
        )
        mock_model.from_pretrained.return_value = Mock()
        
        generator = ResponseGenerationService()
        generator.load_model()
        
        assert generator.is_initialized() is True
