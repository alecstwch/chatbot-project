"""
Unit tests for DialoGPTChatbot.

Tests cover initialization, model loading, response generation,
conversation history management, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from src.infrastructure.ml.chatbots.dialogpt_chatbot import DialoGPTChatbot


class TestDialoGPTChatbotInitialization:
    """Test chatbot initialization."""
    
    def test_init_with_default_model(self):
        """Should initialize with default DialoGPT-small model."""
        bot = DialoGPTChatbot()
        assert bot.model_name == "microsoft/DialoGPT-small"
        assert bot.max_length == 1000
        assert not bot.is_ready()
    
    def test_init_with_custom_model(self):
        """Should initialize with custom model name."""
        bot = DialoGPTChatbot(model_name="microsoft/DialoGPT-medium")
        assert bot.model_name == "microsoft/DialoGPT-medium"
    
    def test_init_with_custom_max_length(self):
        """Should initialize with custom max length."""
        bot = DialoGPTChatbot(max_length=500)
        assert bot.max_length == 500
    
    def test_init_sets_device_cpu_when_no_cuda(self):
        """Should default to CPU when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            bot = DialoGPTChatbot()
            assert bot.device == 'cpu'
    
    def test_init_sets_device_cuda_when_available(self):
        """Should use CUDA when available."""
        with patch('torch.cuda.is_available', return_value=True):
            bot = DialoGPTChatbot()
            assert bot.device == 'cuda'
    
    def test_init_with_explicit_device(self):
        """Should use explicitly specified device."""
        bot = DialoGPTChatbot(device='cpu')
        assert bot.device == 'cpu'
    
    def test_init_creates_empty_history(self):
        """Should start with no conversation history."""
        bot = DialoGPTChatbot()
        assert bot.chat_history_ids is None
        assert bot.get_conversation_length() == 0


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class):
        """Should successfully load model and tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        assert bot.is_ready()
        assert bot.tokenizer is not None
        assert bot.model is not None
        mock_tokenizer_class.from_pretrained.assert_called_once_with("microsoft/DialoGPT-small")
        mock_model_class.from_pretrained.assert_called_once_with("microsoft/DialoGPT-small")
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_load_model_sets_pad_token(self, mock_model_class, mock_tokenizer_class):
        """Should set pad_token to eos_token if not present."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        assert mock_tokenizer.pad_token == '<eos>'
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_load_model_moves_to_device(self, mock_model_class, mock_tokenizer_class):
        """Should move model to specified device."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot(device='cpu')
        bot.load_model()
        
        mock_model.to.assert_called_once_with('cpu')
        mock_model.eval.assert_called_once()
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    def test_load_model_handles_errors(self, mock_tokenizer_class):
        """Should raise RuntimeError on model loading failure."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        bot = DialoGPTChatbot()
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            bot.load_model()
        
        assert not bot.is_ready()


class TestResponseGeneration:
    """Test response generation."""
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_get_response_basic(self, mock_model_class, mock_tokenizer_class):
        """Should generate response to user input."""
        mock_tokenizer = self._create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = self._create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        response = bot.get_response("Hello")
        
        assert isinstance(response, str)
        assert len(response) > 0
        mock_model.generate.assert_called_once()
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_get_response_maintains_history(self, mock_model_class, mock_tokenizer_class):
        """Should maintain conversation history across turns."""
        mock_tokenizer = self._create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = self._create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        bot.get_response("Hello")
        assert bot.chat_history_ids is not None
        
        first_history_length = bot.get_conversation_length()
        
        bot.get_response("How are you?")
        second_history_length = bot.get_conversation_length()
        
        assert second_history_length >= first_history_length
    
    def test_get_response_not_initialized(self):
        """Should raise error if model not loaded."""
        bot = DialoGPTChatbot()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            bot.get_response("Hello")
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_get_response_empty_input(self, mock_model_class, mock_tokenizer_class):
        """Should handle empty input gracefully."""
        mock_tokenizer = self._create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = self._create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        response = bot.get_response("")
        assert "didn't catch that" in response.lower()
        
        response = bot.get_response("   ")
        assert "didn't catch that" in response.lower()
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_get_response_with_custom_temperature(self, mock_model_class, mock_tokenizer_class):
        """Should use custom temperature parameter."""
        mock_tokenizer = self._create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = self._create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        bot.get_response("Hello", temperature=0.9)
        
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs['temperature'] == 0.9
        assert 'attention_mask' in call_kwargs  # Verify attention mask is passed
        assert 'repetition_penalty' in call_kwargs  # Verify new parameters
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_chat_alias(self, mock_model_class, mock_tokenizer_class):
        """Should support chat() as alias for get_response()."""
        mock_tokenizer = self._create_mock_tokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = self._create_mock_model()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        response = bot.chat("Hello")
        assert isinstance(response, str)
    
    @staticmethod
    def _create_mock_tokenizer():
        """Helper to create mock tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer.eos_token_id = 50256
        
        # Mock encode_plus to return dict with input_ids and attention_mask
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.encode_plus.return_value = mock_encoded
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Hi there!"
        return mock_tokenizer
    
    @staticmethod
    def _create_mock_model():
        """Helper to create mock model."""
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        return mock_model


class TestConversationManagement:
    """Test conversation history management."""
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_reset_clears_history(self, mock_model_class, mock_tokenizer_class):
        """Should clear conversation history on reset."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer.eos_token_id = 50256
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.encode_plus.return_value = mock_encoded
        mock_tokenizer.decode.return_value = "Hello"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        bot.get_response("Hello")
        assert bot.chat_history_ids is not None
        
        bot.reset()
        assert bot.chat_history_ids is None
        assert bot.get_conversation_length() == 0
    
    def test_get_conversation_length_empty(self):
        """Should return 0 for empty conversation."""
        bot = DialoGPTChatbot()
        assert bot.get_conversation_length() == 0
    
    def test_truncate_history_no_history(self):
        """Should handle truncation when no history exists."""
        bot = DialoGPTChatbot()
        bot.truncate_history()
        assert bot.chat_history_ids is None
    
    def test_truncate_history_below_threshold(self):
        """Should not truncate if below threshold."""
        bot = DialoGPTChatbot()
        bot.chat_history_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        bot.truncate_history(max_tokens=100)
        
        assert bot.chat_history_ids.shape[-1] == 5
    
    def test_truncate_history_above_threshold(self):
        """Should truncate if above threshold."""
        bot = DialoGPTChatbot()
        bot.chat_history_ids = torch.tensor([[i for i in range(100)]])
        
        bot.truncate_history(max_tokens=50)
        
        assert bot.chat_history_ids.shape[-1] == 50
        assert bot.chat_history_ids[0, 0].item() == 50


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_response_generation_error_handling(self, mock_model_class, mock_tokenizer_class):
        """Should handle errors during response generation gracefully."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.encode.side_effect = Exception("Encoding error")
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        response = bot.get_response("Hello")
        assert "error" in response.lower()


class TestIntegration:
    """Integration tests for full conversation flows."""
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_full_conversation_flow(self, mock_model_class, mock_tokenizer_class):
        """Should handle complete multi-turn conversation."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer.eos_token_id = 50256
        
        responses = ["Hi!", "I'm doing well!", "Nice to meet you too!"]
        call_count = [0]
        
        def mock_encode_plus(text, return_tensors=None, add_special_tokens=True):
            return {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
        
        def mock_decode(tokens, skip_special_tokens=False):
            result = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return result
        
        mock_tokenizer.encode_plus = mock_encode_plus
        mock_tokenizer.decode = mock_decode
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        # Multi-turn conversation
        r1 = bot.get_response("Hello")
        assert r1 == "Hi!"
        
        r2 = bot.get_response("How are you?")
        assert r2 == "I'm doing well!"
        
        r3 = bot.get_response("Nice to meet you")
        assert r3 == "Nice to meet you too!"
        
        # Conversation history should have grown
        assert bot.get_conversation_length() > 0
    
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoTokenizer')
    @patch('src.infrastructure.ml.chatbots.dialogpt_chatbot.AutoModelForCausalLM')
    def test_conversation_reset_and_restart(self, mock_model_class, mock_tokenizer_class):
        """Should properly reset and start new conversation."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.pad_token = '<pad>'
        mock_tokenizer.eos_token_id = 50256
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.encode_plus.return_value = mock_encoded
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        bot = DialoGPTChatbot()
        bot.load_model()
        
        # First conversation
        bot.get_response("Hello")
        first_length = bot.get_conversation_length()
        assert first_length > 0
        
        # Reset
        bot.reset()
        assert bot.get_conversation_length() == 0
        
        # New conversation
        bot.get_response("Hi again")
        new_length = bot.get_conversation_length()
        assert new_length > 0
        assert new_length <= first_length  # New conversation should be same or shorter
