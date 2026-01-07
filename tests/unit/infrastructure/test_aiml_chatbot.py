"""
Unit tests for AIML chatbot.

Tests the AimlChatbot class for initialization, AIML file loading,
response generation, and state management.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.ml.chatbots.aiml_chatbot import AimlChatbot


@pytest.fixture
def temp_aiml_dir(tmp_path):
    """Create temporary AIML directory with test files."""
    aiml_dir = tmp_path / "aiml"
    aiml_dir.mkdir()
    
    # Create a simple test AIML file
    test_aiml = aiml_dir / "test.aiml"
    test_aiml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<aiml version="1.0.1">
    <category>
        <pattern>HELLO</pattern>
        <template>Hi there!</template>
    </category>
    <category>
        <pattern>HOW ARE YOU</pattern>
        <template>I'm doing well, thank you!</template>
    </category>
</aiml>""")
    
    return aiml_dir


@pytest.fixture
def chatbot(temp_aiml_dir):
    """Create AimlChatbot instance with temp directory."""
    return AimlChatbot(aiml_dir=temp_aiml_dir)


class TestAimlChatbotInitialization:
    """Test AIML chatbot initialization."""
    
    def test_init_with_custom_dir(self, temp_aiml_dir):
        """Test initialization with custom AIML directory."""
        bot = AimlChatbot(aiml_dir=temp_aiml_dir)
        
        assert bot.aiml_dir == temp_aiml_dir
        assert bot.loaded_files == []
        assert not bot.is_ready()
    
    def test_init_with_default_dir(self):
        """Test initialization with default directory."""
        bot = AimlChatbot()
        
        assert bot.aiml_dir == Path("data/knowledge_bases/aiml")
        assert not bot.is_ready()
    
    def test_init_creates_kernel(self, chatbot):
        """Test that initialization creates AIML kernel."""
        assert chatbot.kernel is not None
        assert hasattr(chatbot.kernel, 'respond')


class TestAimlFileLoading:
    """Test AIML file loading functionality."""
    
    def test_load_aiml_files_success(self, chatbot):
        """Test successfully loading AIML files."""
        num_loaded = chatbot.load_aiml_files()
        
        assert num_loaded == 1  # One test.aiml file
        assert chatbot.is_ready()
        assert 'test.aiml' in chatbot.get_loaded_files()
    
    def test_load_aiml_files_nonexistent_dir(self):
        """Test error when AIML directory doesn't exist."""
        bot = AimlChatbot(aiml_dir="/nonexistent/path")
        
        with pytest.raises(ValueError, match="AIML directory not found"):
            bot.load_aiml_files()
    
    def test_load_aiml_files_no_files(self, tmp_path):
        """Test error when no AIML files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        bot = AimlChatbot(aiml_dir=empty_dir)
        
        with pytest.raises(ValueError, match="No AIML files found"):
            bot.load_aiml_files()
    
    def test_load_aiml_files_multiple(self, temp_aiml_dir):
        """Test loading multiple AIML files."""
        # Create second AIML file
        second_aiml = temp_aiml_dir / "second.aiml"
        second_aiml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<aiml version="1.0.1">
    <category>
        <pattern>BYE</pattern>
        <template>Goodbye!</template>
    </category>
</aiml>""")
        
        bot = AimlChatbot(aiml_dir=temp_aiml_dir)
        num_loaded = bot.load_aiml_files()
        
        assert num_loaded == 2
        assert len(bot.get_loaded_files()) == 2
    
    def test_load_specific_file_success(self, chatbot):
        """Test loading a specific AIML file."""
        chatbot.load_specific_file("test.aiml")
        
        assert chatbot.is_ready()
        assert 'test.aiml' in chatbot.get_loaded_files()
    
    def test_load_specific_file_not_found(self, chatbot):
        """Test error when specific file doesn't exist."""
        with pytest.raises(ValueError, match="AIML file not found"):
            chatbot.load_specific_file("nonexistent.aiml")
    
    def test_load_files_sets_initialized_flag(self, chatbot):
        """Test that loading files sets initialized flag."""
        assert not chatbot._is_initialized
        chatbot.load_aiml_files()
        assert chatbot._is_initialized


class TestResponseGeneration:
    """Test chatbot response generation."""
    
    def test_get_response_basic(self, chatbot):
        """Test generating response to user input."""
        chatbot.load_aiml_files()
        
        response = chatbot.get_response("HELLO")
        assert response == "Hi there!"
    
    def test_get_response_case_insensitive(self, chatbot):
        """Test that AIML matching is case-insensitive."""
        chatbot.load_aiml_files()
        
        response1 = chatbot.get_response("hello")
        response2 = chatbot.get_response("HELLO")
        response3 = chatbot.get_response("HeLLo")
        
        # All should return same response
        assert response1 == response2 == response3
    
    def test_get_response_multiple_patterns(self, chatbot):
        """Test responses to different patterns."""
        chatbot.load_aiml_files()
        
        response1 = chatbot.get_response("HELLO")
        response2 = chatbot.get_response("HOW ARE YOU")
        
        assert response1 == "Hi there!"
        assert response2 == "I'm doing well, thank you!"
    
    def test_get_response_no_match(self, chatbot):
        """Test fallback response when no pattern matches."""
        chatbot.load_aiml_files()
        
        response = chatbot.get_response("UNKNOWN PATTERN")
        assert "not sure" in response.lower() or "rephrase" in response.lower()
    
    def test_get_response_empty_input(self, chatbot):
        """Test handling of empty input."""
        chatbot.load_aiml_files()
        
        response = chatbot.get_response("")
        assert "didn't quite catch" in response.lower()
    
    def test_get_response_whitespace_only(self, chatbot):
        """Test handling of whitespace-only input."""
        chatbot.load_aiml_files()
        
        response = chatbot.get_response("   ")
        assert "didn't quite catch" in response.lower()
    
    def test_get_response_not_initialized(self, chatbot):
        """Test error when trying to respond before initialization."""
        with pytest.raises(RuntimeError, match="not initialized"):
            chatbot.get_response("HELLO")
    
    def test_chat_alias(self, chatbot):
        """Test that chat() is an alias for get_response()."""
        chatbot.load_aiml_files()
        
        response1 = chatbot.get_response("HELLO")
        response2 = chatbot.chat("HELLO")
        
        assert response1 == response2


class TestChatbotState:
    """Test chatbot state management."""
    
    def test_reset_clears_state(self, chatbot):
        """Test that reset clears chatbot state."""
        chatbot.load_aiml_files()
        assert chatbot.is_ready()
        
        chatbot.reset()
        
        assert not chatbot.is_ready()
        assert chatbot.get_loaded_files() == []
    
    def test_is_ready_after_loading(self, chatbot):
        """Test is_ready returns correct status."""
        assert not chatbot.is_ready()
        
        chatbot.load_aiml_files()
        assert chatbot.is_ready()
    
    def test_get_loaded_files_returns_copy(self, chatbot):
        """Test that get_loaded_files returns a copy."""
        chatbot.load_aiml_files()
        
        files1 = chatbot.get_loaded_files()
        files2 = chatbot.get_loaded_files()
        
        # Should be equal but not same object
        assert files1 == files2
        assert files1 is not files2


class TestPredicates:
    """Test AIML predicate (variable) functionality."""
    
    def test_set_and_get_predicate(self, chatbot):
        """Test setting and getting predicates."""
        chatbot.load_aiml_files()
        
        chatbot.set_predicate("name", "Alice")
        value = chatbot.get_predicate("name")
        
        assert value == "Alice"
    
    def test_get_predicate_not_set(self, chatbot):
        """Test getting unset predicate returns empty string."""
        chatbot.load_aiml_files()
        
        value = chatbot.get_predicate("nonexistent")
        assert value == ""
    
    def test_predicate_persistence(self, chatbot):
        """Test that predicates persist across responses."""
        chatbot.load_aiml_files()
        
        chatbot.set_predicate("user", "Bob")
        chatbot.get_response("HELLO")
        
        # Predicate should still be set
        assert chatbot.get_predicate("user") == "Bob"


class TestCategoryCount:
    """Test AIML category counting."""
    
    def test_get_num_categories(self, chatbot):
        """Test getting number of loaded categories."""
        chatbot.load_aiml_files()
        
        num_categories = chatbot.get_num_categories()
        assert num_categories == 2  # Two patterns in test.aiml


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_malformed_aiml_file(self, temp_aiml_dir):
        """Test handling of malformed AIML file."""
        bad_aiml = temp_aiml_dir / "bad.aiml"
        bad_aiml.write_text("This is not valid AIML")
        
        bot = AimlChatbot(aiml_dir=temp_aiml_dir)
        
        # AIML library doesn't raise exceptions for malformed files,
        # it just prints errors and continues. Test that loading completes
        # but we should still have valid AIML from test.aiml
        num_loaded = bot.load_aiml_files()
        assert num_loaded >= 1  # At least test.aiml should load
    
    def test_response_strips_whitespace(self, chatbot):
        """Test that input whitespace is stripped."""
        chatbot.load_aiml_files()
        
        response1 = chatbot.get_response("  HELLO  ")
        response2 = chatbot.get_response("HELLO")
        
        assert response1 == response2


class TestIntegration:
    """Integration tests with real AIML files."""
    
    def test_full_conversation_flow(self, chatbot):
        """Test a full conversation flow."""
        chatbot.load_aiml_files()
        
        # Start conversation
        response1 = chatbot.chat("HELLO")
        assert response1 == "Hi there!"
        
        # Continue
        response2 = chatbot.chat("HOW ARE YOU")
        assert response2 == "I'm doing well, thank you!"
    
    def test_reload_after_reset(self, chatbot):
        """Test reloading files after reset."""
        chatbot.load_aiml_files()
        response1 = chatbot.chat("HELLO")
        
        chatbot.reset()
        chatbot.load_aiml_files()
        response2 = chatbot.chat("HELLO")
        
        assert response1 == response2
