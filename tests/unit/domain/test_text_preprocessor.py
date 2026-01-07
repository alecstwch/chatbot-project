"""
Unit tests for TextPreprocessingService

Tests the domain logic in isolation without external dependencies.
"""

import pytest
from src.domain.services.text_preprocessor import TextPreprocessingService


class TestTextPreprocessingService:
    """Test suite for TextPreprocessingService"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing"""
        return TextPreprocessingService()
    
    def test_initialization(self, preprocessor):
        """Test service initialization"""
        assert preprocessor.language == 'english'
        assert len(preprocessor.stop_words) > 0
        assert preprocessor.stemmer is not None
        assert preprocessor.lemmatizer is not None
    
    def test_clean_text_removes_urls(self, preprocessor):
        """Test URL removal"""
        text = "Check this out https://example.com and www.test.com"
        result = preprocessor.clean_text(text)
        assert "https://example.com" not in result
        assert "www.test.com" not in result
        assert "Check this out" in result
    
    def test_clean_text_removes_html_tags(self, preprocessor):
        """Test HTML tag removal"""
        text = "This is <b>bold</b> and <i>italic</i> text"
        result = preprocessor.clean_text(text)
        assert "<b>" not in result
        assert "</b>" not in result
        assert "bold" in result
        assert "italic" in result
    
    def test_clean_text_removes_emails(self, preprocessor):
        """Test email address removal"""
        text = "Contact me at user@example.com for info"
        result = preprocessor.clean_text(text)
        assert "user@example.com" not in result
        assert "Contact me at" in result
    
    def test_clean_text_removes_special_characters(self, preprocessor):
        """Test special character removal"""
        text = "Hello! @#$% How are you?"
        result = preprocessor.clean_text(text)
        assert "@#$%" not in result
        assert "Hello!" in result  # Basic punctuation kept
        assert "How are you?" in result
    
    def test_clean_text_removes_extra_whitespace(self, preprocessor):
        """Test whitespace normalization"""
        text = "Too    many     spaces"
        result = preprocessor.clean_text(text)
        assert "  " not in result
        assert result == "Too many spaces"
    
    def test_tokenize(self, preprocessor):
        """Test tokenization"""
        text = "Hello, how are you?"
        tokens = preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert "Hello" in tokens or "hello" in [t.lower() for t in tokens]
        assert len(tokens) > 0
    
    def test_to_lowercase(self, preprocessor):
        """Test lowercase conversion"""
        tokens = ["Hello", "WORLD", "Test"]
        result = preprocessor.to_lowercase(tokens)
        assert result == ["hello", "world", "test"]
    
    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal"""
        tokens = ["the", "quick", "brown", "fox"]
        result = preprocessor.remove_stopwords(tokens)
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result
    
    def test_apply_stemming(self, preprocessor):
        """Test Porter Stemmer"""
        tokens = ["running", "flies", "easily"]
        result = preprocessor.apply_stemming(tokens)
        # Stemmer reduces to root forms
        assert "run" in result or "runn" in result
        assert len(result) == len(tokens)
    
    def test_apply_lemmatization(self, preprocessor):
        """Test WordNet Lemmatizer"""
        tokens = ["running", "flies", "better"]
        result = preprocessor.apply_lemmatization(tokens)
        # Lemmatizer converts to dictionary forms
        assert isinstance(result, list)
        assert len(result) == len(tokens)
    
    def test_preprocess_default_settings(self, preprocessor):
        """Test full preprocessing with default settings"""
        text = "I'm running to the store!"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be lowercase
        assert result.islower() or not result.isalpha()
    
    def test_preprocess_without_lowercase(self, preprocessor):
        """Test preprocessing without lowercasing"""
        text = "Hello World"
        result = preprocessor.preprocess(text, lowercase=False)
        # Original case should be preserved (after other operations)
        assert "Hello" in result or "World" in result or result != result.lower()
    
    def test_preprocess_with_stopwords_kept(self, preprocessor):
        """Test preprocessing keeping stopwords"""
        text = "The quick brown fox"
        result_with = preprocessor.preprocess(text, remove_stopwords=False)
        result_without = preprocessor.preprocess(text, remove_stopwords=True)
        # With stopwords should be longer
        assert len(result_with.split()) >= len(result_without.split())
    
    def test_preprocess_stemming_vs_lemmatization(self, preprocessor):
        """Test stemming vs lemmatization"""
        text = "running easily"
        stemmed = preprocessor.preprocess(text, stemming=True, lemmatization=False)
        lemmatized = preprocessor.preprocess(text, stemming=False, lemmatization=True)
        # Results may differ
        assert isinstance(stemmed, str)
        assert isinstance(lemmatized, str)
    
    def test_compare_preprocessing_methods(self, preprocessor):
        """Test preprocessing comparison"""
        text = "I'm feeling very anxious today!"
        results = preprocessor.compare_preprocessing_methods(text)
        
        assert 'original' in results
        assert 'cleaned_only' in results
        assert 'with_stopwords' in results
        assert 'without_stopwords' in results
        assert 'stemmed' in results
        assert 'lemmatized' in results
        
        # Original should be unchanged
        assert results['original'] == text
        
        # Different methods should produce different results
        assert results['with_stopwords'] != results['without_stopwords']
    
    def test_batch_preprocess(self, preprocessor):
        """Test batch preprocessing"""
        texts = [
            "Hello world!",
            "How are you?",
            "I'm fine, thanks!"
        ]
        results = preprocessor.batch_preprocess(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 0 for r in results)
    
    def test_batch_preprocess_with_options(self, preprocessor):
        """Test batch preprocessing with custom options"""
        texts = ["The cat", "The dog"]
        results = preprocessor.batch_preprocess(
            texts,
            remove_stopwords=False,
            lowercase=True
        )
        assert len(results) == 2
        # Should keep "the" since stopwords not removed
        assert any("the" in r for r in results)
    
    def test_empty_text_handling(self, preprocessor):
        """Test handling of empty text"""
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_whitespace_only_text(self, preprocessor):
        """Test handling of whitespace-only text"""
        result = preprocessor.preprocess("   \n\t  ")
        assert result == "" or result.strip() == ""
    
    def test_special_characters_only(self, preprocessor):
        """Test text with only special characters"""
        text = "@#$%^&*()"
        result = preprocessor.clean_text(text)
        assert result == "" or result.strip() == ""
    
    def test_mixed_content_preservation(self, preprocessor):
        """Test that meaningful content is preserved"""
        text = "I feel anxious about my health"
        result = preprocessor.preprocess(text)
        # Important words should be preserved
        assert "anxious" in result or "anxiou" in result  # May be lemmatized
        assert "health" in result
    
    def test_multiple_sentences(self, preprocessor):
        """Test preprocessing of multiple sentences"""
        text = "Hello! How are you? I am fine."
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0


class TestTextPreprocessingEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessingService()
    
    def test_very_long_text(self, preprocessor):
        """Test with very long text"""
        text = "word " * 1000
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
    
    def test_unicode_characters(self, preprocessor):
        """Test handling of unicode characters"""
        text = "Hello 你好 مرحبا"
        result = preprocessor.clean_text(text)
        # Should handle gracefully (may remove non-ASCII)
        assert isinstance(result, str)
    
    def test_numbers_handling(self, preprocessor):
        """Test handling of numbers"""
        text = "I have 5 cats and 3 dogs"
        result = preprocessor.clean_text(text)
        assert "5" in result
        assert "3" in result
    
    def test_multiple_urls(self, preprocessor):
        """Test removal of multiple URLs"""
        text = "Visit https://site1.com or www.site2.com or http://site3.org"
        result = preprocessor.clean_text(text)
        assert "https://site1.com" not in result
        assert "www.site2.com" not in result
        assert "http://site3.org" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
