"""
Text Preprocessing Domain Service

Pure business logic for text preprocessing.
No external dependencies beyond NLP libraries.
Follows Domain-Driven Design principles.
"""

import re
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessingService:
    """
    Domain service for text preprocessing operations.
    
    This is pure business logic with no I/O operations.
    Demonstrates multiple NLP techniques for the chatbot project.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessing service.
        
        Args:
            language: Language for stopwords (default: 'english')
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Removes:
        - URLs (http/https/www)
        - HTML tags
        - Email addresses
        - Special characters (keeps alphanumeric and basic punctuation)
        - Extra whitespace
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK word tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def to_lowercase(self, tokens: List[str]) -> List[str]:
        """
        Convert all tokens to lowercase.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lowercased tokens
        """
        return [token.lower() for token in tokens]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter Stemmer to tokens.
        
        Stemming reduces words to their root form (e.g., 'running' -> 'run').
        Faster but less accurate than lemmatization.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply WordNet Lemmatizer to tokens.
        
        Lemmatization reduces words to their dictionary form.
        More accurate but slower than stemming.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(
        self, 
        text: str, 
        lowercase: bool = True,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatization: bool = True
    ) -> str:
        """
        Execute full preprocessing pipeline with configurable steps.
        
        Args:
            text: Input text to preprocess
            lowercase: Convert to lowercase (default: True)
            remove_stopwords: Remove stopwords (default: True)
            stemming: Apply Porter Stemmer (default: False)
            lemmatization: Apply Lemmatization (default: True)
            
        Returns:
            Preprocessed text as string
            
        Note:
            If both stemming and lemmatization are True,
            only lemmatization will be applied.
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(text)
        
        # Step 3: Lowercase
        if lowercase:
            tokens = self.to_lowercase(tokens)
        
        # Step 4: Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 5: Stemming (mutually exclusive with lemmatization)
        if stemming and not lemmatization:
            tokens = self.apply_stemming(tokens)
        
        # Step 6: Lemmatization
        if lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        # Join tokens back into string
        return ' '.join(tokens)
    
    def compare_preprocessing_methods(self, text: str) -> Dict[str, str]:
        """
        Compare different preprocessing approaches on the same text.
        
        Useful for documenting preprocessing impact in research paper.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping method names to preprocessed results
        """
        return {
            'original': text,
            'cleaned_only': self.clean_text(text),
            'with_stopwords': self.preprocess(text, remove_stopwords=False),
            'without_stopwords': self.preprocess(text, remove_stopwords=True),
            'stemmed': self.preprocess(text, stemming=True, lemmatization=False),
            'lemmatized': self.preprocess(text, stemming=False, lemmatization=True),
        }
    
    def batch_preprocess(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Process multiple texts with same configuration.
        
        Args:
            texts: List of texts to process
            **kwargs: Arguments passed to preprocess()
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, **kwargs) for text in texts]
