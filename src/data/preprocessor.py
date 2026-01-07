"""
Text Preprocessing Module
Implements various NLP preprocessing techniques for chatbot training
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Dict, Optional
import pandas as pd

# Download required NLTK data (run once)
def download_nltk_data():
    """Download necessary NLTK data"""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_data()


class TextPreprocessor:
    """
    Text preprocessing pipeline for chatbot data
    Demonstrates multiple NLP techniques for the project
    """
    
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        - Remove URLs
        - Remove HTML tags
        - Remove special characters
        - Remove extra whitespace
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
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
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def lowercase(self, tokens: List[str]) -> List[str]:
        """Convert all tokens to lowercase"""
        return [token.lower() for token in tokens]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter Stemmer to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply WordNet Lemmatizer to tokens"""
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
        Full preprocessing pipeline
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_stopwords: Remove stopwords
            stemming: Apply Porter Stemmer
            lemmatization: Apply Lemmatization
            
        Returns:
            Preprocessed text as string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Lowercase
        if lowercase:
            tokens = self.lowercase(tokens)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Stemming (mutually exclusive with lemmatization)
        if stemming and not lemmatization:
            tokens = self.stem_tokens(tokens)
        
        # Lemmatization
        if lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        return ' '.join(tokens)
    
    def compare_preprocessing_methods(self, text: str) -> Dict[str, str]:
        """
        Compare different preprocessing methods for analysis
        Useful for documenting in the paper
        
        Returns:
            Dictionary with different preprocessing results
        """
        results = {
            'original': text,
            'cleaned_only': self.clean_text(text),
            'with_stopwords': self.preprocess(text, remove_stopwords=False),
            'without_stopwords': self.preprocess(text, remove_stopwords=True),
            'stemmed': self.preprocess(text, stemming=True, lemmatization=False),
            'lemmatized': self.preprocess(text, stemming=False, lemmatization=True),
        }
        
        return results
    
    def batch_preprocess(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Process multiple texts
        
        Args:
            texts: List of texts to process
            **kwargs: Arguments passed to preprocess()
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, **kwargs) for text in texts]


def preprocess_dataframe(
    df: pd.DataFrame, 
    text_column: str,
    **preprocess_kwargs
) -> pd.DataFrame:
    """
    Preprocess text column in a DataFrame
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text
        **preprocess_kwargs: Arguments for preprocessing
        
    Returns:
        DataFrame with additional preprocessed column
    """
    preprocessor = TextPreprocessor()
    df['preprocessed_text'] = preprocessor.batch_preprocess(
        df[text_column].tolist(),
        **preprocess_kwargs
    )
    return df


# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    sample_text = """
    Hello! I'm feeling very anxious today. 
    I can't stop worrying about everything. 
    Check this link: https://example.com
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Compare different methods
    results = preprocessor.compare_preprocessing_methods(sample_text)
    
    for method, result in results.items():
        print(f"{method}:")
        print(f"  {result}")
        print()
