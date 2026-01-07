"""
Unit tests for EDA service.

Tests the EDAService class for exploratory data analysis including
statistics calculation and visualization generation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from src.application.analysis.eda_service import EDAService


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for visualizations."""
    return tmp_path / "output"


@pytest.fixture
def eda_service(temp_output_dir):
    """Create EDAService instance with temp directory."""
    return EDAService(output_dir=temp_output_dir)


@pytest.fixture
def sample_texts():
    """Create sample text data for testing."""
    return [
        "Hello world",
        "How are you doing today?",
        "I am feeling great",
        "This is a test message",
        "",  # Empty text
        "Short",
        "A much longer message with many more words than the others to test edge cases"
    ]


@pytest.fixture
def sample_dataset(sample_texts):
    """Create sample HuggingFace dataset."""
    return Dataset.from_dict({
        'text': sample_texts,
        'label': [0, 1, 0, 1, 0, 1, 0]
    })


class TestEDAServiceInitialization:
    """Test EDAService initialization."""
    
    def test_init_creates_output_directory(self, temp_output_dir):
        """Test that initialization creates output directory."""
        service = EDAService(output_dir=temp_output_dir)
        
        assert service.output_dir == temp_output_dir
        assert service.output_dir.exists()
    
    def test_init_with_none_uses_current_dir(self):
        """Test initialization with None uses current directory."""
        service = EDAService(output_dir=None)
        assert service.output_dir == Path.cwd()
    
    def test_init_with_existing_directory(self, temp_output_dir):
        """Test initialization with pre-existing directory."""
        temp_output_dir.mkdir(parents=True)
        service = EDAService(output_dir=temp_output_dir)
        assert service.output_dir.exists()


class TestStatisticsCalculation:
    """Test statistical analysis methods."""
    
    def test_calculate_statistics_basic(self, eda_service, sample_texts):
        """Test basic statistics calculation."""
        stats = eda_service.calculate_statistics(sample_texts)
        
        assert stats['total_samples'] == 7
        assert stats['empty_texts'] == 1
        assert stats['min_length'] == 0  # Empty text
        assert stats['max_length'] > stats['min_length']
        assert stats['avg_length'] > 0
        assert stats['avg_word_count'] > 0
    
    def test_calculate_statistics_empty_list(self, eda_service):
        """Test statistics with empty list."""
        stats = eda_service.calculate_statistics([])
        
        assert stats['total_samples'] == 0
        assert stats['avg_length'] == 0
        assert stats['empty_texts'] == 0
    
    def test_calculate_statistics_single_text(self, eda_service):
        """Test statistics with single text."""
        stats = eda_service.calculate_statistics(["Hello world"])
        
        assert stats['total_samples'] == 1
        assert stats['avg_length'] == 11
        assert stats['avg_word_count'] == 2
    
    def test_calculate_statistics_word_counts(self, eda_service, sample_texts):
        """Test word count calculations."""
        stats = eda_service.calculate_statistics(sample_texts)
        
        assert 'min_word_count' in stats
        assert 'max_word_count' in stats
        assert 'avg_word_count' in stats
        assert stats['min_word_count'] <= stats['avg_word_count']
        assert stats['avg_word_count'] <= stats['max_word_count']


class TestVocabularyAnalysis:
    """Test vocabulary statistics methods."""
    
    def test_calculate_vocabulary_stats_basic(self, eda_service, sample_texts):
        """Test basic vocabulary statistics."""
        stats = eda_service.calculate_vocabulary_stats(sample_texts)
        
        assert 'total_words' in stats
        assert 'unique_words' in stats
        assert 'vocabulary_size' in stats
        assert 'most_common_words' in stats
        assert stats['total_words'] > 0
        assert stats['unique_words'] > 0
    
    def test_calculate_vocabulary_stats_case_insensitive(self, eda_service):
        """Test that vocabulary calculation is case-insensitive."""
        texts = ["Hello World", "hello world", "HELLO"]
        stats = eda_service.calculate_vocabulary_stats(texts)
        
        # "hello" and "world" should be counted together regardless of case
        assert stats['unique_words'] == 2  # hello, world
    
    def test_calculate_vocabulary_most_common(self, eda_service):
        """Test most common words extraction."""
        texts = ["the cat", "the dog", "the bird", "a cat"]
        stats = eda_service.calculate_vocabulary_stats(texts)
        
        most_common = stats['most_common_words']
        assert len(most_common) > 0
        assert most_common[0][0] == 'the'  # Most common word
        assert most_common[0][1] == 3  # Appears 3 times
    
    def test_calculate_vocabulary_empty_texts(self, eda_service):
        """Test vocabulary stats with empty texts."""
        stats = eda_service.calculate_vocabulary_stats([])
        
        assert stats['total_words'] == 0
        assert stats['unique_words'] == 0


class TestDatasetAnalysis:
    """Test dataset analysis methods."""
    
    def test_analyze_dataset(self, eda_service, sample_dataset):
        """Test full dataset analysis."""
        stats = eda_service.analyze_dataset(sample_dataset)
        
        assert 'total_samples' in stats
        assert stats['total_samples'] == len(sample_dataset)
    
    def test_analyze_dataset_custom_column(self, eda_service):
        """Test analysis with custom text column name."""
        dataset = Dataset.from_dict({
            'content': ["Hello", "World"],
            'label': [0, 1]
        })
        
        stats = eda_service.analyze_dataset(dataset, text_column='content')
        assert stats['total_samples'] == 2


class TestVisualizationGeneration:
    """Test visualization methods."""
    
    def test_visualize_length_distribution_save(
        self, 
        eda_service, 
        sample_texts
    ):
        """Test saving length distribution plot."""
        save_path = eda_service.visualize_length_distribution(
            sample_texts,
            save_name="test_length.png"
        )
        
        assert save_path is not None
        assert save_path.exists()
        assert save_path.name == "test_length.png"
    
    def test_visualize_length_distribution_no_save(
        self, 
        eda_service, 
        sample_texts
    ):
        """Test not saving plot returns None."""
        with patch('matplotlib.pyplot.show'):
            result = eda_service.visualize_length_distribution(
                sample_texts,
                save_name=None
            )
        assert result is None
    
    def test_visualize_word_count_distribution_save(
        self, 
        eda_service, 
        sample_texts
    ):
        """Test saving word count distribution plot."""
        save_path = eda_service.visualize_word_count_distribution(
            sample_texts,
            save_name="test_words.png"
        )
        
        assert save_path is not None
        assert save_path.exists()
        assert save_path.name == "test_words.png"
    
    def test_visualize_custom_title(self, eda_service, sample_texts):
        """Test visualization with custom title."""
        save_path = eda_service.visualize_length_distribution(
            sample_texts,
            title="Custom Title",
            save_name="custom.png"
        )
        
        assert save_path.exists()


class TestComparisonTable:
    """Test dataset comparison methods."""
    
    def test_create_comparison_table(self, eda_service, sample_dataset):
        """Test creating comparison table for multiple datasets."""
        datasets = {
            'Dataset1': sample_dataset,
            'Dataset2': sample_dataset
        }
        
        df = eda_service.create_comparison_table(datasets)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Dataset' in df.columns
        assert 'Samples' in df.columns
        assert 'Avg Length' in df.columns
    
    def test_create_comparison_table_custom_column(self, eda_service):
        """Test comparison with custom text column."""
        dataset = Dataset.from_dict({
            'content': ["Hello", "World"],
            'label': [0, 1]
        })
        
        datasets = {'Test': dataset}
        df = eda_service.create_comparison_table(
            datasets, 
            text_column='content'
        )
        
        assert len(df) == 1
        assert df.iloc[0]['Samples'] == 2


class TestFullReport:
    """Test comprehensive report generation."""
    
    def test_generate_full_report(self, eda_service, sample_dataset):
        """Test generating complete EDA report."""
        report = eda_service.generate_full_report(
            sample_dataset,
            dataset_name="TestDataset"
        )
        
        assert 'dataset_name' in report
        assert 'basic_stats' in report
        assert 'vocabulary_stats' in report
        assert 'visualizations' in report
        assert report['dataset_name'] == "TestDataset"
    
    def test_generate_full_report_creates_visualizations(
        self, 
        eda_service, 
        sample_dataset
    ):
        """Test that full report creates visualization files."""
        report = eda_service.generate_full_report(
            sample_dataset,
            dataset_name="Test"
        )
        
        viz = report['visualizations']
        assert viz['length_distribution'] is not None
        assert viz['word_count_distribution'] is not None
        
        # Check files exist
        length_path = Path(viz['length_distribution'])
        word_path = Path(viz['word_count_distribution'])
        assert length_path.exists()
        assert word_path.exists()
    
    def test_generate_full_report_statistics_included(
        self, 
        eda_service, 
        sample_dataset
    ):
        """Test that full report includes all statistics."""
        report = eda_service.generate_full_report(
            sample_dataset,
            dataset_name="Test"
        )
        
        basic = report['basic_stats']
        assert 'total_samples' in basic
        assert 'avg_length' in basic
        
        vocab = report['vocabulary_stats']
        assert 'total_words' in vocab
        assert 'unique_words' in vocab


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset_analysis(self, eda_service):
        """Test analysis with empty dataset."""
        empty_dataset = Dataset.from_dict({'text': []})
        stats = eda_service.analyze_dataset(empty_dataset)
        
        assert stats['total_samples'] == 0
    
    def test_all_empty_texts(self, eda_service):
        """Test with all empty strings."""
        texts = ["", "", ""]
        stats = eda_service.calculate_statistics(texts)
        
        assert stats['total_samples'] == 3
        assert stats['empty_texts'] == 3
        assert stats['avg_length'] == 0
    
    def test_single_word_texts(self, eda_service):
        """Test with single-word texts."""
        texts = ["hello", "world", "test"]
        stats = eda_service.calculate_statistics(texts)
        
        assert stats['avg_word_count'] == 1
        assert stats['min_word_count'] == 1
        assert stats['max_word_count'] == 1
