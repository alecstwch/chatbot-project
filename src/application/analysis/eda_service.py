"""
Exploratory Data Analysis (EDA) service.

Provides comprehensive analysis of text datasets including:
- Statistical summaries
- Text length distributions
- Vocabulary analysis
- Data quality checks
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class EDAService:
    """
    Service for performing exploratory data analysis on text datasets.
    
    Follows application layer principles:
    - Orchestrates domain and infrastructure services
    - Generates insights from raw data
    - Produces visualizations for analysis
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize EDA service.
        
        Args:
            output_dir: Directory to save visualizations (default: current dir)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def analyze_dataset(
        self, 
        dataset: Dataset, 
        text_column: str = 'text'
    ) -> Dict[str, any]:
        """
        Perform comprehensive analysis on a dataset.
        
        Args:
            dataset: Dataset to analyze
            text_column: Name of the text column
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing dataset with {len(dataset)} samples")
        
        # Extract text data
        texts = dataset[text_column]
        
        # Calculate statistics
        stats = self.calculate_statistics(texts)
        
        logger.info(f"Analysis complete: {stats['total_samples']} samples analyzed")
        return stats
    
    def calculate_statistics(self, texts: List[str]) -> Dict[str, any]:
        """
        Calculate statistical summaries of text data.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with statistics
        """
        # Text lengths
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Calculate statistics
        stats = {
            'total_samples': len(texts),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_word_count': min(word_counts) if word_counts else 0,
            'max_word_count': max(word_counts) if word_counts else 0,
            'empty_texts': sum(1 for text in texts if not text.strip()),
        }
        
        return stats
    
    def calculate_vocabulary_stats(
        self, 
        texts: List[str]
    ) -> Dict[str, any]:
        """
        Calculate vocabulary statistics.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with vocabulary statistics
        """
        # Collect all words
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        # Calculate statistics
        word_freq = Counter(all_words)
        
        stats = {
            'total_words': len(all_words),
            'unique_words': len(word_freq),
            'vocabulary_size': len(word_freq),
            'most_common_words': word_freq.most_common(10),
        }
        
        return stats
    
    def visualize_length_distribution(
        self,
        texts: List[str],
        title: str = "Text Length Distribution",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create histogram of text length distribution.
        
        Args:
            texts: List of text strings
            title: Plot title
            save_name: Filename to save plot (if None, only displays)
            
        Returns:
            Path to saved file if save_name provided, None otherwise
        """
        lengths = [len(text) for text in texts]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def visualize_word_count_distribution(
        self,
        texts: List[str],
        title: str = "Word Count Distribution",
        save_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create histogram of word count distribution.
        
        Args:
            texts: List of text strings
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Path to saved file if save_name provided
        """
        word_counts = [len(text.split()) for text in texts]
        
        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=50, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def create_comparison_table(
        self,
        datasets: Dict[str, Dataset],
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple datasets.
        
        Args:
            datasets: Dictionary of dataset_name: dataset
            text_column: Name of text column
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for name, dataset in datasets.items():
            texts = dataset[text_column]
            stats = self.calculate_statistics(texts)
            
            comparison_data.append({
                'Dataset': name,
                'Samples': stats['total_samples'],
                'Avg Length': f"{stats['avg_length']:.1f}",
                'Avg Words': f"{stats['avg_word_count']:.1f}",
                'Min Words': stats['min_word_count'],
                'Max Words': stats['max_word_count'],
                'Empty': stats['empty_texts']
            })
        
        df = pd.DataFrame(comparison_data)
        logger.info(f"Created comparison table for {len(datasets)} datasets")
        return df
    
    def generate_full_report(
        self,
        dataset: Dataset,
        dataset_name: str,
        text_column: str = 'text'
    ) -> Dict[str, any]:
        """
        Generate comprehensive EDA report with statistics and visualizations.
        
        Args:
            dataset: Dataset to analyze
            dataset_name: Name for labeling
            text_column: Name of text column
            
        Returns:
            Dictionary with all analysis results and file paths
        """
        logger.info(f"Generating full EDA report for {dataset_name}")
        
        texts = dataset[text_column]
        
        # Calculate statistics
        basic_stats = self.calculate_statistics(texts)
        vocab_stats = self.calculate_vocabulary_stats(texts)
        
        # Create visualizations
        length_plot = self.visualize_length_distribution(
            texts,
            title=f"{dataset_name} - Text Length Distribution",
            save_name=f"{dataset_name.lower()}_length_dist.png"
        )
        
        word_count_plot = self.visualize_word_count_distribution(
            texts,
            title=f"{dataset_name} - Word Count Distribution",
            save_name=f"{dataset_name.lower()}_word_count_dist.png"
        )
        
        report = {
            'dataset_name': dataset_name,
            'basic_stats': basic_stats,
            'vocabulary_stats': vocab_stats,
            'visualizations': {
                'length_distribution': str(length_plot) if length_plot else None,
                'word_count_distribution': str(word_count_plot) if word_count_plot else None,
            }
        }
        
        logger.info(f"EDA report complete for {dataset_name}")
        return report
