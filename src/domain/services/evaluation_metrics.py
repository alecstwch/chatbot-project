"""
Evaluation metrics for chatbot performance assessment.

This module provides comprehensive evaluation metrics for both
intent classification and response generation tasks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import logging

logger = logging.getLogger(__name__)


class IntentClassificationMetrics:
    """Metrics for intent classification evaluation."""
    
    @staticmethod
    def calculate_metrics(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        logger.info(f"Classification metrics calculated: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def get_classification_report(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> str:
        """Get detailed classification report."""
        return classification_report(y_true, y_pred, labels=labels, zero_division=0)


class ResponseGenerationMetrics:
    """Metrics for response generation evaluation."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
    
    def calculate_bleu(
        self,
        references: List[str],
        candidates: List[str],
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores (1-4).
        
        Args:
            references: Reference responses
            candidates: Generated responses
            max_n: Maximum n-gram (default: 4)
            
        Returns:
            Dictionary of BLEU scores
        """
        # Tokenize
        ref_tokens = [[ref.split()] for ref in references]
        cand_tokens = [cand.split() for cand in candidates]
        
        scores = {}
        
        # Individual BLEU scores
        for n in range(1, max_n + 1):
            weights = tuple([1/n if i < n else 0 for i in range(4)])
            score = corpus_bleu(
                ref_tokens,
                cand_tokens,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            scores[f'bleu_{n}'] = score
        
        # Overall BLEU-4
        scores['bleu'] = corpus_bleu(
            ref_tokens,
            cand_tokens,
            smoothing_function=self.smoothing.method1
        )
        
        logger.info(f"BLEU scores calculated: BLEU-4={scores['bleu']:.4f}")
        
        return scores
    
    def calculate_rouge(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            references: Reference responses
            candidates: Generated responses
            
        Returns:
            Dictionary of ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, cand in zip(references, candidates):
            scores = self.rouge_scorer.score(ref, cand)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        results = {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
        
        logger.info(f"ROUGE scores calculated: ROUGE-L={results['rougeL']:.4f}")
        
        return results
    
    def calculate_meteor(
        self,
        references: List[str],
        candidates: List[str]
    ) -> float:
        """
        Calculate METEOR score.
        
        Args:
            references: Reference responses
            candidates: Generated responses
            
        Returns:
            Average METEOR score
        """
        scores = []
        
        for ref, cand in zip(references, candidates):
            score = meteor_score([ref.split()], cand.split())
            scores.append(score)
        
        avg_score = np.mean(scores)
        logger.info(f"METEOR score calculated: {avg_score:.4f}")
        
        return avg_score
    
    def calculate_all_metrics(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all generation metrics.
        
        Args:
            references: Reference responses
            candidates: Generated responses
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # BLEU scores
        bleu_scores = self.calculate_bleu(references, candidates)
        metrics.update(bleu_scores)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(references, candidates)
        metrics.update(rouge_scores)
        
        # METEOR score
        meteor = self.calculate_meteor(references, candidates)
        metrics['meteor'] = meteor
        
        logger.info(f"All generation metrics calculated: {len(metrics)} metrics")
        
        return metrics


class DialogueMetrics:
    """Metrics for end-to-end dialogue evaluation."""
    
    @staticmethod
    def calculate_response_diversity(responses: List[str]) -> float:
        """
        Calculate response diversity (unique n-grams ratio).
        
        Args:
            responses: List of generated responses
            
        Returns:
            Diversity score (0-1)
        """
        all_tokens = []
        for response in responses:
            all_tokens.extend(response.split())
        
        if not all_tokens:
            return 0.0
        
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        
        diversity = unique_tokens / total_tokens
        logger.info(f"Response diversity: {diversity:.4f}")
        
        return diversity
    
    @staticmethod
    def calculate_average_length(responses: List[str]) -> Dict[str, float]:
        """
        Calculate average response length statistics.
        
        Args:
            responses: List of responses
            
        Returns:
            Dictionary of length statistics
        """
        lengths = [len(response.split()) for response in responses]
        
        stats = {
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths)
        }
        
        logger.info(f"Average response length: {stats['mean_length']:.2f} words")
        
        return stats


def evaluate_chatbot_performance(
    intent_true: List[str],
    intent_pred: List[str],
    references: List[str],
    candidates: List[str],
    intent_labels: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Comprehensive chatbot evaluation.
    
    Args:
        intent_true: True intents
        intent_pred: Predicted intents
        references: Reference responses
        candidates: Generated responses
        intent_labels: List of intent labels
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Starting comprehensive chatbot evaluation")
    
    results = {}
    
    # Intent classification metrics
    intent_metrics = IntentClassificationMetrics.calculate_metrics(
        intent_true, intent_pred, intent_labels
    )
    results['intent_classification'] = intent_metrics
    
    # Response generation metrics
    gen_metrics_calc = ResponseGenerationMetrics()
    gen_metrics = gen_metrics_calc.calculate_all_metrics(references, candidates)
    results['response_generation'] = gen_metrics
    
    # Dialogue metrics
    diversity = DialogueMetrics.calculate_response_diversity(candidates)
    length_stats = DialogueMetrics.calculate_average_length(candidates)
    
    results['dialogue_quality'] = {
        'diversity': diversity,
        **length_stats
    }
    
    logger.info("Comprehensive evaluation completed")
    
    return results
