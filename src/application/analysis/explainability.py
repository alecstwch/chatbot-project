"""
Explainability tools for chatbot analysis.

This module provides tools for explaining model predictions including
LIME for intent classification and attention visualization.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


class IntentExplainer:
    """Explains intent classification predictions using LIME."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize intent explainer.
        
        Args:
            class_names: List of intent class names
        """
        self.explainer = LimeTextExplainer(class_names=class_names)
        self.class_names = class_names
        logger.info(f"IntentExplainer initialized with {len(class_names)} classes")
    
    def explain_prediction(
        self,
        text: str,
        classifier_fn,
        num_features: int = 10,
        num_samples: int = 500
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            text: Input text to explain
            classifier_fn: Function that returns prediction probabilities
            num_features: Number of features to show
            num_samples: Number of perturbed samples for LIME
            
        Returns:
            Dictionary with explanation details
        """
        logger.info(f"Explaining prediction for text: '{text[:50]}...'")
        
        try:
            # Generate explanation
            exp = self.explainer.explain_instance(
                text,
                classifier_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract results
            predicted_class = exp.available_labels()[0]
            predicted_class_name = self.class_names[predicted_class]
            
            # Get feature importance
            feature_importance = exp.as_list(label=predicted_class)
            
            # Get prediction probabilities
            probs = classifier_fn([text])[0]
            
            explanation = {
                'text': text,
                'predicted_class': predicted_class_name,
                'confidence': probs[predicted_class],
                'all_probabilities': {
                    self.class_names[i]: float(probs[i])
                    for i in range(len(self.class_names))
                },
                'feature_importance': feature_importance,
                'top_features': feature_importance[:5]
            }
            
            logger.info(f"Explanation generated: {predicted_class_name} (confidence: {probs[predicted_class]:.3f})")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                'text': text,
                'error': str(e)
            }
    
    def explain_batch(
        self,
        texts: List[str],
        classifier_fn,
        num_features: int = 10
    ) -> List[Dict]:
        """
        Explain multiple predictions.
        
        Args:
            texts: List of input texts
            classifier_fn: Classifier function
            num_features: Number of features per explanation
            
        Returns:
            List of explanation dictionaries
        """
        logger.info(f"Explaining {len(texts)} predictions")
        
        explanations = []
        for text in texts:
            exp = self.explain_prediction(text, classifier_fn, num_features)
            explanations.append(exp)
        
        return explanations
    
    def get_global_feature_importance(
        self,
        texts: List[str],
        classifier_fn,
        num_features: int = 20
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple predictions.
        
        Args:
            texts: List of input texts
            classifier_fn: Classifier function
            num_features: Number of top features
            
        Returns:
            Dictionary of aggregated feature importance
        """
        logger.info(f"Calculating global feature importance from {len(texts)} examples")
        
        all_features = {}
        
        for text in texts:
            exp = self.explain_prediction(text, classifier_fn, num_features)
            if 'feature_importance' in exp:
                for feature, importance in exp['feature_importance']:
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(abs(importance))
        
        # Average importance
        global_importance = {
            feature: np.mean(importances)
            for feature, importances in all_features.items()
        }
        
        # Sort by importance
        sorted_features = dict(
            sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"Global feature importance calculated: {len(sorted_features)} features")
        
        return sorted_features


class AttentionVisualizer:
    """Visualizes attention weights for transformer models."""
    
    @staticmethod
    def extract_attention_weights(
        model,
        tokenizer,
        text: str
    ) -> Tuple[List[str], np.ndarray]:
        """
        Extract attention weights from a transformer model.
        
        Args:
            model: Transformer model
            tokenizer: Model tokenizer
            text: Input text
            
        Returns:
            Tuple of (tokens, attention_weights)
        """
        import torch
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Extract attention from last layer
        attention = outputs.attentions[-1]  # Last layer
        attention = attention[0].mean(dim=0)  # Average over heads
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Convert to numpy
        attention_weights = attention.cpu().numpy()
        
        logger.info(f"Extracted attention weights: {attention_weights.shape}")
        
        return tokens, attention_weights
    
    @staticmethod
    def get_top_attended_tokens(
        tokens: List[str],
        attention_weights: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get tokens with highest attention.
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weight matrix
            top_k: Number of top tokens
            
        Returns:
            List of (token, attention_score) tuples
        """
        # Average attention for each token
        token_importance = attention_weights.mean(axis=0)
        
        # Get top-k
        top_indices = np.argsort(token_importance)[-top_k:][::-1]
        
        top_tokens = [
            (tokens[i], float(token_importance[i]))
            for i in top_indices
        ]
        
        logger.info(f"Top {top_k} attended tokens extracted")
        
        return top_tokens
    
    @staticmethod
    def format_attention_matrix(
        tokens: List[str],
        attention_weights: np.ndarray,
        precision: int = 3
    ) -> str:
        """
        Format attention matrix as readable string.
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weight matrix
            precision: Decimal precision
            
        Returns:
            Formatted attention matrix string
        """
        lines = []
        lines.append("Attention Matrix:")
        lines.append("-" * 70)
        
        # Header
        header = "Token".ljust(15) + " | " + " ".join(
            [t[:8].ljust(8) for t in tokens[:10]]
        )
        lines.append(header)
        lines.append("-" * 70)
        
        # Rows
        for i, token in enumerate(tokens[:10]):
            row_values = " ".join([
                f"{attention_weights[i, j]:.{precision}f}".ljust(8)
                for j in range(min(10, len(tokens)))
            ])
            row = token[:15].ljust(15) + " | " + row_values
            lines.append(row)
        
        return "\n".join(lines)


class ModelComparison:
    """Tools for comparing different chatbot models."""
    
    @staticmethod
    def compare_metrics(
        model_results: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Create a comparison table of model metrics.
        
        Args:
            model_results: Dictionary mapping model names to metrics
            
        Returns:
            Formatted comparison table
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL COMPARISON")
        lines.append("=" * 80)
        lines.append("")
        
        # Get all metric names
        all_metrics = set()
        for metrics in model_results.values():
            all_metrics.update(metrics.keys())
        
        # Sort metrics
        sorted_metrics = sorted(all_metrics)
        
        # Header
        header = "Metric".ljust(25) + " | " + " | ".join([
            name[:15].ljust(15) for name in model_results.keys()
        ])
        lines.append(header)
        lines.append("-" * 80)
        
        # Rows
        for metric in sorted_metrics:
            row = metric[:25].ljust(25) + " | "
            values = []
            for model_name in model_results.keys():
                value = model_results[model_name].get(metric, 0.0)
                if isinstance(value, float):
                    values.append(f"{value:.4f}".ljust(15))
                else:
                    values.append(str(value)[:15].ljust(15))
            row += " | ".join(values)
            lines.append(row)
        
        lines.append("")
        lines.append("=" * 80)
        
        logger.info(f"Model comparison table created for {len(model_results)} models")
        
        return "\n".join(lines)
    
    @staticmethod
    def rank_models(
        model_results: Dict[str, Dict[str, float]],
        primary_metric: str = 'f1_macro'
    ) -> List[Tuple[str, float]]:
        """
        Rank models by a primary metric.
        
        Args:
            model_results: Dictionary mapping model names to metrics
            primary_metric: Metric to use for ranking
            
        Returns:
            List of (model_name, score) tuples in descending order
        """
        rankings = [
            (model_name, metrics.get(primary_metric, 0.0))
            for model_name, metrics in model_results.items()
        ]
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Models ranked by {primary_metric}")
        
        return rankings
    
    @staticmethod
    def find_best_model(
        model_results: Dict[str, Dict[str, float]],
        metrics_to_optimize: List[str] = ['accuracy', 'f1_macro', 'bleu']
    ) -> str:
        """
        Find best model across multiple metrics.
        
        Args:
            model_results: Dictionary mapping model names to metrics
            metrics_to_optimize: List of metrics to consider
            
        Returns:
            Name of best model
        """
        scores = {}
        
        for model_name, metrics in model_results.items():
            # Calculate average rank across metrics
            ranks = []
            for metric in metrics_to_optimize:
                if metric in metrics:
                    # Get all values for this metric
                    values = [m.get(metric, 0.0) for m in model_results.values()]
                    # Calculate rank (higher is better)
                    rank = sorted(values, reverse=True).index(metrics[metric])
                    ranks.append(rank)
            
            # Average rank (lower is better)
            scores[model_name] = np.mean(ranks) if ranks else float('inf')
        
        best_model = min(scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Best model: {best_model}")
        
        return best_model
