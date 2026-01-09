"""
Model comparison and benchmarking utilities.

This module provides tools for comparing different chatbot models,
generating comparison tables, and visualizing performance differences.
"""

from typing import Dict, List, Optional
import json
import csv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmarking tool for chatbot models."""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        Add evaluation results for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric values
            metadata: Optional metadata (hyperparameters, etc.)
        """
        self.results[model_name] = {
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        logger.info(f"Added results for model: {model_name}")
    
    def compare_models(self, metrics_to_compare: Optional[List[str]] = None) -> str:
        """
        Generate comparison table.
        
        Args:
            metrics_to_compare: List of specific metrics to compare
            
        Returns:
            Formatted comparison table
        """
        if not self.results:
            return "No model results available for comparison"
        
        # Get all metrics if not specified
        if metrics_to_compare is None:
            all_metrics = set()
            for result in self.results.values():
                all_metrics.update(result['metrics'].keys())
            metrics_to_compare = sorted(all_metrics)
        
        lines = []
        lines.append("=" * 100)
        lines.append("MODEL BENCHMARK COMPARISON")
        lines.append("=" * 100)
        lines.append("")
        
        # Header
        header = "Metric".ljust(30)
        for model_name in self.results.keys():
            header += f" | {model_name[:18].ljust(18)}"
        lines.append(header)
        lines.append("-" * 100)
        
        # Metrics rows
        for metric in metrics_to_compare:
            row = metric[:30].ljust(30)
            values = []
            
            for model_name, result in self.results.items():
                value = result['metrics'].get(metric)
                if value is not None:
                    if isinstance(value, float):
                        values.append((f"{value:.4f}", value))
                    else:
                        values.append((str(value)[:18], value))
                else:
                    values.append(("N/A", None))
            
            # Highlight best value
            numeric_values = [v[1] for v in values if v[1] is not None]
            best_value = max(numeric_values) if numeric_values else None
            
            for val_str, val_num in values:
                if val_num == best_value and best_value is not None:
                    row += f" | *{val_str.ljust(17)}"
                else:
                    row += f" | {val_str.ljust(18)}"
            
            lines.append(row)
        
        lines.append("")
        lines.append("* = Best performance")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def get_rankings(self, metric: str) -> List[tuple]:
        """
        Get model rankings for a specific metric.
        
        Args:
            metric: Metric name
            
        Returns:
            List of (model_name, score, rank) tuples
        """
        scores = []
        
        for model_name, result in self.results.items():
            score = result['metrics'].get(metric)
            if score is not None:
                scores.append((model_name, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add rankings
        rankings = [
            (name, score, rank + 1)
            for rank, (name, score) in enumerate(scores)
        ]
        
        return rankings
    
    def export_to_json(self, filepath: str):
        """
        Export results to JSON file.
        
        Args:
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")
    
    def export_to_csv(self, filepath: str):
        """
        Export results to CSV file.
        
        Args:
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all metrics
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result['metrics'].keys())
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Model'] + sorted(all_metrics)
            writer.writerow(header)
            
            # Data rows
            for model_name, result in self.results.items():
                row = [model_name]
                for metric in sorted(all_metrics):
                    value = result['metrics'].get(metric, '')
                    row.append(value)
                writer.writerow(row)
        
        logger.info(f"Results exported to {filepath}")
    
    def generate_summary(self) -> str:
        """
        Generate summary statistics.
        
        Returns:
            Formatted summary
        """
        lines = []
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Total models evaluated: {len(self.results)}")
        lines.append("")
        
        # Collect all metrics
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result['metrics'].keys())
        
        # Best performers per metric
        lines.append("Best Performers by Metric:")
        lines.append("-" * 70)
        
        for metric in sorted(all_metrics):
            rankings = self.get_rankings(metric)
            if rankings:
                best_model, best_score, _ = rankings[0]
                lines.append(f"  {metric:30s}: {best_model:20s} ({best_score:.4f})")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class CrossValidationAnalyzer:
    """Analyzes cross-validation results."""
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.fold_results = []
    
    def add_fold_result(self, fold_idx: int, metrics: Dict[str, float]):
        """
        Add results from a single fold.
        
        Args:
            fold_idx: Fold index
            metrics: Metrics for this fold
        """
        self.fold_results.append({
            'fold': fold_idx,
            'metrics': metrics
        })
        
        logger.info(f"Added results for fold {fold_idx}")
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate mean and std for each metric across folds.
        
        Returns:
            Dictionary of statistics
        """
        import numpy as np
        
        if not self.fold_results:
            return {}
        
        # Collect metrics
        all_metrics = set()
        for result in self.fold_results:
            all_metrics.update(result['metrics'].keys())
        
        # Calculate statistics
        stats = {}
        
        for metric in all_metrics:
            values = [
                result['metrics'].get(metric)
                for result in self.fold_results
                if metric in result['metrics']
            ]
            
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        logger.info(f"Calculated statistics for {len(stats)} metrics")
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generate cross-validation report.
        
        Returns:
            Formatted report
        """
        stats = self.calculate_statistics()
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"CROSS-VALIDATION REPORT ({self.n_folds}-fold)")
        lines.append("=" * 70)
        lines.append("")
        
        if not stats:
            lines.append("No results available")
        else:
            lines.append("Metric".ljust(30) + "Mean ± Std".ljust(20) + "Range")
            lines.append("-" * 70)
            
            for metric, stat in sorted(stats.items()):
                mean_std = f"{stat['mean']:.4f} ± {stat['std']:.4f}"
                range_str = f"[{stat['min']:.4f}, {stat['max']:.4f}]"
                lines.append(
                    f"{metric[:30].ljust(30)}{mean_std.ljust(20)}{range_str}"
                )
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
