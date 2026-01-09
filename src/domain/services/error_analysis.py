"""
Error analysis tools for chatbot evaluation.

This module provides utilities for analyzing chatbot errors,
categorizing failure modes, and generating insights.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Represents a single error case."""
    input_text: str
    expected: str
    predicted: str
    error_type: str
    confidence: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class ErrorAnalyzer:
    """Analyzes chatbot errors and failure patterns."""
    
    def __init__(self):
        self.errors: List[ErrorCase] = []
        self.error_categories = {
            'intent_misclassification': [],
            'inappropriate_response': [],
            'repetitive_response': [],
            'out_of_vocabulary': [],
            'context_loss': [],
            'length_mismatch': [],
            'generic_response': [],
            'other': []
        }
    
    def add_error(
        self,
        input_text: str,
        expected: str,
        predicted: str,
        error_type: str = 'other',
        confidence: Optional[float] = None,
        **metadata
    ):
        """
        Add an error case for analysis.
        
        Args:
            input_text: User input
            expected: Expected output
            predicted: Predicted output
            error_type: Category of error
            confidence: Prediction confidence
            **metadata: Additional metadata
        """
        error = ErrorCase(
            input_text=input_text,
            expected=expected,
            predicted=predicted,
            error_type=error_type,
            confidence=confidence,
            metadata=metadata
        )
        
        self.errors.append(error)
        
        if error_type in self.error_categories:
            self.error_categories[error_type].append(error)
        else:
            self.error_categories['other'].append(error)
        
        logger.debug(f"Added error case: {error_type}")
    
    def categorize_error(
        self,
        input_text: str,
        expected: str,
        predicted: str,
        intent_expected: Optional[str] = None,
        intent_predicted: Optional[str] = None
    ) -> str:
        """
        Automatically categorize an error.
        
        Args:
            input_text: User input
            expected: Expected output
            predicted: Predicted output
            intent_expected: Expected intent
            intent_predicted: Predicted intent
            
        Returns:
            Error category
        """
        # Intent misclassification
        if intent_expected and intent_predicted and intent_expected != intent_predicted:
            return 'intent_misclassification'
        
        # Repetitive response (same as input)
        if predicted.lower() in input_text.lower():
            return 'repetitive_response'
        
        # Generic response (very short)
        if len(predicted.split()) <= 3:
            return 'generic_response'
        
        # Length mismatch (response too short or too long)
        expected_len = len(expected.split())
        predicted_len = len(predicted.split())
        if abs(expected_len - predicted_len) > max(expected_len, predicted_len) * 0.5:
            return 'length_mismatch'
        
        # Default
        return 'inappropriate_response'
    
    def get_error_distribution(self) -> Dict[str, int]:
        """
        Get distribution of error types.
        
        Returns:
            Dictionary mapping error types to counts
        """
        distribution = {
            error_type: len(cases)
            for error_type, cases in self.error_categories.items()
            if len(cases) > 0
        }
        
        logger.info(f"Error distribution: {len(distribution)} error types found")
        
        return distribution
    
    def get_most_common_errors(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get most common error types.
        
        Args:
            top_n: Number of top errors to return
            
        Returns:
            List of (error_type, count) tuples
        """
        distribution = self.get_error_distribution()
        sorted_errors = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_errors[:top_n]
    
    def analyze_confidence_distribution(self) -> Dict[str, List[float]]:
        """
        Analyze confidence scores by error type.
        
        Returns:
            Dictionary mapping error types to confidence scores
        """
        confidence_by_type = defaultdict(list)
        
        for error in self.errors:
            if error.confidence is not None:
                confidence_by_type[error.error_type].append(error.confidence)
        
        return dict(confidence_by_type)
    
    def get_low_confidence_errors(self, threshold: float = 0.5) -> List[ErrorCase]:
        """
        Get errors with low confidence scores.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            List of low-confidence errors
        """
        low_conf_errors = [
            error for error in self.errors
            if error.confidence is not None and error.confidence < threshold
        ]
        
        logger.info(f"Found {len(low_conf_errors)} low-confidence errors (< {threshold})")
        
        return low_conf_errors
    
    def generate_error_report(self) -> str:
        """
        Generate a comprehensive error analysis report.
        
        Returns:
            Formatted error report
        """
        report = []
        report.append("=" * 70)
        report.append("CHATBOT ERROR ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Total errors
        report.append(f"Total errors: {len(self.errors)}")
        report.append("")
        
        # Error distribution
        report.append("Error Distribution:")
        report.append("-" * 50)
        distribution = self.get_error_distribution()
        for error_type, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.errors)) * 100 if self.errors else 0
            report.append(f"  {error_type:30s}: {count:4d} ({percentage:5.2f}%)")
        report.append("")
        
        # Confidence analysis
        confidence_dist = self.analyze_confidence_distribution()
        if confidence_dist:
            report.append("Confidence Analysis:")
            report.append("-" * 50)
            for error_type, confidences in confidence_dist.items():
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    report.append(f"  {error_type:30s}: avg={avg_conf:.3f}")
            report.append("")
        
        # Sample errors from each category
        report.append("Sample Errors by Category:")
        report.append("-" * 50)
        for error_type, cases in self.error_categories.items():
            if cases:
                report.append(f"\n{error_type.upper()} ({len(cases)} cases):")
                for i, error in enumerate(cases[:3], 1):  # Show first 3
                    report.append(f"  Example {i}:")
                    report.append(f"    Input: {error.input_text[:60]}...")
                    report.append(f"    Expected: {error.expected[:60]}...")
                    report.append(f"    Got: {error.predicted[:60]}...")
                    if error.confidence:
                        report.append(f"    Confidence: {error.confidence:.3f}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def export_errors_to_dict(self) -> List[Dict]:
        """
        Export all errors as list of dictionaries.
        
        Returns:
            List of error dictionaries
        """
        return [
            {
                'input': error.input_text,
                'expected': error.expected,
                'predicted': error.predicted,
                'error_type': error.error_type,
                'confidence': error.confidence,
                **error.metadata
            }
            for error in self.errors
        ]


class FailurePatternDetector:
    """Detects patterns in chatbot failures."""
    
    @staticmethod
    def detect_oov_words(
        inputs: List[str],
        vocabulary: set
    ) -> Dict[str, List[str]]:
        """
        Detect out-of-vocabulary words in failed cases.
        
        Args:
            inputs: List of input texts
            vocabulary: Known vocabulary
            
        Returns:
            Dictionary mapping inputs to OOV words
        """
        oov_by_input = {}
        
        for input_text in inputs:
            words = input_text.lower().split()
            oov_words = [word for word in words if word not in vocabulary]
            if oov_words:
                oov_by_input[input_text] = oov_words
        
        logger.info(f"Detected OOV words in {len(oov_by_input)} inputs")
        
        return oov_by_input
    
    @staticmethod
    def detect_length_anomalies(
        inputs: List[str],
        responses: List[str],
        threshold: float = 3.0
    ) -> List[Tuple[str, str, float]]:
        """
        Detect responses with unusual length ratios.
        
        Args:
            inputs: List of input texts
            responses: List of responses
            threshold: Standard deviations for anomaly
            
        Returns:
            List of (input, response, ratio) anomalies
        """
        import numpy as np
        
        ratios = []
        for inp, resp in zip(inputs, responses):
            input_len = len(inp.split())
            resp_len = len(resp.split())
            ratio = resp_len / max(input_len, 1)
            ratios.append((inp, resp, ratio))
        
        # Calculate mean and std
        ratio_values = [r[2] for r in ratios]
        mean_ratio = np.mean(ratio_values)
        std_ratio = np.std(ratio_values)
        
        # Find anomalies
        anomalies = [
            (inp, resp, ratio)
            for inp, resp, ratio in ratios
            if abs(ratio - mean_ratio) > threshold * std_ratio
        ]
        
        logger.info(f"Detected {len(anomalies)} length anomalies")
        
        return anomalies
    
    @staticmethod
    def detect_repetitive_responses(
        responses: List[str],
        threshold: int = 3
    ) -> Dict[str, int]:
        """
        Detect frequently repeated responses.
        
        Args:
            responses: List of responses
            threshold: Minimum count to be considered repetitive
            
        Returns:
            Dictionary of repetitive responses and counts
        """
        response_counts = Counter(responses)
        repetitive = {
            resp: count
            for resp, count in response_counts.items()
            if count >= threshold
        }
        
        logger.info(f"Detected {len(repetitive)} repetitive responses")
        
        return repetitive
