"""
Day 5: Evaluation & Analysis Demo

This script demonstrates comprehensive evaluation of chatbot models including:
- Intent classification metrics (accuracy, precision, recall, F1)
- Response generation metrics (BLEU, ROUGE, METEOR)
- Error analysis and failure pattern detection
- Model explainability (LIME)
- Cross-model comparison
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import logging
from pathlib import Path
import json

# Domain services
from src.domain.services.evaluation_metrics import (
    IntentClassificationMetrics,
    ResponseGenerationMetrics,
    DialogueMetrics,
    evaluate_chatbot_performance
)
from src.domain.services.error_analysis import (
    ErrorAnalyzer,
    FailurePatternDetector
)
from src.domain.services.intent_classifier import IntentClassificationService

# Application services
from src.application.analysis.explainability import (
    IntentExplainer,
    ModelComparison
)
from src.application.analysis.model_comparison import (
    ModelBenchmark,
    CrossValidationAnalyzer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_intent_classification_metrics():
    """Demonstrate intent classification evaluation."""
    print_section("1. INTENT CLASSIFICATION METRICS")
    
    # Sample data
    y_true = [
        'greeting', 'help', 'booking', 'greeting', 'complaint',
        'help', 'booking', 'greeting', 'complaint', 'help'
    ]
    
    y_pred = [
        'greeting', 'help', 'booking', 'greeting', 'help',  # Last one is wrong
        'help', 'complaint', 'greeting', 'complaint', 'help'  # booking->complaint is wrong
    ]
    
    labels = ['greeting', 'help', 'booking', 'complaint']
    
    print("Sample Data:")
    print(f"  True labels: {y_true}")
    print(f"  Predictions: {y_pred}")
    print()
    
    # Calculate metrics
    metrics = IntentClassificationMetrics.calculate_metrics(y_true, y_pred, labels)
    
    print("Classification Metrics:")
    print("-" * 60)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:25s}: {value:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = IntentClassificationMetrics.get_confusion_matrix(y_true, y_pred, labels)
    print("              " + "  ".join([l[:8].ljust(8) for l in labels]))
    for i, label in enumerate(labels):
        row = f"  {label[:12].ljust(12)}"
        for j in range(len(labels)):
            row += f"  {cm[i,j]:8d}"
        print(row)
    
    # Detailed report
    print("\nClassification Report:")
    print("-" * 60)
    report = IntentClassificationMetrics.get_classification_report(y_true, y_pred, labels)
    print(report)


def demo_response_generation_metrics():
    """Demonstrate response generation evaluation."""
    print_section("2. RESPONSE GENERATION METRICS")
    
    # Sample data
    references = [
        "Hello! How can I help you today?",
        "I'm sorry to hear that. Let me assist you.",
        "Your booking is confirmed for tomorrow at 3 PM.",
        "Thank you for choosing our service!",
        "You can find more information on our website."
    ]
    
    candidates = [
        "Hi! How may I assist you?",  # Similar
        "I apologize. Let me help.",  # Similar but shorter
        "Booking confirmed tomorrow 3pm.",  # Similar but less formal
        "Thanks for using our service!",  # Similar
        "Check our website for details."  # Similar but different structure
    ]
    
    print("Sample Data:")
    print("-" * 60)
    for i, (ref, cand) in enumerate(zip(references, candidates), 1):
        print(f"{i}. Reference: {ref}")
        print(f"   Generated: {cand}")
        print()
    
    # Calculate metrics
    gen_metrics = ResponseGenerationMetrics()
    
    # BLEU scores
    print("BLEU Scores:")
    bleu_scores = gen_metrics.calculate_bleu(references, candidates)
    for metric, score in bleu_scores.items():
        print(f"  {metric:15s}: {score:.4f}")
    
    # ROUGE scores
    print("\nROUGE Scores:")
    rouge_scores = gen_metrics.calculate_rouge(references, candidates)
    for metric, score in rouge_scores.items():
        print(f"  {metric:15s}: {score:.4f}")
    
    # METEOR score
    print("\nMETEOR Score:")
    meteor = gen_metrics.calculate_meteor(references, candidates)
    print(f"  meteor         : {meteor:.4f}")
    
    # All metrics together
    print("\nAll Metrics Combined:")
    all_metrics = gen_metrics.calculate_all_metrics(references, candidates)
    print("-" * 60)
    for metric, score in all_metrics.items():
        print(f"  {metric:15s}: {score:.4f}")


def demo_dialogue_metrics():
    """Demonstrate dialogue quality metrics."""
    print_section("3. DIALOGUE QUALITY METRICS")
    
    responses = [
        "Hello! How can I help you today?",
        "I'm sorry to hear that. Let me assist you.",
        "Your booking is confirmed.",
        "Thank you for choosing our service!",
        "You can find more information on our website.",
        "Hello! How can I help you today?",  # Duplicate
        "Is there anything else I can help with?",
        "Have a great day!"
    ]
    
    print("Sample Responses:")
    for i, response in enumerate(responses, 1):
        print(f"  {i}. {response}")
    print()
    
    # Diversity
    diversity = DialogueMetrics.calculate_response_diversity(responses)
    print(f"Response Diversity: {diversity:.4f}")
    print("  (Ratio of unique tokens to total tokens)")
    
    # Length statistics
    print("\nLength Statistics:")
    length_stats = DialogueMetrics.calculate_average_length(responses)
    for stat, value in length_stats.items():
        print(f"  {stat:15s}: {value:.2f} words")


def demo_error_analysis():
    """Demonstrate error analysis."""
    print_section("4. ERROR ANALYSIS")
    
    # Create error analyzer
    analyzer = ErrorAnalyzer()
    
    # Add sample errors
    errors = [
        {
            'input': "I need help with my booking",
            'expected': "help",
            'predicted': "booking",
            'intent_exp': "help",
            'intent_pred': "booking"
        },
        {
            'input': "Hello there",
            'expected': "greeting",
            'predicted': "greeting",  # Correct but we'll add response error
        },
        {
            'input': "This is unacceptable",
            'expected': "complaint",
            'predicted': "help",
            'intent_exp': "complaint",
            'intent_pred': "help"
        },
        {
            'input': "Can you help?",
            'expected': "help",
            'predicted': "help",  # Correct
        },
        {
            'input': "I want to cancel",
            'expected': "cancellation",
            'predicted': "booking",
            'intent_exp': "cancellation",
            'intent_pred': "booking"
        }
    ]
    
    print("Adding error cases...")
    for error in errors:
        error_type = analyzer.categorize_error(
            error['input'],
            error.get('expected', ''),
            error.get('predicted', ''),
            error.get('intent_exp'),
            error.get('intent_pred')
        )
        
        analyzer.add_error(
            error['input'],
            error.get('expected', ''),
            error.get('predicted', ''),
            error_type=error_type,
            confidence=0.65 if error.get('intent_exp') != error.get('intent_pred') else 0.95
        )
    
    print(f"\nTotal errors collected: {len(analyzer.errors)}")
    
    # Error distribution
    print("\nError Distribution:")
    distribution = analyzer.get_error_distribution()
    for error_type, count in distribution.items():
        print(f"  {error_type:30s}: {count}")
    
    # Most common errors
    print("\nMost Common Errors:")
    common_errors = analyzer.get_most_common_errors(top_n=5)
    for error_type, count in common_errors:
        print(f"  {error_type:30s}: {count} occurrences")
    
    # Low confidence errors
    print("\nLow Confidence Errors:")
    low_conf = analyzer.get_low_confidence_errors(threshold=0.7)
    print(f"  Found {len(low_conf)} errors with confidence < 0.7")
    
    # Full report
    print("\n" + "=" * 70)
    print(analyzer.generate_error_report())


def demo_failure_patterns():
    """Demonstrate failure pattern detection."""
    print_section("5. FAILURE PATTERN DETECTION")
    
    # Sample failed inputs
    failed_inputs = [
        "Hello there friend",
        "I need help ASAP",
        "Can you assist with xyz problem",
        "What is the status of my order",
        "Hello there friend",  # Duplicate
    ]
    
    responses = [
        "Hi",  # Too short
        "I can help you",
        "I can assist you with that issue",
        "Let me check the status for you",
        "Hi",  # Too short, duplicate
    ]
    
    # Known vocabulary
    vocabulary = {
        'hello', 'hi', 'help', 'can', 'you', 'i', 'need',
        'assist', 'with', 'what', 'is', 'the', 'status',
        'of', 'my', 'order', 'let', 'me', 'check', 'for'
    }
    
    # Detect OOV words
    print("Out-of-Vocabulary Words:")
    oov_words = FailurePatternDetector.detect_oov_words(failed_inputs, vocabulary)
    for inp, words in oov_words.items():
        print(f"  '{inp}': {words}")
    
    # Detect repetitive responses
    print("\nRepetitive Responses:")
    repetitive = FailurePatternDetector.detect_repetitive_responses(responses, threshold=2)
    for response, count in repetitive.items():
        print(f"  '{response}': {count} times")
    
    # Length anomalies
    print("\nLength Anomalies:")
    anomalies = FailurePatternDetector.detect_length_anomalies(
        failed_inputs, responses, threshold=2.0
    )
    for inp, resp, ratio in anomalies:
        print(f"  Input: '{inp}'")
        print(f"  Response: '{resp}'")
        print(f"  Length ratio: {ratio:.2f}\n")


def demo_model_explainability():
    """Demonstrate model explainability with LIME."""
    print_section("6. MODEL EXPLAINABILITY (LIME)")
    
    print("Loading intent classifier...")
    
    try:
        # Initialize classifier
        classifier = IntentClassificationService()
        
        # Define classes
        class_names = [
            'therapy_support',
            'therapy_greeting',
            'therapy_farewell',
            'therapy_crisis',
            'general'
        ]
        
        # Create explainer
        explainer = IntentExplainer(class_names)
        
        # Sample texts to explain
        texts = [
            "I'm feeling really anxious today",
            "Hello, I need someone to talk to",
            "Thank you for listening, goodbye"
        ]
        
        print("\nGenerating explanations...\n")
        
        # Wrapper function for classifier
        def classifier_fn(texts_batch):
            """Wrapper for LIME."""
            results = []
            for text in texts_batch:
                probs = classifier.classify(text, class_names)
                # Convert to list of probabilities
                prob_list = [probs.get(name, 0.0) for name in class_names]
                results.append(prob_list)
            return results
        
        # Explain each prediction
        for i, text in enumerate(texts, 1):
            print(f"Example {i}: '{text}'")
            print("-" * 60)
            
            explanation = explainer.explain_prediction(
                text,
                classifier_fn,
                num_features=5,
                num_samples=100
            )
            
            if 'error' not in explanation:
                print(f"Predicted: {explanation['predicted_class']}")
                print(f"Confidence: {explanation['confidence']:.3f}")
                
                print("\nAll Probabilities:")
                for class_name, prob in explanation['all_probabilities'].items():
                    print(f"  {class_name:25s}: {prob:.3f}")
                
                print("\nTop Contributing Features:")
                for feature, importance in explanation['top_features']:
                    direction = "supports" if importance > 0 else "contradicts"
                    print(f"  '{feature}': {abs(importance):.3f} ({direction})")
            else:
                print(f"Error: {explanation['error']}")
            
            print()
    
    except Exception as e:
        logger.error(f"Error in explainability demo: {e}")
        print(f"Note: LIME explainability requires the intent classifier model.")
        print(f"Error details: {e}")


def demo_model_comparison():
    """Demonstrate model comparison."""
    print_section("7. MODEL COMPARISON")
    
    # Create benchmark
    benchmark = ModelBenchmark()
    
    # Add results for different models
    print("Adding results for 3 different models...\n")
    
    # AIML Model (rule-based)
    benchmark.add_model_results(
        'AIML (Rule-based)',
        {
            'accuracy': 0.72,
            'precision_macro': 0.68,
            'recall_macro': 0.70,
            'f1_macro': 0.69,
            'bleu': 0.45,
            'rouge1': 0.52,
            'rougeL': 0.48
        },
        {'type': 'rule-based', 'rules': 150}
    )
    
    # DialoGPT Model
    benchmark.add_model_results(
        'DialoGPT',
        {
            'accuracy': 0.78,
            'precision_macro': 0.75,
            'recall_macro': 0.76,
            'f1_macro': 0.75,
            'bleu': 0.58,
            'rouge1': 0.64,
            'rougeL': 0.60
        },
        {'type': 'neural', 'parameters': '117M'}
    )
    
    # GPT-2 + Intent
    benchmark.add_model_results(
        'GPT-2 + Intent',
        {
            'accuracy': 0.85,
            'precision_macro': 0.83,
            'recall_macro': 0.84,
            'f1_macro': 0.83,
            'bleu': 0.62,
            'rouge1': 0.68,
            'rougeL': 0.65
        },
        {'type': 'hybrid', 'parameters': '124M'}
    )
    
    # Show comparison
    print(benchmark.compare_models())
    
    # Rankings
    print("\n\nRankings by F1 Score:")
    print("-" * 60)
    rankings = benchmark.get_rankings('f1_macro')
    for model_name, score, rank in rankings:
        print(f"  #{rank} {model_name:25s}: {score:.4f}")
    
    # Summary
    print("\n")
    print(benchmark.generate_summary())
    
    # Export results
    output_dir = Path('evaluation/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark.export_to_json('evaluation/results/benchmark_results.json')
    benchmark.export_to_csv('evaluation/results/benchmark_results.csv')
    
    print(f"\nResults exported to evaluation/results/")


def demo_cross_validation():
    """Demonstrate cross-validation analysis."""
    print_section("8. CROSS-VALIDATION ANALYSIS")
    
    # Create analyzer
    cv_analyzer = CrossValidationAnalyzer(n_folds=5)
    
    print("Simulating 5-fold cross-validation results...\n")
    
    # Simulate fold results with some variance
    import random
    random.seed(42)
    
    base_metrics = {
        'accuracy': 0.82,
        'precision': 0.80,
        'recall': 0.81,
        'f1_score': 0.80,
        'bleu': 0.58
    }
    
    for fold in range(5):
        # Add some random variance
        fold_metrics = {
            metric: value + random.uniform(-0.03, 0.03)
            for metric, value in base_metrics.items()
        }
        cv_analyzer.add_fold_result(fold, fold_metrics)
        
        print(f"Fold {fold}: F1={fold_metrics['f1_score']:.4f}, BLEU={fold_metrics['bleu']:.4f}")
    
    # Generate report
    print("\n")
    print(cv_analyzer.generate_report())
    
    # Statistics
    stats = cv_analyzer.calculate_statistics()
    print("\nDetailed Statistics:")
    print("-" * 70)
    for metric, stat in stats.items():
        print(f"{metric:15s}: {stat['mean']:.4f} ± {stat['std']:.4f} "
              f"(range: [{stat['min']:.4f}, {stat['max']:.4f}])")


def demo_comprehensive_evaluation():
    """Demonstrate comprehensive chatbot evaluation."""
    print_section("9. COMPREHENSIVE EVALUATION")
    
    print("Performing end-to-end evaluation...\n")
    
    # Sample data
    intent_true = ['greeting', 'help', 'booking', 'complaint', 'help']
    intent_pred = ['greeting', 'help', 'booking', 'help', 'help']  # One error
    
    references = [
        "Hello! How can I help you?",
        "I'll assist you with that.",
        "Your booking is confirmed.",
        "I apologize for the inconvenience.",
        "Let me help you with that."
    ]
    
    candidates = [
        "Hi! How may I assist?",
        "I can help with that.",
        "Booking confirmed.",
        "Sorry about that.",
        "I'll help you."
    ]
    
    # Comprehensive evaluation
    results = evaluate_chatbot_performance(
        intent_true=intent_true,
        intent_pred=intent_pred,
        references=references,
        candidates=candidates,
        intent_labels=['greeting', 'help', 'booking', 'complaint']
    )
    
    # Display results
    print("INTENT CLASSIFICATION:")
    print("-" * 60)
    for metric, value in results['intent_classification'].items():
        print(f"  {metric:25s}: {value:.4f}")
    
    print("\nRESPONSE GENERATION:")
    print("-" * 60)
    for metric, value in results['response_generation'].items():
        print(f"  {metric:25s}: {value:.4f}")
    
    print("\nDIALOGUE QUALITY:")
    print("-" * 60)
    for metric, value in results['dialogue_quality'].items():
        print(f"  {metric:25s}: {value:.4f}")
    
    # Export results
    output_file = 'evaluation/results/comprehensive_evaluation.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nFull results saved to {output_file}")


def main():
    """Run all evaluation demos."""
    print("\n" + "=" * 80)
    print("  DAY 5: COMPREHENSIVE CHATBOT EVALUATION & ANALYSIS")
    print("=" * 80)
    print("\nThis demo showcases:")
    print("  • Intent classification metrics (accuracy, precision, recall, F1)")
    print("  • Response generation metrics (BLEU, ROUGE, METEOR)")
    print("  • Dialogue quality metrics (diversity, length)")
    print("  • Error analysis and failure pattern detection")
    print("  • Model explainability (LIME)")
    print("  • Multi-model comparison and benchmarking")
    print("  • Cross-validation analysis")
    
    try:
        demo_intent_classification_metrics()
        demo_response_generation_metrics()
        demo_dialogue_metrics()
        demo_error_analysis()
        demo_failure_patterns()
        demo_model_explainability()
        demo_model_comparison()
        demo_cross_validation()
        demo_comprehensive_evaluation()
        
        print_section("EVALUATION COMPLETE!")
        print("All evaluation modules demonstrated successfully.")
        print("\nGenerated outputs:")
        print("  • evaluation/results/benchmark_results.json")
        print("  • evaluation/results/benchmark_results.csv")
        print("  • evaluation/results/comprehensive_evaluation.json")
        print("\nDay 5 evaluation infrastructure is ready for use!")
        
    except Exception as e:
        logger.error(f"Error during evaluation demo: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
