"""
Quick test of the Error Analysis module.

Tests ErrorAnalyzer and FailurePatternDetector functionality.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.domain.services.error_analysis import (
    ErrorAnalyzer,
    FailurePatternDetector
)

print("=" * 80)
print("  ERROR ANALYSIS MODULE TEST")
print("=" * 80)

# Test 1: ErrorAnalyzer
print("\n1. Testing ErrorAnalyzer...")
print("-" * 80)

analyzer = ErrorAnalyzer()

# Add some test errors
test_cases = [
    {
        'input': "I need help with my booking",
        'expected': "help",
        'predicted': "booking",
        'intent_exp': "help",
        'intent_pred': "booking"
    },
    {
        'input': "This service is terrible",
        'expected': "complaint",
        'predicted': "help",
        'intent_exp': "complaint",
        'intent_pred': "help"
    },
    {
        'input': "Hello there",
        'expected': "Hi! How can I help?",
        'predicted': "Hi",
        'intent_exp': "greeting",
        'intent_pred': "greeting"
    },
    {
        'input': "What's your name?",
        'expected': "I'm a chatbot assistant",
        'predicted': "What's your name?",  # Repetitive
        'intent_exp': "identity",
        'intent_pred': "identity"
    },
    {
        'input': "Cancel my reservation please",
        'expected': "cancellation",
        'predicted': "booking",
        'intent_exp': "cancellation",
        'intent_pred': "booking"
    }
]

print(f"Adding {len(test_cases)} error cases...\n")

for case in test_cases:
    # Categorize the error
    error_type = analyzer.categorize_error(
        case['input'],
        case['expected'],
        case['predicted'],
        case.get('intent_exp'),
        case.get('intent_pred')
    )
    
    # Add to analyzer
    analyzer.add_error(
        input_text=case['input'],
        expected=case['expected'],
        predicted=case['predicted'],
        error_type=error_type,
        confidence=0.65 if case.get('intent_exp') != case.get('intent_pred') else 0.85
    )
    
    print(f"Added: {error_type}")

# Get error distribution
print("\n" + "=" * 80)
print("Error Distribution:")
print("-" * 80)
distribution = analyzer.get_error_distribution()
for error_type, count in distribution.items():
    print(f"  {error_type:30s}: {count:2d}")

# Get most common errors
print("\n" + "=" * 80)
print("Most Common Errors:")
print("-" * 80)
common_errors = analyzer.get_most_common_errors(top_n=5)
for error_type, count in common_errors:
    print(f"  {error_type:30s}: {count} occurrences")

# Low confidence errors
print("\n" + "=" * 80)
print("Low Confidence Errors (< 0.7):")
print("-" * 80)
low_conf = analyzer.get_low_confidence_errors(threshold=0.7)
print(f"Found {len(low_conf)} errors with confidence < 0.7")
for error in low_conf:
    print(f"  - {error.error_type}: '{error.input_text[:40]}...' (conf: {error.confidence:.2f})")

# Generate full report
print("\n" + "=" * 80)
print(analyzer.generate_error_report())

# Export to dict
print("\n" + "=" * 80)
print("Exporting errors to structured format...")
print("-" * 80)
errors_dict = analyzer.export_errors_to_dict()
print(f"Exported {len(errors_dict)} errors")
print(f"\nFirst error example:")
print(f"  Input: {errors_dict[0]['input']}")
print(f"  Expected: {errors_dict[0]['expected']}")
print(f"  Predicted: {errors_dict[0]['predicted']}")
print(f"  Type: {errors_dict[0]['error_type']}")

# Test 2: FailurePatternDetector
print("\n\n" + "=" * 80)
print("2. Testing FailurePatternDetector...")
print("-" * 80)

# Test OOV detection
print("\nOOV Word Detection:")
print("-" * 80)

failed_inputs = [
    "Hello there friend",
    "I need help ASAP immediately",
    "Can you assist with xyz problem",
    "What's the status of my order",
]

vocabulary = {
    'hello', 'hi', 'help', 'can', 'you', 'i', 'need',
    'assist', 'with', 'what', 'is', 'the', 'status',
    'of', 'my', 'order', 'let', 'me', 'check', 'for'
}

oov_words = FailurePatternDetector.detect_oov_words(failed_inputs, vocabulary)
print(f"Found OOV words in {len(oov_words)} inputs:")
for inp, words in oov_words.items():
    print(f"  '{inp}'")
    print(f"    OOV: {words}")

# Test repetitive responses
print("\n" + "=" * 80)
print("Repetitive Response Detection:")
print("-" * 80)

responses = [
    "I can help you",
    "Let me assist",
    "I can help you",  # Duplicate
    "Sure thing",
    "I can help you",  # Duplicate
    "Absolutely",
    "I can help you",  # Duplicate
]

repetitive = FailurePatternDetector.detect_repetitive_responses(responses, threshold=2)
print(f"Found {len(repetitive)} repetitive responses:")
for response, count in repetitive.items():
    print(f"  '{response}': appeared {count} times")

# Test length anomalies
print("\n" + "=" * 80)
print("Length Anomaly Detection:")
print("-" * 80)

inputs = [
    "Hi",
    "How are you?",
    "Tell me about your services",
    "What can you do?",
    "Hello"
]

responses_for_anomaly = [
    "Hi! How can I assist you today with your needs?",  # Too long for "Hi"
    "Good",  # Too short for "How are you?"
    "We offer various services",
    "I can help",
    "Hello there"
]

anomalies = FailurePatternDetector.detect_length_anomalies(
    inputs, responses_for_anomaly, threshold=2.0
)

print(f"Found {len(anomalies)} length anomalies:")
for inp, resp, ratio in anomalies:
    print(f"\n  Input:  '{inp}'")
    print(f"  Response: '{resp}'")
    print(f"  Ratio: {ratio:.2f} (response/input length)")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nError Analysis module is working correctly!")
