"""
Day 4 Demo: All Three Model Types

This script demonstrates all three model architectures:
1. Traditional/Rule-Based (AIML)
2. Neural Network (DialoGPT)
3. Transformer with Intent Classification (Hybrid + Enhanced)

Run this to see all chatbot implementations in action.
"""

import sys
import os
from pathlib import Path
import logging

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.ml.chatbots.aiml_chatbot import AimlChatbot
from src.infrastructure.ml.chatbots.dialogpt_chatbot import DialoGPTChatbot
from src.infrastructure.ml.chatbots.hybrid_chatbot import HybridChatbot
from src.infrastructure.ml.chatbots.transformer_enhanced_chatbot import TransformerEnhancedChatbot
from src.domain.services.intent_classifier import IntentClassificationService
from src.infrastructure.ml.models.response_generator import ResponseGenerationService

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings/errors for demo
    format='%(levelname)s - %(message)s'
)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_aiml_chatbot():
    """Demonstrate AIML (rule-based) chatbot."""
    print_header("MODEL 1: TRADITIONAL/RULE-BASED (AIML)")
    print("\nPattern matching with AIML - fast and deterministic")
    print("Best for: Structured conversations, FAQs, therapy patterns\n")
    
    try:
        # Initialize AIML chatbot
        bot = AimlChatbot(aiml_dir=Path("data/knowledge_bases/aiml"))
        num_files = bot.load_aiml_files()
        print(f"✓ Loaded {num_files} AIML files\n")
        
        # Test cases
        test_inputs = [
            "I feel anxious",
            "I am depressed",
            "Hello",
            "Can you help me?"
        ]
        
        print("Test Conversations:")
        for user_input in test_inputs:
            response = bot.get_response(user_input)
            print(f"  You: {user_input}")
            print(f"  Bot: {response}\n")
        
        print("✓ AIML chatbot demo complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_dialogpt_chatbot():
    """Demonstrate DialoGPT (neural) chatbot."""
    print_header("MODEL 2: NEURAL NETWORK (DialoGPT)")
    print("\nPre-trained transformer model for natural conversations")
    print("Best for: Open-domain chitchat, contextual responses\n")
    
    try:
        # Initialize DialoGPT
        print("Loading DialoGPT model (this may take a moment)...")
        bot = DialoGPTChatbot(model_name="microsoft/DialoGPT-small")
        bot.load_model()
        print("✓ DialoGPT model loaded\n")
        
        # Test conversation
        test_inputs = [
            "Hi, how are you?",
            "Tell me about chatbots",
            "What can you help me with?"
        ]
        
        print("Test Conversations:")
        for user_input in test_inputs:
            response = bot.get_response(user_input)
            print(f"  You: {user_input}")
            print(f"  Bot: {response}\n")
        
        print("✓ DialoGPT chatbot demo complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: DialoGPT requires transformers library and model download")


def demo_intent_classification():
    """Demonstrate intent classification service."""
    print_header("TRANSFORMER COMPONENT: Intent Classification")
    print("\nZero-shot classification for understanding user intent")
    print("Detects: depression, anxiety, stress, grief, etc.\n")
    
    try:
        # Initialize classifier
        print("Loading intent classifier (this may take a moment)...")
        classifier = IntentClassificationService()
        classifier.load_model()
        print("✓ Intent classifier loaded\n")
        
        # Test cases
        test_inputs = [
            "I'm feeling very anxious and worried",
            "I've been so sad and hopeless lately",
            "I'm really stressed about work",
            "Hello, can you help me?",
            "Thank you for your help"
        ]
        
        print("Intent Classification Results:")
        for text in test_inputs:
            result = classifier.classify(text)
            print(f"  Input: {text}")
            print(f"  Intent: {result.intent} (confidence: {result.confidence:.2f})")
            print()
        
        print("✓ Intent classification demo complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: Intent classification requires transformers library")


def demo_response_generation():
    """Demonstrate GPT-2 response generation."""
    print_header("TRANSFORMER COMPONENT: Response Generation (GPT-2)")
    print("\nContextual response generation for complex queries")
    print("Best for: Therapy-focused responses, creative generation\n")
    
    try:
        # Initialize generator
        print("Loading GPT-2 model (this may take a moment)...")
        generator = ResponseGenerationService(model_name="gpt2")
        generator.load_model()
        print("✓ GPT-2 model loaded\n")
        
        # Test therapy responses
        test_cases = [
            ("I'm feeling anxious", "anxiety"),
            ("I've been depressed", "depression"),
            ("I need help", None)
        ]
        
        print("Therapy Response Generation:")
        for user_input, intent in test_cases:
            response = generator.generate_therapy_response(user_input, intent)
            print(f"  Patient: {user_input}")
            print(f"  Intent: {intent or 'not specified'}")
            print(f"  Therapist: {response}\n")
        
        print("✓ Response generation demo complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: Response generation requires transformers and torch")


def demo_hybrid_chatbot():
    """Demonstrate hybrid chatbot (AIML + GPT-2)."""
    print_header("MODEL 3A: HYBRID CHATBOT (AIML + GPT-2 + Intent)")
    print("\nCombines AIML patterns with GPT-2 generation")
    print("Strategy: AIML first → Intent classification → GPT-2 fallback\n")
    
    try:
        # Initialize hybrid bot
        print("Loading hybrid chatbot components...")
        bot = HybridChatbot(
            aiml_dir=Path("data/knowledge_bases/aiml"),
            gpt2_model="gpt2",
            use_intent_classification=True
        )
        bot.initialize()
        print("✓ Hybrid chatbot initialized\n")
        
        # Test cases - mix of AIML patterns and complex queries
        test_inputs = [
            "I feel anxious",  # Should match AIML
            "What is the meaning of existence?",  # GPT-2
            "I'm overwhelmed with everything",  # GPT-2 with intent
            "Hello"  # AIML
        ]
        
        print("Test Conversations (with strategy info):")
        for user_input in test_inputs:
            result = bot.get_response(user_input, return_metadata=True)
            print(f"  You: {user_input}")
            print(f"  Bot: {result['response']}")
            print(f"  Strategy: {result['metadata']['strategy']}")
            if result['metadata']['intent']:
                print(f"  Intent: {result['metadata']['intent']}")
            print()
        
        # Show statistics
        stats = bot.get_statistics()
        print("Session Statistics:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  AIML responses: {stats['aiml_responses']}")
        print(f"  GPT-2 responses: {stats['gpt2_responses']}")
        
        print("\n✓ Hybrid chatbot demo complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_transformer_enhanced():
    """Demonstrate transformer-enhanced DialoGPT."""
    print_header("MODEL 3B: TRANSFORMER-ENHANCED (DialoGPT + Intent)")
    print("\nDialoGPT with intent classification for context awareness")
    print("Adapts response generation based on detected intent\n")
    
    try:
        # Initialize enhanced bot
        print("Loading transformer-enhanced chatbot...")
        bot = TransformerEnhancedChatbot(use_intent_classification=True)
        bot.load_models()
        print("✓ Transformer-enhanced chatbot initialized\n")
        
        # Test conversation
        test_inputs = [
            "I'm feeling really anxious",
            "Can you help me feel better?",
            "Thank you for listening"
        ]
        
        print("Test Conversations (with intent detection):")
        for user_input in test_inputs:
            result = bot.get_response(user_input, return_metadata=True)
            print(f"  You: {user_input}")
            print(f"  Bot: {result['response']}")
            if result['metadata']['intent']:
                print(f"  Intent: {result['metadata']['intent']} ({result['metadata']['confidence']:.2f})")
            print()
        
        print("✓ Transformer-enhanced chatbot demo complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("  DAY 4 DEMONSTRATION: ALL THREE MODEL TYPES")
    print("  Showcasing Traditional, Neural, and Transformer Approaches")
    print("=" * 80)
    
    print("\nThis demo will show:")
    print("  1. AIML (Traditional/Rule-Based)")
    print("  2. DialoGPT (Neural Network)")
    print("  3. Intent Classification (Transformer)")
    print("  4. Response Generation with GPT-2 (Transformer)")
    print("  5. Hybrid Chatbot (AIML + GPT-2 + Intent)")
    print("  6. Transformer-Enhanced DialoGPT (DialoGPT + Intent)")
    
    print("\n" + "=" * 80)
    print("Starting demos...")
    print("=" * 80)
    
    # Run all demos
    try:
        demo_aiml_chatbot()
        
        print("\n" + "-" * 80)
        input("Press Enter to continue to Neural Network demo...")
        
        demo_dialogpt_chatbot()
        
        print("\n" + "-" * 80)
        input("Press Enter to continue to Intent Classification demo...")
        
        demo_intent_classification()
        
        print("\n" + "-" * 80)
        input("Press Enter to continue to Response Generation demo...")
        
        demo_response_generation()
        
        print("\n" + "-" * 80)
        input("Press Enter to continue to Hybrid Chatbot demo...")
        
        demo_hybrid_chatbot()
        
        print("\n" + "-" * 80)
        input("Press Enter to continue to Transformer-Enhanced demo...")
        
        demo_transformer_enhanced()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print_header("DEMO COMPLETE")
    print("\nAll three model types have been demonstrated:")
    print("  ✓ Traditional (AIML) - Fast pattern matching")
    print("  ✓ Neural (DialoGPT) - Contextual conversations")
    print("  ✓ Transformer (Intent + GPT-2) - Adaptive, intent-aware responses")
    print("\nDay 4 objectives achieved:")
    print("  ✓ Intent classification with transformers")
    print("  ✓ Response generation with GPT-2")
    print("  ✓ Hybrid approach combining all techniques")
    print("\nNext: Day 5 - Evaluation & Analysis")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
