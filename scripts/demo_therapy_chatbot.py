"""
Interactive Therapy Chatbot Demo - Day 4

This demonstrates the therapy chatbot with intent classification.
The chatbot uses:
- AIML pattern matching for therapy responses
- Intent classification to understand emotional states
- Configuration-driven approach (all intents in YAML)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.ml.chatbots.aiml_chatbot import AimlChatbot
from src.domain.services.intent_classifier import IntentClassificationService


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("  THERAPY CHATBOT - Intent-Aware Conversation")
    print("\nThis chatbot combines:")
    print("   AIML pattern matching for therapy responses")
    print("   Intent classification to understand emotional states")
    print("   11 therapy intents (depression, anxiety, stress, etc.)")
    print("\nType 'quit' to exit\n")
    print("-" * 70 + "\n")


def main():
    """Run interactive therapy chatbot demo."""
    print_header()
    
    # Initialize AIML chatbot
    print("Loading therapy chatbot...")
    try:
        aiml_bot = AimlChatbot(aiml_dir=Path("data/knowledge_bases/aiml"))
        num_files = aiml_bot.load_aiml_files()
        print(f"Loaded {num_files} AIML files")
    except Exception as e:
        print(f"Error loading AIML: {e}")
        return
    
    # Initialize intent classifier
    print("Loading intent classifier...")
    try:
        intent_classifier = IntentClassificationService(domain="therapy_intents")
        info = intent_classifier.get_domain_info()
        print(f"Intent classifier ready ({info['num_intents']} intents)")
        print(f"  Intents: {', '.join(info['intent_labels'][:5])}...\n")
    except Exception as e:
        print(f"Error loading intent classifier: {e}")
        return
    
    print("Ready! Start chatting...\n")
    
    # Conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for sharing. Take care!")
                break
            
            # Classify intent
            intent_result = intent_classifier._keyword_classify(user_input)
            
            # Get AIML response
            response = aiml_bot.get_response(user_input)
            
            # Display intent and response
            print(f"\n[Intent detected: {intent_result.intent} ({intent_result.confidence:.2f})]\n")
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
