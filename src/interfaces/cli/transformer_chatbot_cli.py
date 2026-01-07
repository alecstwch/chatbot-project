"""
CLI for Transformer-Enhanced DialoGPT Chatbot.

Command-line interface for the transformer chatbot that combines
DialoGPT with intent classification for better context awareness.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.infrastructure.ml.chatbots.transformer_enhanced_chatbot import TransformerEnhancedChatbot
from src.infrastructure.config.chatbot_settings import DialoGPTSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run transformer-enhanced chatbot CLI."""
    try:
        print("=" * 70)
        print("TRANSFORMER-ENHANCED CHATBOT")
        print("DialoGPT + Intent Classification")
        print("=" * 70)
        print("\nInitializing chatbot (this may take a minute)...\n")
        
        # Load settings
        settings = DialoGPTSettings()
        
        # Create transformer-enhanced chatbot
        chatbot = TransformerEnhancedChatbot(
            settings=settings,
            use_intent_classification=True
        )
        
        # Load models
        print("Loading DialoGPT model...")
        print("Loading intent classifier...")
        chatbot.load_models()
        
        print("\n✓ All models loaded successfully!")
        print("\nThis chatbot features:")
        print("  • DialoGPT for natural conversation")
        print("  • Intent classification for context awareness")
        print("  • Adaptive response generation")
        print("  • Conversation history tracking")
        print()
        
        # Start chat
        chatbot.chat()
        
        # Show conversation summary
        print("\n" + "=" * 70)
        print("CONVERSATION SUMMARY")
        print("=" * 70)
        history = chatbot.get_conversation_history()
        print(f"Total turns: {len(history)}")
        
        if history:
            print("\nDetected intents:")
            intent_counts = {}
            for turn in history:
                if turn['intent']:
                    intent_counts[turn['intent']] = intent_counts.get(turn['intent'], 0) + 1
            
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {intent}: {count}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Error running chatbot: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Please check that all required models are available.")
        sys.exit(1)


if __name__ == "__main__":
    main()
