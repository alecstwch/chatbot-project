"""
CLI for Hybrid Chatbot (AIML + GPT-2).

Command-line interface for the hybrid therapy chatbot that combines
AIML pattern matching with GPT-2 response generation.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.infrastructure.ml.chatbots.hybrid_chatbot import HybridChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run hybrid chatbot CLI."""
    try:
        print("=" * 70)
        print("HYBRID THERAPY CHATBOT")
        print("Combining AIML Pattern Matching + GPT-2 Generation")
        print("=" * 70)
        print("\nInitializing chatbot (this may take a minute)...\n")
        
        # Create hybrid chatbot
        chatbot = HybridChatbot(
            aiml_dir=Path("data/knowledge_bases/aiml"),
            gpt2_model="gpt2",
            intent_model="facebook/bart-large-mnli",
            use_intent_classification=True,
            aiml_confidence_threshold=10
        )
        
        # Initialize all components
        print("Loading AIML files...")
        print("Loading GPT-2 model...")
        print("Loading intent classifier...")
        chatbot.initialize()
        
        print("\nAll models loaded successfully!")
        print("\nThis chatbot uses a hybrid approach:")
        print("  1. Tries AIML pattern matching first (fast)")
        print("  2. Classifies your intent (depression, anxiety, etc.)")
        print("  3. Falls back to GPT-2 for complex queries")
        print()
        
        # Start chat
        chatbot.chat()
        
        # Show statistics
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        stats = chatbot.get_statistics()
        print(f"Total queries: {stats['total_queries']}")
        if stats['total_queries'] > 0:
            print(f"AIML responses: {stats['aiml_responses']} ({stats['aiml_percentage']:.1f}%)")
            print(f"GPT-2 responses: {stats['gpt2_responses']} ({stats['gpt2_percentage']:.1f}%)")
            print(f"Fallback responses: {stats['fallback_responses']} ({stats['fallback_percentage']:.1f}%)")
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
