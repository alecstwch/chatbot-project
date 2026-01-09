"""
MongoDB Integration Demo

Demonstrates storing conversations with user identification and intent tagging.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import logging
from src.domain.services.conversation_storage import ConversationStorageService
from src.domain.services.intent_classifier import IntentClassificationService
from src.infrastructure.ml.chatbots.aiml_chatbot import AimlChatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("  MONGODB CONVERSATION STORAGE DEMO")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Store conversations with user identification")
    print("  • Tag messages with intent classification")
    print("  • Track user statistics and history")
    print("  • Query conversations by intent")
    print("\n" + "-" * 70 + "\n")


def demo_basic_storage():
    """Demo: Basic conversation storage."""
    print("=" * 70)
    print("Demo 1: Basic Conversation Storage")
    print("=" * 70 + "\n")
    
    # Initialize storage service
    storage = ConversationStorageService()
    
    if not storage.initialize():
        print("Failed to connect to MongoDB")
        print("\nMake sure MongoDB is running:")
        print("  Option 1: docker run -d -p 27017:27017 mongo")
        print("  Option 2: Install MongoDB Community Edition\n")
        return False
    
    print("Connected to MongoDB\n")
    
    # Start a session for user "alice"
    session_id = storage.start_session(
        user_name="alice",
        metadata={"platform": "cli", "version": "1.0"}
    )
    print(f"Started session: {session_id}\n")
    
    # Simulate conversation with intent detection
    conversations = [
        ("I feel anxious about work", "anxiety", 0.95),
        ("I can't stop worrying", "anxiety", 0.88),
        ("Thank you for listening", "gratitude", 0.92)
    ]
    
    for user_msg, intent, confidence in conversations:
        # Log user message with intent
        storage.log_user_message(
            content=user_msg,
            intent=intent,
            confidence=confidence
        )
        
        # Simulate bot response
        bot_response = f"I understand you're experiencing {intent}. Tell me more."
        storage.log_bot_response(
            content=bot_response,
            metadata={"model": "aiml", "response_time_ms": 50}
        )
        
        print(f"  User: {user_msg}")
        print(f"  Intent: {intent} ({confidence:.2f})")
        print(f"  Bot: {bot_response}\n")
    
    # End and save session
    storage.end_session()
    print("Session saved to MongoDB\n")
    
    # Retrieve user history
    print("Retrieving user history...")
    sessions = storage.get_user_history("alice", limit=5)
    print(f"Found {len(sessions)} session(s) for user 'alice'\n")
    
    # Get user profile
    profile = storage.get_user_profile("alice")
    if profile:
        print(f"User Profile: alice")
        print(f"  Total sessions: {profile.total_sessions}")
        print(f"  Total messages: {profile.total_messages}")
        print(f"  Primary intents: {', '.join(profile.primary_intents[:5])}")
        print(f"  Last seen: {profile.last_seen}\n")
    
    storage.shutdown()
    return True


def demo_intent_search():
    """Demo: Search conversations by intent."""
    print("\n" + "=" * 70)
    print("Demo 2: Search by Intent")
    print("=" * 70 + "\n")
    
    storage = ConversationStorageService()
    if not storage.initialize():
        return False
    
    # Create sessions for different users with various intents
    test_data = [
        ("bob", [("I'm depressed", "depression", 0.93), ("I feel hopeless", "depression", 0.91)]),
        ("carol", [("I'm stressed about exams", "stress", 0.89), ("Too much pressure", "stress", 0.85)]),
        ("alice", [("I'm anxious again", "anxiety", 0.94)])
    ]
    
    for user_name, messages in test_data:
        storage.start_session(user_name)
        for msg, intent, conf in messages:
            storage.log_user_message(msg, intent=intent, confidence=conf)
            storage.log_bot_response(f"I understand your {intent}. Let's talk about it.")
        storage.end_session()
    
    print("Created test sessions\n")
    
    # Search by intent
    print("Searching for 'anxiety' conversations...")
    anxiety_sessions = storage.search_by_intent("anxiety")
    print(f"Found {len(anxiety_sessions)} session(s) with 'anxiety' intent\n")
    
    for session in anxiety_sessions:
        print(f"  User: {session.user_name}")
        print(f"  Date: {session.started_at}")
        print(f"  Messages: {len(session.messages)}")
        print(f"  Intents: {', '.join(session.get_intents())}\n")
    
    storage.shutdown()
    return True


def demo_analytics():
    """Demo: Intent analytics."""
    print("\n" + "=" * 70)
    print("Demo 3: Intent Analytics")
    print("=" * 70 + "\n")
    
    storage = ConversationStorageService()
    if not storage.initialize():
        return False
    
    # Get intent statistics for last 7 days
    stats = storage.get_intent_analytics(days=7)
    
    if stats:
        print("Intent Statistics (Last 7 Days):\n")
        for intent, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {intent:20s}: {count:3d} occurrences")
        print()
    else:
        print("No intent data found (create some sessions first)\n")
    
    storage.shutdown()
    return True


def demo_with_chatbot():
    """Demo: Integrated with AIML chatbot and intent classifier."""
    print("\n" + "=" * 70)
    print("Demo 4: Integrated Chatbot with MongoDB Storage")
    print("=" * 70 + "\n")
    
    # Initialize components
    storage = ConversationStorageService()
    if not storage.initialize():
        return False
    
    try:
        chatbot = AimlChatbot(aiml_dir=Path("data/knowledge_bases/aiml"))
        chatbot.load_aiml_files()
        print("AIML chatbot loaded")
        
        classifier = IntentClassificationService(domain="therapy_intents")
        print("Intent classifier loaded\n")
        
    except Exception as e:
        print(f"Failed to load chatbot: {e}\n")
        storage.shutdown()
        return False
    
    # Start session
    user_name = "demo_user"
    storage.start_session(user_name, metadata={"demo": True})
    print(f"Session started for: {user_name}\n")
    
    # Test conversations
    test_inputs = [
        "I feel very anxious",
        "I'm stressed about everything",
        "Hello, can you help me?"
    ]
    
    for user_input in test_inputs:
        # Classify intent
        intent_result = classifier._keyword_classify(user_input)
        
        # Get chatbot response
        bot_response = chatbot.get_response(user_input)
        
        # Store in MongoDB
        storage.log_user_message(
            content=user_input,
            intent=intent_result.intent,
            confidence=intent_result.confidence
        )
        storage.log_bot_response(
            content=bot_response,
            metadata={"model": "aiml", "strategy": "pattern_match"}
        )
        
        # Display
        print(f"  You: {user_input}")
        print(f"  [Intent: {intent_result.intent} ({intent_result.confidence:.2f})]")
        print(f"  Bot: {bot_response}\n")
    
    # End session
    storage.end_session()
    print("Session saved with intent tags\n")
    
    # Show user profile
    profile = storage.get_user_profile(user_name)
    if profile:
        print(f"Updated Profile: {user_name}")
        print(f"  Sessions: {profile.total_sessions}")
        print(f"  Intents detected: {', '.join(profile.primary_intents)}\n")
    
    storage.shutdown()
    return True


def main():
    """Run all demos."""
    print_header()
    
    try:
        # Demo 1: Basic storage
        if not demo_basic_storage():
            print("\nNote: MongoDB must be running to use conversation storage")
            print("See docs/MONGODB_SETUP.md for installation instructions\n")
            return
        
        # Demo 2: Intent search
        demo_intent_search()
        
        # Demo 3: Analytics
        demo_analytics()
        
        # Demo 4: Integrated chatbot
        demo_with_chatbot()
        
        print("=" * 70)
        print("  ALL DEMOS COMPLETE")
        print("=" * 70)
        print("\nMongoDB Features Demonstrated:")
        print("  User identification and session management")
        print("  Intent classification and tagging")
        print("  Conversation history storage")
        print("  User profile tracking")
        print("  Intent-based search")
        print("  Analytics and statistics")
        print("\nNext Steps:")
        print("  - Explore MongoDB Compass to view stored data")
        print("  - Run: python scripts/demo_mongodb_queries.py")
        print("  - Check user profiles in the 'users' collection\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user\n")
    except Exception as e:
        print(f"\nDemo error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
