#!/usr/bin/env python3
"""
Demo script for RAG-enhanced chatbot.

This script demonstrates the RAG (Retrieval-Augmented Generation) capabilities
of the chatbot, showing how it retrieves and uses context from past conversations.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


def demo_rag_memory():
    """Demonstrate RAG memory service independently."""
    print("=" * 60)
    print("RAG Memory Service Demo")
    print("=" * 60)
    
    from src.infrastructure.memory.rag_memory_service import RAGMemoryService
    from src.infrastructure.config.qdrant_settings import QdrantSettings
    
    # Use local storage for demo
    settings = QdrantSettings()
    settings.use_local_storage = True
    settings.local_path = "./data/qdrant_demo"
    
    # Initialize service
    print("\n1. Initializing RAG memory service...")
    memory = RAGMemoryService(settings=settings)
    
    if not memory.initialize():
        print("❌ Failed to initialize. Is Qdrant running?")
        return False
    
    print("✅ RAG memory service initialized")
    
    # Store some demo messages
    print("\n2. Storing demo conversation history...")
    
    demo_user = "demo_user"
    
    messages = [
        ("My name is Alex and I'm 28 years old", "user"),
        ("Nice to meet you, Alex! How can I help you today?", "assistant"),
        ("I've been struggling with anxiety lately", "user"),
        ("I'm sorry to hear that. Can you tell me more about what's been causing your anxiety?", "assistant"),
        ("Work has been really stressful, and I had depression last year", "user"),
        ("That sounds challenging. Given your history with depression, it's important to address these feelings.", "assistant"),
        ("I also have trouble sleeping at night", "user"),
        ("Sleep issues often accompany stress and anxiety. Have you tried any relaxation techniques?", "assistant"),
    ]
    
    for text, role in messages:
        memory.store_message(
            user_id=demo_user,
            message=text,
            role=role,
            session_id="demo_session"
        )
        print(f"  Stored: [{role}] {text[:50]}...")
    
    print(f"\n✅ Stored {len(messages)} messages")
    
    # Search by semantic similarity
    print("\n3. Testing semantic search...")
    
    queries = [
        "feeling anxious today",
        "what is my name",
        "mental health issues",
        "problems with sleep"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = memory.retrieve_relevant_context(
            user_id=demo_user,
            query=query,
            top_k=2
        )
        
        for r in results:
            print(f"   [{r['similarity']:.0%}] {r['content'][:60]}...")
    
    # Show message count
    count = memory.get_user_message_count(demo_user)
    print(f"\n📊 Total messages for user: {count}")
    
    # Cleanup
    memory.shutdown()
    print("\n✅ Demo complete!")
    return True


def demo_rag_chatbot():
    """Demonstrate full RAG chatbot."""
    print("\n" + "=" * 60)
    print("RAG Chatbot Demo")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  No Gemini API key found.")
        print("Set GEMINI_API_KEY in your .env file to test the full chatbot.")
        return False
    
    from src.infrastructure.ml.chatbots.rag_chatbot import RAGChatbot
    from src.infrastructure.config.qdrant_settings import QdrantSettings
    
    # Configure for local storage
    qdrant_settings = QdrantSettings()
    qdrant_settings.use_local_storage = True
    qdrant_settings.local_path = "./data/qdrant_demo"
    
    print("\n1. Initializing RAG chatbot...")
    
    chatbot = RAGChatbot(
        qdrant_settings=qdrant_settings,
        user_id="demo_user",
        user_name="Alex",
        use_therapy_mode=True
    )
    
    try:
        chatbot.load_model()
        print("✅ RAG chatbot ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Show memory stats
    stats = chatbot.get_memory_stats()
    print(f"\n📚 Memory contains {stats.get('message_count', 0)} messages")
    
    # Test a query that should use past context
    print("\n2. Testing context-aware response...")
    print("   Query: 'I'm feeling stressed today'")
    print("   (This should reference past conversations about anxiety/depression)\n")
    
    response = chatbot.get_response("I'm feeling stressed today")
    
    print(f"🤖 Response: {response}")
    print(f"\n📊 Used {chatbot.get_last_context_count()} context items from memory")
    
    # Cleanup
    chatbot.shutdown()
    print("\n✅ RAG chatbot demo complete!")
    return True


def main():
    """Run all demos."""
    print("\n" + "🧠" * 30)
    print("\n  RAG (Retrieval-Augmented Generation) Demo")
    print("\n" + "🧠" * 30)
    
    # Demo 1: RAG Memory Service
    demo_rag_memory()
    
    # Demo 2: Full RAG Chatbot
    demo_rag_chatbot()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("""
1. Run the interactive RAG chatbot:
   python -m src.interfaces.cli.rag_chatbot_cli --user-id myname

2. With therapy mode:
   python -m src.interfaces.cli.rag_chatbot_cli --therapy --user-name "Alex"

3. With Qdrant server (Podman):
   podman run -d -p 6333:6333 --name qdrant qdrant/qdrant
   python -m src.interfaces.cli.rag_chatbot_cli --qdrant-server
""")


if __name__ == "__main__":
    main()

