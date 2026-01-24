"""
Command-line interface for the RAG-enhanced conversational chatbot.

This CLI provides an interactive conversation interface with semantic memory,
allowing the chatbot to recall and use context from past conversations.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional

from src.infrastructure.ml.chatbots.rag_chatbot import RAGChatbot
from src.infrastructure.ml.chatbots.enhanced_rag_chatbot import EnhancedRAGChatbot
from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings
from src.infrastructure.config.qdrant_settings import QdrantSettings
from src.infrastructure.config.mongodb_settings import MongoDBSettings


class RAGChatbotCLI:
    """
    Interactive CLI for RAG-enhanced conversational chatbot.
    
    Provides commands for conversation management, memory search,
    and context-aware chatbot interaction.
    """
    
    def __init__(
        self,
        user_id: str = "default_user",
        user_name: Optional[str] = None,
        use_therapy_mode: bool = False,
        use_local_qdrant: bool = True,
        use_enhanced_mode: bool = True
    ):
        """
        Initialize the CLI with RAG chatbot.

        Args:
            user_id: Unique user identifier for memory
            user_name: Display name for personalization
            use_therapy_mode: Use therapy-optimized prompts
            use_local_qdrant: Use local file storage instead of server
            use_enhanced_mode: Use enhanced mode with MongoDB and emotion detection
        """
        # Load settings
        settings = NeuralChatbotSettings()

        qdrant_settings = QdrantSettings()
        if use_local_qdrant:
            qdrant_settings.use_local_storage = True
            qdrant_settings.local_path = "./data/qdrant_db"

        # Initialize chatbot based on mode
        self.use_enhanced_mode = use_enhanced_mode
        self.user_id = user_id
        self.user_name = user_name
        self.use_therapy_mode = use_therapy_mode
        self.conversation_history: List[Tuple[str, str]] = []
        self.running = False

        if use_enhanced_mode:
            # Enhanced mode with MongoDB and emotion detection
            mongodb_settings = MongoDBSettings()
            self.chatbot = EnhancedRAGChatbot(
                settings=settings,
                qdrant_settings=qdrant_settings,
                mongodb_settings=mongodb_settings,
                user_id=user_id,
                user_name=user_name,
                use_therapy_mode=use_therapy_mode
            )
        else:
            # Standard RAG mode
            self.chatbot = RAGChatbot(
                settings=settings,
                qdrant_settings=qdrant_settings,
                user_id=user_id,
                user_name=user_name,
                use_therapy_mode=use_therapy_mode
            )
    
    def print_welcome(self) -> None:
        """Display welcome message and instructions."""
        print("\n" + "=" * 70)
        if self.use_enhanced_mode:
            print("  🧠 Enhanced RAG Chatbot (Patient Data + Emotion Tracking)")
        else:
            print("  🧠 RAG-Enhanced Conversational Bot")
        print("=" * 70)

        mode = "Therapy Support" if self.use_therapy_mode else "General Conversation"
        print(f"\nMode: {mode}")
        print(f"User: {self.user_name or self.user_id}")

        if self.use_enhanced_mode:
            print(f"✨ Enhanced Mode: Emotion detection + MongoDB patient files")
            if hasattr(self.chatbot, 'is_mongodb_enabled') and self.chatbot.is_mongodb_enabled():
                profile = self.chatbot.get_patient_profile()
                if profile:
                    print(f"👤 Patient Profile: Loaded")
                    if profile.get('risk_level'):
                        risk_emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                        print(f"   Risk Level: {risk_emojis.get(profile['risk_level'], '⚪')} {profile['risk_level'].upper()}")
                    if profile.get('total_conversations', 0) > 0:
                        print(f"   Previous Sessions: {profile['total_conversations']}")

        if self.chatbot.is_rag_enabled():
            stats = self.chatbot.get_memory_stats()
            print(f"📚 Memory: {stats.get('message_count', 0)} stored messages")
        else:
            print("⚠️  Memory: Disabled (Qdrant not connected)")

        print("\n" + "-" * 70)
        if self.use_enhanced_mode:
            print("This chatbot provides:")
            print("  • Emotion detection and tracking")
            print("  • Behavior pattern analysis")
            print("  • Personalized responses using past context")
            print("  • Patient profile management")
        else:
            print("This chatbot remembers your past conversations and uses them")
            print("to provide personalized, context-aware responses.")
        print("-" * 70)

        print("\nCommands:")
        print("  help     - Show all commands")
        print("  search   - Search past conversations")
        if self.use_enhanced_mode:
            print("  emotions - View emotion history")
            print("  profile  - View patient profile")
        print("  memory   - View memory statistics")
        print("  quit     - Exit the chatbot")
        print("\n" + "-" * 70 + "\n")
    
    def print_help(self) -> None:
        """Display available commands."""
        print("\n" + "=" * 50)
        print("Available Commands")
        print("=" * 50)
        print("  help      - Show this help message")
        print("  reset     - Clear current conversation (keeps memory)")
        print("  history   - Show current conversation history")
        print("  search    - Search past conversations by topic")
        print("  data      - View your stored conversations")
        print("  alldata   - View ALL stored data (all users)")
        print("  memory    - View memory statistics")
        print("  forget    - Delete all your stored data")
        print("  clear     - Clear the screen")
        print("  quit/exit - End the conversation")
        print("=" * 50 + "\n")
    
    def process_command(self, user_input: str) -> bool:
        """
        Process special commands.
        
        Args:
            user_input: User's input
            
        Returns:
            True if should exit, False otherwise
        """
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            print("\n🤖 Goodbye! Your memories are saved for next time!")
            return True
        
        if command == 'help':
            self.print_help()
            return False
        
        if command == 'reset':
            self.chatbot.reset()
            self.conversation_history = []
            print("\n✅ Conversation cleared. Memory preserved.\n")
            return False
        
        if command == 'history':
            self.print_history()
            return False
        
        if command == 'search':
            self.search_memory()
            return False
        
        if command == 'memory':
            self.show_memory_stats()
            return False
        
        if command == 'data':
            self.show_stored_data()
            return False
        
        if command == 'alldata':
            self.show_all_data()
            return False
        
        if command == 'forget':
            self.forget_user_data()
            return False

        if command == 'emotions':
            if self.use_enhanced_mode:
                self.show_emotion_summary()
            return False

        if command == 'profile':
            if self.use_enhanced_mode:
                self.show_patient_profile()
            return False

        if command == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            self.print_welcome()
            return False

        return False
    
    def print_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.\n")
            return
        
        print("\n" + "=" * 50)
        print("Current Conversation")
        print("=" * 50)
        for i, (user_msg, bot_msg) in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] You: {user_msg}")
            print(f"    Bot: {bot_msg}")
        print("\n" + "=" * 50 + "\n")
    
    def search_memory(self) -> None:
        """Interactive memory search."""
        if not self.chatbot.is_rag_enabled():
            print("\n⚠️  Memory is not enabled.\n")
            return
        
        print("\n🔍 Search your conversation history")
        query = input("Enter topic to search (e.g., 'anxiety', 'work stress'): ").strip()
        
        if not query:
            print("Search cancelled.\n")
            return
        
        results = self.chatbot.search_memory(query, top_k=5)
        
        if not results:
            print(f"\n📭 No results found for '{query}'.\n")
            return
        
        print(f"\n📚 Found {len(results)} relevant messages:\n")
        for i, item in enumerate(results, 1):
            role = "You" if item.get("role") == "user" else "Bot"
            content = item.get("content", "")[:100]
            similarity = item.get("similarity", 0)
            print(f"  {i}. [{role}] ({similarity:.0%} match): \"{content}...\"")
        print()
    
    def show_memory_stats(self) -> None:
        """Display memory statistics."""
        stats = self.chatbot.get_memory_stats()
        
        print("\n" + "=" * 50)
        print("Memory Statistics")
        print("=" * 50)
        
        if stats.get("enabled"):
            print(f"  Status: ✅ Enabled")
            print(f"  User ID: {stats.get('user_id')}")
            print(f"  Stored Messages: {stats.get('message_count', 0)}")
            print(f"  Collection: {stats.get('collection')}")
        else:
            print("  Status: ❌ Disabled")
            print("  Reason: Qdrant not connected")
        
        print("=" * 50 + "\n")
    
    def show_stored_data(self) -> None:
        """Display all stored conversations in a readable format."""
        if not self.chatbot.is_rag_enabled():
            print("\n⚠️  Memory is not enabled.\n")
            return
        
        messages = self.chatbot.rag_memory.get_all_user_messages(self.user_id, limit=100)
        
        if not messages:
            print("\n📭 No stored conversations found.\n")
            return
        
        print("\n" + "=" * 70)
        print(f"📚 Stored Conversations for: {self.user_id}")
        print(f"   Total messages: {len(messages)}")
        print("=" * 70)
        
        # Group by session
        sessions = {}
        for msg in messages:
            session_id = msg.get("session_id", "unknown")[:8]
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(msg)
        
        for session_id, session_messages in sessions.items():
            print(f"\n┌─ Session: {session_id}... ({len(session_messages)} messages)")
            print("│")
            
            for msg in session_messages:
                role = msg.get("role", "user")
                text = msg.get("text", "")
                timestamp = msg.get("timestamp", "")
                
                # Format timestamp (show only date and time)
                if timestamp:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        time_str = dt.strftime("%m/%d %H:%M")
                    except:
                        time_str = ""
                else:
                    time_str = ""
                
                # Truncate long messages
                if len(text) > 60:
                    text = text[:57] + "..."
                
                # Format role
                if role == "user":
                    role_icon = "👤 You"
                else:
                    role_icon = "🤖 Bot"
                
                print(f"│  [{time_str}] {role_icon}: {text}")
                
                # Show vector info
                vector_dim = msg.get("vector_dim", 0)
                vector_preview = msg.get("vector_preview", [])
                if vector_preview:
                    preview_str = ", ".join(f"{v:.3f}" for v in vector_preview[:3])
                    print(f"│     📊 Vector: [{preview_str}, ...] ({vector_dim} dims)")
            
            print("│")
            print("└" + "─" * 50)
        
        print("\n" + "=" * 70)
        print("💡 Tip: Use 'search' to find specific topics")
        print("=" * 70 + "\n")
    
    def show_all_data(self) -> None:
        """Display ALL stored conversations from all users (admin/debug)."""
        if not self.chatbot.is_rag_enabled():
            print("\n⚠️  Memory is not enabled.\n")
            return
        
        messages = self.chatbot.rag_memory.get_all_messages(limit=200)
        
        if not messages:
            print("\n📭 No stored data found in Qdrant.\n")
            return
        
        # Count unique users
        users = set(msg.get("user_id", "unknown") for msg in messages)
        
        print("\n" + "=" * 70)
        print(f"📚 ALL Data in Qdrant (Admin View)")
        print(f"   Total messages: {len(messages)}")
        print(f"   Total users: {len(users)}")
        print("=" * 70)
        
        # Group by user, then by session
        user_messages = {}
        for msg in messages:
            user_id = msg.get("user_id", "unknown")
            if user_id not in user_messages:
                user_messages[user_id] = {}
            
            session_id = msg.get("session_id", "unknown")[:8]
            if session_id not in user_messages[user_id]:
                user_messages[user_id][session_id] = []
            user_messages[user_id][session_id].append(msg)
        
        for user_id, sessions in user_messages.items():
            total_user_msgs = sum(len(msgs) for msgs in sessions.values())
            print(f"\n{'─' * 70}")
            print(f"👤 USER: {user_id} ({total_user_msgs} messages)")
            print(f"{'─' * 70}")
            
            for session_id, session_messages in sessions.items():
                print(f"\n  ┌─ Session: {session_id}... ({len(session_messages)} messages)")
                print("  │")
                
                for msg in session_messages:
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    timestamp = msg.get("timestamp", "")
                    
                    # Format timestamp
                    if timestamp:
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            time_str = dt.strftime("%m/%d %H:%M")
                        except:
                            time_str = ""
                    else:
                        time_str = ""
                    
                    # Truncate long messages
                    if len(text) > 50:
                        text = text[:47] + "..."
                    
                    # Format role
                    role_icon = "💬" if role == "user" else "🤖"
                    
                    print(f"  │  [{time_str}] {role_icon} {role}: {text}")
                    
                    # Show vector info
                    vector_dim = msg.get("vector_dim", 0)
                    vector_preview = msg.get("vector_preview", [])
                    if vector_preview:
                        preview_str = ", ".join(f"{v:.3f}" for v in vector_preview[:3])
                        print(f"  │     📊 Vector: [{preview_str}, ...] ({vector_dim} dims)")
                
                print("  │")
                print("  └" + "─" * 50)
        
        print("\n" + "=" * 70)
        print("💡 This shows ALL data from ALL users in Qdrant")
        print("=" * 70 + "\n")
    
    def forget_user_data(self) -> None:
        """Delete all user data from memory."""
        if not self.chatbot.is_rag_enabled():
            print("\n⚠️  Memory is not enabled.\n")
            return
        
        confirm = input("\n⚠️  This will delete ALL your stored conversations. Type 'DELETE' to confirm: ")
        
        if confirm == "DELETE":
            if self.chatbot.rag_memory.delete_user_data(self.user_id):
                print("✅ All your data has been deleted.\n")
            else:
                print("❌ Failed to delete data.\n")
        else:
            print("Cancelled.\n")

    def show_emotion_summary(self) -> None:
        """Display emotion summary for enhanced mode."""
        if not self.use_enhanced_mode or not hasattr(self.chatbot, 'get_emotion_summary'):
            print("\n⚠️  Emotion tracking not available in this mode.\n")
            return

        summary = self.chatbot.get_emotion_summary()

        if not summary:
            print("\n📭 No emotion data available yet.\n")
            return

        print("\n" + "=" * 50)
        print("Emotion Summary")
        print("=" * 50)

        if summary.get("most_common_emotion"):
            print(f"Most Common Emotion: {summary['most_common_emotion']}")

        if summary.get("emotion_distribution"):
            print("\nEmotion Distribution:")
            for emotion, count in summary["emotion_distribution"].items():
                print(f"  - {emotion}: {count} occurrences")

        if summary.get("intensity_distribution"):
            print("\nIntensity Distribution:")
            for intensity, count in summary["intensity_distribution"].items():
                print(f"  - {intensity}: {count} occurrences")

        if summary.get("sentiment_distribution"):
            sentiment = summary["sentiment_distribution"]
            print(f"\nSentiment: {sentiment.get('positive', 0)} positive, "
                  f"{sentiment.get('negative', 0)} negative, "
                  f"{sentiment.get('neutral', 0)} neutral")

        print(f"\nTotal Messages Analyzed: {summary.get('total_messages_analyzed', 0)}")
        print("=" * 50 + "\n")

    def show_patient_profile(self) -> None:
        """Display patient profile information."""
        if not self.use_enhanced_mode or not hasattr(self.chatbot, 'get_patient_profile'):
            print("\n⚠️  Patient profile not available in this mode.\n")
            return

        profile = self.chatbot.get_patient_profile()

        if not profile:
            print("\n📭 No patient profile found.\n")
            return

        print("\n" + "=" * 50)
        print("Patient Profile")
        print("=" * 50)

        if profile.get("name"):
            print(f"Name: {profile['name']}")
        print(f"User ID: {profile['user_id']}")

        if profile.get("age"):
            print(f"Age: {profile['age']}")

        if profile.get("known_conditions"):
            print(f"\nKnown Conditions:")
            for condition in profile['known_conditions']:
                print(f"  - {condition}")

        if profile.get("medications"):
            print(f"\nMedications:")
            for med in profile['medications']:
                print(f"  - {med}")

        if profile.get("triggers"):
            print(f"\nKnown Triggers:")
            for trigger in profile['triggers']:
                print(f"  - {trigger}")

        if profile.get("behavior_patterns"):
            print(f"\nDetected Behavior Patterns:")
            for pattern in profile['behavior_patterns']:
                pattern_type = pattern.get("pattern_type", "unknown")
                severity = pattern.get("severity", "unknown")
                frequency = pattern.get("frequency", 0)
                print(f"  - {pattern_type}: {severity} severity ({frequency} occurrences)")

        risk_level = profile.get("risk_level", "low")
        risk_emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        print(f"\nRisk Level: {risk_emojis.get(risk_level, '⚪')} {risk_level.upper()}")

        if profile.get("total_conversations", 0) > 0:
            print(f"Total Sessions: {profile['total_conversations']}")

        if profile.get("treatment_goals"):
            print(f"\nTreatment Goals:")
            for goal in profile['treatment_goals']:
                print(f"  - {goal}")

        print("=" * 50 + "\n")

    def run(self) -> None:
        """Run the interactive chat loop."""
        self.print_welcome()
        
        # Load model
        print("🔄 Loading RAG chatbot...")
        print(f"   Model: {self.chatbot.model_name}")
        
        try:
            self.chatbot.load_model()
            print("✅ Chatbot ready!\n")
        except Exception as e:
            print(f"\n❌ Error loading chatbot: {e}")
            print("Please check your configuration and API keys.")
            return
        
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                commands = ['quit', 'exit', 'bye', 'help', 'reset',
                           'history', 'search', 'memory', 'data', 'alldata', 'forget', 'clear']
                if self.use_enhanced_mode:
                    commands.extend(['emotions', 'profile'])

                if user_input.lower() in commands:
                    if self.process_command(user_input):
                        break
                    continue

                # Get bot response
                print("🤖 Thinking...", end='\r')

                if self.use_enhanced_mode:
                    # Enhanced mode returns structured JSON
                    response_data = self.chatbot.get_response(user_input)

                    # Get stats
                    resp_time, tokens, tok_per_sec = self.chatbot.get_benchmark_stats()
                    context_count = self.chatbot.get_last_context_count()

                    # Display response
                    response_text = response_data.get("response", "")
                    next_question = response_data.get("next_question", "")
                    behavior_update = response_data.get("behavior_update")
                    emotion_update = response_data.get("emotion_update")
                    risk_assessment = response_data.get("risk_assessment", {})

                    print(f"\nBot: {response_text}")

                    # Display next question
                    if next_question:
                        print(f"\n💬 {next_question}")

                    # Display emotion info
                    if emotion_update:
                        emotion = emotion_update.get("primary_emotion", "neutral")
                        intensity = emotion_update.get("intensity", "low")
                        print(f"\n📊 Detected Emotion: {emotion} ({intensity} intensity)")

                    # Display behavior pattern update
                    if behavior_update:
                        pattern_type = behavior_update.get("pattern_type", "")
                        severity = behavior_update.get("severity", "")
                        print(f"\n⚠️  Pattern Detected: {pattern_type} ({severity} severity)")

                    # Display risk assessment
                    if risk_assessment:
                        level = risk_assessment.get("level", "low")
                        risk_emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                        emoji = risk_emojis.get(level, "⚪")
                        print(f"\n{emoji} Risk Assessment: {level.upper()}")
                        if risk_assessment.get("reasoning"):
                            print(f"   Reasoning: {risk_assessment['reasoning']}")

                    # Display metrics
                    memory_indicator = f"📚 {context_count}" if context_count > 0 else "📝 0"
                    print(f"\n[{resp_time:.2f}s | {tokens} tokens | {memory_indicator} memories]\n")

                    # Save to local history
                    self.conversation_history.append((user_input, response_text))

                else:
                    # Standard mode
                    response = self.chatbot.get_response(user_input)

                    # Get stats
                    resp_time, tokens, tok_per_sec = self.chatbot.get_benchmark_stats()
                    context_count = self.chatbot.get_last_context_count()

                    # Display response
                    print(f"Bot: {response}")

                    # Display metrics
                    memory_indicator = f"📚 {context_count}" if context_count > 0 else "📝 0"
                    print(f"[{resp_time:.2f}s | {tokens} tokens | {memory_indicator} memories used]\n")

                    # Save to local history
                    self.conversation_history.append((user_input, response))
                
            except KeyboardInterrupt:
                print("\n\n🤖 Interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n🤖 End of input. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Type 'quit' to exit or continue chatting.\n")
        
        # Cleanup
        self.chatbot.shutdown()


def main():
    """Main entry point for the CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RAG-Enhanced Conversational Chatbot CLI'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        default="default_user",
        help='Unique user identifier for memory persistence'
    )
    parser.add_argument(
        '--user-name',
        type=str,
        default=None,
        help='Display name for personalization'
    )
    parser.add_argument(
        '--therapy',
        action='store_true',
        help='Enable therapy/mental health support mode'
    )
    parser.add_argument(
        '--qdrant-server',
        action='store_true',
        help='Use Qdrant server instead of local storage'
    )
    parser.add_argument(
        '--qdrant-host',
        type=str,
        default='localhost',
        help='Qdrant server host (default: localhost)'
    )
    parser.add_argument(
        '--qdrant-port',
        type=int,
        default=6333,
        help='Qdrant server port (default: 6333)'
    )
    parser.add_argument(
        '--enhanced',
        action='store_true',
        default=True,
        help='Use enhanced mode with MongoDB and emotion detection (default: True)'
    )
    parser.add_argument(
        '--standard',
        action='store_true',
        help='Use standard RAG mode without MongoDB and emotion detection'
    )

    args = parser.parse_args()

    # Set Qdrant environment variables if using server
    if args.qdrant_server:
        os.environ['QDRANT_USE_LOCAL_STORAGE'] = 'false'
        os.environ['QDRANT_HOST'] = args.qdrant_host
        os.environ['QDRANT_PORT'] = str(args.qdrant_port)

    # Determine mode: enhanced (default) or standard
    use_enhanced = not args.standard  # Enhanced is default, use --standard to disable

    cli = RAGChatbotCLI(
        user_id=args.user_id,
        user_name=args.user_name,
        use_therapy_mode=args.therapy,
        use_local_qdrant=not args.qdrant_server,
        use_enhanced_mode=use_enhanced
    )
    cli.run()


if __name__ == "__main__":
    main()

