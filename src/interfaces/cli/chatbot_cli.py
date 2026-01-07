"""
CLI interface for AIML therapy chatbot.

Provides a command-line interface for interacting with the
psychotherapy AIML chatbot.
"""

import sys
from pathlib import Path
from typing import Optional

from src.infrastructure.ml.chatbots.aiml_chatbot import AimlChatbot


class ChatbotCLI:
    """
    Command-line interface for chatbot interaction.
    
    Provides a simple REPL (Read-Eval-Print Loop) for chatting
    with the AIML-based chatbot.
    """
    
    def __init__(self, chatbot: AimlChatbot, chatbot_name: str = "Therapy Bot"):
        """
        Initialize CLI.
        
        Args:
            chatbot: Initialized AimlChatbot instance
            chatbot_name: Display name for the chatbot
        """
        self.chatbot = chatbot
        self.chatbot_name = chatbot_name
        self.conversation_history = []
    
    def print_welcome(self) -> None:
        """Print welcome message."""
        print("\n" + "="*60)
        print(f"  {self.chatbot_name}")
        print("="*60)
        print("\nWelcome! I'm here to listen and help.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for more commands.")
        print("\n" + "-"*60 + "\n")
    
    def print_help(self) -> None:
        """Print help information."""
        print("\nAvailable commands:")
        print("  quit, exit, bye  - End the conversation")
        print("  help             - Show this help message")
        print("  reset            - Reset the conversation")
        print("  history          - Show conversation history")
        print("  clear            - Clear the screen")
        print()
    
    def print_history(self) -> None:
        """Print conversation history."""
        if not self.conversation_history:
            print("\nNo conversation history yet.\n")
            return
        
        print("\n" + "="*60)
        print("  Conversation History")
        print("="*60 + "\n")
        
        for i, (user_msg, bot_msg) in enumerate(self.conversation_history, 1):
            print(f"[{i}] You: {user_msg}")
            print(f"    Bot: {bot_msg}\n")
    
    def clear_screen(self) -> None:
        """Clear terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def reset_conversation(self) -> None:
        """Reset chatbot and conversation history."""
        self.chatbot.reset()
        self.chatbot.load_aiml_files()
        self.conversation_history.clear()
        print("\n[OK] Conversation reset successfully.\n")
    
    def run(self) -> None:
        """
        Run the interactive chat loop.
        
        Continuously prompts for user input and displays bot responses
        until user decides to quit.
        """
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"\n{self.chatbot_name}: Goodbye! Take care.\n")
                    break
                
                if user_input.lower() == 'help':
                    self.print_help()
                    continue
                
                if user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue
                
                if user_input.lower() == 'history':
                    self.print_history()
                    continue
                
                if user_input.lower() == 'clear':
                    self.clear_screen()
                    self.print_welcome()
                    continue
                
                # Get bot response
                response = self.chatbot.chat(user_input)
                
                # Store in history
                self.conversation_history.append((user_input, response))
                
                # Display response
                print(f"{self.chatbot_name}: {response}\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{self.chatbot_name}: Goodbye! Take care.\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """
    Main entry point for therapy chatbot CLI.
    
    Initializes the AIML chatbot and starts the CLI interface.
    """
    try:
        # Initialize chatbot
        print("Loading therapy chatbot...")
        chatbot = AimlChatbot()
        chatbot.load_aiml_files()
        
        # Start CLI
        cli = ChatbotCLI(chatbot, chatbot_name="Therapy Bot")
        cli.run()
        
    except ValueError as e:
        print(f"\nError initializing chatbot: {e}")
        print("Make sure AIML files are in data/knowledge_bases/aiml/")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
