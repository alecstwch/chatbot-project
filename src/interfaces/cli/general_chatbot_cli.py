"""
Command-line interface for the general conversational chatbot using DialoGPT.

This CLI provides an interactive conversation interface for the DialoGPT-based chatbot,
supporting natural multi-turn conversations.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

from src.infrastructure.ml.chatbots.dialogpt_chatbot import DialoGPTChatbot
from src.infrastructure.config.chatbot_settings import DialoGPTSettings


class GeneralChatbotCLI:
    """
    Interactive CLI for DialoGPT conversational chatbot.
    
    Provides commands for conversation management, history viewing,
    and chatbot interaction.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the CLI with DialoGPT chatbot.
        
        Args:
            model_name: DialoGPT model variant (overrides config if provided)
        """
        # Load settings from environment/config
        settings = DialoGPTSettings()
        
        # Initialize chatbot with settings
        self.chatbot = DialoGPTChatbot(
            settings=settings,
            model_name=model_name  # Override if provided
        )
        self.conversation_history: List[Tuple[str, str]] = []
        self.running = False
    
    def print_welcome(self) -> None:
        """Display welcome message and instructions."""
        print("\n" + "=" * 60)
        print("  General Conversational Bot (DialoGPT)")
        print("=" * 60)
        print("\nWelcome! Let's have a friendly conversation.")
        print("\nNote: DialoGPT-small has limited conversational ability.")
        print("For better results, use DialoGPT-medium or -large:")
        print("  Set DIALOGPT_MODEL_NAME in .env file")
        print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for more commands.")
        print("\n" + "-" * 60 + "\n")
    
    def print_help(self) -> None:
        """Display available commands."""
        print("\nAvailable commands:")
        print("  help          - Show this help message")
        print("  reset         - Start a new conversation (clear history)")
        print("  history       - Show conversation history")
        print("  clear         - Clear the screen")
        print("  quit/exit/bye - End the conversation")
        print()
    
    def print_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            print("\nNo conversation history yet.\n")
            return
        
        print("\n" + "-" * 60)
        print("Conversation History:")
        print("-" * 60)
        for i, (user_msg, bot_msg) in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] You: {user_msg}")
            print(f"    Bot: {bot_msg}")
        print("-" * 60 + "\n")
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def reset_conversation(self) -> None:
        """Reset the chatbot conversation and history."""
        self.chatbot.reset()
        self.conversation_history.clear()
        print("\n[Conversation reset. Starting fresh!]\n")
    
    def process_command(self, user_input: str) -> bool:
        """
        Process special commands.
        
        Args:
            user_input: User's input to check for commands
            
        Returns:
            True if input was a command, False otherwise
        """
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            self.running = False
            print("\nBot: Goodbye! It was nice chatting with you!\n")
            return True
        
        if command == 'help':
            self.print_help()
            return True
        
        if command == 'reset':
            self.reset_conversation()
            return True
        
        if command == 'history':
            self.print_history()
            return True
        
        if command == 'clear':
            self.clear_screen()
            self.print_welcome()
            return True
        
        return False
    
    def run(self) -> None:
        """
        Run the interactive chatbot CLI.
        
        Main conversation loop that handles user input and bot responses.
        """
        self.print_welcome()
        
        # Load the model
        print("Loading DialoGPT conversational chatbot...")
        print(f"Model: {self.chatbot.model_name}")
        print(f"Settings: temp={self.chatbot.settings.temperature}, "
              f"max_tokens={self.chatbot.settings.max_new_tokens}")
        try:
            self.chatbot.load_model()
            print("Model loaded successfully!\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to load model: {e}")
            print("Please check your internet connection and try again.\n")
            return
        
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if self.process_command(user_input):
                    continue
                
                # Get bot response
                try:
                    import sys
                    print("Bot: ", end='', flush=True)
                    sys.stdout.write("Thinking...")
                    sys.stdout.flush()
                    
                    response = self.chatbot.get_response(user_input)
                    
                    # Clear "Thinking..." and print response
                    sys.stdout.write('\r' + ' ' * 20 + '\r')
                    sys.stdout.flush()
                    print(f"Bot: {response}\n")
                    
                    # Store in history
                    self.conversation_history.append((user_input, response))
                    
                except Exception as e:
                    print(f"\rBot: Sorry, I encountered an error: {e}\n")
            
            except KeyboardInterrupt:
                print("\n\n[Interrupted by user]\n")
                self.running = False
            
            except EOFError:
                print("\n")
                self.running = False


def main():
    """Main entry point for the general chatbot CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive DialoGPT conversational chatbot"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=[
            'microsoft/DialoGPT-small',
            'microsoft/DialoGPT-medium',
            'microsoft/DialoGPT-large'
        ],
        help='DialoGPT model variant to use (overrides DIALOGPT_MODEL_NAME in .env)'
    )
    
    args = parser.parse_args()
    
    cli = GeneralChatbotCLI(model_name=args.model)
    cli.run()


if __name__ == "__main__":
    main()
