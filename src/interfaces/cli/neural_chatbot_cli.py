"""
Command-line interface for the Mistral-7B conversational chatbot.

This CLI provides an interactive conversation interface for the Mistral-7B-based chatbot,
supporting high-quality natural conversations.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

from src.infrastructure.ml.chatbots.neural_chatbot import NeuralChatbot
from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings


class NeuralChatbotCLI:
    """
    Interactive CLI for neural conversational chatbot.
    
    Provides commands for conversation management, history viewing,
    and chatbot interaction.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the CLI with neural chatbot.
        
        Args:
            model_name: Model variant (overrides config if provided)
        """
        # Load settings from environment/config
        settings = NeuralChatbotSettings()
        
        # Initialize chatbot with settings
        self.chatbot = NeuralChatbot(
            settings=settings,
            model_name=model_name  # Override if provided
        )
        self.conversation_history: List[Tuple[str, str]] = []
        self.running = False
    
    def print_welcome(self) -> None:
        """Display welcome message and instructions."""
        print("\n" + "=" * 60)
        print("  Phi-2 Conversational Bot")
        print("=" * 60)
        print("\nWelcome! Let's have an intelligent conversation.")
        print("\nUsing Microsoft Phi-2 (2.7B parameters).")
        
        device = self.chatbot.settings.device
        vram_gb = self.chatbot.settings.vram_size_gb
        use_8bit = self.chatbot.settings.use_8bit_quantization
        
        # Display VRAM info only for GPU
        if device == "cuda" or (device == "auto" and self.chatbot.device == "cuda"):
            if use_8bit:
                print(f"Optimized for {vram_gb}GB VRAM with 8-bit quantization.")
            else:
                print(f"Running in FP16 mode ({vram_gb}GB VRAM).")
        
        print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for more commands.")
        print("\n" + "-" * 60 + "\n")
    
    def print_help(self) -> None:
        """Display available commands."""
        print("\nAvailable commands:")
        print("  help      - Show this help message")
        print("  reset     - Clear conversation history and start fresh")
        print("  history   - Show conversation history")
        print("  clear     - Clear the screen")
        print("  quit/exit - End the conversation")
        print()
    
    def process_command(self, user_input: str) -> bool:
        """
        Process special commands.
        
        Args:
            user_input: User's input
            
        Returns:
            True if command was processed, False if normal conversation
        """
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'bye']:
            print("\nBot: Goodbye! It was great talking with you!")
            return True
        
        if command == 'help':
            self.print_help()
            return False
        
        if command == 'reset':
            self.chatbot.reset()
            self.conversation_history = []
            print("\nConversation history cleared. Starting fresh!\n")
            return False
        
        if command == 'history':
            self.print_history()
            return False
        
        if command == 'clear':
            # Clear screen (cross-platform)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            self.print_welcome()
            return False
        
        return False
    
    def print_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            print("\nNo conversation history yet.\n")
            return
        
        print("\n" + "=" * 60)
        print("Conversation History")
        print("=" * 60)
        for i, (user_msg, bot_msg) in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] You: {user_msg}")
            print(f"    Bot: {bot_msg}")
        print("\n" + "=" * 60 + "\n")
    
    def run(self) -> None:
        """Run the interactive chat loop."""
        self.print_welcome()
        
        # Load model
        print("Loading Phi-2 conversational chatbot...")
        print(f"Model: {self.chatbot.model_name}")
        print(f"Settings: temp={self.chatbot.settings.temperature}, max_tokens={self.chatbot.settings.max_new_tokens}")
        
        try:
            self.chatbot.load_model()
            print("Model loaded successfully!\n")
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("Please check your configuration and ensure you have enough memory/disk space.")
            return
        
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                if self.process_command(user_input):
                    break
                
                # Get bot response
                print("Bot: Thinking...", end='\r')
                response = self.chatbot.get_response(user_input)
                
                # Get benchmark stats
                resp_time, tokens, tok_per_sec = self.chatbot.get_benchmark_stats()
                
                print(f"Bot: {response}")
                print(f"[Benchmark: {resp_time:.2f}s | {tokens} tokens | {tok_per_sec:.2f} tok/s]\n")
                
                # Save to history
                self.conversation_history.append((user_input, response))
                
            except KeyboardInterrupt:
                print("\n\nBot: Interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\nBot: End of input. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Type 'quit' to exit or continue chatting.\n")


def main():
    """Main entry point for the CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Neural Conversational Chatbot CLI (Phi-2)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model variant to use (overrides NEURAL_MODEL_NAME in .env)'
    )
    
    args = parser.parse_args()
    
    cli = NeuralChatbotCLI(model_name=args.model)
    cli.run()


if __name__ == "__main__":
    main()
