"""
AIML-based chatbot implementation.

Provides a rule-based conversational agent using AIML (Artificial Intelligence
Markup Language) for pattern matching and response generation.
"""

import logging
from pathlib import Path
from typing import Optional, List

import aiml

logger = logging.getLogger(__name__)


class AimlChatbot:
    """
    AIML-based chatbot for rule-based conversations.
    
    This chatbot uses pattern matching with AIML files to generate responses.
    Ideal for structured, domain-specific conversations like therapy or FAQs.
    
    Follows infrastructure layer principles:
    - Wraps external AIML library
    - Manages chatbot state and resources
    - Provides clean interface to application layer
    """
    
    def __init__(self, aiml_dir: Optional[Path] = None):
        """
        Initialize AIML chatbot.
        
        Args:
            aiml_dir: Directory containing AIML files (default: data/knowledge_bases/aiml/)
        """
        self.kernel = aiml.Kernel()
        self.aiml_dir = Path(aiml_dir) if aiml_dir else Path("data/knowledge_bases/aiml")
        self.loaded_files: List[str] = []
        self._is_initialized = False
        
        logger.info(f"AIML chatbot created with directory: {self.aiml_dir}")
    
    def load_aiml_files(self, file_pattern: str = "*.aiml") -> int:
        """
        Load AIML files from directory.
        
        Args:
            file_pattern: Glob pattern for AIML files (default: *.aiml)
            
        Returns:
            Number of files loaded
            
        Raises:
            ValueError: If AIML directory doesn't exist or no files found
        """
        if not self.aiml_dir.exists():
            raise ValueError(f"AIML directory not found: {self.aiml_dir}")
        
        aiml_files = list(self.aiml_dir.glob(file_pattern))
        
        if not aiml_files:
            raise ValueError(f"No AIML files found in {self.aiml_dir}")
        
        logger.info(f"Loading {len(aiml_files)} AIML files...")
        
        for aiml_file in aiml_files:
            try:
                self.kernel.learn(str(aiml_file))
                self.loaded_files.append(aiml_file.name)
                logger.debug(f"Loaded: {aiml_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {aiml_file.name}: {e}")
                raise ValueError(f"Failed to load AIML file {aiml_file.name}: {e}") from e
        
        self._is_initialized = True
        logger.info(f"Successfully loaded {len(self.loaded_files)} AIML files")
        return len(self.loaded_files)
    
    def load_specific_file(self, filename: str) -> None:
        """
        Load a specific AIML file.
        
        Args:
            filename: Name of AIML file to load
            
        Raises:
            ValueError: If file doesn't exist
        """
        file_path = self.aiml_dir / filename
        
        if not file_path.exists():
            raise ValueError(f"AIML file not found: {file_path}")
        
        try:
            self.kernel.learn(str(file_path))
            self.loaded_files.append(filename)
            self._is_initialized = True
            logger.info(f"Loaded AIML file: {filename}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            raise ValueError(f"Failed to load AIML file: {e}") from e
    
    def get_response(self, user_input: str) -> str:
        """
        Generate response to user input.
        
        Args:
            user_input: User's message
            
        Returns:
            Bot's response
            
        Raises:
            RuntimeError: If chatbot not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Chatbot not initialized. Load AIML files first.")
        
        if not user_input or not user_input.strip():
            logger.warning("Empty input received")
            return "I didn't quite catch that. Could you say something?"
        
        try:
            response = self.kernel.respond(user_input.strip())
            
            # Handle case where AIML has no matching pattern
            if not response:
                response = "I'm not sure how to respond to that. Could you rephrase?"
                logger.debug(f"No AIML match for: {user_input}")
            
            logger.debug(f"User: {user_input} | Bot: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble understanding. Please try again."
    
    def chat(self, user_input: str) -> str:
        """
        Alias for get_response for more natural API.
        
        Args:
            user_input: User's message
            
        Returns:
            Bot's response
        """
        return self.get_response(user_input)
    
    def reset(self) -> None:
        """
        Reset the chatbot state.
        
        Clears conversation history and resets the AIML kernel.
        """
        logger.info("Resetting chatbot state")
        self.kernel.resetBrain()
        self._is_initialized = False
        self.loaded_files.clear()
    
    def set_predicate(self, name: str, value: str) -> None:
        """
        Set a predicate (variable) in the AIML kernel.
        
        Useful for personalization (e.g., storing user's name).
        
        Args:
            name: Predicate name
            value: Predicate value
        """
        self.kernel.setPredicate(name, value)
        logger.debug(f"Set predicate: {name} = {value}")
    
    def get_predicate(self, name: str) -> str:
        """
        Get a predicate value from the AIML kernel.
        
        Args:
            name: Predicate name
            
        Returns:
            Predicate value (empty string if not set)
        """
        return self.kernel.getPredicate(name)
    
    def is_ready(self) -> bool:
        """
        Check if chatbot is ready to respond.
        
        Returns:
            True if initialized and ready
        """
        return self._is_initialized
    
    def get_loaded_files(self) -> List[str]:
        """
        Get list of loaded AIML files.
        
        Returns:
            List of filenames
        """
        return self.loaded_files.copy()
    
    def get_num_categories(self) -> int:
        """
        Get number of AIML categories loaded.
        
        Returns:
            Number of pattern-response pairs
        """
        return self.kernel.numCategories()
