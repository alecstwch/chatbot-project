"""
RAG-Enhanced Conversational Chatbot.

This module implements a chatbot with RAG (Retrieval-Augmented Generation)
capabilities using Qdrant for semantic memory and Gemini for generation.
"""

import logging
import time
import os
from typing import Optional, List, Tuple, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings
from src.infrastructure.config.qdrant_settings import QdrantSettings
from src.infrastructure.memory.rag_memory_service import RAGMemoryService
from src.infrastructure.memory.rag_prompt_builder import RAGPromptBuilder, TherapyRAGPromptBuilder

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    RAG-enhanced conversational chatbot.
    
    Combines Gemini API for generation with Qdrant vector database
    for semantic memory retrieval, enabling the chatbot to recall
    and use relevant context from past conversations.
    
    Attributes:
        model_name: Gemini model identifier
        user_id: Current user identifier for memory retrieval
        rag_memory: RAG memory service for semantic search
        prompt_builder: Builds context-augmented prompts
    """
    
    def __init__(
        self,
        settings: Optional[NeuralChatbotSettings] = None,
        qdrant_settings: Optional[QdrantSettings] = None,
        model_name: Optional[str] = None,
        user_id: str = "default_user",
        user_name: Optional[str] = None,
        use_therapy_mode: bool = False
    ):
        """
        Initialize RAG chatbot.
        
        Args:
            settings: Neural chatbot configuration settings
            qdrant_settings: Qdrant vector database settings
            model_name: Gemini model identifier (overrides settings)
            user_id: Unique user identifier for memory
            user_name: Display name for personalization
            use_therapy_mode: Use therapy-optimized prompts
        """
        # Load settings
        self.settings = settings or NeuralChatbotSettings()
        self.qdrant_settings = qdrant_settings or QdrantSettings()
        
        # Model configuration
        self.model_name = model_name or self.settings.model_name
        
        # User identification
        self.user_id = user_id
        self.user_name = user_name
        
        # Initialize components
        self.client: Optional[genai.Client] = None
        self.rag_memory: Optional[RAGMemoryService] = None
        
        # Choose prompt builder based on mode
        if use_therapy_mode:
            self.prompt_builder = TherapyRAGPromptBuilder(
                max_context_items=5,
                include_similarity_scores=False
            )
        else:
            self.prompt_builder = RAGPromptBuilder(
                max_context_items=5,
                include_similarity_scores=False
            )
        
        # Conversation state
        self.chat_history: List[types.Content] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.current_session_id: Optional[str] = None
        
        # State flags
        self._initialized = False
        self._rag_enabled = False
        
        # Metrics
        self.last_response_time: float = 0.0
        self.last_tokens_generated: int = 0
        self.last_tokens_per_sec: float = 0.0
        self.last_context_items: int = 0
        
        logger.info(f"RAG Chatbot initialized for user: {user_id}")
        logger.info(f"Model: {self.model_name}, Therapy mode: {use_therapy_mode}")
    
    def load_model(self) -> None:
        """
        Initialize the Gemini API client and RAG memory service.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize Gemini API
            logger.info(f"Initializing Gemini API with model: {self.model_name}")
            
            api_key = (
                self.settings.api_key or 
                os.environ.get("GEMINI_API_KEY") or 
                os.environ.get("GOOGLE_API_KEY") or
                os.environ.get("NEURAL_API_KEY")
            )
            
            if not api_key:
                raise RuntimeError(
                    "Gemini API key not found. Set GEMINI_API_KEY in your .env file."
                )
            
            self.client = genai.Client(api_key=api_key)
            self._initialized = True
            logger.info("Gemini API initialized successfully")
            
            # Initialize RAG memory service
            logger.info("Initializing RAG memory service...")
            self.rag_memory = RAGMemoryService(settings=self.qdrant_settings)
            
            if self.rag_memory.initialize():
                self._rag_enabled = True
                msg_count = self.rag_memory.get_user_message_count(self.user_id)
                logger.info(f"RAG memory enabled. User has {msg_count} stored messages.")
            else:
                logger.warning("RAG memory initialization failed. Running without memory.")
                self._rag_enabled = False
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG chatbot: {e}")
            raise RuntimeError(f"RAG chatbot initialization failed: {e}")
    
    def set_user(self, user_id: str, user_name: Optional[str] = None) -> None:
        """
        Set the current user for memory retrieval.
        
        Args:
            user_id: Unique user identifier
            user_name: Optional display name
        """
        self.user_id = user_id
        self.user_name = user_name
        self.reset()  # Clear conversation for new user
        
        if self._rag_enabled:
            msg_count = self.rag_memory.get_user_message_count(user_id)
            logger.info(f"Switched to user {user_id}. Memory contains {msg_count} messages.")
    
    def get_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        store_in_memory: bool = True
    ) -> str:
        """
        Generate a response using RAG-enhanced generation.
        
        Args:
            user_input: User's message
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            store_in_memory: Whether to store this exchange in memory
            
        Returns:
            Generated response string
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you say that again?"
        
        try:
            start_time = time.time()
            
            # Step 1: Retrieve relevant context from memory
            retrieved_context = []
            if self._rag_enabled:
                retrieved_context = self.rag_memory.retrieve_relevant_context(
                    user_id=self.user_id,
                    query=user_input,
                    top_k=5,
                    min_similarity=0.4
                )
                self.last_context_items = len(retrieved_context)
                logger.debug(f"Retrieved {len(retrieved_context)} relevant context items")
            
            # Step 2: Build RAG-augmented context
            rag_context = ""
            if retrieved_context:
                rag_context = self.prompt_builder.build_gemini_prompt(
                    current_message=user_input,
                    retrieved_context=retrieved_context,
                    user_name=self.user_name
                )
            
            # Step 3: Add user message to local history
            self.conversation_history.append({'user': user_input})
            
            # Trim history if needed
            if len(self.conversation_history) > self.settings.max_history_turns:
                self.conversation_history = self.conversation_history[-self.settings.max_history_turns:]
            
            # Step 4: Prepare messages for Gemini
            # If we have RAG context, prepend it as a system-like message
            messages_for_api = []
            
            if rag_context and not self.chat_history:
                # Add RAG context as first "user" message with instruction
                messages_for_api.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"[System Context]\n{rag_context}\n\n[End Context]\n\nPlease acknowledge you understand this context.")]
                    )
                )
                messages_for_api.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text="I understand the context from our past conversations. I'll use this information naturally in my responses.")]
                    )
                )
            
            # Add existing chat history
            messages_for_api.extend(self.chat_history)
            
            # Add current user message
            current_message_content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
            messages_for_api.append(current_message_content)
            
            # Step 5: Generate response
            gen_config = types.GenerateContentConfig(
                temperature=temperature or self.settings.temperature,
                top_p=top_p or self.settings.top_p,
                top_k=top_k or self.settings.top_k,
                max_output_tokens=self.settings.max_new_tokens,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages_for_api,
                config=gen_config,
            )
            
            response_text = response.text.strip()
            
            # Step 6: Update chat history
            self.chat_history.append(current_message_content)
            self.chat_history.append(
                types.Content(role="model", parts=[types.Part(text=response_text)])
            )
            
            # Step 7: Store in RAG memory
            if store_in_memory and self._rag_enabled:
                self.rag_memory.store_conversation_turn(
                    user_id=self.user_id,
                    user_message=user_input,
                    assistant_message=response_text,
                    session_id=self.current_session_id
                )
                logger.debug("Stored conversation turn in RAG memory")
            
            # Calculate metrics
            end_time = time.time()
            self.last_response_time = end_time - start_time
            
            # Estimate tokens
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.last_tokens_generated = getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                self.last_tokens_generated = len(response_text) // 4
            
            self.last_tokens_per_sec = (
                self.last_tokens_generated / self.last_response_time 
                if self.last_response_time > 0 else 0
            )
            
            # Update conversation history
            self.conversation_history[-1]['assistant'] = response_text
            
            logger.info(
                f"Response generated in {self.last_response_time:.2f}s "
                f"({self.last_tokens_generated} tokens, "
                f"{self.last_context_items} context items used)"
            )
            
            return response_text or "I'm not sure how to respond to that."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Cleanup failed state
            if self.conversation_history and 'assistant' not in self.conversation_history[-1]:
                self.conversation_history.pop()
            if self.chat_history and self.chat_history[-1].role == "user":
                self.chat_history.pop()
            return f"Sorry, I encountered an error: {str(e)}"
    
    def chat(self, user_input: str, **kwargs) -> str:
        """Alias for get_response()."""
        return self.get_response(user_input, **kwargs)
    
    def reset(self) -> None:
        """Reset conversation history (keeps memory intact)."""
        self.conversation_history = []
        self.chat_history = []
        self.current_session_id = None
        logger.debug("Conversation history reset (memory preserved)")
    
    def search_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search past conversations by topic.
        
        Args:
            query: Topic or phrase to search for
            top_k: Maximum results
            
        Returns:
            List of matching messages with metadata
        """
        if not self._rag_enabled:
            return []
        
        return self.rag_memory.search_by_topic(
            user_id=self.user_id,
            topic=query,
            top_k=top_k
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the user's memory."""
        if not self._rag_enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "user_id": self.user_id,
            "message_count": self.rag_memory.get_user_message_count(self.user_id),
            "collection": self.qdrant_settings.collection_name
        }
    
    def is_ready(self) -> bool:
        """Check if the chatbot is ready to use."""
        return self._initialized and self.client is not None
    
    def is_rag_enabled(self) -> bool:
        """Check if RAG memory is enabled."""
        return self._rag_enabled
    
    def get_conversation_length(self) -> int:
        """Get current conversation length in turns."""
        return len(self.conversation_history)
    
    def get_benchmark_stats(self) -> Tuple[float, int, float]:
        """Get performance statistics from last response."""
        return (self.last_response_time, self.last_tokens_generated, self.last_tokens_per_sec)
    
    def get_last_context_count(self) -> int:
        """Get number of context items used in last response."""
        return self.last_context_items
    
    def shutdown(self) -> None:
        """Shutdown the chatbot and cleanup resources."""
        if self.rag_memory:
            self.rag_memory.shutdown()
        self._initialized = False
        self._rag_enabled = False
        logger.info("RAG Chatbot shutdown complete")

