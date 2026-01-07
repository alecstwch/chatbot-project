"""
Conversation storage service.

Domain service for managing conversation persistence with MongoDB.
"""

import logging
import uuid
from typing import Optional
from datetime import datetime

from src.domain.models.conversation import (
    ConversationSession,
    ConversationMessage,
    MessageRole,
    UserProfile
)
from src.infrastructure.database.conversation_repository import ConversationRepository
from src.infrastructure.config.mongodb_settings import MongoDBSettings

logger = logging.getLogger(__name__)


class ConversationStorageService:
    """
    Service for storing and retrieving conversations.
    
    This service provides a high-level API for chatbot components
    to persist conversations without knowing about MongoDB details.
    """
    
    def __init__(
        self,
        repository: Optional[ConversationRepository] = None,
        settings: Optional[MongoDBSettings] = None
    ):
        """
        Initialize conversation storage service.
        
        Args:
            repository: MongoDB repository (creates one if not provided)
            settings: MongoDB settings
        """
        self.settings = settings or MongoDBSettings()
        self.repository = repository or ConversationRepository(self.settings)
        self._current_session: Optional[ConversationSession] = None
    
    def initialize(self) -> bool:
        """
        Initialize the storage service and connect to MongoDB.
        
        Returns:
            True if initialization successful
        """
        success = self.repository.connect()
        if success:
            logger.info("Conversation storage service initialized")
        else:
            logger.error("Failed to initialize conversation storage service")
        return success
    
    def shutdown(self) -> None:
        """Shutdown the storage service."""
        self.repository.disconnect()
        logger.info("Conversation storage service shutdown")
    
    def start_session(
        self,
        user_name: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_name: Unique identifier for the user
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        self._current_session = ConversationSession(
            session_id=session_id,
            user_name=user_name,
            metadata=metadata or {}
        )
        
        logger.info(f"Started session {session_id} for user {user_name}")
        return session_id
    
    def end_session(self) -> bool:
        """
        End the current conversation session and save to database.
        
        Returns:
            True if saved successfully
        """
        if not self._current_session:
            logger.warning("No active session to end")
            return False
        
        # Mark session as ended
        self._current_session.end_session()
        
        # Save session
        success = self.repository.save_session(self._current_session)
        
        if success:
            # Update user statistics
            self.repository.update_user_stats(
                self._current_session.user_name,
                self._current_session
            )
            logger.info(f"Ended session {self._current_session.session_id}")
        
        self._current_session = None
        return success
    
    def log_user_message(
        self,
        content: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Log a user message in the current session.
        
        Args:
            content: Message text
            intent: Detected intent
            confidence: Intent confidence score
            metadata: Additional message metadata
            
        Returns:
            True if logged successfully
        """
        if not self._current_session:
            logger.warning("No active session - cannot log message")
            return False
        
        message = ConversationMessage(
            role=MessageRole.USER,
            content=content,
            intent=intent,
            intent_confidence=confidence,
            metadata=metadata or {}
        )
        
        self._current_session.add_message(message)
        logger.debug(f"Logged user message with intent: {intent}")
        return True
    
    def log_bot_response(
        self,
        content: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Log a bot response in the current session.
        
        Args:
            content: Response text
            metadata: Additional metadata (model used, strategy, etc.)
            
        Returns:
            True if logged successfully
        """
        if not self._current_session:
            logger.warning("No active session - cannot log response")
            return False
        
        message = ConversationMessage(
            role=MessageRole.BOT,
            content=content,
            metadata=metadata or {}
        )
        
        self._current_session.add_message(message)
        logger.debug("Logged bot response")
        return True
    
    def save_current_session(self) -> bool:
        """
        Save the current session to database without ending it.
        
        Returns:
            True if saved successfully
        """
        if not self._current_session:
            return False
        
        return self.repository.save_session(self._current_session)
    
    def get_user_history(
        self,
        user_name: str,
        limit: int = 10
    ) -> list:
        """
        Retrieve conversation history for a user.
        
        Args:
            user_name: User identifier
            limit: Maximum number of sessions to retrieve
            
        Returns:
            List of conversation sessions
        """
        sessions = self.repository.get_user_sessions(user_name, limit=limit)
        logger.info(f"Retrieved {len(sessions)} sessions for user {user_name}")
        return sessions
    
    def get_user_profile(self, user_name: str) -> Optional[UserProfile]:
        """
        Get user profile with statistics.
        
        Args:
            user_name: User identifier
            
        Returns:
            UserProfile if found
        """
        return self.repository.get_user_profile(user_name)
    
    def search_by_intent(
        self,
        intent: str,
        user_name: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        Search conversations by detected intent.
        
        Args:
            intent: Intent to search for
            user_name: Optional user filter
            limit: Maximum results
            
        Returns:
            List of matching sessions
        """
        sessions = self.repository.get_sessions_by_intent(
            intent,
            user_name=user_name,
            limit=limit
        )
        logger.info(f"Found {len(sessions)} sessions with intent '{intent}'")
        return sessions
    
    def get_intent_analytics(
        self,
        user_name: Optional[str] = None,
        days: int = 7
    ) -> dict:
        """
        Get intent statistics for analytics.
        
        Args:
            user_name: Optional user filter
            days: Number of days to analyze
            
        Returns:
            Dictionary mapping intent to count
        """
        stats = self.repository.get_intent_statistics(
            user_name=user_name,
            days=days
        )
        logger.info(f"Retrieved intent statistics: {len(stats)} intents")
        return stats
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session.session_id if self._current_session else None
    
    def is_session_active(self) -> bool:
        """Check if there's an active session."""
        return self._current_session is not None
