"""
MongoDB repository for conversation persistence.

This is the infrastructure layer - handles all database operations.
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from src.domain.models.conversation import (
    ConversationSession,
    UserProfile,
    ConversationMessage
)
from src.infrastructure.config.mongodb_settings import MongoDBSettings

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository for managing conversation persistence in MongoDB.
    
    Responsibilities:
    - Store and retrieve conversation sessions
    - Manage user profiles
    - Query conversations by user, date, intent
    - Aggregate conversation statistics
    """
    
    def __init__(self, settings: Optional[MongoDBSettings] = None):
        """
        Initialize MongoDB connection.
        
        Args:
            settings: MongoDB configuration settings
        """
        self.settings = settings or MongoDBSettings()
        self._client: Optional[MongoClient] = None
        self._db = None
        self._conversations = None
        self._users = None
        self._connected = False
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._client = MongoClient(
                self.settings.mongodb_uri,
                maxPoolSize=self.settings.mongodb_max_pool_size,
                minPoolSize=self.settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=self.settings.mongodb_timeout_ms
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            # Get database and collections
            self._db = self._client[self.settings.mongodb_database]
            self._conversations = self._db[self.settings.conversations_collection]
            self._users = self._db[self.settings.users_collection]
            
            # Create indexes
            if self.settings.create_indexes:
                self._create_indexes()
            
            self._connected = True
            logger.info(f"Connected to MongoDB: {self.settings.mongodb_database}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
    
    def _create_indexes(self) -> None:
        """Create database indexes for efficient queries."""
        try:
            # Conversation indexes
            self._conversations.create_index([("session_id", ASCENDING)], unique=True)
            self._conversations.create_index([("user_name", ASCENDING)])
            self._conversations.create_index([("started_at", DESCENDING)])
            self._conversations.create_index([("user_name", ASCENDING), ("started_at", DESCENDING)])
            
            # User profile indexes
            self._users.create_index([("user_name", ASCENDING)], unique=True)
            self._users.create_index([("last_seen", DESCENDING)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self._connected
    
    # === Conversation Session Operations ===
    
    def save_session(self, session: ConversationSession) -> bool:
        """
        Save or update a conversation session.
        
        Args:
            session: Conversation session to save
            
        Returns:
            True if saved successfully
        """
        if not self._connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            session_dict = session.to_dict()
            self._conversations.update_one(
                {"session_id": session.session_id},
                {"$set": session_dict},
                upsert=True
            )
            logger.debug(f"Saved session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a conversation session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession if found, None otherwise
        """
        if not self._connected:
            return None
        
        try:
            data = self._conversations.find_one({"session_id": session_id})
            if data:
                return ConversationSession.from_dict(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def get_user_sessions(
        self,
        user_name: str,
        limit: int = 10,
        skip: int = 0
    ) -> List[ConversationSession]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_name: User identifier
            limit: Maximum number of sessions to return
            skip: Number of sessions to skip
            
        Returns:
            List of conversation sessions
        """
        if not self._connected:
            return []
        
        try:
            cursor = self._conversations.find(
                {"user_name": user_name}
            ).sort("started_at", DESCENDING).skip(skip).limit(limit)
            
            return [ConversationSession.from_dict(data) for data in cursor]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def get_sessions_by_intent(
        self,
        intent: str,
        user_name: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationSession]:
        """
        Find sessions containing a specific intent.
        
        Args:
            intent: Intent to search for
            user_name: Optional user filter
            limit: Maximum results
            
        Returns:
            List of matching sessions
        """
        if not self._connected:
            return []
        
        try:
            query = {"messages.intent": intent}
            if user_name:
                query["user_name"] = user_name
            
            cursor = self._conversations.find(query).sort(
                "started_at", DESCENDING
            ).limit(limit)
            
            return [ConversationSession.from_dict(data) for data in cursor]
            
        except Exception as e:
            logger.error(f"Failed to get sessions by intent: {e}")
            return []
    
    # === User Profile Operations ===
    
    def save_user_profile(self, profile: UserProfile) -> bool:
        """
        Save or update a user profile.
        
        Args:
            profile: User profile to save
            
        Returns:
            True if saved successfully
        """
        if not self._connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            profile_dict = profile.to_dict()
            self._users.update_one(
                {"user_name": profile.user_name},
                {"$set": profile_dict},
                upsert=True
            )
            logger.debug(f"Saved user profile: {profile.user_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")
            return False
    
    def get_user_profile(self, user_name: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile.
        
        Args:
            user_name: User identifier
            
        Returns:
            UserProfile if found, None otherwise
        """
        if not self._connected:
            return None
        
        try:
            data = self._users.find_one({"user_name": user_name})
            if data:
                return UserProfile.from_dict(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    def update_user_stats(self, user_name: str, session: ConversationSession) -> None:
        """
        Update user statistics based on a new session.
        
        Args:
            user_name: User identifier
            session: New conversation session
        """
        if not self._connected:
            return
        
        try:
            # Get or create profile
            profile = self.get_user_profile(user_name)
            if not profile:
                profile = UserProfile(
                    user_name=user_name,
                    first_seen=session.started_at
                )
            
            # Update statistics
            profile.total_sessions += 1
            profile.total_messages += len([m for m in session.messages if m.role.value == "user"])
            profile.last_seen = session.started_at
            
            # Update primary intents
            intents = session.get_intents()
            for intent in intents:
                if intent not in profile.primary_intents:
                    profile.primary_intents.append(intent)
            
            # Keep only top 10 intents
            profile.primary_intents = profile.primary_intents[:10]
            
            # Save updated profile
            self.save_user_profile(profile)
            
        except Exception as e:
            logger.error(f"Failed to update user stats: {e}")
    
    # === Analytics & Queries ===
    
    def get_recent_sessions(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[ConversationSession]:
        """
        Get recent conversation sessions.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of sessions
            
        Returns:
            List of recent sessions
        """
        if not self._connected:
            return []
        
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            cursor = self._conversations.find(
                {"started_at": {"$gte": cutoff}}
            ).sort("started_at", DESCENDING).limit(limit)
            
            return [ConversationSession.from_dict(data) for data in cursor]
            
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []
    
    def get_intent_statistics(
        self,
        user_name: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, int]:
        """
        Get statistics on detected intents.
        
        Args:
            user_name: Optional user filter
            days: Number of days to analyze
            
        Returns:
            Dictionary mapping intent to count
        """
        if not self._connected:
            return {}
        
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            match_stage = {"started_at": {"$gte": cutoff}}
            if user_name:
                match_stage["user_name"] = user_name
            
            pipeline = [
                {"$match": match_stage},
                {"$unwind": "$messages"},
                {"$match": {"messages.intent": {"$ne": None}}},
                {"$group": {
                    "_id": "$messages.intent",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            results = self._conversations.aggregate(pipeline)
            return {item["_id"]: item["count"] for item in results}
            
        except Exception as e:
            logger.error(f"Failed to get intent statistics: {e}")
            return {}
