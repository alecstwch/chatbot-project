"""
RAG Memory Service using Qdrant Vector Database.

This service provides semantic search over past conversations,
enabling the chatbot to recall relevant context from history.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from sentence_transformers import SentenceTransformer

from src.infrastructure.config.qdrant_settings import QdrantSettings

logger = logging.getLogger(__name__)


class RAGMemoryService:
    """
    RAG-based conversational memory using Qdrant.
    
    Stores all messages as vector embeddings and retrieves
    semantically similar past conversations for context augmentation.
    
    Attributes:
        settings: Qdrant configuration settings
        client: Qdrant client instance
        embedder: Sentence transformer model for embeddings
        collection_name: Name of the vector collection
    """
    
    def __init__(
        self,
        settings: Optional[QdrantSettings] = None,
        embedder: Optional[SentenceTransformer] = None
    ):
        """
        Initialize RAG memory service.
        
        Args:
            settings: Qdrant configuration settings
            embedder: Pre-loaded sentence transformer (optional)
        """
        self.settings = settings or QdrantSettings()
        self.collection_name = self.settings.collection_name
        self.client: Optional[QdrantClient] = None
        self._embedder = embedder
        self._initialized = False
        
        logger.info(f"RAG Memory Service created with collection: {self.collection_name}")
    
    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._embedder = SentenceTransformer(self.settings.embedding_model)
            logger.info("Embedding model loaded successfully")
        return self._embedder
    
    def initialize(self) -> bool:
        """
        Initialize connection to Qdrant and ensure collection exists.
        
        Returns:
            True if initialization successful
        """
        try:
            # Connect to Qdrant
            if self.settings.use_local_storage:
                logger.info(f"Using local Qdrant storage: {self.settings.local_path}")
                self.client = QdrantClient(path=self.settings.local_path)
            else:
                logger.info(f"Connecting to Qdrant server: {self.settings.host}:{self.settings.port}")
                self.client = QdrantClient(
                    host=self.settings.host,
                    port=self.settings.port,
                )
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG memory service: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the service and close connections."""
        if self.client:
            self.client.close()
            self.client = None
        self._initialized = False
        logger.info("RAG Memory Service shutdown")
    
    def store_message(
        self,
        user_id: str,
        message: str,
        role: str,
        session_id: Optional[str] = None,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a message with its embedding in Qdrant.
        
        Args:
            user_id: Unique user identifier
            message: The message text
            role: "user" or "assistant"
            session_id: Optional session identifier
            intent: Optional detected intent
            metadata: Additional metadata
            
        Returns:
            Message ID (UUID)
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized. Call initialize() first.")
        
        if not message or not message.strip():
            logger.warning("Attempted to store empty message, skipping")
            return ""
        
        try:
            message_id = str(uuid.uuid4())
            
            # Create embedding
            embedding = self.embedder.encode(message).tolist()
            
            # Prepare payload (metadata)
            payload = {
                "user_id": user_id,
                "role": role,
                "text": message,
                "session_id": session_id or "default",
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if intent:
                payload["intent"] = intent
            
            if metadata:
                payload.update(metadata)
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=message_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.debug(f"Stored message {message_id} for user {user_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            raise
    
    def store_conversation_turn(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        session_id: Optional[str] = None,
        user_intent: Optional[str] = None,
        emotion_data: Optional[Dict[str, Any]] = None,
        behavior_patterns: Optional[List[str]] = None,
        next_question: Optional[str] = None
    ) -> tuple:
        """
        Store both user and assistant messages from a conversation turn.

        Args:
            user_id: Unique user identifier
            user_message: User's message
            assistant_message: Assistant's response
            session_id: Optional session identifier
            user_intent: Optional detected intent for user message
            emotion_data: Optional emotion analysis data for user message
            behavior_patterns: Optional list of detected behavior patterns
            next_question: Optional follow-up question suggested by assistant

        Returns:
            Tuple of (user_message_id, assistant_message_id)
        """
        # Prepare metadata with emotion data and behavior patterns
        user_metadata = {}
        if emotion_data:
            user_metadata["emotion"] = emotion_data.get("primary_emotion")
            user_metadata["emotion_confidence"] = emotion_data.get("confidence")
            user_metadata["emotion_intensity"] = emotion_data.get("intensity")
            user_metadata["sentiment"] = emotion_data.get("sentiment")
            user_metadata["emotion_keywords"] = emotion_data.get("keywords", [])

        # Add behavior patterns to user metadata
        if behavior_patterns:
            user_metadata["behavior_patterns"] = behavior_patterns
            logger.debug(f"Storing {len(behavior_patterns)} behavior patterns for user message")

        user_msg_id = self.store_message(
            user_id=user_id,
            message=user_message,
            role="user",
            session_id=session_id,
            intent=user_intent,
            metadata=user_metadata if user_metadata else None
        )

        # Prepare assistant metadata with next question
        assistant_metadata = {}
        if next_question:
            assistant_metadata["next_question"] = next_question
            logger.debug(f"Storing next_question for assistant message")

        assistant_msg_id = self.store_message(
            user_id=user_id,
            message=assistant_message,
            role="assistant",
            session_id=session_id,
            metadata=assistant_metadata if assistant_metadata else None
        )

        return user_msg_id, assistant_msg_id
    
    def retrieve_relevant_context(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        include_assistant: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages most similar to the query.
        
        Args:
            user_id: Filter by user
            query: Current message/query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            include_assistant: Whether to include assistant messages
            
        Returns:
            List of relevant past messages with metadata and similarity scores
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized. Call initialize() first.")
        
        top_k = top_k or self.settings.search_limit
        min_similarity = min_similarity or self.settings.min_similarity
        
        try:
            # Create query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Build filter
            filter_conditions = [
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
            
            if not include_assistant:
                filter_conditions.append(
                    FieldCondition(
                        key="role",
                        match=MatchValue(value="user")
                    )
                )
            
            # Search using query_points (newer qdrant-client API)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=Filter(must=filter_conditions),
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            relevant_messages = []
            for result in results.points:
                # Qdrant returns score as similarity (for cosine)
                similarity = result.score
                
                if similarity >= min_similarity:
                    relevant_messages.append({
                        "id": result.id,
                        "content": result.payload.get("text", ""),
                        "role": result.payload.get("role", "user"),
                        "similarity": similarity,
                        "intent": result.payload.get("intent"),
                        "timestamp": result.payload.get("timestamp"),
                        "session_id": result.payload.get("session_id"),
                        "metadata": {
                            k: v for k, v in result.payload.items()
                            if k not in ["text", "role", "user_id", "intent", "timestamp", "session_id"]
                        }
                    })
            
            logger.debug(f"Retrieved {len(relevant_messages)} relevant messages for query")
            return relevant_messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def search_by_topic(
        self,
        user_id: str,
        topic: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search past conversations by topic.
        
        Args:
            user_id: User identifier
            topic: Topic to search for (e.g., "anxiety", "depression")
            top_k: Maximum results
            
        Returns:
            List of matching messages
        """
        return self.retrieve_relevant_context(
            user_id=user_id,
            query=topic,
            top_k=top_k,
            min_similarity=0.4  # Lower threshold for topic search
        )
    
    def search_by_intent(
        self,
        user_id: str,
        intent: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search messages by detected intent.
        
        Args:
            user_id: User identifier
            intent: Intent to filter by
            top_k: Maximum results
            
        Returns:
            List of matching messages
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized.")
        
        try:
            # Build filter for intent
            query_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="intent", match=MatchValue(value=intent))
                ]
            )
            
            # Use scroll to get all matching points
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            return [
                {
                    "id": r.id,
                    "content": r.payload.get("text", ""),
                    "role": r.payload.get("role"),
                    "intent": r.payload.get("intent"),
                    "timestamp": r.payload.get("timestamp"),
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to search by intent: {e}")
            return []

    def search_by_emotion(
        self,
        user_id: str,
        emotion: str,
        top_k: int = 10,
        intensity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search messages by detected emotion.

        IMPORTANT: This method filters by BOTH user_id AND emotion to ensure
        data isolation between patients. You will ONLY get messages from the
        specified user, not all users with the same emotion.

        Args:
            user_id: User identifier (REQUIRED for data isolation)
            emotion: Emotion to filter by (e.g., "joy", "sadness", "anxiety")
            top_k: Maximum results
            intensity: Optional intensity filter ("low", "medium", "high")

        Returns:
            List of matching messages from the specified user only
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized.")

        try:
            # Build filter - MUST include user_id for data isolation
            # This ensures we only retrieve messages from THIS patient, not all patients
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="emotion", match=MatchValue(value=emotion))
            ]

            if intensity:
                filter_conditions.append(
                    FieldCondition(key="emotion_intensity", match=MatchValue(value=intensity))
                )

            query_filter = Filter(must=filter_conditions)

            # Use scroll to get all matching points
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True
            )

            logger.debug(
                f"Retrieved {len(results)} messages for user '{user_id}' "
                f"with emotion '{emotion}' (data isolation enforced)"
            )

            return [
                {
                    "id": r.id,
                    "content": r.payload.get("text", ""),
                    "role": r.payload.get("role"),
                    "emotion": r.payload.get("emotion"),
                    "emotion_confidence": r.payload.get("emotion_confidence"),
                    "emotion_intensity": r.payload.get("emotion_intensity"),
                    "sentiment": r.payload.get("sentiment"),
                    "timestamp": r.payload.get("timestamp"),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Failed to search by emotion: {e}")
            return []

    def search_by_behavior_pattern(
        self,
        user_id: str,
        behavior_pattern: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search messages by detected behavior pattern.

        IMPORTANT: This method filters by BOTH user_id AND behavior_pattern to ensure
        data isolation between patients.

        Args:
            user_id: User identifier (REQUIRED for data isolation)
            behavior_pattern: Behavior pattern to filter by (e.g., "social_withdrawal")
            top_k: Maximum results

        Returns:
            List of matching messages from the specified user only
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized.")

        try:
            # Build filter - MUST include user_id for data isolation
            # This ensures we only retrieve messages from THIS patient, not all patients
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="behavior_patterns", match=MatchValue(value=behavior_pattern))
            ]

            query_filter = Filter(must=filter_conditions)

            # Use scroll to get all matching points
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True
            )

            logger.debug(
                f"Retrieved {len(results)} messages for user '{user_id}' "
                f"with behavior_pattern='{behavior_pattern}' (data isolation enforced)"
            )

            return [
                {
                    "id": r.id,
                    "content": r.payload.get("text", ""),
                    "role": r.payload.get("role"),
                    "behavior_patterns": r.payload.get("behavior_patterns", []),
                    "emotion": r.payload.get("emotion"),
                    "emotion_intensity": r.payload.get("emotion_intensity"),
                    "timestamp": r.payload.get("timestamp", ""),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Failed to search by behavior pattern: {e}")
            return []

    def retrieve_by_behavior_pattern_context(
        self,
        user_id: str,
        query: str,
        target_pattern: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages with semantic similarity AND optional behavior pattern filter.

        IMPORTANT: This method ALWAYS filters by user_id to ensure data isolation
        between patients. Even when filtering by behavior pattern, you will ONLY get messages
        from the specified user.

        This is useful for finding similar behavioral patterns from the past.
        For example, finding past times the user exhibited similar behaviors.

        Args:
            user_id: User identifier (REQUIRED - ensures patient data isolation)
            query: Current message to find similar contexts for
            target_pattern: Optional behavior pattern to filter by
            top_k: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant past messages from the specified user only,
            with behavior pattern metadata
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized.")

        top_k = top_k or self.settings.search_limit

        try:
            # Create query embedding
            query_embedding = self.embedder.encode(query).tolist()

            # Build filter - user_id is ALWAYS required for data isolation
            # This ensures we only search within THIS patient's messages
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            # Add behavior pattern filter if specified (still scoped to this user only)
            if target_pattern:
                filter_conditions.append(
                    FieldCondition(key="behavior_patterns", match=MatchValue(value=target_pattern))
                )

            # Search using query_points
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=top_k,
                with_payload=True
            )

            # Format results with behavior pattern metadata
            relevant_messages = []
            for result in results.points:
                similarity = result.score

                if similarity >= min_similarity:
                    relevant_messages.append({
                        "id": result.id,
                        "content": result.payload.get("text", ""),
                        "role": result.payload.get("role", "user"),
                        "similarity": similarity,
                        "behavior_patterns": result.payload.get("behavior_patterns", []),
                        "emotion": result.payload.get("emotion"),
                        "emotion_confidence": result.payload.get("emotion_confidence"),
                        "timestamp": result.payload.get("timestamp"),
                        "session_id": result.payload.get("session_id"),
                    })

            logger.debug(
                f"Retrieved {len(relevant_messages)} messages for user '{user_id}' "
                f"with behavior_pattern filter='{target_pattern or 'all'}' "
                f"(data isolation enforced)"
            )
            return relevant_messages

        except Exception as e:
            logger.error(f"Failed to retrieve by behavior pattern context: {e}")
            return []

    def retrieve_by_emotion_context(
        self,
        user_id: str,
        query: str,
        target_emotion: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages with semantic similarity AND optional emotion filter.

        IMPORTANT: This method ALWAYS filters by user_id to ensure data isolation
        between patients. Even when filtering by emotion, you will ONLY get messages
        from the specified user, not all users with the same emotion.

        This is useful for finding similar emotional states from the past.
        For example, finding past times the user felt similarly to now.

        Args:
            user_id: User identifier (REQUIRED - ensures patient data isolation)
            query: Current message to find similar contexts for
            target_emotion: Optional emotion to filter by (finds similar past states
                          for THIS USER ONLY)
            top_k: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant past messages from the specified user only,
            with emotion metadata
        """
        if not self._initialized:
            raise RuntimeError("RAG Memory Service not initialized.")

        top_k = top_k or self.settings.search_limit

        try:
            # Create query embedding
            query_embedding = self.embedder.encode(query).tolist()

            # Build filter - user_id is ALWAYS required for data isolation
            # This ensures we only search within THIS patient's messages
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            # Add emotion filter if specified (still scoped to this user only)
            if target_emotion:
                filter_conditions.append(
                    FieldCondition(key="emotion", match=MatchValue(value=target_emotion))
                )

            # Search using query_points
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=top_k,
                with_payload=True
            )

            # Format results with emotion metadata
            relevant_messages = []
            for result in results.points:
                similarity = result.score

                if similarity >= min_similarity:
                    relevant_messages.append({
                        "id": result.id,
                        "content": result.payload.get("text", ""),
                        "role": result.payload.get("role", "user"),
                        "similarity": similarity,
                        "emotion": result.payload.get("emotion"),
                        "emotion_confidence": result.payload.get("emotion_confidence"),
                        "emotion_intensity": result.payload.get("emotion_intensity"),
                        "sentiment": result.payload.get("sentiment"),
                        "timestamp": result.payload.get("timestamp"),
                        "session_id": result.payload.get("session_id"),
                    })

            logger.debug(
                f"Retrieved {len(relevant_messages)} messages for user '{user_id}' "
                f"with emotion filter='{target_emotion or 'all'}' "
                f"(data isolation enforced)"
            )
            return relevant_messages

        except Exception as e:
            logger.error(f"Failed to retrieve by emotion context: {e}")
            return []

    def get_emotion_summary(
        self,
        user_id: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get emotion distribution summary for a user.

        Args:
            user_id: User identifier
            limit: Maximum messages to analyze

        Returns:
            Dictionary with emotion statistics
        """
        if not self._initialized:
            return {}

        try:
            # Get all user messages
            messages = self.get_all_user_messages(user_id, limit=limit)

            # Count emotions
            emotion_counts = {}
            intensity_counts = {}
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

            for msg in messages:
                # Emotion data is now at top level, not nested in metadata
                emotion = msg.get("emotion")
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                intensity = msg.get("emotion_intensity")
                if intensity:
                    intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1

                sentiment = msg.get("sentiment")
                if sentiment:
                    sentiment_counts[sentiment] += 1

            return {
                "emotion_distribution": emotion_counts,
                "intensity_distribution": intensity_counts,
                "sentiment_distribution": sentiment_counts,
                "total_messages_analyzed": len(messages)
            }

        except Exception as e:
            logger.error(f"Failed to get emotion summary: {e}")
            return {}

    def get_user_message_count(self, user_id: str) -> int:
        """
        Get total messages stored for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of messages
        """
        if not self._initialized:
            return 0
        
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            return result.count
        except Exception as e:
            logger.error(f"Failed to count messages: {e}")
            return 0
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user (GDPR compliance).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        if not self._initialized:
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(key="user_id", match=MatchValue(value=user_id))
                        ]
                    )
                )
            )
            logger.info(f"Deleted all data for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            return False
    
    def get_all_user_messages(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all stored messages for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum messages to retrieve
            
        Returns:
            List of all messages with metadata, sorted by timestamp
        """
        if not self._initialized:
            return []
        
        try:
            # Build filter for user
            query_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                ]
            )
            
            # Use scroll to get all matching points (with vectors!)
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True  # Include the embedding vectors
            )
            
            # Format and sort by timestamp
            messages = [
                {
                    "id": r.id,
                    "vector": r.vector,  # The 384-dimensional embedding
                    "vector_preview": r.vector[:5] if r.vector else None,  # First 5 values for display
                    "vector_dim": len(r.vector) if r.vector else 0,  # Vector dimension
                    "role": r.payload.get("role", "user"),
                    "text": r.payload.get("text", ""),
                    "timestamp": r.payload.get("timestamp", ""),
                    "session_id": r.payload.get("session_id", ""),
                    "intent": r.payload.get("intent"),
                    # Emotion metadata (stored directly in payload)
                    "emotion": r.payload.get("emotion"),
                    "emotion_confidence": r.payload.get("emotion_confidence"),
                    "emotion_intensity": r.payload.get("emotion_intensity"),
                    "sentiment": r.payload.get("sentiment"),
                    "emotion_keywords": r.payload.get("emotion_keywords", []),
                    # Behavior pattern metadata
                    "behavior_patterns": r.payload.get("behavior_patterns", []),
                    # Next question (for assistant messages)
                    "next_question": r.payload.get("next_question"),
                }
                for r in results
            ]
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.get("timestamp", ""))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get user messages: {e}")
            return []
    
    def get_all_messages(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Get ALL stored messages from all users (admin/debug).
        
        Args:
            limit: Maximum messages to retrieve
            
        Returns:
            List of all messages with metadata, sorted by timestamp
        """
        if not self._initialized:
            return []
        
        try:
            # Scroll through all points (no filter, with vectors)
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=True  # Include the embedding vectors
            )
            
            # Format and sort by timestamp
            messages = [
                {
                    "id": r.id,
                    "user_id": r.payload.get("user_id", "unknown"),
                    "vector": r.vector,  # The full embedding vector
                    "vector_preview": r.vector[:5] if r.vector else None,  # First 5 values
                    "vector_dim": len(r.vector) if r.vector else 0,  # Vector dimension
                    "role": r.payload.get("role", "user"),
                    "text": r.payload.get("text", ""),
                    "timestamp": r.payload.get("timestamp", ""),
                    "session_id": r.payload.get("session_id", ""),
                    # Emotion metadata (stored directly in payload)
                    "emotion": r.payload.get("emotion"),
                    "emotion_confidence": r.payload.get("emotion_confidence"),
                    "emotion_intensity": r.payload.get("emotion_intensity"),
                    "sentiment": r.payload.get("sentiment"),
                    "emotion_keywords": r.payload.get("emotion_keywords", []),
                    # Behavior pattern metadata
                    "behavior_patterns": r.payload.get("behavior_patterns", []),
                    # Next question (for assistant messages)
                    "next_question": r.payload.get("next_question"),
                }
                for r in results
            ]
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.get("timestamp", ""))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get all messages: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if the service is initialized and ready."""
        return self._initialized and self.client is not None

