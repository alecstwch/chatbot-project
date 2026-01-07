"""
Domain models for conversation storage.

Following DDD principles - these are pure domain models
independent of infrastructure concerns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class MessageRole(Enum):
    """Role of the message sender."""
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: Who sent the message (user/bot/system)
        content: The message text
        timestamp: When the message was sent
        intent: Detected intent (if applicable)
        intent_confidence: Confidence score for intent detection
        metadata: Additional context (model used, response time, etc.)
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'intent': self.intent,
            'intent_confidence': self.intent_confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationMessage':
        """Create from dictionary."""
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            timestamp=data['timestamp'],
            intent=data.get('intent'),
            intent_confidence=data.get('intent_confidence'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ConversationSession:
    """
    Represents a conversation session between a user and the chatbot.
    
    Attributes:
        session_id: Unique identifier for this session
        user_name: Unique identifier for the user
        messages: List of messages in this session
        started_at: When the session began
        ended_at: When the session ended (None if active)
        metadata: Session-level metadata (device, location, etc.)
    """
    session_id: str
    user_name: str
    messages: List[ConversationMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    
    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to this session."""
        self.messages.append(message)
    
    def end_session(self) -> None:
        """Mark the session as ended."""
        self.ended_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None
    
    def get_intents(self) -> List[str]:
        """Get all unique intents detected in this session."""
        return list(set(
            msg.intent for msg in self.messages 
            if msg.intent is not None
        ))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'session_id': self.session_id,
            'user_name': self.user_name,
            'messages': [msg.to_dict() for msg in self.messages],
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationSession':
        """Create from dictionary."""
        return cls(
            session_id=data['session_id'],
            user_name=data['user_name'],
            messages=[ConversationMessage.from_dict(msg) for msg in data.get('messages', [])],
            started_at=data['started_at'],
            ended_at=data.get('ended_at'),
            metadata=data.get('metadata', {})
        )


@dataclass
class UserProfile:
    """
    Aggregated profile for a user across all sessions.
    
    Attributes:
        user_name: Unique identifier for the user
        total_sessions: Number of conversation sessions
        total_messages: Total messages sent by user
        primary_intents: Most common intents detected
        first_seen: When user first started chatting
        last_seen: Most recent interaction
        metadata: User-level metadata (preferences, settings, etc.)
    """
    user_name: str
    total_sessions: int = 0
    total_messages: int = 0
    primary_intents: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'user_name': self.user_name,
            'total_sessions': self.total_sessions,
            'total_messages': self.total_messages,
            'primary_intents': self.primary_intents,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create from dictionary."""
        return cls(
            user_name=data['user_name'],
            total_sessions=data.get('total_sessions', 0),
            total_messages=data.get('total_messages', 0),
            primary_intents=data.get('primary_intents', []),
            first_seen=data['first_seen'],
            last_seen=data.get('last_seen', data['first_seen']),
            metadata=data.get('metadata', {})
        )
