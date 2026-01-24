"""
MongoDB Patient File Repository.

Manages patient profiles, behavior patterns, and conversation history
for therapy/mental health chatbot applications.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from src.infrastructure.config.mongodb_settings import MongoDBSettings

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BehaviorPattern:
    """Represents a detected behavior pattern."""
    pattern_type: str  # e.g., "social_withdrawal", "sleep_disruption", "mood_swings"
    description: str
    frequency: int  # How often this pattern occurs
    first_detected: str  # ISO timestamp
    last_detected: str  # ISO timestamp
    severity: str  # "mild", "moderate", "severe"
    associated_emotions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)  # Example messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PatientProfile:
    """Patient profile file."""
    user_id: str
    name: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Basic information
    age: Optional[int] = None
    gender: Optional[str] = None

    # Mental health profile
    known_conditions: List[str] = field(default_factory=list)  # e.g., ["anxiety", "depression"]
    medications: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # Known triggers

    # Behavioral patterns detected over time
    behavior_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Emotional summary
    emotion_summary: Dict[str, Any] = field(default_factory=dict)

    # Risk assessment
    risk_level: str = RiskLevel.LOW.value
    last_risk_assessment: Optional[str] = None

    # Conversation metadata
    total_conversations: int = 0
    last_conversation_date: Optional[str] = None

    # Notes from previous sessions
    session_notes: List[Dict[str, Any]] = field(default_factory=list)

    # Goals and progress
    treatment_goals: List[str] = field(default_factory=list)
    progress_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PatientRepository:
    """
    Repository for managing patient files in MongoDB.

    Provides CRUD operations for patient profiles and behavior pattern tracking.
    """

    def __init__(self, settings: Optional[MongoDBSettings] = None):
        """
        Initialize the patient repository.

        Args:
            settings: MongoDB configuration settings
        """
        self.settings = settings or MongoDBSettings()
        self.client: Optional[MongoClient] = None
        self.db = None
        self._collection = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize MongoDB connection and create indexes.

        Returns:
            True if initialization successful
        """
        try:
            logger.info(f"Connecting to MongoDB: {self.settings.mongodb_uri}")

            self.client = MongoClient(
                self.settings.mongodb_uri,
                maxPoolSize=self.settings.mongodb_max_pool_size,
                minPoolSize=self.settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=self.settings.mongodb_timeout_ms
            )

            # Test connection
            self.client.admin.command('ping')

            self.db = self.client[self.settings.mongodb_database]
            self._collection = self.db["patients"]

            # Create indexes
            if self.settings.create_indexes:
                self._create_indexes()

            self._initialized = True
            logger.info(f"MongoDB connected: {self.settings.mongodb_database}")
            return True

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"MongoDB initialization error: {e}")
            return False

    def _create_indexes(self) -> None:
        """Create database indexes for efficient queries."""
        try:
            # Unique index on user_id
            self._collection.create_index(
                [("user_id", ASCENDING)],
                unique=True,
                name="user_id_unique"
            )

            # Index for risk level queries
            self._collection.create_index(
                [("risk_level", ASCENDING)],
                name="risk_level_idx"
            )

            # Index for last conversation date
            self._collection.create_index(
                [("last_conversation_date", DESCENDING)],
                name="last_conversation_idx"
            )

            # Index for behavior patterns
            self._collection.create_index(
                [("behavior_patterns.pattern_type", ASCENDING)],
                name="behavior_pattern_idx"
            )

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def create_patient(self, profile: PatientProfile) -> bool:
        """
        Create a new patient profile.

        Args:
            profile: PatientProfile object

        Returns:
            True if created successfully
        """
        if not self._initialized:
            raise RuntimeError("Repository not initialized")

        try:
            doc = profile.to_dict()
            self._collection.insert_one(doc)
            logger.info(f"Created patient profile: {profile.user_id}")
            return True

        except DuplicateKeyError:
            logger.warning(f"Patient {profile.user_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to create patient: {e}")
            return False

    def get_patient(self, user_id: str) -> Optional[PatientProfile]:
        """
        Get a patient profile by user ID.

        Args:
            user_id: User identifier

        Returns:
            PatientProfile if found, None otherwise
        """
        if not self._initialized:
            return None

        try:
            doc = self._collection.find_one({"user_id": user_id})
            if doc:
                doc.pop("_id", None)  # Remove MongoDB _id
                return PatientProfile(**doc)
            return None

        except Exception as e:
            logger.error(f"Failed to get patient: {e}")
            return None

    def update_patient(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update patient profile fields.

        Args:
            user_id: User identifier
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully
        """
        if not self._initialized:
            return False

        try:
            # Always update the updated_at timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()

            result = self._collection.update_one(
                {"user_id": user_id},
                {"$set": updates}
            )

            if result.modified_count > 0:
                logger.debug(f"Updated patient {user_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to update patient: {e}")
            return False

    def add_behavior_pattern(
        self,
        user_id: str,
        pattern: BehaviorPattern
    ) -> bool:
        """
        Add or update a behavior pattern for a patient.

        Args:
            user_id: User identifier
            pattern: BehaviorPattern object

        Returns:
            True if added/updated successfully
        """
        if not self._initialized:
            return False

        try:
            # Check if pattern already exists
            patient = self.get_patient(user_id)
            if not patient:
                return False

            # Update existing pattern or add new one
            pattern_dict = pattern.to_dict()
            updated_patterns = [
                p for p in patient.behavior_patterns
                if p["pattern_type"] != pattern.pattern_type
            ]
            updated_patterns.append(pattern_dict)

            return self.update_patient(
                user_id,
                {"behavior_patterns": updated_patterns}
            )

        except Exception as e:
            logger.error(f"Failed to add behavior pattern: {e}")
            return False

    def add_session_note(
        self,
        user_id: str,
        note: str,
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a session note for a patient.

        Args:
            user_id: User identifier
            note: Note content
            emotion_data: Optional emotion data from the session

        Returns:
            True if added successfully
        """
        if not self._initialized:
            return False

        try:
            patient = self.get_patient(user_id)
            if not patient:
                return False

            session_note = {
                "date": datetime.utcnow().isoformat(),
                "note": note,
                "emotion_data": emotion_data or {}
            }

            session_notes = patient.session_notes + [session_note]

            # Keep only last 50 notes
            if len(session_notes) > 50:
                session_notes = session_notes[-50:]

            return self.update_patient(
                user_id,
                {
                    "session_notes": session_notes,
                    "last_conversation_date": datetime.utcnow().isoformat(),
                    "total_conversations": patient.total_conversations + 1
                }
            )

        except Exception as e:
            logger.error(f"Failed to add session note: {e}")
            return False

    def update_emotion_summary(
        self,
        user_id: str,
        emotion_summary: Dict[str, Any]
    ) -> bool:
        """
        Update the emotion summary for a patient.

        Args:
            user_id: User identifier
            emotion_summary: Emotion analysis summary

        Returns:
            True if updated successfully
        """
        return self.update_patient(
            user_id,
            {"emotion_summary": emotion_summary}
        )

    def update_risk_level(
        self,
        user_id: str,
        risk_level: RiskLevel,
        reason: Optional[str] = None
    ) -> bool:
        """
        Update the risk assessment for a patient.

        Args:
            user_id: User identifier
            risk_level: New risk level
            reason: Optional reason for the change

        Returns:
            True if updated successfully
        """
        updates = {
            "risk_level": risk_level.value,
            "last_risk_assessment": datetime.utcnow().isoformat()
        }

        if reason:
            patient = self.get_patient(user_id)
            if patient:
                notes = patient.session_notes + [{
                    "date": datetime.utcnow().isoformat(),
                    "note": f"Risk level updated to {risk_level.value}. Reason: {reason}",
                    "type": "risk_assessment"
                }]
                updates["session_notes"] = notes

        return self.update_patient(user_id, updates)

    def get_patients_by_risk_level(
        self,
        risk_level: RiskLevel
    ) -> List[PatientProfile]:
        """
        Get all patients with a specific risk level.

        Args:
            risk_level: Risk level to filter by

        Returns:
            List of PatientProfile objects
        """
        if not self._initialized:
            return []

        try:
            cursor = self._collection.find({"risk_level": risk_level.value})
            patients = []
            for doc in cursor:
                doc.pop("_id", None)
                patients.append(PatientProfile(**doc))
            return patients

        except Exception as e:
            logger.error(f"Failed to get patients by risk level: {e}")
            return []

    def get_recent_conversations(
        self,
        days: int = 7
    ) -> List[PatientProfile]:
        """
        Get patients who have had conversations in the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of PatientProfile objects
        """
        if not self._initialized:
            return []

        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            cursor = self._collection.find({
                "last_conversation_date": {"$gte": cutoff_date}
            })
            patients = []
            for doc in cursor:
                doc.pop("_id", None)
                patients.append(PatientProfile(**doc))
            return patients

        except Exception as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []

    def delete_patient(self, user_id: str) -> bool:
        """
        Delete a patient profile.

        Args:
            user_id: User identifier

        Returns:
            True if deleted successfully
        """
        if not self._initialized:
            return False

        try:
            result = self._collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted patient: {user_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete patient: {e}")
            return False

    def shutdown(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._initialized = False
            logger.info("MongoDB connection closed")
