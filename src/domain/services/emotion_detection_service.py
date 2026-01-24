"""
Emotion Detection Service for analyzing user messages.

This service detects emotions in user messages to provide context
for the chatbot and enable emotion-based retrieval from memory.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class Emotion(str, Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "anxiety"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    HOPE = "hope"
    GRATITUDE = "gratitude"
    LONELINESS = "loneliness"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"


@dataclass
class EmotionAnalysis:
    """Result of emotion analysis."""
    primary_emotion: Emotion
    confidence: float
    emotions: Dict[str, float]  # All emotions with scores
    intensity: str  # "low", "medium", "high"
    keywords: List[str]  # Keywords that triggered the emotion
    sentiment: str  # "positive", "negative", "neutral"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_emotion": self.primary_emotion.value,
            "confidence": self.confidence,
            "emotions": {k: v for k, v in self.emotions.items()},
            "intensity": self.intensity,
            "keywords": self.keywords,
            "sentiment": self.sentiment
        }


class EmotionDetectionService:
    """
    Service for detecting emotions in user messages.

    Uses a combination of keyword analysis and pattern matching
    to identify emotional content in messages.
    """

    # Emotion keyword mappings
    EMOTION_KEYWORDS = {
        Emotion.JOY: [
            "happy", "glad", "excited", "wonderful", "amazing", "great",
            "love", "fantastic", "awesome", "thrilled", "delighted",
            "cheerful", "pleased", "ecstatic", "overjoyed", "blessed"
        ],
        Emotion.SADNESS: [
            "sad", "depressed", "unhappy", "down", "miserable", "heartbroken",
            "crying", "tears", "grief", "sorrow", "hopeless", "empty",
            "lonely", "missing", "loss", "hurt", "pain"
        ],
        Emotion.ANGER: [
            "angry", "furious", "mad", "irate", "annoyed", "frustrated",
            "rage", "hate", "irritated", "upset", "livid", "outraged",
            "hostile", "resentful", "disgusted"
        ],
        Emotion.FEAR: [
            "anxious", "afraid", "scared", "frightened", "worried", "nervous",
            "terrified", "panic", "fear", "dread", "uneasy", "tense",
            "overwhelmed", "stressed", "restless"
        ],
        Emotion.DISGUST: [
            "disgusted", "repulsed", "revolted", "sick", "appalled",
            "horrible", "terrible", "awful", "gross"
        ],
        Emotion.SURPRISE: [
            "surprised", "shocked", "amazed", "astonished", "stunned",
            "unexpected", "sudden", "wow"
        ],
        Emotion.HOPE: [
            "hope", "hopeful", "optimistic", "looking forward", "wish",
            "believe", "faith", "confidence", "expect", "anticipate"
        ],
        Emotion.GRATITUDE: [
            "thank", "grateful", "appreciate", "grateful", "thanks",
            "blessed", "fortunate", "indebted", "thankful"
        ],
        Emotion.LONELINESS: [
            "lonely", "alone", "isolated", "nobody", "no one", "by myself",
            "solitude", "withdrawn", "separated"
        ],
        Emotion.FRUSTRATION: [
            "frustrated", "stuck", "pointless", "giving up", "hopeless",
            "impossible", "can't", "unable", "difficult", "hard"
        ],
        Emotion.CONFUSION: [
            "confused", "don't understand", "unclear", "puzzled", "unsure",
            "uncertain", "lost", "bewildered", "uncertain"
        ]
    }

    # Sentiment keywords
    POSITIVE_WORDS = [
        "good", "great", "excellent", "wonderful", "amazing", "love",
        "happy", "best", "better", "beautiful", "perfect", "enjoy"
    ]

    NEGATIVE_WORDS = [
        "bad", "terrible", "awful", "horrible", "hate", "worst",
        "sad", "angry", "pain", "hurt", "scary", "afraid", "worry"
    ]

    def __init__(self):
        """Initialize the emotion detection service."""
        logger.info("Emotion Detection Service initialized")

    def detect_emotion(self, message: str) -> EmotionAnalysis:
        """
        Detect the primary emotion in a message.

        Args:
            message: The user's message

        Returns:
            EmotionAnalysis with detected emotion and metadata
        """
        if not message or not message.strip():
            return EmotionAnalysis(
                primary_emotion=Emotion.NEUTRAL,
                confidence=0.0,
                emotions={},
                intensity="low",
                keywords=[],
                sentiment="neutral"
            )

        message_lower = message.lower()
        words = re.findall(r'\b\w+\b', message_lower)

        # Score each emotion
        emotion_scores = {}
        found_keywords = {}

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            matched_keywords = [kw for kw in keywords if kw in message_lower]
            if matched_keywords:
                # Calculate score based on keyword matches
                score = min(1.0, len(matched_keywords) * 0.3 +
                           sum(1 for word in words if word in keywords) * 0.1)
                emotion_scores[emotion.value] = score
                found_keywords[emotion.value] = matched_keywords

        # Determine primary emotion
        if emotion_scores:
            primary_emotion_value = max(emotion_scores, key=emotion_scores.get)
            primary_emotion = Emotion(primary_emotion_value)
            confidence = emotion_scores[primary_emotion_value]
            keywords = found_keywords[primary_emotion_value]
        else:
            primary_emotion = Emotion.NEUTRAL
            confidence = 0.5
            keywords = []

        # Determine intensity
        intensity = self._determine_intensity(confidence, len(words))

        # Determine sentiment
        sentiment = self._determine_sentiment(message_lower, emotion_scores)

        # Normalize all emotion scores to ensure they sum to ~1
        if emotion_scores:
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v / total for k, v in emotion_scores.items()}

        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            confidence=confidence,
            emotions=emotion_scores,
            intensity=intensity,
            keywords=keywords,
            sentiment=sentiment
        )

    def _determine_intensity(self, confidence: float, word_count: int) -> str:
        """Determine emotion intensity level."""
        if confidence >= 0.7 or word_count > 15:
            return "high"
        elif confidence >= 0.4 or word_count > 8:
            return "medium"
        else:
            return "low"

    def _determine_sentiment(
        self,
        message_lower: str,
        emotion_scores: Dict[str, float]
    ) -> str:
        """Determine overall sentiment."""
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in message_lower)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in message_lower)

        # Also consider emotion scores
        positive_emotions = sum(
            score for emotion, score in emotion_scores.items()
            if emotion in [Emotion.JOY.value, Emotion.HOPE.value, Emotion.GRATITUDE.value]
        )
        negative_emotions = sum(
            score for emotion, score in emotion_scores.items()
            if emotion in [Emotion.SADNESS.value, Emotion.ANGER.value,
                          Emotion.FEAR.value, Emotion.LONELINESS.value,
                          Emotion.FRUSTRATION.value]
        )

        total_positive = positive_count + positive_emotions * 2
        total_negative = negative_count + negative_emotions * 2

        if total_positive > total_negative:
            return "positive"
        elif total_negative > total_positive:
            return "negative"
        else:
            return "neutral"

    def get_emotional_summary(
        self,
        analyses: List[EmotionAnalysis]
    ) -> Dict[str, Any]:
        """
        Get summary statistics from multiple emotion analyses.

        Args:
            analyses: List of EmotionAnalysis objects

        Returns:
            Summary statistics
        """
        if not analyses:
            return {
                "most_common_emotion": Emotion.NEUTRAL.value,
                "average_confidence": 0.0,
                "emotion_distribution": {},
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
            }

        # Count emotions
        emotion_counts = {}
        total_confidence = 0
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for analysis in analyses:
            emotion = analysis.primary_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += analysis.confidence
            sentiment_counts[analysis.sentiment] += 1

        # Find most common emotion
        most_common = max(emotion_counts, key=emotion_counts.get)

        return {
            "most_common_emotion": most_common,
            "average_confidence": total_confidence / len(analyses),
            "emotion_distribution": emotion_counts,
            "sentiment_distribution": sentiment_counts,
            "total_analyzed": len(analyses)
        }

    def detect_emotions_batch(
        self,
        messages: List[str]
    ) -> List[EmotionAnalysis]:
        """
        Detect emotions for multiple messages.

        Args:
            messages: List of messages

        Returns:
            List of EmotionAnalysis objects
        """
        return [self.detect_emotion(msg) for msg in messages]
