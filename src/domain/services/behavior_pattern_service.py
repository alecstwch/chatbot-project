"""
Behavior Pattern Detection Service for analyzing user messages.

This service detects behavior patterns in user messages to provide
comprehensive behavioral health tracking and pattern recognition.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BehaviorPattern(str, Enum):
    """Behavior pattern categories."""
    SOCIAL_WITHDRAWAL = "social_withdrawal"
    SLEEP_DISRUPTION = "sleep_disruption"
    MOOD_SWINGS = "mood_swings"
    APPETITE_CHANGE = "appetite_change"
    ANXIETY_INCREASE = "anxiety_increase"
    DEPRESSION_SYMPTOMS = "depression_symptoms"
    SELF_HARM_RISK = "self_harm_risk"
    SUBSTANCE_USE = "substance_use"
    IRRITABILITY = "irritability"
    CONCENTRATION_DIFFICULTY = "concentration_difficulty"
    HOPELESSNESS = "hopelessness"
    AGITATION = "agitation"
    FATIGUE = "fatigue"
    GUILT = "guilt"
    AVOIDANCE = "avoidance"


class Severity(str, Enum):
    """Severity levels for behavior patterns."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class PatternDetection:
    """Result of behavior pattern detection."""
    pattern_type: BehaviorPattern
    confidence: float
    severity: Severity
    indicators: List[str]  # Keywords/phrases that triggered detection
    description: str
    suggested_response: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "indicators": self.indicators,
            "description": self.description,
            "suggested_response": self.suggested_response
        }


@dataclass
class PatternHistory:
    """Historical data for a behavior pattern."""
    pattern_type: BehaviorPattern
    first_detected: str  # ISO timestamp
    last_detected: str  # ISO timestamp
    frequency: int  # Number of occurrences
    severity_history: List[Tuple[str, Severity]]  # (timestamp, severity)
    associated_emotions: List[str]
    examples: List[str]  # Example messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "frequency": self.frequency,
            "severity_history": [
                {"timestamp": ts, "severity": sev.value}
                for ts, sev in self.severity_history
            ],
            "associated_emotions": self.associated_emotions,
            "examples": self.examples
        }


class BehaviorPatternService:
    """
    Service for detecting and tracking behavior patterns in user messages.

    Analyzes messages for signs of behavioral health patterns and
    provides tools for searching historical pattern data.
    """

    # Pattern keywords and indicators
    PATTERN_INDICATORS = {
        BehaviorPattern.SOCIAL_WITHDRAWAL: {
            "keywords": [
                "alone", "lonely", "isolate", "withdraw", "avoid people",
                "don't want to see", "stay home", "isolated", "by myself",
                "not socializing", "avoiding friends", "skip plans", "cancel plans"
            ],
            "phrases": [
                "haven't seen my friends",
                "don't want to go out",
                "staying home alone",
                "avoiding people",
                "feel isolated"
            ]
        },
        BehaviorPattern.SLEEP_DISRUPTION: {
            "keywords": [
                "sleep", "insomnia", "awake", "tired", "exhausted", "nightmare",
                "can't sleep", "restless", "up all night", "sleeping too much",
                "fatigue", "no energy", "trouble sleeping"
            ],
            "phrases": [
                "can't fall asleep",
                "wake up at night",
                "sleeping all day",
                "tired all the time",
                "insomnia"
            ]
        },
        BehaviorPattern.MOOD_SWINGS: {
            "keywords": [
                "mood", "emotional", "up and down", "unpredictable", "irritable",
                "sudden change", "emotional rollercoaster", "fine one minute",
                "angry the next", "mood swings", "unstable"
            ],
            "phrases": [
                "mood changes suddenly",
                "up and down",
                "emotional rollercoaster",
                "one minute fine",
                "next minute angry"
            ]
        },
        BehaviorPattern.APPETITE_CHANGE: {
            "keywords": [
                "appetite", "eating", "hungry", "food", "lose weight", "gain weight",
                "not eating", "overeating", "no appetite", "eating too much",
                "lost appetite", "binge eating"
            ],
            "phrases": [
                "lost my appetite",
                "not hungry anymore",
                "eating too much",
                "gaining weight",
                "losing weight"
            ]
        },
        BehaviorPattern.ANXIETY_INCREASE: {
            "keywords": [
                "anxious", "worried", "nervous", "panic", "overwhelmed", "stressed",
                "racing thoughts", "can't relax", "on edge", "tense", "restless",
                "anxiety", "panic attack"
            ],
            "phrases": [
                "feeling overwhelmed",
                "can't stop worrying",
                "racing thoughts",
                "panic attack",
                "always anxious"
            ]
        },
        BehaviorPattern.DEPRESSION_SYMPTOMS: {
            "keywords": [
                "depressed", "hopeless", "empty", "numb", "sad", "unhappy",
                "no motivation", "don't care", "pointless", "worthless",
                "can't enjoy", "nothing matters"
            ],
            "phrases": [
                "feel hopeless",
                "nothing matters anymore",
                "can't enjoy anything",
                "feel worthless",
                "no motivation"
            ]
        },
        BehaviorPattern.SELF_HARM_RISK: {
            "keywords": [
                "hurt myself", "harm", "cut", "suicide", "kill myself",
                "end it", "die", "suicidal", "self-harm", "want to die"
            ],
            "phrases": [
                "want to kill myself",
                "want to hurt myself",
                "thinking of suicide",
                "end it all",
                "self-harm"
            ]
        },
        BehaviorPattern.SUBSTANCE_USE: {
            "keywords": [
                "drinking", "alcohol", "drugs", "medication", "pills", "substance",
                "addicted", "dependence", "using", "abuse"
            ],
            "phrases": [
                "drinking too much",
                "using drugs",
                "relying on alcohol",
                "taking more pills"
            ]
        },
        BehaviorPattern.IRRITABILITY: {
            "keywords": [
                "irritable", "annoyed", "frustrated", "angry", "agitated",
                "short temper", "snapping", "lose patience", "easily angered"
            ],
            "phrases": [
                "getting annoyed easily",
                "snapping at people",
                "losing my temper",
                "irritable all the time"
            ]
        },
        BehaviorPattern.CONCENTRATION_DIFFICULTY: {
            "keywords": [
                "concentrate", "focus", "distracted", "can't focus", "forgetful",
                "brain fog", "confused", "scatterbrained", "attention"
            ],
            "phrases": [
                "can't concentrate",
                "trouble focusing",
                "brain fog",
                "easily distracted",
                "forget things"
            ]
        },
        BehaviorPattern.HOPELESSNESS: {
            "keywords": [
                "hopeless", "no hope", "future", "pointless", "give up",
                "nothing will change", "no point", "doomed", "helpless"
            ],
            "phrases": [
                "feel hopeless",
                "no point in trying",
                "nothing will get better",
                "giving up",
                "no hope for the future"
            ]
        },
        BehaviorPattern.AGITATION: {
            "keywords": [
                "agitated", "restless", "can't sit still", "pacing", "fidgeting",
                "on edge", "tense", "unable to relax"
            ],
            "phrases": [
                "can't sit still",
                "feeling agitated",
                "restless energy",
                "pacing around",
                "on edge"
            ]
        },
        BehaviorPattern.FATIGUE: {
            "keywords": [
                "tired", "exhausted", "weary", "drained", "no energy", "lethargic",
                "worn out", "sleepy", "burnout"
            ],
            "phrases": [
                "exhausted all the time",
                "no energy at all",
                "feel drained",
                "constant fatigue",
                "burned out"
            ]
        },
        BehaviorPattern.GUILT: {
            "keywords": [
                "guilt", "guilty", "my fault", "blame", "ashamed", "regret",
                "should have", "fault", "sorry"
            ],
            "phrases": [
                "feel guilty",
                "my fault",
                "blaming myself",
                "feel ashamed",
                "should have done"
            ]
        },
        BehaviorPattern.AVOIDANCE: {
            "keywords": [
                "avoid", "procrastinate", "putting off", "ignoring", "delaying",
                "can't face", "stay away", "dodging", "evading"
            ],
            "phrases": [
                "avoiding everything",
                "putting things off",
                "can't face it",
                "ignoring problems",
                "procrastinating"
            ]
        }
    }

    # Suggested responses for each pattern
    SUGGESTED_RESPONSES = {
        BehaviorPattern.SOCIAL_WITHDRAWAL: "It sounds like you've been spending more time alone lately. Social connections are important for our wellbeing. Would you like to talk about what might be making social situations difficult?",
        BehaviorPattern.SLEEP_DISRUPTION: "Sleep issues can really impact how we feel. Are you having trouble falling asleep, staying asleep, or both? Let's explore what might be affecting your rest.",
        BehaviorPattern.MOOD_SWINGS: "I notice you've mentioned experiencing ups and downs. Tracking these patterns can help us understand what triggers these changes. Have you noticed anything specific that sets off these mood shifts?",
        BehaviorPattern.APPETITE_CHANGE: "Changes in eating habits can be a sign that our body or mind is under stress. Are you eating less than usual, or has your relationship with food changed?",
        BehaviorPattern.ANXIETY_INCREASE: "Anxiety can feel overwhelming. Let's work together to identify what's triggering these feelings and develop some coping strategies.",
        BehaviorPattern.DEPRESSION_SYMPTOMS: "I hear you're going through a difficult time. Your feelings are valid, and I'm here to support you. Would you like to talk more about what's been making you feel this way?",
        BehaviorPattern.SELF_HARM_RISK: "I'm concerned about what you're sharing. Please know that there are people who want to help. If you're in immediate danger, please contact crisis services. Your life matters.",
        BehaviorPattern.SUBSTANCE_USE: "I've noticed some patterns that suggest changes in how you're coping. It's brave of you to share this. Would you be open to discussing healthier ways to manage these feelings?",
        BehaviorPattern.IRRITABILITY: "It sounds like things have been frustrating you lately. Irritability often builds up when we're under stress. What's been the main source of this frustration?",
        BehaviorPattern.CONCENTRATION_DIFFICULTY: "Trouble focusing can make everything feel more difficult. This often happens when our minds are overloaded. What's been occupying your thoughts lately?",
        BehaviorPattern.HOPELESSNESS: "I can hear how heavy things feel right now. Please know that difficult times don't last forever, even when it feels like they will. What would help you feel even a little bit better today?",
        BehaviorPattern.AGITATION: "That restless, on-edge feeling is really uncomfortable. Let's talk about what might be causing this agitation and find ways to help you feel more settled.",
        BehaviorPattern.FATIGUE: "Being exhausted all the time makes everything harder. Let's explore whether this fatigue is physical, emotional, or both.",
        BehaviorPattern.GUILT: "Guilt can be a heavy burden to carry. Everyone makes mistakes, and you deserve compassion. Would you like to talk about what's making you feel this way?",
        BehaviorPattern.AVOIDANCE: "Sometimes avoiding things feels safer in the moment, but it can make problems grow over time. What feels too overwhelming to face right now?"
    }

    def __init__(self):
        """Initialize the behavior pattern service."""
        logger.info("Behavior Pattern Service initialized")

    def detect_patterns(
        self,
        message: str,
        emotion: Optional[str] = None
    ) -> List[PatternDetection]:
        """
        Detect behavior patterns in a user message.

        Args:
            message: The user's message
            emotion: Optional detected emotion to add context

        Returns:
            List of detected patterns with confidence scores
        """
        if not message or not message.strip():
            return []

        message_lower = message.lower()
        detected_patterns = []

        for pattern_type, pattern_data in self.PATTERN_INDICATORS.items():
            keywords = pattern_data["keywords"]
            phrases = pattern_data["phrases"]

            # Count keyword matches
            keyword_matches = [kw for kw in keywords if kw in message_lower]

            # Count phrase matches (check if phrase is contained in message)
            phrase_matches = [ph for ph in phrases if ph in message_lower]

            total_matches = len(keyword_matches) + (len(phrase_matches) * 2)

            if total_matches > 0:
                # Calculate confidence based on matches
                confidence = min(1.0, total_matches * 0.15)

                # Determine severity based on confidence and emotion
                severity = self._determine_severity(confidence, emotion, message_lower)

                # Gather indicators
                indicators = keyword_matches + phrase_matches

                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    severity=severity,
                    indicators=indicators[:5],  # Keep top 5
                    description=self._get_pattern_description(pattern_type),
                    suggested_response=self.SUGGESTED_RESPONSES.get(pattern_type, "")
                )
                detected_patterns.append(detection)

        # Sort by confidence and return
        detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        return detected_patterns

    def _determine_severity(
        self,
        confidence: float,
        emotion: Optional[str],
        message: str
    ) -> Severity:
        """Determine severity level of a detected pattern."""
        # High severity indicators
        severe_indicators = ["suicide", "kill myself", "self-harm", "end it", "die"]
        if any(indicator in message for indicator in severe_indicators):
            return Severity.SEVERE

        # Moderate severity indicators
        if confidence >= 0.6:
            return Severity.MODERATE

        # Consider emotion in severity
        if emotion in ["anger", "fear", "sadness"]:
            return Severity.MODERATE

        # Default to mild
        return Severity.MILD

    def _get_pattern_description(self, pattern_type: BehaviorPattern) -> str:
        """Get description for a pattern type."""
        descriptions = {
            BehaviorPattern.SOCIAL_WITHDRAWAL: "Reduced social interaction and isolation",
            BehaviorPattern.SLEEP_DISRUPTION: "Disturbances in sleep patterns or quality",
            BehaviorPattern.MOOD_SWINGS: "Rapid or intense changes in emotional state",
            BehaviorPattern.APPETITE_CHANGE: "Significant changes in eating habits",
            BehaviorPattern.ANXIETY_INCREASE: "Heightened anxiety or worry",
            BehaviorPattern.DEPRESSION_SYMPTOMS: "Signs of depression or low mood",
            BehaviorPattern.SELF_HARM_RISK: "Indicators of self-harm or suicidal thoughts",
            BehaviorPattern.SUBSTANCE_USE: "References to substance use or dependency",
            BehaviorPattern.IRRITABILITY: "Increased irritability or short temper",
            BehaviorPattern.CONCENTRATION_DIFFICULTY: "Trouble focusing or concentrating",
            BehaviorPattern.HOPELESSNESS: "Feelings of hopelessness about the future",
            BehaviorPattern.AGITATION: "Restlessness or inability to relax",
            BehaviorPattern.FATIGUE: "Persistent tiredness or lack of energy",
            BehaviorPattern.GUILT: "Feelings of guilt or self-blame",
            BehaviorPattern.AVOIDANCE: "Avoiding difficult situations or tasks"
        }
        return descriptions.get(pattern_type, "Behavioral pattern detected")

    def search_pattern_history(
        self,
        rag_messages: List[Dict[str, Any]],
        pattern_type: BehaviorPattern,
        min_confidence: float = 0.3
    ) -> PatternHistory:
        """
        Search RAG messages for historical occurrences of a pattern.

        Args:
            rag_messages: List of messages from RAG memory
            pattern_type: The pattern to search for
            min_confidence: Minimum confidence threshold

        Returns:
            PatternHistory with aggregated data
        """
        current_time = datetime.utcnow()
        examples = []
        emotion_counts = {}
        severity_counts = {Severity.MILD: 0, Severity.MODERATE: 0, Severity.SEVERE: 0}
        first_detected = None
        last_detected = None

        for msg in rag_messages:
            # Support both "content" and "text" field names for compatibility
            content = msg.get("content") or msg.get("text", "")
            timestamp = msg.get("timestamp", "")
            emotion = msg.get("emotion")

            # Detect pattern in this historical message
            detections = self.detect_patterns(content, emotion)

            for detection in detections:
                if detection.pattern_type == pattern_type and detection.confidence >= min_confidence:
                    if not first_detected:
                        first_detected = timestamp

                    last_detected = timestamp
                    examples.append(content[:100])  # Truncate

                    if emotion:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                    severity_counts[detection.severity] += 1

        if not examples:
            return None

        # Build severity history
        severity_history = [(last_detected or current_time.isoformat(),
                           max(severity_counts, key=severity_counts.get))]

        return PatternHistory(
            pattern_type=pattern_type,
            first_detected=first_detected or current_time.isoformat(),
            last_detected=last_detected or current_time.isoformat(),
            frequency=len(examples),
            severity_history=severity_history,
            associated_emotions=list(emotion_counts.keys()),
            examples=examples[:5]  # Keep top 5 examples
        )

    def get_trending_patterns(
        self,
        rag_messages: List[Dict[str, Any]],
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get patterns that have increased in frequency recently.

        Args:
            rag_messages: List of messages from RAG memory
            days: Number of days to look back

        Returns:
            List of trending patterns with frequency data
        """
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        recent_messages = [
            msg for msg in rag_messages
            if msg.get("timestamp", "") >= cutoff_date
        ]

        pattern_counts = {}
        pattern_severity = {}

        for msg in recent_messages:
            # Support both "content" and "text" field names for compatibility
            message_content = msg.get("content") or msg.get("text", "")
            detections = self.detect_patterns(message_content)

            for detection in detections:
                pattern = detection.pattern_type.value
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

                if pattern not in pattern_severity:
                    pattern_severity[pattern] = []
                pattern_severity[pattern].append(detection.severity.value)

        # Calculate trends
        trending = []
        for pattern, count in pattern_counts.items():
            if count >= 2:  # At least 2 occurrences to be trending
                severities = pattern_severity.get(pattern, [])
                most_severe = self._get_most_severe(severities)

                trending.append({
                    "pattern_type": pattern,
                    "frequency": count,
                    "avg_severity": most_severe,
                    "period_days": days
                })

        # Sort by frequency
        trending.sort(key=lambda x: x["frequency"], reverse=True)
        return trending

    def _get_most_severe(self, severities: List[str]) -> str:
        """Get the most severe level from a list."""
        severity_order = {"severe": 3, "moderate": 2, "mild": 1}
        if not severities:
            return "mild"

        return max(severities, key=lambda s: severity_order.get(s, 0))

    def compare_patterns(
        self,
        rag_messages: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Compare current patterns with historical baseline.

        Args:
            rag_messages: List of messages from RAG memory
            user_id: User identifier

        Returns:
            Comparison analysis
        """
        all_messages = rag_messages
        total_messages = len(all_messages)

        if total_messages == 0:
            return {"error": "No messages to analyze"}

        # Detect all patterns
        all_patterns = []
        for msg in all_messages:
            # Support both "content" and "text" field names for compatibility
            message_content = msg.get("content") or msg.get("text", "")
            detections = self.detect_patterns(message_content)
            all_patterns.extend(detections)

        # Count patterns
        pattern_counts = {}
        for detection in all_patterns:
            pattern = detection.pattern_type.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Calculate percentages
        pattern_percentages = {
            pattern: (count / total_messages) * 100
            for pattern, count in pattern_counts.items()
        }

        # Identify dominant patterns (>10% of messages)
        dominant_patterns = {
            pattern: pct for pattern, pct in pattern_percentages.items()
            if pct > 10
        }

        return {
            "total_messages_analyzed": total_messages,
            "pattern_counts": pattern_counts,
            "pattern_percentages": pattern_percentages,
            "dominant_patterns": dominant_patterns,
            "total_patterns_detected": len(all_patterns)
        }
