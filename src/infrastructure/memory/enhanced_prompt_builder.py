"""
Enhanced Prompt Builder for therapy chatbot with patient data integration.

Builds comprehensive prompts that integrate:
- Retrieved conversation context from Qdrant
- Patient profile data from MongoDB
- Emotion analysis
- Behavior patterns
- Returns structured JSON responses with next question and updated patient data
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedPromptBuilder:
    """
    Enhanced prompt builder for therapy chatbot applications.

    Integrates patient data, emotion analysis, and conversation context
    to generate structured responses that include behavior pattern updates.
    """

    def __init__(
        self,
        max_context_items: int = 5,
        include_timestamps: bool = False,
        include_similarity_scores: bool = False
    ):
        """
        Initialize the enhanced prompt builder.

        Args:
            max_context_items: Maximum number of retrieved context items
            include_timestamps: Whether to show timestamps
            include_similarity_scores: Whether to show similarity scores
        """
        self.max_context_items = max_context_items
        self.include_timestamps = include_timestamps
        self.include_similarity_scores = include_similarity_scores

    def build_system_prompt(
        self,
        patient_profile: Optional[Dict[str, Any]] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        emotion_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the system prompt with all available context.

        Args:
            patient_profile: Patient profile from MongoDB
            retrieved_context: Retrieved conversation chunks from Qdrant
            emotion_summary: Emotion analysis summary

        Returns:
            Formatted system prompt string
        """
        sections = []

        # System instructions
        sections.append(self._get_system_instructions())

        # Patient information section
        if patient_profile:
            sections.append(self._format_patient_profile(patient_profile))

        # Behavior patterns section
        if patient_profile and patient_profile.get("behavior_patterns"):
            sections.append(self._format_behavior_patterns(
                patient_profile["behavior_patterns"]
            ))

        # Retrieved conversation context
        if retrieved_context:
            sections.append(self._format_retrieved_context_with_emotions(
                retrieved_context
            ))

        # Emotion summary
        if emotion_summary:
            sections.append(self._format_emotion_summary(emotion_summary))

        # Response format instructions
        sections.append(self._get_response_format_instructions())

        return "\n\n".join(sections)

    def _get_system_instructions(self) -> str:
        """Core system instructions for the chatbot."""
        return """You are a compassionate mental health support companion. Your role is to provide
empathetic support while tracking and understanding the patient's emotional journey.

IMPORTANT: You must respond with a valid JSON object containing:
1. "response": Your conversational response to the user
2. "next_question": A thoughtful follow-up question to deepen understanding
3. "behavior_update": Any detected behavior patterns (or null)
4. "emotion_update": The current emotional state (or null)
5. "risk_assessment": Current risk level (low/medium/high/critical) and reasoning

Your approach:
- Be warm, empathetic, and non-judgmental
- Track emotional patterns across conversations
- Notice behavior changes (social withdrawal, sleep issues, mood swings, etc.)
- Ask open-ended questions that encourage sharing
- Validate feelings before offering suggestions
- Reference past experiences naturally when relevant"""

    def _format_patient_profile(self, profile: Dict[str, Any]) -> str:
        """Format patient profile information."""
        sections = []

        sections.append("📋 PATIENT PROFILE:")

        if profile.get("name"):
            sections.append(f"Name: {profile['name']}")

        if profile.get("age"):
            sections.append(f"Age: {profile['age']}")

        if profile.get("known_conditions"):
            sections.append(f"Known Conditions: {', '.join(profile['known_conditions'])}")

        if profile.get("medications"):
            sections.append(f"Medications: {', '.join(profile['medications'])}")

        if profile.get("triggers"):
            sections.append(f"Known Triggers: {', '.join(profile['triggers'])}")

        if profile.get("treatment_goals"):
            sections.append(f"Treatment Goals:")
            for goal in profile['treatment_goals']:
                sections.append(f"  - {goal}")

        if profile.get("risk_level"):
            risk_emoji = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🟠",
                "critical": "🔴"
            }
            sections.append(
                f"Current Risk Level: {risk_emoji.get(profile['risk_level'], '⚪')} "
                f"{profile['risk_level'].upper()}"
            )

        if profile.get("total_conversations", 0) > 0:
            sections.append(f"Previous Sessions: {profile['total_conversations']}")

        return "\n".join(sections)

    def _format_behavior_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format detected behavior patterns."""
        lines = ["\n🔍 DETECTED BEHAVIOR PATTERNS:"]

        if not patterns:
            lines.append("  No significant patterns detected yet.")
            return "\n".join(lines)

        for pattern in patterns[-5:]:  # Show last 5 patterns
            pattern_type = pattern.get("pattern_type", "unknown")
            description = pattern.get("description", "")
            severity = pattern.get("severity", "unknown")
            frequency = pattern.get("frequency", 0)

            lines.append(f"\n  ⚠️  {pattern_type.replace('_', ' ').title()}")
            lines.append(f"     Description: {description}")
            lines.append(f"     Severity: {severity} | Occurrences: {frequency}")

            if pattern.get("associated_emotions"):
                lines.append(f"     Associated Emotions: {', '.join(pattern['associated_emotions'])}")

        return "\n".join(lines)

    def _format_retrieved_context_with_emotions(
        self,
        retrieved_context: List[Dict[str, Any]]
    ) -> str:
        """Format retrieved context with emotion metadata."""
        lines = ["\n💭 RELEVANT PAST CONVERSATIONS:"]

        for i, item in enumerate(retrieved_context[:self.max_context_items], 1):
            content = item.get("content", "")[:150]
            role = item.get("role", "user")
            emotion = item.get("emotion")
            intensity = item.get("emotion_intensity", "")
            timestamp = item.get("timestamp", "")

            # Format role
            if role == "user":
                role_label = "User"
                emotion_icon = self._get_emotion_icon(emotion)
            else:
                role_label = "You"
                emotion_icon = ""

            # Build entry
            lines.append(f"\n  {i}. [{role_label}] {emotion_icon}")
            lines.append(f"     \"{content}...\"")

            # Add emotion info if available
            if emotion:
                lines.append(f"     Emotion: {emotion} ({intensity} intensity)")

            if self.include_timestamps and timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    lines.append(f"     Date: {dt.strftime('%b %d, %Y')}")
                except:
                    pass

        return "\n".join(lines)

    def _format_emotion_summary(self, summary: Dict[str, Any]) -> str:
        """Format emotion analysis summary."""
        lines = ["\n📊 EMOTION ANALYSIS:"]

        most_common = summary.get("most_common_emotion", "neutral")
        lines.append(f"Most Common Emotion: {most_common}")

        if summary.get("emotion_distribution"):
            lines.append("\nEmotion Distribution:")
            for emotion, count in summary["emotion_distribution"].items():
                lines.append(f"  - {emotion}: {count} occurrences")

        if summary.get("sentiment_distribution"):
            sentiment = summary["sentiment_distribution"]
            lines.append(f"\nOverall Sentiment: {sentiment.get('positive', 0)} positive, "
                        f"{sentiment.get('negative', 0)} negative, "
                        f"{sentiment.get('neutral', 0)} neutral")

        return "\n".join(lines)

    def _get_emotion_icon(self, emotion: Optional[str]) -> str:
        """Get emoji for emotion."""
        emotion_icons = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "anxiety": "😰",
            "fear": "😨",
            "disgust": "🤢",
            "surprise": "😲",
            "hope": "🌟",
            "gratitude": "🙏",
            "loneliness": "😔",
            "frustration": "😤",
            "confusion": "😕",
            "neutral": "😐"
        }
        return emotion_icons.get(emotion, "")

    def _get_response_format_instructions(self) -> str:
        """Instructions for JSON response format."""
        return """\n📝 RESPONSE FORMAT:

You MUST respond with valid JSON in this exact format:

{
  "response": "Your conversational response to the user",
  "next_question": "A thoughtful follow-up question to deepen understanding",
  "behavior_update": {
    "pattern_type": "e.g., social_withdrawal, sleep_disruption, mood_swings",
    "description": "Brief description of the pattern",
    "severity": "mild/moderate/severe",
    "associated_emotions": ["emotion1", "emotion2"]
  } | null,
  "emotion_update": {
    "primary_emotion": "detected_emotion",
    "sentiment": "positive/negative/neutral",
    "intensity": "low/medium/high",
    "keywords": ["triggering", "keywords"]
  } | null,
  "risk_assessment": {
    "level": "low/medium/high/critical",
    "reasoning": "Brief explanation for the risk level",
    "recommendations": ["recommendation1", "recommendation2"]
  }
}

Guidelines:
- behavior_update: Only include if you detect a NEW or CHANGED pattern
- emotion_update: Include the current emotional state
- risk_assessment: Always include, update level if concerning content is detected
- next_question: Should be open-ended and relevant to the conversation
- response: Should be warm, empathetic, and natural"""

    def parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's JSON response.

        Args:
            llm_response: Raw response string from LLM

        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to extract JSON from the response
            # Handle cases where the model might add text before/after the JSON
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                response_data = json.loads(json_str)

                # Validate required fields
                required_fields = ["response", "next_question", "risk_assessment"]
                for field in required_fields:
                    if field not in response_data:
                        logger.warning(f"Missing required field: {field}")
                        response_data[field] = self._get_default_value(field)

                # Ensure optional fields exist
                if "behavior_update" not in response_data:
                    response_data["behavior_update"] = None
                if "emotion_update" not in response_data:
                    response_data["emotion_update"] = None

                return response_data
            else:
                logger.warning("No JSON found in response, using fallback")
                return self._create_fallback_response(llm_response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._create_fallback_response(llm_response)

    def _get_default_value(self, field: str) -> Any:
        """Get default value for a field."""
        defaults = {
            "response": "I'm here to listen and support you.",
            "next_question": "How are you feeling right now?",
            "risk_assessment": {
                "level": "low",
                "reasoning": "No concerning content detected",
                "recommendations": []
            }
        }
        return defaults.get(field)

    def _create_fallback_response(self, raw_response: str) -> Dict[str, Any]:
        """Create a fallback response when JSON parsing fails."""
        return {
            "response": raw_response,
            "next_question": "Can you tell me more about how you're feeling?",
            "behavior_update": None,
            "emotion_update": None,
            "risk_assessment": {
                "level": "low",
                "reasoning": "Unable to analyze - JSON parsing failed",
                "recommendations": []
            }
        }


class TherapyEnhancedPromptBuilder(EnhancedPromptBuilder):
    """
    Specialized enhanced prompt builder for therapy applications.

    Includes additional context for mental health support scenarios.
    """

    def _get_system_instructions(self) -> str:
        """Therapy-specific system instructions."""
        return """You are a compassionate mental health support companion working alongside
healthcare professionals. Your role is to provide empathetic support while tracking
the patient's emotional journey and behavior patterns.

CRITICAL REMINDERS:
- You are NOT a replacement for professional mental health care
- If someone expresses thoughts of self-harm, provide crisis resources
- Always recommend professional help for serious mental health concerns
- Document any concerning patterns for healthcare provider review

RESPONSE REQUIREMENTS:
You must respond with a valid JSON object containing:
1. "response": Your warm, empathetic conversational response
2. "next_question": A thoughtful follow-up question to deepen understanding
3. "behavior_update": Any NEW or CHANGED behavior patterns (or null)
4. "emotion_update": Current emotional state analysis (or null)
5. "risk_assessment": Current risk level with reasoning

Your approach:
- Validate emotions first: "It sounds like you're feeling..."
- Notice patterns: "I've noticed you've mentioned feeling..."
- Ask gentle questions: "Can you tell me more about..."
- Share coping strategies when appropriate
- Always maintain hope and connection
- Reference past conversations when relevant to show continuity"""
