# Enhanced RAG Chatbot System

**Date**: January 24, 2026
**Author**: Claude Code
**Version**: 1.0.0

## Overview

This document describes the Enhanced RAG Chatbot system that integrates emotion detection, MongoDB patient file management, and structured JSON responses. The system is designed for therapy/mental health support applications with comprehensive behavior pattern tracking.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Data Isolation & Privacy](#data-isolation--privacy)
3. [New Components](#new-components)
4. [Modified Files](#modified-files)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Data Models](#data-models)
9. [Examples](#examples)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Enhanced RAG Chatbot                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Emotion         │    │   MongoDB        │              │
│  │  Detection       │◄──►│   Patient        │              │
│  │  Service         │    │   Repository     │              │
│  └──────────────────┘    └──────────────────┘              │
│           │                       ▲                         │
│           ▼                       │                         │
│  ┌──────────────────────────────────────────────────┐      │
│  │        Enhanced RAG Chatbot                       │      │
│  │  - Integrates all components                      │      │
│  │  - Returns structured JSON responses              │      │
│  └──────────────────────────────────────────────────┘      │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Enhanced        │    │   Qdrant RAG     │              │
│  │  Prompt Builder  │    │   Memory         │              │
│  │                  │    │   (with emotion) │              │
│  └──────────────────┘    └──────────────────┘              │
│           │                       ▲                         │
│           └───────────────────────┘                         │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     CLI       │
                    │   Interface   │
                    └───────────────┘
```

---

## Data Isolation & Privacy

### Critical Privacy Feature

**Every retrieval operation in the system enforces strict patient data isolation.**

When searching or retrieving messages by emotion (or any other criteria), the system **ALWAYS** filters by `user_id` first. This ensures that:

- ✅ Patient A can NEVER see Patient B's messages
- ✅ Emotion-based searches only return messages from THAT specific patient
- ✅ Behavior patterns are tracked per-patient, never across patients
- ✅ MongoDB collections are scoped to individual user IDs

### How It Works

All Qdrant retrieval methods use mandatory `user_id` filtering:

```python
# Example: Searching for "anxiety" emotions
filter_conditions = [
    FieldCondition(key="user_id", match=MatchValue(value="patient_123")),  # REQUIRED
    FieldCondition(key="emotion", match=MatchValue(value="anxiety"))
]
```

**Result**: Only returns messages from `patient_123` with `anxiety` emotion.

### Retrieval Methods with Data Isolation

| Method | user_id Filter | Description |
|--------|----------------|-------------|
| `retrieve_relevant_context()` | ✅ Always | Semantic search within patient's messages |
| `search_by_emotion()` | ✅ Always | Filter by emotion within patient's messages |
| `retrieve_by_emotion_context()` | ✅ Always | Semantic + emotion filter within patient's messages |
| `search_by_intent()` | ✅ Always | Filter by intent within patient's messages |
| `get_all_user_messages()` | ✅ Always | Get all messages for specific patient |
| `get_user_message_count()` | ✅ Always | Count messages for specific patient |
| `delete_user_data()` | ✅ Always | Delete all data for specific patient (GDPR) |

### Log Verification

The system logs all retrieval operations with explicit user_id tracking:

```
DEBUG: Retrieved 5 messages for user 'patient_123' with emotion 'anxiety' (data isolation enforced)
```

### Security Considerations

1. **user_id is Required**: All retrieval methods require a user_id parameter
2. **No Cross-Patient Access**: There is no method to retrieve messages across multiple patients
3. **GDPR Compliance**: Each patient's data can be independently deleted via `delete_user_data()`
4. **Session Isolation**: Each conversation is tagged with both `user_id` and `session_id`

---

## New Components

### 1. Emotion Detection Service

**File**: `src/domain/services/emotion_detection_service.py`

**Purpose**: Analyzes user messages to detect emotions and sentiment.

**Features**:
- Detects 12 emotion categories: joy, sadness, anger, anxiety, fear, disgust, surprise, hope, gratitude, loneliness, frustration, confusion
- Calculates confidence scores and intensity levels (low/medium/high)
- Determines overall sentiment (positive/negative/neutral)
- Batch processing support

**Key Classes**:
```python
class Emotion(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "anxiety"
    # ... etc

@dataclass
class EmotionAnalysis:
    primary_emotion: Emotion
    confidence: float
    emotions: Dict[str, float]
    intensity: str
    keywords: List[str]
    sentiment: str

class EmotionDetectionService:
    def detect_emotion(message: str) -> EmotionAnalysis
    def get_emotional_summary(analyses: List[EmotionAnalysis]) -> Dict
    def detect_emotions_batch(messages: List[str]) -> List[EmotionAnalysis]
```

**Usage Example**:
```python
from src.domain.services.emotion_detection_service import EmotionDetectionService

service = EmotionDetectionService()
analysis = service.detect_emotion("I'm feeling really anxious about my presentation")

print(analysis.primary_emotion)  # "anxiety"
print(analysis.intensity)         # "medium"
print(analysis.sentiment)         # "negative"
```

---

### 2. MongoDB Patient Repository

**File**: `src/infrastructure/database/patient_repository.py`

**Purpose**: Manages patient profiles and behavior patterns in MongoDB.

**Features**:
- Patient profile CRUD operations
- Behavior pattern tracking
- Risk level assessment (low/medium/high/critical)
- Session notes and treatment goals
- Emotion summary storage

**Key Classes**:
```python
@dataclass
class BehaviorPattern:
    pattern_type: str
    description: str
    frequency: int
    first_detected: str
    last_detected: str
    severity: str
    associated_emotions: List[str]
    examples: List[str]

@dataclass
class PatientProfile:
    user_id: str
    name: Optional[str]
    age: Optional[int]
    known_conditions: List[str]
    medications: List[str]
    triggers: List[str]
    behavior_patterns: List[Dict[str, Any]]
    risk_level: str
    treatment_goals: List[str]
    # ... etc

class PatientRepository:
    def initialize() -> bool
    def create_patient(profile: PatientProfile) -> bool
    def get_patient(user_id: str) -> Optional[PatientProfile]
    def update_patient(user_id: str, updates: Dict) -> bool
    def add_behavior_pattern(user_id: str, pattern: BehaviorPattern) -> bool
    def add_session_note(user_id: str, note: str, emotion_data: Dict) -> bool
    def update_emotion_summary(user_id: str, summary: Dict) -> bool
    def update_risk_level(user_id: str, level: RiskLevel, reason: str) -> bool
    def get_patients_by_risk_level(level: RiskLevel) -> List[PatientProfile]
```

**Usage Example**:
```python
from src.infrastructure.database.patient_repository import (
    PatientRepository, PatientProfile, BehaviorPattern, RiskLevel
)

repo = PatientRepository()
repo.initialize()

# Create patient
profile = PatientProfile(
    user_id="patient_123",
    name="John Doe",
    age=35,
    known_conditions=["anxiety", "depression"],
    risk_level=RiskLevel.MEDIUM.value
)
repo.create_patient(profile)

# Add behavior pattern
pattern = BehaviorPattern(
    pattern_type="social_withdrawal",
    description="Patient avoids social interactions",
    frequency=5,
    first_detected="2026-01-20T10:00:00",
    last_detected="2026-01-24T15:30:00",
    severity="moderate",
    associated_emotions=["anxiety", "sadness"]
)
repo.add_behavior_pattern("patient_123", pattern)
```

---

### 3. Enhanced RAG Memory Service (Updated)

**File**: `src/infrastructure/memory/rag_memory_service.py`

**Changes**: Added emotion metadata support and emotion-based retrieval.

**New Methods**:
```python
class RAGMemoryService:
    # Updated method with emotion data
    def store_conversation_turn(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        session_id: Optional[str] = None,
        user_intent: Optional[str] = None,
        emotion_data: Optional[Dict[str, Any]] = None  # NEW
    ) -> tuple

    # New methods
    def search_by_emotion(
        self,
        user_id: str,
        emotion: str,
        top_k: int = 10,
        intensity: Optional[str] = None
    ) -> List[Dict[str, Any]]

    def retrieve_by_emotion_context(
        self,
        user_id: str,
        query: str,
        target_emotion: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.4
    ) -> List[Dict[str, Any]]

    def get_emotion_summary(
        self,
        user_id: str,
        limit: int = 100
    ) -> Dict[str, Any]
```

**Emotion Metadata Stored**:
- `emotion`: Primary emotion detected
- `emotion_confidence`: Confidence score (0-1)
- `emotion_intensity`: Intensity level (low/medium/high)
- `sentiment`: Overall sentiment (positive/negative/neutral)
- `emotion_keywords`: Keywords that triggered the emotion

---

### 4. Enhanced Prompt Builder

**File**: `src/infrastructure/memory/enhanced_prompt_builder.py`

**Purpose**: Builds comprehensive prompts that integrate patient data, emotion analysis, and conversation context.

**Key Classes**:
```python
class EnhancedPromptBuilder:
    def build_system_prompt(
        self,
        patient_profile: Optional[Dict[str, Any]] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        emotion_summary: Optional[Dict[str, Any]] = None
    ) -> str

    def parse_llm_response(self, llm_response: str) -> Dict[str, Any]

class TherapyEnhancedPromptBuilder(EnhancedPromptBuilder):
    # Specialized for therapy applications
```

**Prompt Components**:
1. System instructions for therapy support
2. Patient profile information (name, conditions, medications, triggers, risk level)
3. Detected behavior patterns
4. Relevant past conversations with emotion metadata
5. Emotion analysis summary
6. JSON response format instructions

---

### 5. Enhanced RAG Chatbot

**File**: `src/infrastructure/ml/chatbots/enhanced_rag_chatbot.py`

**Purpose**: Main chatbot class that integrates all components.

**Key Features**:
- Emotion detection on every user message
- Patient profile retrieval and management
- Emotion-aware context retrieval from Qdrant
- Structured JSON responses with behavior pattern updates

**Key Methods**:
```python
class EnhancedRAGChatbot:
    def load_model() -> None
    def get_response(user_input: str) -> Dict[str, Any]
    def get_patient_profile() -> Optional[Dict[str, Any]]
    def update_patient_profile(updates: Dict) -> bool
    def search_by_emotion(emotion: str, top_k: int) -> List[Dict]
    def get_emotion_summary() -> Dict[str, Any]
```

---

## Modified Files

### 1. RAG Chatbot CLI

**File**: `src/interfaces/cli/rag_chatbot_cli.py`

**Changes**:
- Added `use_enhanced_mode` parameter (default: `True`)
- Integrated `EnhancedRAGChatbot` alongside standard `RAGChatbot`
- New commands: `emotions`, `profile`
- Enhanced response display with:
  - Detected emotion and intensity
  - Behavior pattern updates
  - Risk assessment
  - Next question suggestions

**New CLI Commands**:
```
emotions  - View emotion history and statistics
profile   - View patient profile information
```

**Command Line Arguments**:
```bash
--enhanced   # Use enhanced mode with MongoDB and emotion detection (default)
--standard   # Use standard RAG mode without enhancements
```

---

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=chatbot_db
MONGODB_MAX_POOL_SIZE=10
MONGODB_MIN_POOL_SIZE=1
MONGODB_TIMEOUT_MS=5000

# Gemini API (already required)
GEMINI_API_KEY=your_api_key_here

# Qdrant Configuration (already required)
QDRANT_USE_LOCAL_STORAGE=true
QDRANT_LOCAL_PATH=./data/qdrant_db
```

### MongoDB Setup

1. Install and start MongoDB:
   ```bash
   # Windows (if using PowerShell script)
   .\install_mongodb.ps1

   # Or manual installation
   # Download from: https://www.mongodb.com/try/download/community
   ```

2. Verify connection:
   ```bash
   mongosh
   > show dbs
   ```

---

## Usage

### Basic Usage (Enhanced Mode)

Enhanced mode is enabled by default:

```bash
python -m src.interfaces.cli.rag_chatbot_cli \
    --user-id patient_123 \
    --user-name "John Doe" \
    --therapy
```

### Standard Mode (Without Enhancements)

```bash
python -m src.interfaces.cli.rag_chatbot_cli --standard
```

### Programmatic Usage

```python
from src.infrastructure.ml.chatbots.enhanced_rag_chatbot import EnhancedRAGChatbot
from src.infrastructure.config.chatbot_settings import NeuralChatbotSettings
from src.infrastructure.config.qdrant_settings import QdrantSettings
from src.infrastructure.config.mongodb_settings import MongoDBSettings

# Initialize chatbot
chatbot = EnhancedRAGChatbot(
    user_id="patient_123",
    user_name="John Doe",
    use_therapy_mode=True
)

# Load all services
chatbot.load_model()

# Get response
response = chatbot.get_response("I've been feeling really anxious lately")

# Response contains:
print(response["response"])          # Main conversational response
print(response["next_question"])     # Follow-up question
print(response["emotion_update"])    # Current emotion state
print(response["behavior_update"])   # Detected patterns (if any)
print(response["risk_assessment"])   # Risk assessment

# View patient profile
profile = chatbot.get_patient_profile()

# Search for past emotions
anxious_moments = chatbot.search_by_emotion("anxiety")

# Get emotion summary
summary = chatbot.get_emotion_summary()
```

---

## API Reference

### Response Format

The chatbot returns structured JSON with the following fields:

```json
{
  "response": "Conversational response to the user",
  "next_question": "Follow-up question to deepen understanding",
  "behavior_update": {
    "pattern_type": "social_withdrawal|sleep_disruption|mood_swings|...",
    "description": "Brief description of the pattern",
    "severity": "mild|moderate|severe",
    "associated_emotions": ["anxiety", "sadness"]
  } | null,
  "emotion_update": {
    "primary_emotion": "detected_emotion",
    "sentiment": "positive|negative|neutral",
    "intensity": "low|medium|high",
    "keywords": ["triggering", "keywords"]
  } | null,
  "risk_assessment": {
    "level": "low|medium|high|critical",
    "reasoning": "Brief explanation for the risk level",
    "recommendations": ["recommendation1", "recommendation2"]
  }
}
```

### Emotion Categories

| Emotion | Keywords (examples) |
|---------|---------------------|
| joy | happy, glad, excited, wonderful, amazing |
| sadness | sad, depressed, unhappy, down, miserable |
| anger | angry, furious, mad, annoyed, frustrated |
| anxiety | anxious, afraid, scared, worried, nervous |
| fear | terrified, panic, fear, dread |
| disgust | disgusted, repulsed, sick, horrible |
| surprise | surprised, shocked, amazed, astonished |
| hope | hope, hopeful, optimistic, wish |
| gratitude | thank, grateful, appreciate, thankful |
| loneliness | lonely, alone, isolated, nobody |
| frustration | frustrated, stuck, pointless, giving up |
| confusion | confused, unclear, puzzled, unsure |

### Behavior Pattern Types

| Pattern Type | Description |
|--------------|-------------|
| social_withdrawal | Avoiding social interactions |
| sleep_disruption | Changes in sleep patterns |
| mood_swings | Rapid emotional changes |
| appetite_changes | Changes in eating habits |
| anxiety_increase | Heightened anxiety levels |
| depression_symptoms | Signs of depression |
| self_harm_risk | Concerning self-harm indicators |

### Risk Levels

| Level | Description | Emoji |
|-------|-------------|-------|
| low | Normal, no concerns | 🟢 |
| medium | Some concerns, monitor | 🟡 |
| high | Significant concerns | 🟠 |
| critical | Immediate attention needed | 🔴 |

---

## Data Models

### PatientProfile Schema

```python
{
    "user_id": str,                    # Required
    "name": Optional[str],
    "created_at": str,                 # ISO timestamp
    "updated_at": str,                 # ISO timestamp
    "age": Optional[int],
    "gender": Optional[str],
    "known_conditions": List[str],     # e.g., ["anxiety", "depression"]
    "medications": List[str],
    "triggers": List[str],
    "behavior_patterns": List[Dict],   # BehaviorPattern objects
    "emotion_summary": Dict,           # Emotion statistics
    "risk_level": str,                 # "low" | "medium" | "high" | "critical"
    "last_risk_assessment": Optional[str],
    "total_conversations": int,
    "last_conversation_date": Optional[str],
    "session_notes": List[Dict],
    "treatment_goals": List[str],
    "progress_notes": List[str]
}
```

### BehaviorPattern Schema

```python
{
    "pattern_type": str,               # Pattern identifier
    "description": str,                # Human-readable description
    "frequency": int,                  # Number of occurrences
    "first_detected": str,             # ISO timestamp
    "last_detected": str,              # ISO timestamp
    "severity": str,                   # "mild" | "moderate" | "severe"
    "associated_emotions": List[str],  # Related emotions
    "examples": List[str]              # Example messages
}
```

### EmotionAnalysis Schema

```python
{
    "primary_emotion": str,            # Main emotion detected
    "confidence": float,               # 0.0 to 1.0
    "emotions": Dict[str, float],      # All emotion scores
    "intensity": str,                  # "low" | "medium" | "high"
    "keywords": List[str],             # Triggering keywords
    "sentiment": str                   # "positive" | "negative" | "neutral"
}
```

---

## Examples

### Example 1: Basic Conversation

**User Input**:
```
I've been having trouble sleeping lately and feel anxious about work
```

**Response**:
```json
{
  "response": "I hear you're going through a difficult time. Sleep issues and work anxiety often go together, and it's completely understandable to feel this way when under stress.",
  "next_question": "Can you tell me more about what specifically at work is causing you anxiety?",
  "emotion_update": {
    "primary_emotion": "anxiety",
    "sentiment": "negative",
    "intensity": "medium",
    "keywords": ["anxious", "trouble", "sleeping"]
  },
  "behavior_update": {
    "pattern_type": "sleep_disruption",
    "description": "Patient reporting sleep difficulties",
    "severity": "mild",
    "associated_emotions": ["anxiety"]
  },
  "risk_assessment": {
    "level": "low",
    "reasoning": "User expressing work stress and sleep issues, no immediate concerns",
    "recommendations": ["Monitor sleep patterns", "Discuss stress management techniques"]
  }
}
```

### Example 2: Pattern Detection

**User Input**:
```
I haven't wanted to see my friends lately, I just prefer staying home alone
```

**Response**:
```json
{
  "response": "It's not uncommon to feel like withdrawing from social activities when we're going through tough times. Sometimes we need space, but isolation can also make feelings worse.",
  "next_question": "How long have you been feeling this way about seeing your friends?",
  "emotion_update": {
    "primary_emotion": "loneliness",
    "sentiment": "negative",
    "intensity": "medium",
    "keywords": ["alone", "friends", "staying home"]
  },
  "behavior_update": {
    "pattern_type": "social_withdrawal",
    "description": "Patient avoiding social interactions, preferring isolation",
    "severity": "moderate",
    "associated_emotions": ["loneliness", "sadness"]
  },
  "risk_assessment": {
    "level": "medium",
    "reasoning": "Social withdrawal can indicate worsening mental health",
    "recommendations": ["Monitor for depression symptoms", "Encourage social connection"]
  }
}
```

### Example 3: CLI Session

```
======================================================================
  🧠 Enhanced RAG Chatbot (Patient Data + Emotion Tracking)
======================================================================

Mode: Therapy Support
User: patient_123
✨ Enhanced Mode: Emotion detection + MongoDB patient files
👤 Patient Profile: Loaded
   Risk Level: 🟡 MEDIUM
   Previous Sessions: 5
📚 Memory: 12 stored messages

----------------------------------------------------------------------
This chatbot provides:
  • Emotion detection and tracking
  • Behavior pattern analysis
  • Personalized responses using past context
  • Patient profile management
----------------------------------------------------------------------

Commands:
  help     - Show all commands
  search   - Search past conversations
  emotions - View emotion history
  profile  - View patient profile
  memory   - View memory statistics
  quit     - Exit the chatbot

----------------------------------------------------------------------

You: I've been having trouble sleeping lately and feel anxious about work

🤖 Thinking...

Bot: I hear you're going through a difficult time. Sleep issues and work anxiety often go together, and it's completely understandable to feel this way when under stress.

💬 Can you tell me more about what specifically at work is causing you anxiety?

📊 Detected Emotion: anxiety (medium intensity)

⚠️ Pattern Detected: sleep_disruption (mild severity)

🟢 Risk Assessment: LOW
   Reasoning: User expressing work stress and sleep issues, no immediate concerns

[2.34s | 145 tokens | 📚 3 memories]

You: profile

==================================================
Patient Profile
==================================================
User ID: patient_123

Known Conditions:
  - anxiety
  - depression

Detected Behavior Patterns:
  - sleep_disruption: mild severity (3 occurrences)
  - social_withdrawal: moderate severity (1 occurrence)

🟡 Risk Level: MEDIUM
Total Sessions: 5

==================================================

```

---

## Integration Checklist

Before using the enhanced chatbot, ensure:

- [x] MongoDB is installed and running
- [x] MongoDB connection is configured in `.env`
- [x] Gemini API key is set in `.env`
- [x] Qdrant configuration is correct
- [x] Python dependencies are installed:
  - `pymongo`
  - `sentence-transformers`
  - `qdrant-client`
  - `google-genai`
  - `pydantic`
  - `python-dotenv`

---

## Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
mongosh

# Start MongoDB (Windows)
net start MongoDB

# Start MongoDB (Linux/Mac)
brew services start mongodb-community
```

### Import Errors

```bash
# Install missing dependencies
pip install pymongo sentence-transformers qdrant-client google-genai
```

### Emotion Detection Not Working

- Verify the emotion detection service is initialized
- Check that keywords are being matched correctly
- Review logs for detection errors

---

## Future Enhancements

Potential improvements for future versions:

1. **Advanced Emotion Models**: Integrate transformer-based emotion detection
2. **Behavior Pattern Prediction**: ML-based prediction of future patterns
3. **Multi-user Support**: Support for group therapy scenarios
4. **Analytics Dashboard**: Web interface for viewing patient data
5. **Export Functionality**: Export patient data for analysis
6. **Integration with EHR**: Connect to electronic health records
7. **Voice Input**: Add speech-to-text capabilities
8. **Multilingual Support**: Emotion detection for multiple languages

---

## License

This enhanced RAG chatbot system is part of the NLP Foundations chatbot project.

---

## Contact

For questions or issues, please refer to the project repository or contact the development team.
