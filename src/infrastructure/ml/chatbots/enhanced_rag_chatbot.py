"""
Enhanced RAG-Enhanced Chatbot with Patient Data Integration.

Integrates:
- Emotion detection
- MongoDB patient files
- Qdrant RAG memory with emotion metadata
- Structured JSON responses with behavior pattern tracking
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
from src.infrastructure.config.mongodb_settings import MongoDBSettings
from src.infrastructure.memory.rag_memory_service import RAGMemoryService
from src.infrastructure.memory.enhanced_prompt_builder import EnhancedPromptBuilder, TherapyEnhancedPromptBuilder
from src.infrastructure.database.patient_repository import PatientRepository, PatientProfile
from src.domain.services.emotion_detection_service import EmotionDetectionService
from src.domain.services.behavior_pattern_service import BehaviorPatternService, BehaviorPattern

load_dotenv()

logger = logging.getLogger(__name__)


class EnhancedRAGChatbot:
    """
    Enhanced RAG chatbot with patient data integration and emotion tracking.

    Features:
    - Emotion detection on user messages
    - Patient profile management in MongoDB
    - Emotion-aware retrieval from Qdrant
    - Structured JSON responses with behavior pattern updates
    """

    def __init__(
        self,
        settings: Optional[NeuralChatbotSettings] = None,
        qdrant_settings: Optional[QdrantSettings] = None,
        mongodb_settings: Optional[MongoDBSettings] = None,
        model_name: Optional[str] = None,
        user_id: str = "default_user",
        user_name: Optional[str] = None,
        use_therapy_mode: bool = True,
        initial_message: Optional[str] = None
    ):
        """
        Initialize enhanced RAG chatbot.

        Args:
            settings: Neural chatbot configuration
            qdrant_settings: Qdrant configuration
            mongodb_settings: MongoDB configuration
            model_name: Gemini model identifier
            user_id: User identifier
            user_name: User display name
            use_therapy_mode: Use therapy-optimized prompts
            initial_message: Custom initial greeting/question
        """
        # Load settings
        self.settings = settings or NeuralChatbotSettings()
        self.qdrant_settings = qdrant_settings or QdrantSettings()
        self.mongodb_settings = mongodb_settings or MongoDBSettings()

        # Model configuration
        self.model_name = model_name or self.settings.model_name

        # User identification
        self.user_id = user_id
        self.user_name = user_name

        # Initialize components
        self.client: Optional[genai.Client] = None
        self.rag_memory: Optional[RAGMemoryService] = None
        self.patient_repo: Optional[PatientRepository] = None
        self.emotion_service: Optional[EmotionDetectionService] = None
        self.behavior_pattern_service: Optional[BehaviorPatternService] = None

        # Choose prompt builder
        if use_therapy_mode:
            self.prompt_builder = TherapyEnhancedPromptBuilder(
                max_context_items=5,
                include_similarity_scores=False
            )
        else:
            self.prompt_builder = EnhancedPromptBuilder(
                max_context_items=5,
                include_similarity_scores=False
            )

        # Conversation state
        self.chat_history: List[types.Content] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.current_session_id: Optional[str] = None
        self.initial_message = initial_message

        # State flags
        self._initialized = False
        self._rag_enabled = False
        self._mongodb_enabled = False

        # Model context window (Gemini 2.0 Flash: 1,048,576 tokens)
        self.max_context_window: int = 1_048_576

        # Metrics
        self.last_response_time: float = 0.0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_total_tokens: int = 0
        self.last_context_usage_percent: float = 0.0
        self.last_tokens_per_sec: float = 0.0
        self.last_context_items: int = 0

        logger.info(f"Enhanced RAG Chatbot initialized for user: {user_id}")

    def load_model(self) -> None:
        """Initialize all services and models."""
        try:
            # Initialize Gemini API
            logger.info(f"Initializing Gemini API: {self.model_name}")

            api_key = (
                self.settings.api_key or
                os.environ.get("GEMINI_API_KEY") or
                os.environ.get("GOOGLE_API_KEY") or
                os.environ.get("NEURAL_API_KEY")
            )

            if not api_key:
                raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY in .env")

            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini API initialized")

            # Initialize emotion detection
            self.emotion_service = EmotionDetectionService()
            logger.info("Emotion detection service initialized")

            # Initialize behavior pattern detection
            self.behavior_pattern_service = BehaviorPatternService()
            logger.info("Behavior pattern detection service initialized")

            # Initialize RAG memory
            logger.info("Initializing RAG memory service...")
            self.rag_memory = RAGMemoryService(settings=self.qdrant_settings)

            if self.rag_memory.initialize():
                self._rag_enabled = True
                msg_count = self.rag_memory.get_user_message_count(self.user_id)
                logger.info(f"RAG memory enabled. User has {msg_count} messages")
            else:
                logger.warning("RAG memory initialization failed")

            # Initialize MongoDB
            logger.info("Initializing MongoDB patient repository...")
            self.patient_repo = PatientRepository(settings=self.mongodb_settings)

            if self.patient_repo.initialize():
                self._mongodb_enabled = True
                # Create patient profile if it doesn't exist
                patient = self.patient_repo.get_patient(self.user_id)
                if not patient:
                    logger.info(f"Creating new patient profile for {self.user_id}")
                    profile = PatientProfile(
                        user_id=self.user_id,
                        name=self.user_name
                    )
                    self.patient_repo.create_patient(profile)
                else:
                    logger.info(f"Loaded existing patient profile for {self.user_id}")
            else:
                logger.warning("MongoDB initialization failed")

            self._initialized = True
            logger.info("Enhanced RAG Chatbot fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def get_response(
        self,
        user_input: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        store_in_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response with full patient data integration.

        Args:
            user_input: User's message
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            store_in_memory: Whether to store in memory

        Returns:
            Dictionary with:
            - response: Bot's conversational response
            - next_question: Follow-up question
            - behavior_update: Detected behavior patterns
            - emotion_update: Current emotion state
            - risk_assessment: Risk level assessment
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        if not user_input or not user_input.strip():
            return {
                "response": "I didn't catch that. Could you say that again?",
                "next_question": "What's on your mind?",
                "behavior_update": None,
                "emotion_update": None,
                "risk_assessment": {
                    "level": "low",
                    "reasoning": "No input to analyze",
                    "recommendations": []
                }
            }

        try:
            start_time = time.time()

            # Step 1: Detect emotion in user input
            emotion_analysis = self.emotion_service.detect_emotion(user_input)
            emotion_data = emotion_analysis.to_dict()
            logger.debug(f"Detected emotion: {emotion_data['primary_emotion']} "
                        f"(confidence: {emotion_data['confidence']:.2f})")

            # Step 1b: Detect behavior patterns
            detected_patterns = []
            if self.behavior_pattern_service:
                detected_patterns = self.behavior_pattern_service.detect_patterns(
                    user_input,
                    emotion_data.get('primary_emotion')
                )
                if detected_patterns:
                    logger.debug(f"Detected {len(detected_patterns)} behavior patterns: "
                               f"{[p.pattern_type.value for p in detected_patterns]}")

            # Step 2: Get patient profile
            patient_profile = None
            if self._mongodb_enabled:
                patient = self.patient_repo.get_patient(self.user_id)
                if patient:
                    patient_profile = patient.to_dict()

            # Step 3: Retrieve relevant context with emotion filter
            retrieved_context = []
            if self._rag_enabled:
                # Get context filtered by current emotion
                retrieved_context = self.rag_memory.retrieve_by_emotion_context(
                    user_id=self.user_id,
                    query=user_input,
                    target_emotion=emotion_data['primary_emotion'],
                    top_k=5,
                    min_similarity=0.4
                )
                self.last_context_items = len(retrieved_context)

            # Step 4: Get emotion summary
            emotion_summary = None
            if self._rag_enabled:
                emotion_summary = self.rag_memory.get_emotion_summary(self.user_id)

            # Step 5: Build comprehensive system prompt
            system_prompt = self.prompt_builder.build_system_prompt(
                patient_profile=patient_profile,
                retrieved_context=retrieved_context,
                emotion_summary=emotion_summary
            )

            # Step 6: Prepare messages for Gemini
            messages_for_api = []

            # Add system prompt as first user message
            messages_for_api.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=system_prompt)]
                )
            )
            messages_for_api.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text="I understand. I'm ready to provide supportive "
                                          "responses with behavior pattern tracking. "
                                          "Please send the user's message.")]
                )
            )

            # Add current user message
            current_message_content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )

            # Add conversation history BEFORE current message
            messages_for_api.extend(self.chat_history)
            messages_for_api.append(current_message_content)

            # Step 7: Generate response
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

            # Step 8: Parse JSON response
            parsed_response = self.prompt_builder.parse_llm_response(response_text)

            # Step 8b: Override with keyword-detected behavior patterns
            # The BehaviorPatternService (keyword-based) patterns take priority over LLM detection
            # This ensures consistent, rule-based pattern tracking in MongoDB
            if detected_patterns:
                # Use the highest confidence detected pattern for the response
                top_pattern = detected_patterns[0]
                parsed_response["behavior_update"] = {
                    "pattern_type": top_pattern.pattern_type.value,
                    "description": top_pattern.description,
                    "severity": top_pattern.severity.value,
                    "confidence": top_pattern.confidence,
                    "indicators": top_pattern.indicators,
                    "associated_emotions": [emotion_data.get('primary_emotion')] if emotion_data.get('primary_emotion') else [],
                    "detection_method": "keyword_based"
                }
                logger.debug(f"Using keyword-detected pattern in response: {top_pattern.pattern_type.value} (confidence: {top_pattern.confidence:.2f})")

            # Step 9: Update conversation history (short-term memory)
            self.chat_history.append(current_message_content)

            # Include next_question in the stored history if present
            response_text = parsed_response["response"]
            next_question = parsed_response.get("next_question", "")
            if next_question:
                # Append the next question to the response in history
                # This gives context to the LLM about what question was asked
                response_with_next = f"{response_text}\n\nFollow-up question: {next_question}"
            else:
                response_with_next = response_text

            self.chat_history.append(
                types.Content(role="model", parts=[types.Part(text=response_with_next)])
            )

            # Trim history if needed (keep last 10 turns to manage context window)
            if len(self.chat_history) > 20:  # 10 turns = 20 messages (user + model)
                self.chat_history = self.chat_history[-20:]

            # Step 10: Store in RAG memory with emotion metadata (long-term memory)
            if store_in_memory and self._rag_enabled:
                # Extract behavior patterns from detected patterns
                behavior_pattern_list = [p.pattern_type.value for p in detected_patterns] if detected_patterns else []

                # Get next question from response
                next_question = parsed_response.get("next_question", "")

                self.rag_memory.store_conversation_turn(
                    user_id=self.user_id,
                    user_message=user_input,
                    assistant_message=parsed_response["response"],
                    session_id=self.current_session_id,
                    emotion_data=emotion_data,
                    behavior_patterns=behavior_pattern_list if behavior_pattern_list else None,
                    next_question=next_question if next_question else None
                )

            # Step 11: Update patient profile
            if self._mongodb_enabled and self.patient_repo:
                # Update emotion summary
                if emotion_summary:
                    self.patient_repo.update_emotion_summary(self.user_id, emotion_summary)

                # Add session note with behavior patterns
                behavior_info = ""
                if detected_patterns:
                    patterns_str = ", ".join([p.pattern_type.value for p in detected_patterns[:3]])
                    behavior_info = f", Patterns: {patterns_str}"

                self.patient_repo.add_session_note(
                    user_id=self.user_id,
                    note=f"Emotion: {emotion_data['primary_emotion']}, "
                         f"Sentiment: {emotion_data['sentiment']}"
                         f"{behavior_info}",
                    emotion_data=emotion_data
                )

                # Store behavior patterns detected by keyword-based service in MongoDB
                # These patterns are detected by the BehaviorPatternService, not the LLM
                if detected_patterns:
                    from src.infrastructure.database.patient_repository import BehaviorPattern
                    for pattern in detected_patterns:
                        behavior_pattern = BehaviorPattern(
                            pattern_type=pattern.pattern_type.value,
                            description=pattern.description,
                            frequency=1,
                            first_detected=time.time().__str__(),
                            last_detected=time.time().__str__(),
                            severity=pattern.severity.value,
                            associated_emotions=[emotion_data.get('primary_emotion')] if emotion_data.get('primary_emotion') else []
                        )
                        self.patient_repo.add_behavior_pattern(self.user_id, behavior_pattern)
                        logger.debug(f"Stored behavior pattern in MongoDB: {pattern.pattern_type.value} ({pattern.severity.value})")

                # Update risk level if changed
                risk_assessment = parsed_response.get("risk_assessment", {})
                if risk_assessment.get("level"):
                    from src.infrastructure.database.patient_repository import RiskLevel
                    try:
                        new_risk = RiskLevel(risk_assessment["level"])
                        self.patient_repo.update_risk_level(
                            self.user_id,
                            new_risk,
                            risk_assessment.get("reasoning")
                        )
                    except ValueError:
                        pass  # Invalid risk level

            # Calculate metrics
            end_time = time.time()
            self.last_response_time = end_time - start_time

            # Extract token counts from Gemini usage metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                # prompt_token_count = input tokens sent to the model
                self.last_input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                # candidates_token_count = output tokens generated
                self.last_output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                # total_token_count = sum of both
                self.last_total_tokens = getattr(response.usage_metadata, 'total_token_count',
                                                  self.last_input_tokens + self.last_output_tokens)
            else:
                # Fallback: estimate based on text length
                self.last_input_tokens = len(system_prompt) // 4 + len(user_input) // 4
                self.last_output_tokens = len(response_text) // 4
                self.last_total_tokens = self.last_input_tokens + self.last_output_tokens

            # Calculate context window usage percentage
            self.last_context_usage_percent = (
                (self.last_total_tokens / self.max_context_window) * 100
                if self.max_context_window > 0 else 0
            )

            # Calculate tokens per second (based on output tokens)
            self.last_tokens_per_sec = (
                self.last_output_tokens / self.last_response_time
                if self.last_response_time > 0 else 0
            )

            logger.info(
                f"Response generated in {self.last_response_time:.2f}s - "
                f"Input: {self.last_input_tokens} | Output: {self.last_output_tokens} | "
                f"Total: {self.last_total_tokens} tokens ({self.last_context_usage_percent:.4f}% of {self.max_context_window:,} context window)"
            )

            return parsed_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "next_question": "Would you like to try sharing that again?",
                "behavior_update": None,
                "emotion_update": None,
                "risk_assessment": {
                    "level": "low",
                    "reasoning": "Error during processing",
                    "recommendations": []
                }
            }

    def get_patient_profile(self) -> Optional[Dict[str, Any]]:
        """Get the current patient profile."""
        if not self._mongodb_enabled or not self.patient_repo:
            return None

        patient = self.patient_repo.get_patient(self.user_id)
        return patient.to_dict() if patient else None

    def update_patient_profile(self, updates: Dict[str, Any]) -> bool:
        """Update patient profile fields."""
        if not self._mongodb_enabled or not self.patient_repo:
            return False

        return self.patient_repo.update_patient(self.user_id, updates)

    def search_by_emotion(
        self,
        emotion: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search past conversations by emotion."""
        if not self._rag_enabled:
            return []

        return self.rag_memory.search_by_emotion(
            user_id=self.user_id,
            emotion=emotion,
            top_k=top_k
        )

    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get emotion summary for the user."""
        if not self._rag_enabled:
            return {}

        return self.rag_memory.get_emotion_summary(self.user_id)

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        self.chat_history = []
        self.current_session_id = None
        logger.debug("Conversation reset")

    def is_ready(self) -> bool:
        """Check if chatbot is ready."""
        return self._initialized and self.client is not None

    def is_rag_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self._rag_enabled

    def is_mongodb_enabled(self) -> bool:
        """Check if MongoDB is enabled."""
        return self._mongodb_enabled

    def get_initial_greeting(self) -> str:
        """
        Get the initial greeting/question for starting a conversation.

        Returns:
            Initial greeting message
        """
        if self.initial_message:
            return self.initial_message

        # Default therapy-focused greeting
        if self.user_name:
            return f"Hi {self.user_name}! I'm here to listen and support you. How are you feeling today?"

        # Default general greeting
        return "Hello! I'm here to support you. How are you feeling right now?"

    def get_benchmark_stats(self) -> Tuple[float, int, float]:
        """
        Get performance stats.

        Returns:
            Tuple of (response_time, total_tokens, tokens_per_second)
        """
        return (self.last_response_time, self.last_total_tokens, self.last_tokens_per_sec)

    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get detailed token statistics.

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens,
            context_usage_percent, max_context_window
        """
        return {
            "input_tokens": self.last_input_tokens,
            "output_tokens": self.last_output_tokens,
            "total_tokens": self.last_total_tokens,
            "context_usage_percent": round(self.last_context_usage_percent, 4),
            "max_context_window": self.max_context_window
        }

    def get_last_context_count(self) -> int:
        """Get last context count."""
        return self.last_context_items

    def search_behavior_pattern(
        self,
        pattern_type: str,
        min_confidence: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        """
        Search for historical occurrences of a behavior pattern in RAG.

        Args:
            pattern_type: The pattern type to search for (e.g., "social_withdrawal")
            min_confidence: Minimum confidence threshold for detection

        Returns:
            Pattern history dictionary or None if no matches found
        """
        if not self._rag_enabled or not self.behavior_pattern_service:
            return None

        try:
            # Convert string to enum
            pattern_enum = BehaviorPattern(pattern_type)

            # Get all user messages from RAG
            all_messages = self.rag_memory.get_all_user_messages(self.user_id, limit=200)

            if not all_messages:
                return None

            # Search for pattern history
            pattern_history = self.behavior_pattern_service.search_pattern_history(
                rag_messages=all_messages,
                pattern_type=pattern_enum,
                min_confidence=min_confidence
            )

            return pattern_history.to_dict() if pattern_history else None

        except ValueError:
            logger.error(f"Invalid pattern type: {pattern_type}")
            return None
        except Exception as e:
            logger.error(f"Failed to search behavior pattern: {e}")
            return None

    def get_trending_behavior_patterns(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get behavior patterns that have increased in frequency recently.

        Args:
            days: Number of days to look back

        Returns:
            List of trending patterns
        """
        if not self._rag_enabled or not self.behavior_pattern_service:
            return []

        try:
            # Get recent user messages from RAG
            all_messages = self.rag_memory.get_all_user_messages(self.user_id, limit=200)

            if not all_messages:
                return []

            # Get trending patterns
            trending = self.behavior_pattern_service.get_trending_patterns(
                rag_messages=all_messages,
                days=days
            )

            return trending

        except Exception as e:
            logger.error(f"Failed to get trending patterns: {e}")
            return []

    def detect_behavior_patterns(
        self,
        message: str,
        emotion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect behavior patterns in a message (without storing).

        Args:
            message: The message to analyze
            emotion: Optional emotion to add context

        Returns:
            List of detected pattern dictionaries
        """
        if not self.behavior_pattern_service:
            return []

        try:
            detections = self.behavior_pattern_service.detect_patterns(message, emotion)
            return [d.to_dict() for d in detections]

        except Exception as e:
            logger.error(f"Failed to detect behavior patterns: {e}")
            return []

    def compare_behavior_patterns(self) -> Dict[str, Any]:
        """
        Compare current behavior patterns with historical baseline.

        Returns:
            Comparison analysis dictionary
        """
        if not self._rag_enabled or not self.behavior_pattern_service:
            return {}

        try:
            # Get all user messages from RAG
            all_messages = self.rag_memory.get_all_user_messages(self.user_id, limit=200)

            if not all_messages:
                return {}

            # Compare patterns
            comparison = self.behavior_pattern_service.compare_patterns(
                rag_messages=all_messages,
                user_id=self.user_id
            )

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare behavior patterns: {e}")
            return {}

    def search_by_behavior_pattern(
        self,
        behavior_pattern: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for past messages by behavior pattern.

        Args:
            behavior_pattern: Pattern to search for (e.g., "social_withdrawal")
            top_k: Maximum results

        Returns:
            List of matching messages from this user only
        """
        if not self._rag_enabled:
            return []

        try:
            return self.rag_memory.search_by_behavior_pattern(
                user_id=self.user_id,
                behavior_pattern=behavior_pattern,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Failed to search by behavior pattern: {e}")
            return []

    def retrieve_by_behavior_pattern_context(
        self,
        query: str,
        target_pattern: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages with semantic similarity and optional behavior pattern filter.

        Args:
            query: Current message to find similar contexts for
            target_pattern: Optional behavior pattern to filter by
            top_k: Maximum results

        Returns:
            List of relevant past messages
        """
        if not self._rag_enabled:
            return []

        try:
            return self.rag_memory.retrieve_by_behavior_pattern_context(
                user_id=self.user_id,
                query=query,
                target_pattern=target_pattern,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Failed to retrieve by behavior pattern context: {e}")
            return []

    def shutdown(self) -> None:
        """Shutdown all services."""
        if self.rag_memory:
            self.rag_memory.shutdown()
        if self.patient_repo:
            self.patient_repo.shutdown()
        self._initialized = False
        self._rag_enabled = False
        self._mongodb_enabled = False
        logger.info("Enhanced RAG Chatbot shutdown complete")
