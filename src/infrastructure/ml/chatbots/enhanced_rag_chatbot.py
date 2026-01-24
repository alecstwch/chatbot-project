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
        use_therapy_mode: bool = True
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

        # State flags
        self._initialized = False
        self._rag_enabled = False
        self._mongodb_enabled = False

        # Metrics
        self.last_response_time: float = 0.0
        self.last_tokens_generated: int = 0
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

            # Step 9: Update conversation history (short-term memory)
            self.chat_history.append(current_message_content)
            self.chat_history.append(
                types.Content(role="model", parts=[types.Part(text=parsed_response["response"])])
            )

            # Trim history if needed (keep last 10 turns to manage context window)
            if len(self.chat_history) > 20:  # 10 turns = 20 messages (user + model)
                self.chat_history = self.chat_history[-20:]

            # Step 10: Store in RAG memory with emotion metadata (long-term memory)
            if store_in_memory and self._rag_enabled:
                self.rag_memory.store_conversation_turn(
                    user_id=self.user_id,
                    user_message=user_input,
                    assistant_message=parsed_response["response"],
                    session_id=self.current_session_id,
                    emotion_data=emotion_data
                )

            # Step 11: Update patient profile
            if self._mongodb_enabled and self.patient_repo:
                # Update emotion summary
                if emotion_summary:
                    self.patient_repo.update_emotion_summary(self.user_id, emotion_summary)

                # Add session note
                self.patient_repo.add_session_note(
                    user_id=self.user_id,
                    note=f"Emotion: {emotion_data['primary_emotion']}, "
                         f"Sentiment: {emotion_data['sentiment']}",
                    emotion_data=emotion_data
                )

                # Update behavior pattern if detected
                if parsed_response.get("behavior_update"):
                    from src.infrastructure.database.patient_repository import BehaviorPattern
                    behavior_data = parsed_response["behavior_update"]
                    behavior_pattern = BehaviorPattern(
                        pattern_type=behavior_data.get("pattern_type", "unknown"),
                        description=behavior_data.get("description", ""),
                        frequency=1,
                        first_detected=time.time().__str__(),
                        last_detected=time.time().__str__(),
                        severity=behavior_data.get("severity", "mild"),
                        associated_emotions=behavior_data.get("associated_emotions", [])
                    )
                    self.patient_repo.add_behavior_pattern(self.user_id, behavior_pattern)

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

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.last_tokens_generated = getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                self.last_tokens_generated = len(response_text) // 4

            self.last_tokens_per_sec = (
                self.last_tokens_generated / self.last_response_time
                if self.last_response_time > 0 else 0
            )

            logger.info(
                f"Response generated in {self.last_response_time:.2f}s "
                f"({self.last_tokens_generated} tokens)"
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

    def get_benchmark_stats(self) -> Tuple[float, int, float]:
        """Get performance stats."""
        return (self.last_response_time, self.last_tokens_generated, self.last_tokens_per_sec)

    def get_last_context_count(self) -> int:
        """Get last context count."""
        return self.last_context_items

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
