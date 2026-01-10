"""
RAG Prompt Builder for context-augmented generation.

Builds prompts that include relevant context from past conversations,
enabling the chatbot to provide personalized, context-aware responses.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGPromptBuilder:
    """
    Builds prompts augmented with retrieved conversation context.
    
    This builder formats retrieved memories and conversation history
    into prompts that help the LLM provide contextually relevant responses.
    """
    
    def __init__(
        self,
        max_context_items: int = 5,
        max_history_turns: int = 5,
        include_timestamps: bool = False,
        include_similarity_scores: bool = False
    ):
        """
        Initialize the prompt builder.
        
        Args:
            max_context_items: Maximum number of retrieved context items to include
            max_history_turns: Maximum recent conversation turns to include
            include_timestamps: Whether to show timestamps in context
            include_similarity_scores: Whether to show similarity scores
        """
        self.max_context_items = max_context_items
        self.max_history_turns = max_history_turns
        self.include_timestamps = include_timestamps
        self.include_similarity_scores = include_similarity_scores
    
    def build_prompt(
        self,
        current_message: str,
        retrieved_context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_name: Optional[str] = None,
        system_instructions: Optional[str] = None
    ) -> str:
        """
        Build a RAG-augmented prompt.
        
        Args:
            current_message: User's current input
            retrieved_context: Semantically similar past messages from Qdrant
            conversation_history: Recent conversation turns
            user_name: Optional user name for personalization
            system_instructions: Optional custom system instructions
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # System instructions
        if system_instructions:
            parts.append(system_instructions)
        else:
            parts.append(self._default_system_instructions())
        
        # Retrieved context section
        context_section = self._format_retrieved_context(retrieved_context)
        if context_section:
            parts.append(context_section)
        
        # User info
        if user_name:
            parts.append(f"You are speaking with: {user_name}")
        
        # Recent conversation history
        if conversation_history:
            history_section = self._format_conversation_history(conversation_history)
            if history_section:
                parts.append(history_section)
        
        # Current message
        parts.append(f"Current message: {current_message}")
        
        # Final instruction
        parts.append(self._response_instructions())
        
        return "\n\n".join(parts)
    
    def build_gemini_prompt(
        self,
        current_message: str,
        retrieved_context: List[Dict[str, Any]],
        user_name: Optional[str] = None
    ) -> str:
        """
        Build a prompt optimized for Gemini API (used as system context).
        
        For Gemini, we format the context to be prepended to the conversation.
        
        Args:
            current_message: User's current input
            retrieved_context: Semantically similar past messages
            user_name: Optional user name
            
        Returns:
            Context string to prepend to conversation
        """
        if not retrieved_context:
            return ""
        
        lines = ["[Relevant information from past conversations:]"]
        
        for i, item in enumerate(retrieved_context[:self.max_context_items], 1):
            content = item.get("content", "")
            role = item.get("role", "user")
            
            if role == "user":
                prefix = "User previously said"
            else:
                prefix = "You previously responded"
            
            # Add similarity info if enabled
            if self.include_similarity_scores:
                similarity = item.get("similarity", 0)
                lines.append(f"  {i}. {prefix} (relevance: {similarity:.0%}): \"{content}\"")
            else:
                lines.append(f"  {i}. {prefix}: \"{content}\"")
            
            # Add intent if available
            if item.get("intent"):
                lines.append(f"     (Topic: {item['intent']})")
        
        if user_name:
            lines.append(f"\nUser's name: {user_name}")
        
        lines.append("\n[Use this context naturally in your response. Don't explicitly mention 'based on our past conversation'.]")
        
        return "\n".join(lines)
    
    def _default_system_instructions(self) -> str:
        """Default system instructions for the chatbot."""
        return """You are a helpful, empathetic conversational assistant. 

Your capabilities:
- You have access to relevant information from past conversations with this user
- Use this context to provide personalized, thoughtful responses
- Remember important details the user has shared (name, health history, preferences)
- Be supportive and understanding, especially regarding mental health topics

Guidelines:
- Reference past information naturally, don't say "based on our past conversation"
- If the user mentioned mental health concerns before, be especially thoughtful
- Maintain a warm, supportive tone
- Ask clarifying questions when helpful"""
    
    def _format_retrieved_context(
        self,
        retrieved_context: List[Dict[str, Any]]
    ) -> str:
        """Format the retrieved context section."""
        if not retrieved_context:
            return ""
        
        lines = ["📚 Relevant information from past conversations:"]
        
        for i, item in enumerate(retrieved_context[:self.max_context_items], 1):
            content = item.get("content", "")
            role = item.get("role", "user")
            similarity = item.get("similarity", 0)
            intent = item.get("intent")
            timestamp = item.get("timestamp")
            
            # Format role
            role_label = "User" if role == "user" else "Assistant"
            
            # Build context line
            line_parts = [f"  {i}. [{role_label}]"]
            
            if self.include_timestamps and timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    line_parts.append(f"({dt.strftime('%b %d')})")
                except:
                    pass
            
            if self.include_similarity_scores:
                line_parts.append(f"(relevance: {similarity:.0%})")
            
            line_parts.append(f": \"{content}\"")
            
            lines.append("".join(line_parts))
            
            if intent:
                lines.append(f"     → Topic: {intent}")
        
        return "\n".join(lines)
    
    def _format_conversation_history(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Format recent conversation history."""
        if not conversation_history:
            return ""
        
        # Take only the most recent turns
        recent = conversation_history[-self.max_history_turns:]
        
        lines = ["💬 Recent conversation:"]
        
        for turn in recent:
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")
            
            if user_msg:
                lines.append(f"  User: {user_msg}")
            if assistant_msg:
                lines.append(f"  Assistant: {assistant_msg}")
        
        return "\n".join(lines)
    
    def _response_instructions(self) -> str:
        """Instructions for generating the response."""
        return """Instructions:
- Respond naturally and conversationally
- Use relevant context from past conversations when appropriate
- Be empathetic and supportive
- If you recall important user details, incorporate them naturally"""


class TherapyRAGPromptBuilder(RAGPromptBuilder):
    """
    Specialized prompt builder for therapy/mental health conversations.
    
    Includes additional guidance for handling sensitive topics.
    """
    
    def _default_system_instructions(self) -> str:
        """System instructions optimized for therapy context."""
        return """You are a supportive mental health companion (not a replacement for professional help).

Your role:
- Provide empathetic, non-judgmental support
- Remember and acknowledge the user's mental health journey
- Recognize patterns in emotional states across conversations
- Encourage professional help when appropriate

Important guidelines:
- If the user has mentioned past mental health issues (depression, anxiety, etc.), 
  be mindful and supportive when they discuss current struggles
- Never diagnose or prescribe treatment
- Validate feelings before offering suggestions
- Use the user's name when you know it
- Reference relevant past conversations naturally

Safety:
- If someone expresses thoughts of self-harm, encourage professional help
- Provide crisis resources when appropriate"""
    
    def _response_instructions(self) -> str:
        """Response instructions for therapy context."""
        return """Response guidelines:
- Acknowledge the user's feelings first
- Draw connections to past experiences when relevant
- Be warm and supportive
- Ask open-ended questions to encourage sharing
- Suggest coping strategies when appropriate
- Remember: You're a supportive companion, not a therapist"""

