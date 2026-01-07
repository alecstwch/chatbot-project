"""
Configuration settings for chatbots.

Follows 12-Factor App principles by externalizing all configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class DialoGPTSettings(BaseSettings):
    """DialoGPT chatbot configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="DIALOGPT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields from .env
    )
    
    model_name: str = Field(
        default="microsoft/DialoGPT-small",
        description="HuggingFace model identifier"
    )
    
    max_history_length: int = Field(
        default=1000,
        description="Maximum conversation history length in tokens"
    )
    
    max_new_tokens: int = Field(
        default=50,
        description="Maximum number of new tokens to generate"
    )
    
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (higher = more random)"
    )
    
    top_p: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    top_k: int = Field(
        default=40,
        ge=0,
        description="Top-k sampling (0 = disabled)"
    )
    
    repetition_penalty: float = Field(
        default=1.3,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens"
    )
    
    cache_dir: str = Field(
        default="models/cache",
        description="Directory to cache downloaded models"
    )
    
    device: str = Field(
        default="auto",
        description="Device to run model on (auto/cpu/cuda)"
    )


class NeuralChatbotSettings(BaseSettings):
    """Neural chatbot configuration (supports Phi-2, GPT-2, etc.)."""
    
    model_config = SettingsConfigDict(
        env_prefix="NEURAL_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HuggingFace model identifier"
    )
    
    max_history_turns: int = Field(
        default=10,
        description="Maximum number of conversation turns to keep"
    )
    
    max_context_length: int = Field(
        default=4096,
        description="Maximum context length in tokens"
    )
    
    max_new_tokens: int = Field(
        default=150,
        description="Maximum number of new tokens to generate"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (higher = more random)"
    )
    
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling (0 = disabled)"
    )
    
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens"
    )
    
    cache_dir: str = Field(
        default="models/cache",
        description="Directory to cache downloaded models"
    )
    
    use_8bit_quantization: bool = Field(
        default=True,
        description="Use 8-bit quantization to fit in limited VRAM"
    )
    
    vram_size_gb: int = Field(
        default=4,
        description="GPU VRAM size in GB (for display purposes)"
    )
    
    device: str = Field(
        default="auto",
        description="Device to run model on (auto/cpu/cuda)"
    )


class AimlSettings(BaseSettings):
    """AIML chatbot configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="AIML_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    aiml_dir: str = Field(
        default="data/knowledge_bases/aiml",
        description="Directory containing AIML files"
    )


class ChatbotSettings(BaseSettings):
    """Main chatbot configuration."""
    
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    dialogpt: DialoGPTSettings = Field(default_factory=DialoGPTSettings)
    neural: NeuralChatbotSettings = Field(default_factory=NeuralChatbotSettings)
    aiml: AimlSettings = Field(default_factory=AimlSettings)
