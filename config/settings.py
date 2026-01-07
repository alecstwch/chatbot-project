"""
Configuration Settings Module

Loads configuration from environment variables following 12-Factor App principles.
Configuration is separated from code and can be changed without code changes.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses pydantic for validation and type safety.
    Values are loaded from .env file or environment variables.
    """
    
    # Application Settings
    app_env: str = Field(default="development", description="Application environment")
    app_name: str = Field(default="chatbot-project", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Model Paths
    aiml_model_path: Path = Field(default=Path("models/aiml"), description="AIML model path")
    dialogpt_model_path: Path = Field(default=Path("models/dialogpt"), description="DialoGPT model path")
    transformer_model_path: Path = Field(default=Path("models/transformer"), description="Transformer model path")
    
    # Data Paths
    data_raw_path: Path = Field(default=Path("data/raw"), description="Raw data path")
    data_processed_path: Path = Field(default=Path("data/processed"), description="Processed data path")
    data_embeddings_path: Path = Field(default=Path("data/embeddings"), description="Embeddings path")
    
    # HuggingFace Configuration
    hf_cache_dir: Path = Field(default=Path("models/cache"), description="HuggingFace cache directory")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    hf_datasets_cache: Path = Field(default=Path("data/hf_cache"), description="HuggingFace datasets cache")
    
    # Training Configuration
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    learning_rate: float = Field(default=5e-5, gt=0, description="Learning rate")
    max_epochs: int = Field(default=3, ge=1, description="Maximum training epochs")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    
    # Model Configuration
    max_length: int = Field(default=512, ge=1, description="Maximum sequence length")
    temperature: float = Field(default=0.7, gt=0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0, le=1, description="Top-p sampling")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    
    # Preprocessing Configuration
    preprocess_lowercase: bool = Field(default=True, description="Convert to lowercase")
    preprocess_remove_stopwords: bool = Field(default=True, description="Remove stopwords")
    preprocess_use_lemmatization: bool = Field(default=True, description="Use lemmatization")
    preprocess_use_stemming: bool = Field(default=False, description="Use stemming")
    preprocess_language: str = Field(default="english", description="Language for preprocessing")
    
    # Evaluation Configuration
    eval_batch_size: int = Field(default=64, ge=1, description="Evaluation batch size")
    eval_metrics: str = Field(default="bleu,rouge,f1", description="Evaluation metrics")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, description="API worker count")
    
    # Logging Configuration
    log_file: Path = Field(default=Path("logs/chatbot.log"), description="Log file path")
    log_max_bytes: int = Field(default=10485760, ge=1024, description="Max log file size")
    log_backup_count: int = Field(default=5, ge=0, description="Log backup count")
    
    # External Services
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_org_id: Optional[str] = Field(default=None, description="OpenAI organization ID")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    def get_preprocessing_config(self) -> dict:
        """Get preprocessing configuration as dictionary"""
        return {
            "lowercase": self.preprocess_lowercase,
            "remove_stopwords": self.preprocess_remove_stopwords,
            "lemmatization": self.preprocess_use_lemmatization,
            "stemming": self.preprocess_use_stemming,
            "language": self.preprocess_language,
        }
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.aiml_model_path,
            self.dialogpt_model_path,
            self.transformer_model_path,
            self.data_raw_path,
            self.data_processed_path,
            self.data_embeddings_path,
            self.hf_cache_dir,
            self.hf_datasets_cache,
            self.log_file.parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance (singleton)
settings = Settings()


# Ensure directories exist on import
settings.ensure_directories()
