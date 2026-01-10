"""
Qdrant Vector Database configuration settings.

Follows 12-Factor App principles by externalizing all configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Connection settings
    host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    
    port: int = Field(
        default=6333,
        description="Qdrant server port"
    )
    
    grpc_port: int = Field(
        default=6334,
        description="Qdrant gRPC port"
    )
    
    # For local file-based storage (no server needed)
    use_local_storage: bool = Field(
        default=False,
        description="Use local file storage instead of server"
    )
    
    local_path: str = Field(
        default="./data/qdrant_db",
        description="Path for local Qdrant storage"
    )
    
    # Collection settings
    collection_name: str = Field(
        default="conversations",
        description="Name of the vector collection"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors (384 for MiniLM)"
    )
    
    # Search settings
    search_limit: int = Field(
        default=5,
        description="Default number of results to return"
    )
    
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results"
    )

