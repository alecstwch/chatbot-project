"""
MongoDB configuration settings.

Uses pydantic-settings for configuration from environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoDBSettings(BaseSettings):
    """
    MongoDB connection and database settings.
    
    Configuration priority (highest to lowest):
    1. Environment variables (MONGODB_URI, MONGODB_DATABASE, etc.)
    2. .env file
    3. Default values
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Connection settings
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/",
        description="MongoDB connection URI"
    )
    
    mongodb_database: str = Field(
        default="chatbot_db",
        description="Database name for chatbot data"
    )
    
    # Collection names
    conversations_collection: str = Field(
        default="conversations",
        description="Collection for conversation sessions"
    )
    
    users_collection: str = Field(
        default="users",
        description="Collection for user profiles"
    )
    
    # Connection pool settings
    mongodb_max_pool_size: int = Field(
        default=10,
        description="Maximum connection pool size"
    )
    
    mongodb_min_pool_size: int = Field(
        default=1,
        description="Minimum connection pool size"
    )
    
    mongodb_timeout_ms: int = Field(
        default=5000,
        description="Connection timeout in milliseconds"
    )
    
    # Indexing
    create_indexes: bool = Field(
        default=True,
        description="Automatically create database indexes"
    )
