"""
Configuration settings for LangGraph Lab.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application Settings
    app_name: str = Field(default="LangGraph Lab", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model to use")
    
    # Search Configuration (Optional)
    serper_api_key: Optional[str] = Field(default=None, description="Serper API key for web search")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for web search")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./langgraph_lab.db", description="Database URL")
    
    # LangSmith Configuration (Optional)
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", description="LangSmith endpoint")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="langgraph-lab", description="LangSmith project name")
    
    # Workflow Configuration
    max_iterations: int = Field(default=10, description="Maximum workflow iterations")
    timeout_seconds: int = Field(default=300, description="Workflow timeout in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings