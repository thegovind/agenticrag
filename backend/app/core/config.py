from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
    AZURE_CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
    AZURE_CLIENT_SECRET: str = os.getenv("AZURE_CLIENT_SECRET", "")
    
    AZURE_SEARCH_SERVICE_NAME: str = os.getenv("AZURE_SEARCH_SERVICE_NAME", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "financial-documents")
    
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    AVAILABLE_EMBEDDING_MODELS: List[str] = [
        "text-embedding-ada-002",
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ]
    AVAILABLE_CHAT_MODELS: List[str] = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-35-turbo",
        "financial-llm",  # Industry specific
        "grok-beta",
        "deepseek-chat"
    ]
    
    COSMOS_DB_ENDPOINT: str = os.getenv("COSMOS_DB_ENDPOINT", "")
    COSMOS_DB_DATABASE_NAME: str = os.getenv("COSMOS_DB_DATABASE_NAME", "rag-financial")
    COSMOS_DB_CONTAINER_NAME: str = os.getenv("COSMOS_DB_CONTAINER_NAME", "sessions")
    
    AZURE_FORM_RECOGNIZER_ENDPOINT: str = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT", "")
    
    AI_FOUNDRY_PROJECT_NAME: str = os.getenv("AI_FOUNDRY_PROJECT_NAME", "")
    AI_FOUNDRY_RESOURCE_GROUP: str = os.getenv("AI_FOUNDRY_RESOURCE_GROUP", "")
    AI_FOUNDRY_SUBSCRIPTION_ID: str = os.getenv("AI_FOUNDRY_SUBSCRIPTION_ID", "")
    
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS_PER_REQUEST: int = 4000
    TEMPERATURE: float = 0.1
    
    AUTO_UPDATE_ENABLED: bool = True
    UPDATE_FREQUENCY_HOURS: int = 24
    CREDIBILITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()
