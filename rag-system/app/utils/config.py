import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management for the RAG system."""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # Model deployment names
    AZURE_GPT4_DEPLOYMENT: str = os.getenv("AZURE_GPT4_DEPLOYMENT", "gpt-4")
    AZURE_GPT35_DEPLOYMENT: str = os.getenv("AZURE_GPT35_DEPLOYMENT", "gpt-35-turbo")
    AZURE_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    # Fallback to OpenAI if Azure not configured
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "100"))
    
    # FAISS Configuration (Local Vector Database)
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    FAISS_METADATA_PATH: str = os.getenv("FAISS_METADATA_PATH", "data/faiss_metadata.pkl")
    PINECONE_DIM: int = int(os.getenv("PINECONE_DIM", "1024"))
    PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")
    UPSERT_BATCH_SIZE: int = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Application Configuration
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "500"))
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
    MAX_QUESTIONS_PER_REQUEST: int = int(os.getenv("MAX_QUESTIONS_PER_REQUEST", "10"))
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Advanced RAG Configuration
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MAX_RETRIEVAL_CHUNKS: int = int(os.getenv("MAX_RETRIEVAL_CHUNKS", "10"))
    ENABLE_STRUCTURED_RESPONSES: bool = os.getenv("ENABLE_STRUCTURED_RESPONSES", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """Validate that all required configuration is present."""
        # Check if Azure OpenAI is configured
        azure_configured = all([
            cls.AZURE_OPENAI_API_KEY,
            cls.AZURE_OPENAI_ENDPOINT,
            cls.AZURE_OPENAI_API_VERSION
        ])
        
        # Check if OpenAI is configured (fallback)
        openai_configured = bool(cls.OPENAI_API_KEY)
        
        if not azure_configured and not openai_configured:
            raise ValueError("Either Azure OpenAI or OpenAI configuration is required")
        
        required_fields = [
            ("DATABASE_URL", cls.DATABASE_URL),
        ]

        missing_fields = []
        for field_name, field_value in required_fields:
            if not field_value:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
        
        # Log which service is being used
        if azure_configured:
            print("✅ Using Azure OpenAI")
        else:
            print("⚠️  Using OpenAI (fallback)")
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "embed_model": cls.EMBED_MODEL,
            "batch_size": cls.EMBED_BATCH_SIZE
        }
    
    @classmethod
    def get_faiss_config(cls) -> dict:
        """Get FAISS configuration."""
        return {
            "index_path": cls.FAISS_INDEX_PATH,
            "metadata_path": cls.FAISS_METADATA_PATH,
            "dimension": cls.PINECONE_DIM,
            "metric": cls.PINECONE_METRIC,
            "batch_size": cls.UPSERT_BATCH_SIZE
        }

# Global config instance
config = Config()
