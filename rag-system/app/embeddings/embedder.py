import os
import logging
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# — load keys & config —
load_dotenv()
from ..utils.config import config

# Check if Azure OpenAI is configured
if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
    # Use Azure OpenAI
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION
    )
    EMBED_MODEL = config.AZURE_EMBEDDING_DEPLOYMENT
    print("✅ Using Azure OpenAI for embeddings")
else:
    # Fallback to OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Either Azure OpenAI or OpenAI API key is required")
    openai.api_key = api_key
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
    print("⚠️  Using OpenAI for embeddings")

BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))

# — logger —
from ..utils.logger import setup_logger
logger = setup_logger(__name__)

def _embed_batch_sync(texts: List[str]) -> List[List[float]]:
    """Synchronous embedding for Azure OpenAI"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]

async def _embed_batch_async(texts: List[str]) -> List[List[float]]:
    """Asynchronous embedding for OpenAI"""
    resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Wrapper function that chooses sync or async based on client type"""
    if hasattr(client, 'azure_endpoint'):
        # Azure OpenAI - synchronous
        return _embed_batch_sync(texts)
    else:
        # OpenAI - asynchronous (this won't work in sync context, but we handle it in embed_chunks_openai)
        raise RuntimeError("OpenAI client requires async context")

async def embed_chunks_openai(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Batch‐embed each chunk asynchronously. Adds 'embedding' to each chunk.
    Uses text_for_embedding (with contextual headers) if available, falls back to chunk_text.
    """
    logger.info(f"Embedding {len(chunks)} chunks (batch size={BATCH_SIZE})")
    embeddings: List[List[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        # Use text_for_embedding if available (contains contextual headers), otherwise fall back to chunk_text
        texts = [c.get("text_for_embedding", c["chunk_text"]) for c in batch]
        try:
            # Only use async for OpenAI, not Azure
            is_azure = 'AzureOpenAI' in str(type(client))
            if not is_azure:
                batch_emb = await _embed_batch_async(texts)
            else:
                # For Azure, we should use the sync version
                batch_emb = _embed_batch_sync(texts)
            embeddings.extend(batch_emb)
            logger.info(f"Batch {i//BATCH_SIZE+1} embedded ({len(batch)} items)")
        except Exception as e:
            logger.error(f"Embedding batch {i//BATCH_SIZE+1} failed: {e}")
            raise
    for chunk, vec in zip(chunks, embeddings):
        chunk["embedding"] = vec
    logger.info("All chunks embedded")
    return chunks

def embed_chunks_openai_sync(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Synchronous version for Azure OpenAI. Adds 'embedding' to each chunk.
    Uses text_for_embedding (with contextual headers) if available, falls back to chunk_text.
    """
    logger.info(f"Embedding {len(chunks)} chunks (batch size={BATCH_SIZE})")
    logger.info(f"Using Azure OpenAI client: {'AzureOpenAI' in str(type(client))}")
    embeddings: List[List[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        # Use text_for_embedding if available (contains contextual headers), otherwise fall back to chunk_text
        texts = [c.get("text_for_embedding", c["chunk_text"]) for c in batch]
        try:
            logger.info(f"Calling _embed_batch_sync for batch {i//BATCH_SIZE+1}")
            batch_emb = _embed_batch_sync(texts)
            embeddings.extend(batch_emb)
            logger.info(f"Batch {i//BATCH_SIZE+1} embedded ({len(batch)} items)")
        except Exception as e:
            logger.error(f"Embedding batch {i//BATCH_SIZE+1} failed: {e}")
            raise
    for chunk, vec in zip(chunks, embeddings):
        chunk["embedding"] = vec
    logger.info("All chunks embedded")
    return chunks
