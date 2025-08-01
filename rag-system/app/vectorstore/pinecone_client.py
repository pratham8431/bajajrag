import os
import logging
from typing import List, Dict, Any

import pinecone
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# — setup logger —
from ..utils.logger import setup_logger
logger = setup_logger(__name__)

# — init Pinecone —
_pinecone_initialized = False
_index = None

def _init_pinecone():
    """Initialize Pinecone connection lazily with serverless tier."""
    global _pinecone_initialized, _index
    
    if _pinecone_initialized:
        return _index
    
    # Load environment variables dynamically
    load_dotenv()
    
    # Read all environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENV")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    dim = int(os.getenv("PINECONE_DIM", "1024"))
    metric = os.getenv("PINECONE_METRIC", "cosine")
    
    # Validate required environment variables
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    if not env:
        raise ValueError("PINECONE_ENV environment variable is required")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable is required")
    
    logger.info(f"Initializing Pinecone with environment: {env}")
    logger.info(f"Using index name: {index_name}")
    logger.info(f"Using dimensions: {dim}, metric: {metric}")
    
    try:
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=env)
        
        # Check if index exists
        existing_indexes = pinecone.list_indexes()
        logger.info(f"Available indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            logger.info(f"Creating index {index_name} (dim={dim}) with minimum pods")
            # Use minimum pods (1) to avoid the 0 pods error
            pinecone.create_index(
                name=index_name, 
                dimension=dim, 
                metric=metric,
                pods=1
            )
            logger.info(f"Index {index_name} created successfully")
        else:
            logger.info(f"Index {index_name} already exists, connecting to it")
        
        _index = pinecone.Index(index_name)
        _pinecone_initialized = True
        logger.info("Pinecone initialized successfully")
        return _index
        
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def _upsert_batch(vectors: List[tuple]):
    index = _init_pinecone()
    resp = index.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors")
    return resp

def upsert_to_pinecone(
    chunks: List[Dict[str, Any]]
):
    """
    Upsert embeddings in batched calls with retry.
    """
    # Read batch size dynamically
    batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
    
    total = len(chunks)
    logger.info(f"Upserting {total} chunks to Pinecone (batch={batch_size})")
    
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vectors = [(c["id"], c["embedding"], c["metadata"]) for c in batch]
        try:
            _upsert_batch(vectors)
        except Exception as e:
            logger.error(f"Failed upsert on batch {i//batch_size+1}: {e}")
            raise
    
    logger.info("All chunks upserted successfully")

def query_pinecone(
    query_embedding: List[float],
    top_k: int = 5,
    filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Query top_k nearest chunks & return id, score, metadata.
    """
    index = _init_pinecone()
    resp = index.query(vector=query_embedding, top_k=top_k, filter=filter or {}, include_metadata=True)
    return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in resp.matches]
