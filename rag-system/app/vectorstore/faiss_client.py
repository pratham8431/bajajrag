import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from pathlib import Path

# — setup logger —
from ..utils.logger import setup_logger
logger = setup_logger(__name__)

# — FAISS configuration —
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "data/faiss_metadata.pkl")
DIM = int(os.getenv("PINECONE_DIM", "1024"))
METRIC = os.getenv("PINECONE_METRIC", "cosine")

# — global variables —
_index = None
_metadata = []
_is_initialized = False

def _init_faiss():
    """Initialize FAISS index lazily."""
    global _index, _metadata, _is_initialized
    
    if _is_initialized:
        return _index, _metadata
    
    # Create data directory if it doesn't exist
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create index
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        logger.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        _index = faiss.read_index(FAISS_INDEX_PATH)
        
        with open(FAISS_METADATA_PATH, 'rb') as f:
            _metadata = pickle.load(f)
        
        logger.info(f"Loaded FAISS index with {_index.ntotal} vectors and {len(_metadata)} metadata entries")
    else:
        # We'll create the index when we have the first embedding
        _index = None
        _metadata = []
        logger.info("FAISS index will be created with first embedding dimension")
    
    _is_initialized = True
    return _index, _metadata

def _save_faiss():
    """Save FAISS index and metadata to disk."""
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'wb') as f:
            pickle.dump(_metadata, f)
        logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")

def upsert_to_faiss(chunks: List[Dict[str, Any]]):
    """
    Upsert embeddings to FAISS index.
    """
    global _index, _metadata
    
    index, metadata = _init_faiss()
    
    total = len(chunks)
    logger.info(f"Upserting {total} chunks to FAISS")
    
    # Prepare vectors and metadata
    vectors = []
    new_metadata = []
    
    for chunk in chunks:
        # Convert embedding to numpy array
        embedding = np.array(chunk["embedding"], dtype=np.float32)
        
        # Log the actual dimension
        logger.info(f"Embedding dimension: {len(embedding)}")
        
        # Normalize for cosine similarity if needed
        if METRIC.lower() == "cosine":
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        vectors.append(embedding)
        new_metadata.append({
            "id": chunk["id"],
            "metadata": chunk["metadata"]
        })
    
    # Convert to numpy array
    vectors_array = np.array(vectors, dtype=np.float32)
    
    # Create index if it doesn't exist
    if index is None:
        dim = vectors_array.shape[1]
        logger.info(f"Creating new FAISS index with dimension {dim}")
        
        if METRIC.lower() == "cosine":
            _index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
        else:
            _index = faiss.IndexFlatL2(dim)  # L2 distance
        
        logger.info(f"Created new FAISS index with {METRIC} metric")
        index = _index
    
    # Add to index
    index.add(vectors_array)
    
    # Add metadata
    metadata.extend(new_metadata)
    _metadata = metadata
    
    # Save to disk
    _save_faiss()
    
    logger.info(f"Successfully upserted {total} chunks to FAISS")

def query_faiss(
    query_embedding: List[float],
    top_k: int = 5,
    filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Query top_k nearest chunks & return id, score, metadata.
    """
    index, metadata = _init_faiss()
    
    if index.ntotal == 0:
        logger.warning("FAISS index is empty")
        return []
    
    # Convert query embedding to numpy array
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Normalize for cosine similarity if needed
    if METRIC.lower() == "cosine":
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
    
    # Search
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))
    
    # Prepare results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(metadata):
            result = {
                "id": metadata[idx]["id"],
                "score": float(score),
                "metadata": metadata[idx]["metadata"]
            }
            results.append(result)
    
    logger.info(f"FAISS query returned {len(results)} results")
    return results

def get_faiss_stats() -> Dict[str, Any]:
    """Get FAISS index statistics."""
    index, metadata = _init_faiss()
    
    return {
        "total_vectors": index.ntotal,
        "dimension": index.d,
        "metric": METRIC,
        "metadata_entries": len(metadata),
        "index_path": FAISS_INDEX_PATH,
        "metadata_path": FAISS_METADATA_PATH
    } 