import os
import logging
from typing import List, Dict, Any

import pinecone
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# — load config —
load_dotenv()
API_KEY     = os.getenv("PINECONE_API_KEY")
ENV         = os.getenv("PINECONE_ENV")
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME")
DIM         = int(os.getenv("PINECONE_DIM", "1536"))
METRIC      = os.getenv("PINECONE_METRIC", "cosine")
BATCH_SIZE  = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

# — setup logger —
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# — init Pinecone —
pinecone.init(api_key=API_KEY, environment=ENV)
if INDEX_NAME not in pinecone.list_indexes():
    logger.info(f"Creating index {INDEX_NAME} (dim={DIM})")
    pinecone.create_index(INDEX_NAME, dimension=DIM, metric=METRIC)
index = pinecone.Index(INDEX_NAME)

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def _upsert_batch(vectors: List[tuple]):
    resp = index.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors")
    return resp

def upsert_to_pinecone(
    chunks: List[Dict[str, Any]]
):
    """
    Upsert embeddings in batched calls with retry.
    """
    total = len(chunks)
    logger.info(f"Upserting {total} chunks to Pinecone (batch={BATCH_SIZE})")
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = [(c["id"], c["embedding"], c["metadata"]) for c in batch]
        try:
            _upsert_batch(vectors)
        except Exception as e:
            logger.error(f"Failed upsert on batch {i//BATCH_SIZE+1}: {e}")
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
    resp = index.query(vector=query_embedding, top_k=top_k, filter=filter or {}, include_metadata=True)
    return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in resp.matches]
