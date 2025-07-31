import os
import logging
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# — load keys & config —
load_dotenv()
openai.api_key   = os.getenv("OPENAI_API_KEY")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
BATCH_SIZE       = int(os.getenv("EMBED_BATCH_SIZE", "100"))

# — logger —
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
async def _embed_batch(texts: List[str]) -> List[List[float]]:
    resp = await openai.Embedding.acreate(model=EMBED_MODEL, input=texts)
    return [item["embedding"] for item in resp["data"]]

async def embed_chunks_openai(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Batch‐embed each chunk asynchronously. Adds 'embedding' to each chunk.
    """
    logger.info(f"Embedding {len(chunks)} chunks (batch size={BATCH_SIZE})")
    embeddings: List[List[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]
        try:
            batch_emb = await _embed_batch(texts)
            embeddings.extend(batch_emb)
            logger.info(f"Batch {i//BATCH_SIZE+1} embedded ({len(batch)} items)")
        except Exception as e:
            logger.error(f"Embedding batch {i//BATCH_SIZE+1} failed: {e}")
            raise
    for chunk, vec in zip(chunks, embeddings):
        chunk["embedding"] = vec
    logger.info("All chunks embedded")
    return chunks
