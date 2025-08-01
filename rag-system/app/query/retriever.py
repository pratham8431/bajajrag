import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..vectorstore.faiss_client import query_faiss
from ..embeddings.embedder import _embed_batch_sync, _embed_batch_async, client

class HybridRetriever:
    """
    Advanced hybrid retrieval system combining:
    - Dense embeddings (semantic search)
    - Sparse TF-IDF (keyword search)
    - Exact keyword matching
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.chunk_texts = []
    
    def build_tfidf_index(self, chunks: List[Dict[str, Any]]):
        """Build TF-IDF index from chunks."""
        self.chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        if self.chunk_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_texts)
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform keyword-based search using TF-IDF."""
        if self.tfidf_matrix is None:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "id": f"keyword_{idx}",
                    "score": float(similarities[idx]),
                    "method": "keyword_search",
                    "chunk_text": self.chunk_texts[idx]
                })
        
        return results
    
    def exact_keyword_match(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find exact keyword matches."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        results = []
        
        for chunk in chunks:
            chunk_terms = set(re.findall(r'\b\w+\b', chunk["chunk_text"].lower()))
            matches = query_terms.intersection(chunk_terms)
            
            if matches:
                score = len(matches) / len(query_terms)
                results.append({
                    "id": chunk["id"],
                    "score": score,
                    "method": "exact_match",
                    "matched_terms": list(matches),
                    "chunk_text": chunk["chunk_text"]
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    
    async def hybrid_search(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining multiple retrieval methods.
        """
        # Build TF-IDF index if not already built
        if not self.tfidf_matrix:
            self.build_tfidf_index(chunks)
        
        # 1. Dense vector search (semantic)
        try:
            is_azure = 'AzureOpenAI' in str(type(client))
            if is_azure:
                # Azure OpenAI - synchronous
                query_embedding = (_embed_batch_sync([query]))[0]
            else:
                # OpenAI - asynchronous
                query_embedding = (await _embed_batch_async([query]))[0]
            dense_results = query_faiss(query_embedding, top_k=top_k)
            for result in dense_results:
                result["method"] = "semantic_search"
                # Ensure we have the contextual headers for evaluation
                if "text_for_embedding" in result["metadata"]:
                    result["text_for_embedding"] = result["metadata"]["text_for_embedding"]
                result["chunk_text"] = result["metadata"].get("chunk_text", "")
        except Exception as e:
            dense_results = []
        
        # 2. Keyword search (sparse)
        keyword_results = self.keyword_search(query, top_k=top_k)
        
        # 3. Exact keyword matching
        exact_results = self.exact_keyword_match(query, chunks)
        
        # Combine and rerank results
        all_results = dense_results + keyword_results + exact_results
        
        # Remove duplicates and rerank
        seen_ids = set()
        final_results = []
        
        for result in all_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                final_results.append(result)
        
        # Sort by score and return top_k
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

# Global retriever instance
hybrid_retriever = HybridRetriever()
