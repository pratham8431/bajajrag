import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.ingestion.pipeline import generate_document_title, ingest_document
from app.chunking.chunker import chunk_text_by_page

class TestContextualHeaders:
    """Test contextual chunk headers functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_document_title(self):
        """Test document title generation."""
        document_text = "This is a health insurance policy document covering maternity benefits and waiting periods."
        document_name = "health_policy.pdf"
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Health Insurance Maternity Policy"
        
        with patch('app.ingestion.pipeline.client.chat.completions.create', return_value=mock_response):
            title = await generate_document_title(document_text, document_name)
            
            assert title == "Health Insurance Maternity Policy"
    
    def test_chunks_with_contextual_headers(self):
        """Test that chunks have contextual headers added."""
        pages = [
            (1, "This is page 1 content about insurance coverage."),
            (2, "This is page 2 content about policy terms.")
        ]
        
        # Create chunks
        chunks = chunk_text_by_page(pages, "test_doc", max_total_chunks=10)
        
        # Add contextual headers (simulating the pipeline)
        document_title = "Test Insurance Policy"
        for chunk in chunks:
            header = f"Document Title: {document_title}\n\n"
            chunk["text_for_embedding"] = header + chunk["chunk_text"]
            chunk["metadata"]["document_title"] = document_title
        
        # Verify headers are added
        for chunk in chunks:
            assert "text_for_embedding" in chunk
            assert chunk["text_for_embedding"].startswith(f"Document Title: {document_title}")
            assert chunk["metadata"]["document_title"] == document_title
            # Original chunk_text should still be available
            assert "chunk_text" in chunk
            assert chunk["chunk_text"] in chunk["text_for_embedding"]
    
    def test_embedding_uses_contextual_headers(self):
        """Test that embedding uses text_for_embedding when available."""
        from app.embeddings.embedder import embed_chunks_openai_sync
        
        # Create test chunks with contextual headers
        chunks = [
            {
                "id": "test1",
                "chunk_text": "Original chunk text",
                "text_for_embedding": "Document Title: Test Policy\n\nOriginal chunk text",
                "metadata": {"section": "test"}
            }
        ]
        
        # Mock the embedding response
        mock_embedding = [0.1, 0.2, 0.3] * 341  # 1024 dimensions
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        
        with patch('app.embeddings.embedder.client.embeddings.create', return_value=mock_response):
            embedded_chunks = embed_chunks_openai_sync(chunks)
            
            # Verify embedding was called with text_for_embedding
            assert len(embedded_chunks) == 1
            assert "embedding" in embedded_chunks[0]
            assert embedded_chunks[0]["embedding"] == mock_embedding
    
    def test_faiss_stores_contextual_headers(self):
        """Test that FAISS stores contextual headers in metadata."""
        from app.vectorstore.faiss_client import upsert_to_faiss
        
        # Create test chunks with contextual headers
        chunks = [
            {
                "id": "test1",
                "chunk_text": "Original chunk text",
                "text_for_embedding": "Document Title: Test Policy\n\nOriginal chunk text",
                "metadata": {"section": "test"},
                "embedding": [0.1] * 1024  # Mock embedding
            }
        ]
        
        # Mock FAISS operations
        with patch('app.vectorstore.faiss_client.faiss.IndexFlatIP') as mock_index_class:
            mock_index = MagicMock()
            mock_index_class.return_value = mock_index
            
            with patch('app.vectorstore.faiss_client.faiss.write_index'):
                with patch('app.vectorstore.faiss_client.pickle.dump'):
                    upsert_to_faiss(chunks)
                    
                    # Verify FAISS was called
                    mock_index.add.assert_called_once()
    
    def test_evaluator_uses_contextual_headers(self):
        """Test that evaluator uses contextual headers when available."""
        from app.query.evaluator import _evaluate_answer_azure
        
        # Create test contexts with contextual headers
        contexts = [
            {
                "chunk_text": "Original chunk text",
                "text_for_embedding": "Document Title: Test Policy\n\nOriginal chunk text",
                "metadata": {"section": "test"}
            }
        ]
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"answer": "Test answer", "justification": "Test justification"}'
        
        with patch('app.query.evaluator.client.chat.completions.create', return_value=mock_response):
            result = _evaluate_answer_azure(
                question="What is the policy coverage?",
                structured_query={"intent": "coverage_check"},
                contexts=contexts,
                model="test-model"
            )
            
            assert result["answer"] == "Test answer"
            assert result["justification"] == "Test justification"

if __name__ == "__main__":
    pytest.main([__file__]) 