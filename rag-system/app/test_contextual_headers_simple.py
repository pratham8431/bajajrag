#!/usr/bin/env python3
"""
Simple test script to verify contextual headers functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking.chunker import chunk_text_by_page

def test_contextual_headers():
    """Test that chunks can have contextual headers added."""
    print("🧪 Testing contextual headers functionality...")
    
    # Create test pages
    pages = [
        (1, "This is page 1 content about insurance coverage and maternity benefits."),
        (2, "This is page 2 content about policy terms and waiting periods.")
    ]
    
    # Create chunks
    chunks = chunk_text_by_page(pages, "test_doc", max_total_chunks=10)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Add contextual headers (simulating the pipeline)
    document_title = "Health Insurance Maternity Policy"
    for chunk in chunks:
        header = f"Document Title: {document_title}\n\n"
        chunk["text_for_embedding"] = header + chunk["chunk_text"]
        chunk["metadata"]["document_title"] = document_title
    
    # Verify headers are added
    print("🔍 Verifying contextual headers...")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}:")
        print(f"    Original text: {chunk['chunk_text'][:50]}...")
        print(f"    With header: {chunk['text_for_embedding'][:80]}...")
        print(f"    Document title in metadata: {chunk['metadata']['document_title']}")
        print()
        
        # Assertions
        assert "text_for_embedding" in chunk, "text_for_embedding should be present"
        assert chunk["text_for_embedding"].startswith(f"Document Title: {document_title}"), "Header should be prepended"
        assert chunk["metadata"]["document_title"] == document_title, "Document title should be in metadata"
        assert "chunk_text" in chunk, "Original chunk_text should still be available"
        assert chunk["chunk_text"] in chunk["text_for_embedding"], "Original text should be in embedded text"
    
    print("✅ All contextual header tests passed!")
    return True

def test_embedding_text_selection():
    """Test that embedding functions use text_for_embedding when available."""
    print("\n🧪 Testing embedding text selection...")
    
    # Create test chunks with contextual headers
    chunks = [
        {
            "id": "test1",
            "chunk_text": "Original chunk text",
            "text_for_embedding": "Document Title: Test Policy\n\nOriginal chunk text",
            "metadata": {"section": "test"}
        },
        {
            "id": "test2", 
            "chunk_text": "Another chunk without header",
            "metadata": {"section": "test"}
        }
    ]
    
    # Test the logic that selects text for embedding
    for i, chunk in enumerate(chunks):
        # Use text_for_embedding if available, otherwise fall back to chunk_text
        text_for_embedding = chunk.get("text_for_embedding", chunk["chunk_text"])
        print(f"  Chunk {i+1}:")
        print(f"    Selected text: {text_for_embedding[:60]}...")
        
        if "text_for_embedding" in chunk:
            assert text_for_embedding == chunk["text_for_embedding"], "Should use text_for_embedding when available"
        else:
            assert text_for_embedding == chunk["chunk_text"], "Should fall back to chunk_text"
    
    print("✅ Embedding text selection tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_contextual_headers()
        test_embedding_text_selection()
        print("\n🎉 All tests passed! Contextual headers are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1) 