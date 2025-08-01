import pytest
from app.chunking.chunker import split_into_sections, chunk_sections, chunk_text_by_page

class TestChunker:
    def test_split_into_sections(self):
        """Test section splitting functionality."""
        pages = [
            (1, "PART I\nThis is the first part content."),
            (2, "ARTICLE 1.\nThis is article 1 content."),
            (3, "ARTICLE 2.\nThis is article 2 content.")
        ]
        
        sections = split_into_sections(pages)
        
        assert len(sections) == 2
        assert sections[0]["section_title"] == "PART I"
        assert "first part content" in sections[0]["section_text"]
        assert sections[1]["section_title"] == "ARTICLE 1."
        assert "article 1 content" in sections[1]["section_text"]
    
    def test_chunk_sections(self):
        """Test chunking functionality."""
        sections = [
            {"section_title": "Test Section", "section_text": "This is a test section with some content."}
        ]
        
        chunks = chunk_sections(sections, chunk_size=20, chunk_overlap=5)
        
        assert len(chunks) > 0
        assert all("id" in chunk for chunk in chunks)
        assert all("chunk_text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
    
    def test_chunk_text_by_page(self):
        """Test full chunking pipeline."""
        pages = [
            (1, "Test page content for chunking."),
            (2, "Another page with more content.")
        ]
        
        chunks = chunk_text_by_page(pages, "test_doc", max_total_chunks=10)
        
        assert len(chunks) > 0
        assert all("id" in chunk for chunk in chunks)
        assert all("chunk_text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
