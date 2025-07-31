# HackRx RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for document processing and intelligent question answering.

## Features

- **Document Processing**: Support for PDF, DOCX, and email documents
- **Intelligent Chunking**: Semantic text splitting with configurable parameters
- **Vector Storage**: Pinecone integration for efficient similarity search
- **Question Answering**: GPT-4 powered responses with source citations
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Database Storage**: PostgreSQL for document and chunk metadata

## Architecture

```
app/
├── api/                    # FastAPI endpoints
│   ├── endpoints/         # API route handlers
│   └── main.py           # FastAPI application setup
├── chunking/              # Text chunking logic
├── db/                    # Database models and utilities
├── embeddings/            # OpenAI embedding generation
├── ingestion/             # Document processing pipeline
├── query/                 # Question parsing and evaluation
├── utils/                 # Configuration and logging
├── vectorstore/           # Pinecone vector storage
└── tests/                # Unit tests
```

## Quick Start

### 1. Environment Setup

Create a `.env` file with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4
EMBED_MODEL=text-embedding-ada-002
EMBED_BATCH_SIZE=100

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
PINECONE_DIM=1536
PINECONE_METRIC=cosine
UPSERT_BATCH_SIZE=100

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db

# Application Configuration
MAX_CHUNKS=500
MAX_PDF_SIZE_MB=50
MAX_QUESTIONS_PER_REQUEST=10
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Database

```bash
# The database will be automatically initialized when the app starts
```

### 4. Run the Application

```bash
python run.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```
GET /health
```

### Process Documents and Answer Questions

```
POST /api/v1/hackrx/run
```

**Request Body:**

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the coverage for knee surgery?",
    "What is the waiting period for pre-existing conditions?"
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "question": "What is the coverage for knee surgery?",
      "answer": "Knee surgery is covered up to 80% of the cost...",
      "justification": "Based on Section 3.2 on page 15...",
      "sources": [
        {
          "id": "chunk-uuid",
          "score": 0.95,
          "metadata": { "section": "Surgical Coverage", "page": 15 }
        }
      ]
    }
  ]
}
```

## Configuration

The system uses a centralized configuration management system. All settings can be configured via environment variables:

- **Document Processing**: File size limits, chunking parameters
- **Embedding**: Model selection, batch sizes
- **Vector Storage**: Index configuration, similarity metrics
- **API**: Rate limits, request validation

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Structure

- **Modular Design**: Each component is self-contained
- **Async Support**: Full async/await support for scalability
- **Error Handling**: Comprehensive error handling and logging
- **Type Hints**: Full type annotation for better code quality

## Security Features

- **Input Validation**: All API inputs are validated
- **File Size Limits**: Protection against large file uploads
- **Environment Validation**: Required environment variables are validated
- **Error Sanitization**: Sensitive information is not exposed in errors

## Performance Optimizations

- **Batch Processing**: Embeddings and vector operations are batched
- **Connection Pooling**: Database connections are pooled
- **Caching**: Configurable caching for frequently accessed data
- **Async Operations**: Non-blocking I/O operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details
