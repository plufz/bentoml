# RAG (Retrieval-Augmented Generation) Service

Complete document ingestion and question-answering service using sentence-transformers, Milvus vector database, and Phi-3 Mini for local RAG applications.

## Overview

The RAG service combines document retrieval with language generation to provide accurate, context-aware responses to user questions. It consists of three main components:

- **Embedding Service** (`rag_embedding_service.py`) - Text embeddings using sentence-transformers
- **LLM Service** (`rag_llm_service.py`) - Language generation using Phi-3 Mini via llama-cpp
- **RAG Service** (`rag_service.py`) - Document ingestion, retrieval, and response generation

**Key Features:**
- Multiple document input formats (text, PDF, text files)
- Vector similarity search with Milvus database
- Context-aware response generation
- Local deployment with CPU-optimized models
- Comprehensive document management
- Configurable retrieval and generation parameters

## Quick Start

```bash
# Build RAG service
BENTOFILE=config/bentofiles/rag.yaml ./scripts/run_bentoml.sh build services/rag_service.py

# Start service (downloads models on first run)
./scripts/run_bentoml.sh serve services.rag_service:RAGService

# Test with dedicated script
./scripts/test_rag.sh workflow
```

## API Reference

### Document Ingestion

#### Ingest Text Document
**Endpoint**: `POST /rag_ingest_text`

**Request Format**:
```json
{
  "request": {
    "text": "This is the document content to be indexed for RAG retrieval...",
    "metadata": {
      "source": "manual_input",
      "topic": "machine_learning",
      "author": "user"
    },
    "doc_id": "optional-custom-id"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Document ingested successfully",
  "doc_id": "auto-generated-or-custom-id",
  "chunks_created": 3
}
```

#### Ingest PDF File
**Endpoint**: `POST /rag_ingest_pdf`

**Request**: Multipart form data with PDF file

```bash
curl -X POST http://127.0.0.1:3000/rag_ingest_pdf \
  -F "pdf_file=@./documents/research_paper.pdf"
```

**Response**:
```json
{
  "status": "success", 
  "message": "PDF 'research_paper.pdf' ingested successfully",
  "doc_id": "generated-doc-id",
  "chunks_created": 15
}
```

#### Ingest Text File
**Endpoint**: `POST /rag_ingest_txt_file`

**Request**: Multipart form data with text file

```bash
curl -X POST http://127.0.0.1:3000/rag_ingest_txt_file \
  -F "txt_file=@./documents/notes.txt"
```

### Document Query

#### RAG Query
**Endpoint**: `POST /rag_query`

**Request Format**:
```json
{
  "request": {
    "query": "What is machine learning and how does it work?",
    "max_tokens": 512,
    "temperature": 0.1,
    "top_k": 3,
    "similarity_threshold": 0.7
  }
}
```

**Response**:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed...",
  "sources": [
    {
      "id": 1,
      "text": "Machine learning is a method of data analysis that automates analytical model building...",
      "metadata": {
        "source": "ml_textbook",
        "chapter": "introduction"
      },
      "score": 0.85
    }
  ],
  "tokens_generated": 127,
  "retrieval_score": 0.82
}
```

### Index Management

#### Clear Index
**Endpoint**: `POST /rag_clear_index`

**Request**: `{}`

**Response**:
```json
{
  "status": "success",
  "message": "RAG index cleared successfully"
}
```

#### Health Check
**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "service": "RAGService",
  "components": {
    "embedding_service": {
      "status": "healthy",
      "model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "llm_service": {
      "status": "healthy", 
      "model": "microsoft/Phi-3-mini-4k-instruct-gguf"
    },
    "vector_store": "milvus",
    "index_status": "initialized"
  },
  "version": "1.0.0"
}
```

## Testing

### Using Test Scripts

#### Comprehensive Workflow Test
```bash
# Test complete RAG workflow (ingestion + query)
./scripts/test_rag.sh workflow

# Test individual components
./scripts/test_rag.sh health
./scripts/test_rag.sh ingest-text "Sample document content"
./scripts/test_rag.sh query "What is the main topic?"
```

#### Using pytest
```bash
# Test RAG service specifically
./scripts/test.sh --service rag

# Run with coverage
./scripts/test.sh --service rag --coverage

# Test specific test classes
uv run pytest tests/test_rag_service.py::TestRAGServiceUnit -v
```

### Using Endpoint Script

```bash
# Test document ingestion
./scripts/endpoint.sh rag_ingest_text '{"text": "Artificial intelligence (AI) is the simulation of human intelligence in machines...", "metadata": {"topic": "ai"}}'

# Test with file upload
curl -X POST http://127.0.0.1:3000/rag_ingest_pdf -F "pdf_file=@./test.pdf"

# Test querying
./scripts/endpoint.sh rag_query '{"query": "What is artificial intelligence?", "max_tokens": 256}'

# Clear the index
./scripts/endpoint.sh rag_clear_index '{}'
```

## Configuration

### Model Configuration

The RAG service uses the following models:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Dimension: 384
  - Fast and efficient for semantic similarity
  - CPU-optimized

- **Language Model**: `microsoft/Phi-3-mini-4k-instruct-gguf`
  - Context window: 4K tokens
  - Quantized (Q4) for efficiency
  - CPU-optimized with llama-cpp-python

### Vector Store Configuration

- **Database**: Milvus (SQLite-based for local deployment)
- **Storage**: `./storage/rag_milvus.db`
- **Collection**: `rag_documents`
- **Dimension**: 384 (matches embedding model)

### Text Processing

- **Chunk Size**: 1024 characters
- **Chunk Overlap**: 200 characters
- **Separator**: Space-based splitting
- **Document Types**: Text, PDF, TXT files

## Resource Requirements

### System Requirements
- **Memory**: 16GB+ recommended
- **CPU**: 6+ cores for optimal performance
- **Storage**: 5GB+ for models and vector database
- **Platform**: macOS (Apple Silicon optimized), Linux

### BentoML Resource Configuration
```yaml
# In config/bentofiles/rag.yaml
resources:
  memory: "16Gi"
  cpu: "6"
```

## Architecture

### Component Services

```
RAGService
├── RAGEmbeddingService
│   ├── sentence-transformers/all-MiniLM-L6-v2
│   └── BentoMLEmbeddings (LlamaIndex wrapper)
├── RAGLLMService 
│   ├── microsoft/Phi-3-mini-4k-instruct-gguf
│   └── LlamaCPP backend
└── Vector Store
    ├── MilvusVectorStore (local SQLite)
    ├── VectorStoreIndex (LlamaIndex)
    └── SentenceSplitter (text chunking)
```

### Data Flow

1. **Document Ingestion**:
   - Text/PDF/File → Document object
   - Document → Text chunks (1024 chars)
   - Text chunks → Embeddings (384-dim vectors)
   - Embeddings → Vector store (Milvus)

2. **Query Processing**:
   - Question → Query embedding
   - Query embedding → Vector search (top-k similar chunks)
   - Retrieved chunks → Context for LLM
   - Context + Question → LLM → Generated answer

### Service Dependencies

The RAG service uses `bentoml.depends()` for service composition:

```python
@bentoml.service()
class RAGService:
    embedding_service = bentoml.depends(RAGEmbeddingService)
    llm_service = bentoml.depends(RAGLLMService)
```

This ensures proper service lifecycle management and dependency injection.

## Troubleshooting

### Common Issues

#### Model Download Failures
```bash
# Check internet connection and HuggingFace access
curl -I https://huggingface.co/

# Verify model paths in service code
ls /Volumes/Second/models/  # Check for Phi-3 GGUF file
```

#### Memory Issues
- Reduce batch sizes in embedding service
- Use smaller LLM model variant
- Increase system swap space

#### Vector Store Issues
```bash
# Clear vector database if corrupted
rm -rf ./storage/rag_milvus.db

# Check permissions
ls -la ./storage/
```

#### Service Startup Timeouts
- Increase pytest timeout for integration tests
- Pre-download models before service startup
- Check available system resources

### Performance Optimization

#### Embedding Service
- Use GPU acceleration if available: `device="cuda"`
- Batch multiple texts for embedding efficiency
- Consider smaller embedding models for speed

#### LLM Service  
- Adjust `n_threads` based on CPU cores
- Use appropriate quantization level (Q4/Q8)
- Tune context window size for memory usage

#### Vector Store
- Configure appropriate `similarity_top_k` values
- Use similarity thresholds to filter low-quality results
- Regular index maintenance for large datasets

## Multi-Service Integration

The RAG service integrates with the multi-service architecture:

```python
# In services/multi_service.py
class MultiService:
    def __init__(self):
        self.rag_service = RAGService()
    
    @bentoml.api
    def rag_query(self, request: RAGQueryRequest) -> Dict[str, Any]:
        return self.rag_service.query(request)
```

**Available in Multi-Service:**
- `/rag_ingest_text`
- `/rag_ingest_pdf` 
- `/rag_ingest_txt_file`
- `/rag_query`
- `/rag_clear_index`

## Examples

### Python Client Usage

```python
import requests

# Service endpoint
base_url = "http://127.0.0.1:3000"

# Ingest a document
doc_data = {
    "request": {
        "text": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "metadata": {"topic": "programming", "language": "python"}
    }
}

response = requests.post(f"{base_url}/rag_ingest_text", json=doc_data)
print("Ingestion:", response.json())

# Query the document
query_data = {
    "request": {
        "query": "What programming paradigms does Python support?",
        "max_tokens": 256,
        "temperature": 0.1
    }
}

response = requests.post(f"{base_url}/rag_query", json=query_data)
result = response.json()
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

### Batch Document Processing

```python
import os
import requests
from pathlib import Path

base_url = "http://127.0.0.1:3000"

# Process multiple PDF files
pdf_dir = Path("./documents")
for pdf_file in pdf_dir.glob("*.pdf"):
    with open(pdf_file, "rb") as f:
        files = {"pdf_file": f}
        response = requests.post(f"{base_url}/rag_ingest_pdf", files=files)
        print(f"Processed {pdf_file.name}: {response.json()}")

# Query after batch processing
queries = [
    "What are the main topics covered?",
    "Can you summarize the key findings?",
    "What methodologies were used?"
]

for query in queries:
    query_data = {"request": {"query": query, "top_k": 5}}
    response = requests.post(f"{base_url}/rag_query", json=query_data)
    print(f"Q: {query}")
    print(f"A: {response.json()['answer']}\n")
```

## Advanced Usage

### Custom Metadata Filtering

While the current implementation doesn't support metadata filtering in queries, documents can be organized using metadata for future filtering capabilities:

```python
# Ingest with structured metadata
doc_data = {
    "request": {
        "text": "Machine learning research paper content...",
        "metadata": {
            "document_type": "research_paper",
            "field": "machine_learning", 
            "year": "2024",
            "authors": ["Dr. Smith", "Dr. Johnson"],
            "conference": "ICML 2024"
        }
    }
}
```

### Custom Retrieval Parameters

```python
# Fine-tune retrieval and generation
query_data = {
    "request": {
        "query": "How does attention mechanism work in transformers?",
        "max_tokens": 1024,        # Longer response
        "temperature": 0.2,        # More creative
        "top_k": 5,               # More context sources
        "similarity_threshold": 0.6  # Lower threshold for more results
    }
}
```

This comprehensive RAG service provides a complete solution for document-based question answering with modern RAG techniques, optimized for local deployment and integrated with the BentoML ecosystem.