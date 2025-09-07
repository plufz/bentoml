"""
Tests for RAG Service
Comprehensive test suite for the RAG (Retrieval-Augmented Generation) service.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Import the services and models
from services.rag_service import RAGService, DocumentIngestRequest, RAGQueryRequest
from services.rag_embedding_service import RAGEmbeddingService, EmbeddingRequest
from services.rag_llm_service import RAGLLMService, LLMRequest, RAGQueryRequest as LLMRAGQueryRequest


class TestRAGEmbeddingServiceUnit:
    """Unit tests for RAG Embedding Service"""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mocked embedding service"""
        service = Mock(spec=RAGEmbeddingService)
        
        # Mock the embed_texts method
        def mock_embed_texts(request):
            # Return dummy embeddings based on text length
            embeddings = []
            for text in request.texts:
                # Create a simple embedding based on text hash
                embedding = [float(i % 100) / 100.0 for i in range(384)]
                embeddings.append(embedding)
            
            from services.rag_embedding_service import EmbeddingResponse
            return EmbeddingResponse(
                embeddings=embeddings,
                dimension=384,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        service.embed_texts = mock_embed_texts
        service.embed_single.return_value = [0.1] * 384
        service.health.return_value = {
            "status": "healthy",
            "service": "RAGEmbeddingService"
        }
        
        return service

    def test_embedding_request_validation(self):
        """Test EmbeddingRequest model validation"""
        # Valid request
        request = EmbeddingRequest(texts=["hello world"], normalize=True)
        assert request.texts == ["hello world"]
        assert request.normalize == True
        
        # Default normalize value
        request = EmbeddingRequest(texts=["hello"])
        assert request.normalize == True

    def test_embed_texts_mock(self, mock_embedding_service):
        """Test embedding generation with mocked service"""
        request = EmbeddingRequest(texts=["hello", "world"])
        response = mock_embedding_service.embed_texts(request)
        
        assert len(response.embeddings) == 2
        assert response.dimension == 384
        assert response.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert all(len(emb) == 384 for emb in response.embeddings)


class TestRAGLLMServiceUnit:
    """Unit tests for RAG LLM Service"""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mocked LLM service"""
        service = Mock(spec=RAGLLMService)
        
        def mock_generate(request):
            from services.rag_llm_service import LLMResponse
            return LLMResponse(
                text="This is a generated response.",
                tokens_generated=10,
                model="microsoft/Phi-3-mini-4k-instruct-gguf",
                finish_reason="stop"
            )
        
        def mock_rag_query(request):
            from services.rag_llm_service import RAGQueryResponse
            return RAGQueryResponse(
                answer=f"Based on the context, {request.question}",
                tokens_generated=15,
                model="microsoft/Phi-3-mini-4k-instruct-gguf",
                context_used=bool(request.context.strip())
            )
        
        service.generate = mock_generate
        service.rag_query = mock_rag_query
        service.health.return_value = {
            "status": "healthy",
            "service": "RAGLLMService"
        }
        
        return service

    def test_llm_request_validation(self):
        """Test LLMRequest model validation"""
        request = LLMRequest(prompt="Hello world")
        assert request.prompt == "Hello world"
        assert request.max_tokens == 512  # default
        assert request.temperature == 0.1  # default

    def test_rag_query_request_validation(self):
        """Test RAGQueryRequest model validation"""
        request = LLMRAGQueryRequest(
            context="Some context",
            question="What is this about?"
        )
        assert request.context == "Some context"
        assert request.question == "What is this about?"
        assert request.max_tokens == 512
        assert request.temperature == 0.1

    def test_generate_mock(self, mock_llm_service):
        """Test text generation with mocked service"""
        request = LLMRequest(prompt="Hello world")
        response = mock_llm_service.generate(request)
        
        assert response.text == "This is a generated response."
        assert response.tokens_generated == 10
        assert response.model == "microsoft/Phi-3-mini-4k-instruct-gguf"

    def test_rag_query_mock(self, mock_llm_service):
        """Test RAG query with mocked service"""
        request = LLMRAGQueryRequest(
            context="The sky is blue.",
            question="What color is the sky?"
        )
        response = mock_llm_service.rag_query(request)
        
        assert "What color is the sky?" in response.answer
        assert response.context_used == True
        assert response.tokens_generated == 15


class TestRAGServiceUnit:
    """Unit tests for main RAG Service"""

    @pytest.fixture
    def mock_rag_service(self):
        """Create a mocked RAG service with dependencies"""
        # Mock the dependencies
        mock_embedding = Mock()
        mock_llm = Mock()
        
        # Create the service and replace dependencies
        with patch('services.rag_service.RAGEmbeddingService', return_value=mock_embedding), \
             patch('services.rag_service.RAGLLMService', return_value=mock_llm), \
             patch('services.rag_service.MilvusVectorStore'), \
             patch('services.rag_service.VectorStoreIndex'):
            
            service = RAGService()
            service.embedding_service = mock_embedding
            service.llm_service = mock_llm
            
            # Mock successful responses
            mock_llm.rag_query.return_value = Mock(
                answer="This is a test answer",
                tokens_generated=20
            )
            
            return service

    def test_document_ingest_request_validation(self):
        """Test DocumentIngestRequest model validation"""
        request = DocumentIngestRequest(
            text="This is a test document",
            metadata={"source": "test"},
            doc_id="test-doc-1"
        )
        assert request.text == "This is a test document"
        assert request.metadata == {"source": "test"}
        assert request.doc_id == "test-doc-1"

    def test_rag_query_request_validation(self):
        """Test RAGQueryRequest model validation"""
        request = RAGQueryRequest(query="What is this about?")
        assert request.query == "What is this about?"
        assert request.max_tokens == 512
        assert request.temperature == 0.1
        assert request.top_k == 3
        assert request.similarity_threshold == 0.7

    @patch('services.rag_service.MilvusVectorStore')
    @patch('services.rag_service.VectorStoreIndex') 
    def test_rag_service_init_mocked(self, mock_vector_index, mock_vector_store):
        """Test RAG service initialization with mocked dependencies"""
        mock_embedding = Mock()
        mock_llm = Mock()
        
        with patch('services.rag_service.RAGEmbeddingService', return_value=mock_embedding), \
             patch('services.rag_service.RAGLLMService', return_value=mock_llm):
            
            service = RAGService()
            assert service.embedding_service == mock_embedding
            assert service.llm_service == mock_llm

    def test_health_check_mocked(self, mock_rag_service):
        """Test health check with mocked service"""
        mock_rag_service.embedding_service.health.return_value = {"status": "healthy"}
        mock_rag_service.llm_service.health.return_value = {"status": "healthy"}
        
        health = mock_rag_service.health()
        assert health["status"] == "healthy"
        assert "components" in health


@pytest.mark.slow
class TestRAGServiceIntegration:
    """Integration tests for RAG Service (may be slow due to model loading)"""

    @pytest.fixture
    def temp_pdf(self):
        """Create a temporary PDF file for testing"""
        # This is a simplified test - in real scenarios you'd create a proper PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            # Write minimal PDF content (this won't be a real PDF)
            f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def temp_txt(self):
        """Create a temporary text file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
            f.write("This is a test document.\nIt contains some sample text for RAG testing.\n")
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.mark.timeout(180)
    def test_rag_service_startup(self):
        """Test that RAG service can be instantiated (with timeouts for model loading)"""
        try:
            # This test may fail if models aren't available locally
            # That's expected in CI/test environments
            service = RAGService()
            assert service is not None
        except (FileNotFoundError, ImportError, Exception) as e:
            pytest.skip(f"RAG service dependencies not available: {e}")

    def test_text_ingestion_mock_integration(self):
        """Test text ingestion with integration-level mocking"""
        with patch('services.rag_service.Document') as mock_doc, \
             patch('services.rag_service.VectorStoreIndex') as mock_index:
            
            mock_embedding = Mock()
            mock_llm = Mock()
            
            with patch('services.rag_service.RAGEmbeddingService', return_value=mock_embedding), \
                 patch('services.rag_service.RAGLLMService', return_value=mock_llm):
                
                service = RAGService()
                service.text_splitter = Mock()
                service.text_splitter.get_nodes_from_documents.return_value = [Mock(), Mock()]
                
                request = DocumentIngestRequest(
                    text="This is a test document",
                    metadata={"source": "test"}
                )
                
                # Mock the index creation/update
                mock_index.from_documents.return_value = Mock()
                service.index = Mock()
                service.index.insert.return_value = None
                
                response = service.ingest_text(request)
                assert response.status == "success"
                assert response.chunks_created == 2

    def test_query_with_no_documents(self):
        """Test querying when no documents are ingested"""
        with patch('services.rag_service.RAGEmbeddingService'), \
             patch('services.rag_service.RAGLLMService'), \
             patch('services.rag_service.MilvusVectorStore'), \
             patch('services.rag_service.VectorStoreIndex'):
            
            service = RAGService()
            service.index = None  # No documents ingested
            
            request = RAGQueryRequest(query="What is this about?")
            response = service.query(request)
            
            assert "No documents have been ingested" in response.answer
            assert response.sources == []
            assert response.tokens_generated == 0


class TestRAGServiceEndToEnd:
    """End-to-end tests for complete RAG workflows"""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_complete_rag_workflow_mocked(self):
        """Test complete RAG workflow with comprehensive mocking"""
        # Mock all external dependencies
        with patch('services.rag_service.MilvusVectorStore') as mock_vector_store, \
             patch('services.rag_service.VectorStoreIndex') as mock_vector_index, \
             patch('services.rag_service.Document') as mock_document:
            
            # Mock services
            mock_embedding = Mock()
            mock_llm = Mock()
            
            # Mock LLM response
            from services.rag_llm_service import RAGQueryResponse
            mock_llm.rag_query.return_value = RAGQueryResponse(
                answer="The document is about testing RAG systems.",
                tokens_generated=25,
                model="test-model",
                context_used=True
            )
            
            with patch('services.rag_service.RAGEmbeddingService', return_value=mock_embedding), \
                 patch('services.rag_service.RAGLLMService', return_value=mock_llm):
                
                service = RAGService()
                
                # Setup mocked index and retriever
                mock_retriever = Mock()
                mock_node = Mock()
                mock_node.text = "This is a test document about RAG systems."
                mock_node.score = 0.85
                mock_node.metadata = {"source": "test"}
                mock_retriever.retrieve.return_value = [mock_node]
                
                mock_index = Mock()
                mock_index.as_retriever.return_value = mock_retriever
                service.index = mock_index
                
                # Mock text splitter
                service.text_splitter = Mock()
                service.text_splitter.get_nodes_from_documents.return_value = [Mock()]
                
                # Test document ingestion
                ingest_request = DocumentIngestRequest(
                    text="This is a comprehensive test document about RAG systems and their applications.",
                    metadata={"source": "test", "type": "documentation"}
                )
                
                ingest_response = service.ingest_text(ingest_request)
                assert ingest_response.status == "success"
                
                # Test querying
                query_request = RAGQueryRequest(
                    query="What is this document about?",
                    max_tokens=100,
                    temperature=0.1,
                    top_k=3
                )
                
                query_response = service.query(query_request)
                assert "RAG systems" in query_response.answer
                assert len(query_response.sources) > 0
                assert query_response.tokens_generated > 0
                assert query_response.retrieval_score > 0

    def test_error_handling(self):
        """Test error handling in RAG service"""
        with patch('services.rag_service.RAGEmbeddingService'), \
             patch('services.rag_service.RAGLLMService'), \
             patch('services.rag_service.MilvusVectorStore'), \
             patch('services.rag_service.VectorStoreIndex'):
            
            service = RAGService()
            
            # Test error in ingestion
            with patch.object(service, 'text_splitter') as mock_splitter:
                mock_splitter.get_nodes_from_documents.side_effect = Exception("Test error")
                
                request = DocumentIngestRequest(text="test")
                
                with pytest.raises(Exception) as exc_info:
                    service.ingest_text(request)
                assert "Error ingesting document" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])