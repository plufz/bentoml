"""
RAG Service for BentoML
Complete Retrieval-Augmented Generation service with document ingestion and query capabilities.
"""

from __future__ import annotations
import bentoml
import typing as t
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import our custom services
from services.rag_embedding_service import RAGEmbeddingService, BentoMLEmbeddings
from services.rag_llm_service import RAGLLMService, RAGQueryRequest

# Import LlamaIndex components
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.llama_cpp import LlamaCPP


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion from text"""
    text: str
    metadata: Dict[str, Any] = {}
    doc_id: Optional[str] = None


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion"""
    status: str
    message: str
    doc_id: str
    chunks_created: int


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str
    max_tokens: int = 512
    temperature: float = 0.1
    top_k: int = 3
    similarity_threshold: float = 0.7


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    sources: List[Dict[str, Any]]
    tokens_generated: int
    retrieval_score: float


class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    documents: List[Dict[str, Any]]
    total_count: int
    index_size: int


@bentoml.service(
    traffic={"timeout": 600},
    resources={"memory": "16Gi", "cpu": "6"}
)
class RAGService:
    """
    Complete RAG Service
    Provides document ingestion, embedding, storage, retrieval, and generation capabilities.
    """

    embedding_service = bentoml.depends(RAGEmbeddingService)
    llm_service = bentoml.depends(RAGLLMService)

    def __init__(self):
        """Initialize RAG service with vector store and LLM"""
        
        # Initialize embedding model wrapper
        self.embed_model = BentoMLEmbeddings(self.embedding_service)
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 200
        
        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=1024, 
            chunk_overlap=200,
            separator=" "
        )
        Settings.node_parser = self.text_splitter
        
        # Initialize Milvus vector store
        self.vector_store = MilvusVectorStore(
            uri="./storage/rag_milvus.db",  # Local SQLite-based storage
            dim=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
            overwrite=False,
            collection_name="rag_documents"
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Try to load existing index or create new one
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            print("✅ Loaded existing RAG index")
        except Exception as e:
            print(f"No existing index found, will create new one: {e}")
            self.index = None
        
        print(f"✅ RAG Service initialized")
        print(f"📚 Vector store: Milvus (local SQLite)")
        print(f"🔍 Embedding model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"🧠 LLM: Phi-3 Mini (llama-cpp)")

    @bentoml.api
    def ingest_text(self, request: DocumentIngestRequest) -> DocumentIngestResponse:
        """
        Ingest text document into the RAG system
        
        Args:
            request: DocumentIngestRequest with text and metadata
            
        Returns:
            DocumentIngestResponse with ingestion status
        """
        try:
            # Create document
            doc = Document(
                text=request.text,
                metadata=request.metadata,
                doc_id=request.doc_id
            )
            
            # Create or update index
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [doc], 
                    storage_context=self.storage_context
                )
                print("Created new RAG index")
            else:
                self.index.insert(doc)
                print("Added document to existing index")
            
            # Get number of chunks created
            nodes = self.text_splitter.get_nodes_from_documents([doc])
            chunks_created = len(nodes)
            
            return DocumentIngestResponse(
                status="success",
                message="Document ingested successfully",
                doc_id=doc.doc_id,
                chunks_created=chunks_created
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error ingesting document: {str(e)}")

    @bentoml.api
    def ingest_pdf(self, pdf_file: Path) -> DocumentIngestResponse:
        """
        Ingest PDF file into the RAG system
        
        Args:
            pdf_file: Path to uploaded PDF file
            
        Returns:
            DocumentIngestResponse with ingestion status
        """
        try:
            import pypdf
            
            # Read PDF content
            reader = pypdf.PdfReader(pdf_file)
            texts = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                texts.append(text)
            
            full_text = "\n\n".join(texts)
            
            # Create document with metadata
            metadata = {
                "filename": pdf_file.name,
                "pages": len(reader.pages),
                "source_type": "pdf"
            }
            
            doc = Document(
                text=full_text,
                metadata=metadata
            )
            
            # Create or update index
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [doc], 
                    storage_context=self.storage_context
                )
            else:
                self.index.insert(doc)
            
            # Get number of chunks created
            nodes = self.text_splitter.get_nodes_from_documents([doc])
            chunks_created = len(nodes)
            
            return DocumentIngestResponse(
                status="success",
                message=f"PDF '{pdf_file.name}' ingested successfully",
                doc_id=doc.doc_id,
                chunks_created=chunks_created
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error ingesting PDF: {str(e)}")

    @bentoml.api
    def ingest_txt_file(self, txt_file: Path) -> DocumentIngestResponse:
        """
        Ingest text file into the RAG system
        
        Args:
            txt_file: Path to uploaded text file
            
        Returns:
            DocumentIngestResponse with ingestion status
        """
        try:
            # Read text file
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document with metadata
            metadata = {
                "filename": txt_file.name,
                "source_type": "text_file"
            }
            
            doc = Document(
                text=content,
                metadata=metadata
            )
            
            # Create or update index
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [doc], 
                    storage_context=self.storage_context
                )
            else:
                self.index.insert(doc)
            
            # Get number of chunks created
            nodes = self.text_splitter.get_nodes_from_documents([doc])
            chunks_created = len(nodes)
            
            return DocumentIngestResponse(
                status="success",
                message=f"Text file '{txt_file.name}' ingested successfully",
                doc_id=doc.doc_id,
                chunks_created=chunks_created
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error ingesting text file: {str(e)}")

    @bentoml.api
    def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Query the RAG system
        
        Args:
            request: RAGQueryRequest with query and parameters
            
        Returns:
            RAGQueryResponse with answer and sources
        """
        try:
            if self.index is None:
                return RAGQueryResponse(
                    answer="No documents have been ingested yet. Please ingest some documents first.",
                    sources=[],
                    tokens_generated=0,
                    retrieval_score=0.0
                )
            
            # Create retriever with similarity threshold
            retriever = self.index.as_retriever(
                similarity_top_k=request.top_k,
                # Note: Milvus doesn't directly support similarity threshold in this interface
                # We'll filter results manually if needed
            )
            
            # Retrieve relevant documents
            retrieved_nodes = retriever.retrieve(request.query)
            
            # Filter by similarity threshold if specified
            if request.similarity_threshold > 0:
                retrieved_nodes = [
                    node for node in retrieved_nodes 
                    if node.score and node.score >= request.similarity_threshold
                ]
            
            if not retrieved_nodes:
                return RAGQueryResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    tokens_generated=0,
                    retrieval_score=0.0
                )
            
            # Build context from retrieved nodes
            context_parts = []
            sources = []
            total_score = 0.0
            
            for i, node in enumerate(retrieved_nodes):
                context_parts.append(f"[Source {i+1}]: {node.text}")
                sources.append({
                    "id": i+1,
                    "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "metadata": node.metadata,
                    "score": float(node.score) if node.score else 0.0
                })
                total_score += float(node.score) if node.score else 0.0
            
            context = "\n\n".join(context_parts)
            avg_score = total_score / len(retrieved_nodes) if retrieved_nodes else 0.0
            
            # Generate answer using LLM service
            llm_request = RAGQueryRequest(
                context=context,
                question=request.query,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            llm_response = self.llm_service.rag_query(llm_request)
            
            return RAGQueryResponse(
                answer=llm_response.answer,
                sources=sources,
                tokens_generated=llm_response.tokens_generated,
                retrieval_score=avg_score
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error processing query: {str(e)}")

    @bentoml.api
    def list_documents(self) -> DocumentListResponse:
        """
        List all documents in the RAG system
        
        Returns:
            DocumentListResponse with document information
        """
        try:
            if self.index is None:
                return DocumentListResponse(
                    documents=[],
                    total_count=0,
                    index_size=0
                )
            
            # Get all document IDs from the vector store
            # Note: This is a simplified implementation
            # In production, you might want to maintain a separate document registry
            
            return DocumentListResponse(
                documents=[],  # Simplified - would need vector store introspection
                total_count=0,
                index_size=0
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error listing documents: {str(e)}")

    @bentoml.api
    def clear_index(self) -> Dict[str, Any]:
        """
        Clear all documents from the RAG index
        
        Returns:
            Status message
        """
        try:
            # Recreate the vector store to clear it
            self.vector_store = MilvusVectorStore(
                uri="./storage/rag_milvus.db",
                dim=384,
                overwrite=True,  # This will clear existing data
                collection_name="rag_documents"
            )
            
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = None
            
            return {
                "status": "success",
                "message": "RAG index cleared successfully"
            }
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error clearing index: {str(e)}")

    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            # Check embedding service
            embedding_health = self.embedding_service.health()
            
            # Check LLM service
            llm_health = self.llm_service.health()
            
            # Check index status
            index_status = "initialized" if self.index is not None else "empty"
            
            return {
                "status": "healthy",
                "service": "RAGService",
                "components": {
                    "embedding_service": embedding_health,
                    "llm_service": llm_health,
                    "vector_store": "milvus",
                    "index_status": index_status
                },
                "version": "1.0.0"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @bentoml.api
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        return {
            "service": "RAGService",
            "description": "Complete Retrieval-Augmented Generation service",
            "version": "1.0.0",
            "capabilities": {
                "document_ingestion": ["text", "pdf", "txt_file"],
                "query_processing": True,
                "vector_search": True,
                "language_generation": True
            },
            "components": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_model": "microsoft/Phi-3-mini-4k-instruct-gguf",
                "vector_store": "Milvus (SQLite)",
                "framework": "LlamaIndex + BentoML"
            },
            "endpoints": [
                "/ingest_text",
                "/ingest_pdf", 
                "/ingest_txt_file",
                "/query",
                "/list_documents",
                "/clear_index",
                "/health",
                "/get_service_info"
            ]
        }