"""
RAG Embedding Service for BentoML
Provides text embedding functionality using sentence-transformers for RAG applications.
"""

from __future__ import annotations
import typing as t
import numpy as np
import bentoml
from pydantic import BaseModel
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

# Model configuration
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingRequest(BaseModel):
    """Request model for text embedding"""
    texts: t.List[str]
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    """Response model for text embedding"""
    embeddings: t.List[t.List[float]]
    dimension: int
    model: str


@bentoml.service(
    traffic={"timeout": 300},
    resources={"memory": "2Gi", "cpu": "2"}
)
class RAGEmbeddingService:
    """
    RAG Embedding Service using sentence-transformers
    Provides high-quality text embeddings optimized for semantic search and RAG applications.
    """
    
    def __init__(self) -> None:
        import os
        # Disable GPU for embedding service to save GPU memory for LLM
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        import torch
        from sentence_transformers import SentenceTransformer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {EMBEDDING_MODEL_ID}")
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(EMBEDDING_MODEL_ID)
        self.model.eval()
        
        print(f"✅ RAG Embedding Service initialized on {self.device}")
        print(f"📐 Embedding dimension: {EMBEDDING_DIMENSION}")

    @bentoml.api(batchable=True)
    def embed_texts(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts
        
        Args:
            request: EmbeddingRequest containing texts and normalization option
            
        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                request.texts,
                normalize_embeddings=request.normalize,
                show_progress_bar=False
            )
            
            # Convert to list format for JSON serialization
            embeddings_list = embeddings.tolist()
            
            return EmbeddingResponse(
                embeddings=embeddings_list,
                dimension=EMBEDDING_DIMENSION,
                model=EMBEDDING_MODEL_ID
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error generating embeddings: {str(e)}")

    @bentoml.api
    def embed_single(self, text: str, normalize: bool = True) -> t.List[float]:
        """
        Generate embedding for a single text (convenience method)
        
        Args:
            text: Input text to embed
            normalize: Whether to normalize the embedding vector
            
        Returns:
            List of float values representing the embedding
        """
        try:
            embedding = self.model.encode([text], normalize_embeddings=normalize)[0]
            return embedding.tolist()
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error generating single embedding: {str(e)}")

    @bentoml.api
    def health(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "RAGEmbeddingService",
            "model": EMBEDDING_MODEL_ID,
            "dimension": EMBEDDING_DIMENSION,
            "device": self.device
        }

    @bentoml.api  
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            "model_id": EMBEDDING_MODEL_ID,
            "dimension": EMBEDDING_DIMENSION,
            "max_seq_length": self.model.max_seq_length,
            "device": self.device,
            "description": "Sentence transformer model optimized for semantic similarity"
        }


class BentoMLEmbeddings(BaseEmbedding):
    """
    LlamaIndex-compatible embedding wrapper for BentoML RAG Embedding Service
    """
    _service: bentoml.Service = PrivateAttr()

    def __init__(self, service: bentoml.Service, **kwargs) -> None:
        super().__init__(**kwargs)
        self._service = service
        
    def _get_query_embedding(self, query: str) -> t.List[float]:
        """Get embedding for a query string"""
        return self._service.embed_single(text=query, normalize=True)
    
    def _get_text_embedding(self, text: str) -> t.List[float]:
        """Get embedding for a text string"""
        return self._service.embed_single(text=text, normalize=True)
    
    def _get_text_embeddings(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Get embeddings for multiple texts"""
        request = EmbeddingRequest(texts=texts, normalize=True)
        response = self._service.embed_texts(request=request)
        return response.embeddings
        
    async def _aget_query_embedding(self, query: str) -> t.List[float]:
        """Async version of query embedding"""
        return await self._service.embed_single(text=query, normalize=True)
        
    async def _aget_text_embedding(self, text: str) -> t.List[float]:
        """Async version of text embedding"""
        return await self._service.embed_single(text=text, normalize=True)
        
    async def _aget_text_embeddings(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Async version of text embeddings"""
        request = EmbeddingRequest(texts=texts, normalize=True)
        response = await self._service.embed_texts(request=request)
        return response.embeddings