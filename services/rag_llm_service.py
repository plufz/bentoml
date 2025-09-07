"""
RAG LLM Service for BentoML
Provides language model functionality using llama-cpp-python for RAG applications.
"""

from __future__ import annotations
import typing as t
import bentoml
from pydantic import BaseModel
from pathlib import Path

# Model configuration
LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLM_MODEL_FILE = "Phi-3-mini-4k-instruct-q4.gguf"  # Quantized model for efficiency
LLM_MAX_TOKENS = 4096
LLM_CONTEXT_WINDOW = 3072  # Leave room for response


class LLMRequest(BaseModel):
    """Request model for LLM generation"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: t.List[str] = []


class LLMResponse(BaseModel):
    """Response model for LLM generation"""
    text: str
    tokens_generated: int
    model: str
    finish_reason: str


class RAGQueryRequest(BaseModel):
    """Request model for RAG-optimized generation"""
    context: str
    question: str
    max_tokens: int = 512
    temperature: float = 0.1


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    tokens_generated: int
    model: str
    context_used: bool


@bentoml.service(
    traffic={"timeout": 600},
    resources={"memory": "8Gi", "cpu": "4"}
)
class RAGLLMService:
    """
    RAG LLM Service using llama-cpp-python
    Provides efficient language model inference optimized for RAG applications.
    """
    
    def __init__(self) -> None:
        from llama_cpp import Llama
        import os
        
        # Set model path - try several common locations
        model_paths = [
            f"/Volumes/Second/models/{LLM_MODEL_FILE}",
            f"./models/{LLM_MODEL_FILE}",
            f"~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/{LLM_MODEL_FILE}",
        ]
        
        model_path = None
        for path in model_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                model_path = str(expanded_path)
                break
                
        if not model_path:
            # Try to download the model if not found
            print(f"Model not found locally. You may need to download {LLM_MODEL_FILE}")
            print("You can download it from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
            raise FileNotFoundError(f"Model file {LLM_MODEL_FILE} not found in any of: {model_paths}")
        
        print(f"Loading LLM model from: {model_path}")
        
        # Initialize llama-cpp with optimized settings
        self.llm = Llama(
            model_path=model_path,
            n_ctx=LLM_MAX_TOKENS,  # Context window
            n_threads=4,  # Number of CPU threads
            n_gpu_layers=0,  # Use CPU only for better compatibility
            verbose=False,
            seed=-1,  # Random seed
        )
        
        print(f"✅ RAG LLM Service initialized")
        print(f"📝 Model: {LLM_MODEL_ID}")
        print(f"🧠 Context window: {LLM_MAX_TOKENS}")

    def _create_rag_prompt(self, context: str, question: str) -> str:
        """Create a RAG-optimized prompt template"""
        return f"""<|system|>
You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

<|user|>
{question}

<|assistant|>
"""

    @bentoml.api
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using the language model
        
        Args:
            request: LLMRequest with prompt and generation parameters
            
        Returns:
            LLMResponse with generated text and metadata
        """
        try:
            # Generate response
            output = self.llm(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                stop=request.stop or ["<|user|>", "<|system|>"],
                echo=False
            )
            
            generated_text = output["choices"][0]["text"].strip()
            
            return LLMResponse(
                text=generated_text,
                tokens_generated=output["usage"]["completion_tokens"],
                model=LLM_MODEL_ID,
                finish_reason=output["choices"][0]["finish_reason"]
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error generating text: {str(e)}")

    @bentoml.api
    def rag_query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Generate a RAG-optimized response using context and question
        
        Args:
            request: RAGQueryRequest with context and question
            
        Returns:
            RAGQueryResponse with the answer
        """
        try:
            # Create RAG prompt
            prompt = self._create_rag_prompt(request.context, request.question)
            
            # Generate response
            output = self.llm(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|user|>", "<|system|>", "\n\nContext:", "\n\nQuestion:"],
                echo=False
            )
            
            answer = output["choices"][0]["text"].strip()
            
            return RAGQueryResponse(
                answer=answer,
                tokens_generated=output["usage"]["completion_tokens"],
                model=LLM_MODEL_ID,
                context_used=bool(request.context.strip())
            )
            
        except Exception as e:
            raise bentoml.BentoMLException(f"Error in RAG query: {str(e)}")

    @bentoml.api
    def health(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "RAGLLMService",
            "model": LLM_MODEL_ID,
            "max_tokens": LLM_MAX_TOKENS,
            "context_window": LLM_CONTEXT_WINDOW
        }

    @bentoml.api
    def get_model_info(self) -> dict:
        """Get information about the language model"""
        return {
            "model_id": LLM_MODEL_ID,
            "model_file": LLM_MODEL_FILE,
            "max_tokens": LLM_MAX_TOKENS,
            "context_window": LLM_CONTEXT_WINDOW,
            "backend": "llama-cpp-python",
            "quantization": "q4",
            "description": "Phi-3 Mini model optimized for instruction following and RAG applications"
        }