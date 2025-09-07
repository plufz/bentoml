"""
Multi-Service Composition for BentoML
Combines all individual services into a single unified service with multiple endpoints.
"""

import bentoml
from pydantic import BaseModel
from typing import Dict, Any, List
from pathlib import Path

# Import all individual services
from services.example_service import HelloService
from services.stable_diffusion_service import StableDiffusionService
from services.llava_service import LLaVAService
from services.whisper_service import WhisperService

# Import request models from individual services
from services.example_service import HelloRequest
from services.stable_diffusion_service import ImageGenerationRequest
from services.llava_service import VisionLanguageRequest, VisionLanguageUrlRequest
from services.whisper_service import TranscribeUrlRequest


class ServiceInfo(BaseModel):
    """Information about available services and endpoints"""
    name: str
    description: str
    endpoints: List[str]
    status: str


class MultiServiceResponse(BaseModel):
    """Response model for multi-service information"""
    available_services: List[ServiceInfo]
    total_endpoints: int
    version: str


@bentoml.service(
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
        "memory": "8Gi",
        "cpu": "4"
    },
    traffic={
        "timeout": 600,
        "concurrency": 8
    }
)
class MultiService:
    """
    Multi-Service BentoML Application
    Provides unified access to all AI services through a single endpoint.
    
    Available Services:
    - Hello Service: Simple greeting and health check
    - Stable Diffusion: Text-to-image generation
    - LLaVA: Vision-language understanding and analysis
    - Whisper: Audio transcription from files and URLs
    """
    
    def __init__(self):
        """Initialize all individual services"""
        try:
            self.hello_service = HelloService()
            print("✅ Hello Service initialized")
        except Exception as e:
            print(f"⚠️  Hello Service failed to initialize: {e}")
            self.hello_service = None
            
        try:
            self.stable_diffusion_service = StableDiffusionService()
            print("✅ Stable Diffusion Service initialized")
        except Exception as e:
            print(f"⚠️  Stable Diffusion Service failed to initialize: {e}")
            self.stable_diffusion_service = None
            
        try:
            self.llava_service = LLaVAService()
            print("✅ LLaVA Service initialized")
        except Exception as e:
            print(f"⚠️  LLaVA Service failed to initialize: {e}")
            self.llava_service = None
            
        try:
            self.whisper_service = WhisperService()
            print("✅ Whisper Service initialized")
        except Exception as e:
            print(f"⚠️  Whisper Service failed to initialize: {e}")
            self.whisper_service = None
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Overall health check for all services"""
        services_status = {}
        overall_healthy = True
        
        # Check each service
        if self.hello_service:
            try:
                services_status["hello"] = self.hello_service.health()
            except Exception as e:
                services_status["hello"] = {"status": "error", "error": str(e)}
                overall_healthy = False
        else:
            services_status["hello"] = {"status": "unavailable"}
            overall_healthy = False
            
        if self.stable_diffusion_service:
            try:
                services_status["stable_diffusion"] = self.stable_diffusion_service.health()
            except Exception as e:
                services_status["stable_diffusion"] = {"status": "error", "error": str(e)}
                overall_healthy = False
        else:
            services_status["stable_diffusion"] = {"status": "unavailable"}
            overall_healthy = False
            
        if self.llava_service:
            try:
                services_status["llava"] = self.llava_service.health()
            except Exception as e:
                services_status["llava"] = {"status": "error", "error": str(e)}
                overall_healthy = False
        else:
            services_status["llava"] = {"status": "unavailable"}
            overall_healthy = False
            
        if self.whisper_service:
            try:
                # Whisper doesn't have a health method, so we'll create a simple check
                services_status["whisper"] = {
                    "status": "healthy",
                    "service": "WhisperService",
                    "model": "mlx-community/whisper-large-v3-turbo"
                }
            except Exception as e:
                services_status["whisper"] = {"status": "error", "error": str(e)}
                overall_healthy = False
        else:
            services_status["whisper"] = {"status": "unavailable"}
            overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "service": "MultiService",
            "individual_services": services_status,
            "version": "1.0.0",
            "description": "Multi-Service BentoML Application"
        }
    
    @bentoml.api
    def info(self) -> Dict[str, Any]:
        """Get information about all available services and endpoints"""
        services = []
        
        if self.hello_service:
            services.append({
                "name": "Hello Service",
                "description": "Simple greeting and health check service",
                "endpoints": ["/hello"],
                "status": "available"
            })
        
        if self.stable_diffusion_service:
            services.append({
                "name": "Stable Diffusion",
                "description": "Text-to-image generation using Stable Diffusion",
                "endpoints": ["/generate_image"],
                "status": "available"
            })
        
        if self.llava_service:
            services.append({
                "name": "LLaVA",
                "description": "Vision-language understanding and image analysis",
                "endpoints": ["/analyze_image", "/analyze_image_url", "/example_schemas"],
                "status": "available"
            })
        
        if self.whisper_service:
            services.append({
                "name": "Whisper",
                "description": "Audio transcription from files and URLs",
                "endpoints": ["/transcribe_file", "/transcribe_url"],
                "status": "available"
            })
        
        total_endpoints = sum(len(service["endpoints"]) for service in services) + 2  # +2 for health and info
        
        return {
            "available_services": services,
            "total_endpoints": total_endpoints,
            "version": "1.0.0"
        }
    
    # Hello Service Endpoints
    @bentoml.api
    def hello(self, request: HelloRequest) -> Dict[str, Any]:
        """Hello service endpoint"""
        if not self.hello_service:
            return {"error": "Hello service is not available", "status": "error"}
        return self.hello_service.hello(request)
    
    # Stable Diffusion Endpoints
    @bentoml.api
    def generate_image(self, request: ImageGenerationRequest) -> Dict[str, Any]:
        """Generate image using Stable Diffusion"""
        if not self.stable_diffusion_service:
            return {"error": "Stable Diffusion service is not available", "status": "error"}
        return self.stable_diffusion_service.generate_image(request)
    
    # LLaVA Endpoints
    @bentoml.api
    def analyze_image(self, request: VisionLanguageRequest) -> Dict[str, Any]:
        """Analyze uploaded image using LLaVA"""
        if not self.llava_service:
            return {"error": "LLaVA service is not available", "status": "error"}
        return self.llava_service.analyze_image(request)
    
    @bentoml.api
    def analyze_image_url(self, request: VisionLanguageUrlRequest) -> Dict[str, Any]:
        """Analyze image from URL using LLaVA"""
        if not self.llava_service:
            return {"error": "LLaVA service is not available", "status": "error"}
        return self.llava_service.analyze_image_url(request)
    
    @bentoml.api
    def example_schemas(self) -> Dict[str, Any]:
        """Get example JSON schemas for structured image analysis"""
        if not self.llava_service:
            return {"error": "LLaVA service is not available", "status": "error"}
        return self.llava_service.get_example_schemas()
    
    # Whisper Endpoints
    @bentoml.api
    def transcribe_file(self, audio_file: Path) -> Dict[str, Any]:
        """Transcribe uploaded audio file using Whisper"""
        if not self.whisper_service:
            return {"error": "Whisper service is not available", "status": "error"}
        return self.whisper_service.transcribe_file(audio_file)
    
    @bentoml.api
    def transcribe_url(self, request: TranscribeUrlRequest) -> Dict[str, Any]:
        """Transcribe audio from URL using Whisper"""
        if not self.whisper_service:
            return {"error": "Whisper service is not available", "status": "error"}
        return self.whisper_service.transcribe_url(request)