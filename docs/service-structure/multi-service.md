# Multi-Service Integration Guide

This guide covers integrating new services with the multi-service architecture for unified deployment and endpoint management.

## Multi-Service Overview

The multi-service architecture allows deploying all services through a single unified endpoint, providing:

- **Unified API**: All services accessible through one endpoint
- **Resource Sharing**: Efficient resource utilization across services
- **Simplified Deployment**: Single deployment artifact
- **Consistent Interface**: Standardized request/response patterns

Current multi-service provides **17 endpoints** across 6 integrated services.

## Multi-Service Structure

**File**: `services/multi_service.py`

```python
import bentoml
from services.example_service import ExampleService, HelloRequest
from services.stable_diffusion_service import StableDiffusionService, ImageGenerationRequest
from services.llava_service import LLaVAService, ImageAnalysisRequest, URLImageAnalysisRequest
from services.whisper_service import WhisperService, TranscriptionURLRequest
from services.upscaler_service import PhotoUpscalerService, UpscaleURLRequest
from services.rag_service import RAGService, RAGIngestTextRequest, RAGQueryRequest
# ADD YOUR SERVICE IMPORTS HERE

@bentoml.service(
    name="multi-service",
    resources={"cpu": "4", "memory": "16Gi", "gpu": "1"}
)
class MultiService:
    """Unified multi-service providing all endpoints in a single deployment"""
    
    def __init__(self):
        """Initialize all service instances"""
        print("Initializing Multi-Service...")
        
        # Initialize all services
        self.example_service = ExampleService()
        self.stable_diffusion_service = StableDiffusionService()
        self.llava_service = LLaVAService()
        self.whisper_service = WhisperService()
        self.upscaler_service = PhotoUpscalerService()
        self.rag_service = RAGService()
        
        # ADD YOUR SERVICE INITIALIZATION HERE:
        # self.your_service = YourService()
        
        print("Multi-Service initialized successfully with all endpoints")
    
    # System endpoints
    @bentoml.api
    def health(self) -> dict:
        """System health check"""
        return {
            "status": "healthy",
            "services": ["example", "stable_diffusion", "llava", "whisper", "upscaler", "rag"],
            "endpoints": 17,
            "timestamp": "current_time"
        }
    
    @bentoml.api
    def info(self) -> dict:
        """Service information"""
        return {
            "name": "multi-service",
            "version": "1.0.0",
            "services": {
                "example": "Hello service with customizable greetings",
                "stable_diffusion": "AI image generation from text prompts",
                "llava": "Vision-language analysis of images",
                "whisper": "Audio transcription service",
                "upscaler": "AI-powered image upscaling and enhancement",
                "rag": "Retrieval-Augmented Generation for document Q&A"
            },
            "total_endpoints": 17
        }
```

## Adding Your Service

### 1. Import Your Service

Add imports at the top of `multi_service.py`:

```python
# ADD AFTER EXISTING IMPORTS:
from services.your_service import (
    YourService, 
    YourServiceRequest, 
    YourServiceResponse,
    # Add other request/response models as needed
)
```

### 2. Initialize Your Service

Add initialization in the `__init__` method:

```python
def __init__(self):
    """Initialize all service instances"""
    print("Initializing Multi-Service...")
    
    # ... existing service initializations ...
    
    # ADD YOUR SERVICE:
    self.your_service = YourService()
    
    print("Multi-Service initialized successfully with all endpoints")
```

### 3. Add Service Endpoints

Add your service endpoints to the MultiService class:

```python
# ADD YOUR SERVICE ENDPOINTS:

@bentoml.api
def your_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
    """Your service primary endpoint"""
    return self.your_service.your_endpoint(request)

@bentoml.api  
def your_other_endpoint(self, request: YourOtherRequest) -> YourOtherResponse:
    """Your service secondary endpoint"""
    return self.your_service.your_other_endpoint(request)

# Add file upload endpoints if needed
@bentoml.api
def your_file_endpoint(self, file: bentoml.File, parameter: str = "default") -> dict:
    """Your service file processing endpoint"""
    return self.your_service.process_file(file, parameter)
```

### 4. Update Service Information

Update the health and info endpoints:

```python
@bentoml.api
def health(self) -> dict:
    """System health check"""
    return {
        "status": "healthy",
        "services": ["example", "stable_diffusion", "llava", "whisper", "upscaler", "rag", "your_service"],
        "endpoints": 20,  # Update count
        "timestamp": "current_time"
    }

@bentoml.api
def info(self) -> dict:
    """Service information"""
    return {
        "name": "multi-service",
        "version": "1.0.0",
        "services": {
            # ... existing services ...
            "your_service": "Description of what your service does"
        },
        "total_endpoints": 20  # Update count
    }
```

## Multi-Service Configuration

### Build Configuration

**File**: `config/bentofiles/multi-service.yaml`

```yaml
service: "services.multi_service:MultiService"
name: "multi-service"

labels:
  owner: "platform-team"
  stage: "production"
  service_type: "composition"
  endpoints: "20"  # Update count

description: "Unified multi-service with all available endpoints"

include:
  - "services/"
  - "utils/"
  - "config/bentoml.yaml"

exclude:
  - "tests/"
  - "docs/"
  - "scripts/"
  - "__pycache__/"

python:
  requirements_txt: |
    # Core BentoML
    bentoml[io]>=1.4.0
    pydantic>=2.5.0
    
    # Image generation (Stable Diffusion)
    diffusers>=0.25.0
    transformers>=4.30.0
    torch>=2.1.0
    torchvision>=0.16.0
    accelerate>=0.25.0
    
    # Vision-language (LLaVA)
    llama-cpp-python>=0.2.27
    
    # Audio processing (Whisper)
    whisper>=1.0.0
    
    # Image upscaling (Photo Upscaler)
    realesrgan>=0.3.0
    gfpgan>=1.3.8
    
    # RAG capabilities
    llama-index-core>=0.10.0
    sentence-transformers>=2.2.0
    pymilvus>=2.3.0
    pypdf>=3.0.0
    
    # ADD YOUR SERVICE DEPENDENCIES:
    # your-service-dependency>=1.0.0
    
    # Common utilities
    pillow>=10.0.0
    requests>=2.28.0
    numpy>=1.21.0

docker:
  distro: "debian"
  python_version: "3.11"
  cuda_version: "11.8"
  system_packages:
    - "git"
    - "wget"
    - "curl"
    - "ffmpeg"  # For audio processing
    # ADD YOUR SYSTEM PACKAGES:
    # - "your-system-package"
  run_as_user: "bentoml"
```

## Service Integration Patterns

### 1. Simple Endpoint Integration

For basic request/response endpoints:

```python
@bentoml.api
def simple_endpoint(self, request: SimpleRequest) -> SimpleResponse:
    """Simple endpoint delegation"""
    return self.your_service.simple_endpoint(request)
```

### 2. File Processing Integration

For services that handle file uploads:

```python
@bentoml.api
def process_file(self, file: bentoml.File, config: dict = {}) -> dict:
    """File processing endpoint"""
    return self.your_service.process_file(file, config)

@bentoml.api  
def process_url(self, request: URLProcessRequest) -> URLProcessResponse:
    """URL-based processing endpoint"""
    return self.your_service.process_url(request)
```

### 3. Async Endpoint Integration

For async services:

```python
@bentoml.api
async def async_endpoint(self, request: AsyncRequest) -> AsyncResponse:
    """Async endpoint delegation"""
    return await self.your_service.async_endpoint(request)
```

### 4. Streaming Integration

For streaming endpoints:

```python
from typing import AsyncGenerator

@bentoml.api
async def streaming_endpoint(self, request: StreamRequest) -> AsyncGenerator[str, None]:
    """Streaming endpoint delegation"""
    async for chunk in self.your_service.streaming_endpoint(request):
        yield chunk
```

### 5. Error Handling Integration

Implement consistent error handling:

```python
@bentoml.api
def robust_endpoint(self, request: RobustRequest) -> RobustResponse:
    """Endpoint with integrated error handling"""
    try:
        return self.your_service.robust_endpoint(request)
    except Exception as e:
        # Log error in multi-service context
        logger.error(f"Error in your_service.robust_endpoint: {str(e)}")
        
        # Return consistent error response
        return RobustResponse(
            result="",
            error=str(e),
            status="error"
        )
```

## Resource Management

### 1. Shared Resource Configuration

Configure resources for the entire multi-service:

```python
@bentoml.service(
    name="multi-service",
    resources={
        "cpu": "8",          # Increase for multiple services
        "memory": "32Gi",    # Adequate memory for all services
        "gpu": "1",          # Shared GPU if services need it
        "gpu_type": "nvidia-tesla-v100"
    },
    timeout=600,             # Longer timeout for multiple services
    workers=1                # Single worker to share resources
)
class MultiService:
    # Implementation...
```

### 2. Service-Specific Resource Optimization

Optimize individual service initialization:

```python
def __init__(self):
    """Initialize services with shared resources"""
    # Initialize lightweight services first
    self.example_service = ExampleService()
    
    # Initialize resource-intensive services
    if self._gpu_available():
        self.stable_diffusion_service = StableDiffusionService()
        self.your_gpu_service = YourGPUService()
    
    # Initialize CPU-bound services
    self.whisper_service = WhisperService()
    self.your_cpu_service = YourCPUService()

def _gpu_available(self) -> bool:
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

## Testing Multi-Service Integration

### Integration Test

**File**: `tests/test_multi_service.py`

```python
import pytest
from services.multi_service import MultiService

class TestMultiServiceIntegration:
    def setup_method(self):
        """Setup multi-service for testing"""
        self.multi_service = MultiService()
    
    def test_all_services_initialized(self):
        """Test all services are properly initialized"""
        assert hasattr(self.multi_service, 'your_service')
        assert self.multi_service.your_service is not None
    
    def test_your_service_endpoint(self):
        """Test your service through multi-service"""
        from services.your_service import YourServiceRequest
        
        request = YourServiceRequest(input_field="test")
        response = self.multi_service.your_endpoint(request)
        
        assert response is not None
        assert response.status == "success"
    
    def test_service_health_includes_your_service(self):
        """Test health endpoint includes your service"""
        health = self.multi_service.health()
        
        assert "your_service" in health["services"]
        assert health["endpoints"] >= 18  # Updated count
    
    def test_service_info_includes_your_service(self):
        """Test info endpoint includes your service"""
        info = self.multi_service.info()
        
        assert "your_service" in info["services"]
        assert info["total_endpoints"] >= 18

@pytest.mark.slow
class TestMultiServiceHTTP:
    @pytest.fixture(scope="class")
    def service_client(self):
        """HTTP client for multi-service"""
        from bentoml.testing import get_client
        
        with get_client(MultiService) as client:
            yield client
    
    def test_your_endpoint_http(self, service_client):
        """Test your endpoint over HTTP"""
        payload = {
            "request": {
                "input_field": "http test"
            }
        }
        
        response = service_client.post("/your_endpoint", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
```

## Deployment and Usage

### 1. Building Multi-Service

```bash
# Build multi-service with all dependencies
BENTOFILE=config/bentofiles/multi-service.yaml ./scripts/run_bentoml.sh build services/multi_service.py
```

### 2. Running Multi-Service

```bash
# Start multi-service (includes all endpoints)
./scripts/start.sh

# Or run directly
./scripts/run_bentoml.sh serve services.multi_service:MultiService
```

### 3. Testing Endpoints

```bash
# Test your service endpoints through multi-service
./scripts/endpoint.sh your_endpoint '{"input_field": "test data"}'

# Test health check (should show your service)
./scripts/endpoint.sh health '{}'

# Test service info (should show updated endpoint count)
./scripts/endpoint.sh info '{}'
```

## Benefits and Considerations

### Benefits

1. **Single Deployment**: Deploy all services as one unit
2. **Resource Efficiency**: Shared resources across services
3. **Unified Interface**: Consistent API across all endpoints
4. **Simplified Operations**: One service to monitor and maintain

### Considerations

1. **Startup Time**: All services must initialize
2. **Memory Usage**: All services loaded in memory
3. **Failure Impact**: One service failure affects all
4. **Scaling Complexity**: Cannot scale individual services

### Best Practices

1. **Graceful Degradation**: Handle individual service failures
2. **Resource Monitoring**: Monitor resource usage across services
3. **Error Isolation**: Prevent one service from affecting others
4. **Health Checks**: Implement comprehensive health monitoring
5. **Documentation**: Keep endpoint documentation updated

This multi-service integration approach provides a powerful unified deployment while maintaining the flexibility of individual service development.