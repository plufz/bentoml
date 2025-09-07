# Service Creation Guide

This guide provides a step-by-step walkthrough for creating new BentoML services in this project.

## Step 1: Service Implementation

Create your main service file in the `services/` directory following the established patterns.

### Basic Service Template

**File**: `services/your_service.py`

```python
import bentoml
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Request/Response models
class YourServiceRequest(BaseModel):
    """Request model for your service"""
    input_field: str
    optional_field: Optional[str] = "default"
    parameters: Optional[Dict[str, Any]] = {}

class YourServiceResponse(BaseModel):
    """Response model for your service"""
    result: str
    metadata: Dict[str, Any] = {}
    status: str = "success"

@bentoml.service(
    name="your-service",
    resources={"cpu": "2", "memory": "4Gi"}
)
class YourService:
    """Your service description"""
    
    def __init__(self):
        """Initialize service dependencies"""
        logger.info("Initializing YourService...")
        # Initialize models, load dependencies, etc.
        self._initialized = True
        logger.info("YourService initialized successfully")
    
    @bentoml.api
    def your_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        """Process request and return response"""
        try:
            logger.info(f"Processing request: {request.input_field}")
            
            # Service logic here
            result = self._process_data(request.input_field, request.optional_field)
            
            return YourServiceResponse(
                result=result,
                metadata={
                    "processed_at": "timestamp",
                    "parameters": request.parameters
                },
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return YourServiceResponse(
                result="",
                metadata={"error": str(e)},
                status="error"
            )
    
    def _process_data(self, input_data: str, optional_param: str) -> str:
        """Private method for data processing"""
        # Implementation logic
        return f"Processed: {input_data} with {optional_param}"
```

### Advanced Service Patterns

#### Model-Heavy Service (GPU/CPU intensive)

```python
@bentoml.service(
    name="model-service",
    resources={"cpu": "4", "memory": "8Gi", "gpu": "1", "gpu_type": "nvidia-tesla-v100"}
)
class ModelService:
    def __init__(self):
        # Load heavy models in init
        self.model = self._load_model()
    
    def _load_model(self):
        """Load and initialize model"""
        # Model loading logic
        pass
```

#### Service with File Processing

```python
from bentoml import File

@bentoml.service()
class FileProcessingService:
    @bentoml.api
    def process_file(self, file: File) -> dict:
        """Process uploaded file"""
        try:
            # Read file content
            content = file.read()
            
            # Process content
            result = self._process_content(content)
            
            return {"result": result, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}
```

## Step 2: Utility Functions (Optional)

If your service needs shared utilities, create helper modules in `utils/`.

**File**: `utils/your_service_utils.py`

```python
"""Utility functions for YourService"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def preprocess_data(data: str) -> str:
    """Preprocess input data"""
    logger.debug(f"Preprocessing data: {data[:50]}...")
    return data.strip().lower()

def postprocess_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Postprocess service result"""
    return {
        **result, 
        "processed": True,
        "timestamp": "current_timestamp"
    }

def validate_input(input_data: str) -> bool:
    """Validate input data"""
    if not input_data or len(input_data.strip()) == 0:
        return False
    return True

class YourServiceHelper:
    """Helper class for complex service operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def complex_operation(self, data: str) -> str:
        """Perform complex data transformation"""
        # Implementation
        return data.upper()
```

## Step 3: Service Variants and Patterns

### Async Service Pattern

```python
import asyncio
from typing import AsyncGenerator

@bentoml.service()
class AsyncService:
    @bentoml.api
    async def async_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        """Async endpoint for concurrent processing"""
        result = await self._async_process(request.input_field)
        return YourServiceResponse(result=result)
    
    async def _async_process(self, data: str) -> str:
        """Async processing logic"""
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Async processed: {data}"
    
    @bentoml.api
    async def stream_endpoint(self, request: YourServiceRequest) -> AsyncGenerator[str, None]:
        """Streaming endpoint"""
        for i in range(5):
            await asyncio.sleep(0.1)
            yield f"Stream chunk {i}: {request.input_field}"
```

### Service with Multiple Endpoints

```python
@bentoml.service()
class MultiEndpointService:
    @bentoml.api
    def endpoint_one(self, request: RequestModelA) -> ResponseModelA:
        """First endpoint"""
        return ResponseModelA(result="endpoint_one")
    
    @bentoml.api
    def endpoint_two(self, request: RequestModelB) -> ResponseModelB:
        """Second endpoint"""
        return ResponseModelB(result="endpoint_two")
    
    @bentoml.api
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {"status": "healthy", "service": "multi-endpoint"}
```

## Step 4: Error Handling Patterns

### Comprehensive Error Handling

```python
from enum import Enum

class ErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"

class ServiceError(Exception):
    def __init__(self, error_type: ErrorType, message: str, details: Optional[Dict] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

@bentoml.service()
class RobustService:
    @bentoml.api
    def robust_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        """Endpoint with comprehensive error handling"""
        try:
            # Validation
            if not self._validate_request(request):
                raise ServiceError(
                    ErrorType.VALIDATION_ERROR,
                    "Invalid request data",
                    {"field": "input_field"}
                )
            
            # Processing
            result = self._safe_process(request)
            
            return YourServiceResponse(
                result=result,
                status="success"
            )
            
        except ServiceError as e:
            logger.error(f"Service error: {e.message}")
            return YourServiceResponse(
                result="",
                metadata={
                    "error_type": e.error_type.value,
                    "error_message": e.message,
                    "error_details": e.details
                },
                status="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return YourServiceResponse(
                result="",
                metadata={"error": "Internal server error"},
                status="error"
            )
    
    def _validate_request(self, request: YourServiceRequest) -> bool:
        """Validate request data"""
        return bool(request.input_field and request.input_field.strip())
    
    def _safe_process(self, request: YourServiceRequest) -> str:
        """Process with error handling"""
        try:
            return self._core_logic(request.input_field)
        except Exception as e:
            raise ServiceError(
                ErrorType.PROCESSING_ERROR,
                f"Processing failed: {str(e)}"
            )
```

## Step 5: Service Documentation

Add docstrings and type hints throughout your service:

```python
from typing import Union, List, Optional
import bentoml

@bentoml.service(
    name="documented-service",
    resources={"cpu": "2", "memory": "4Gi"}
)
class DocumentedService:
    """
    A well-documented BentoML service.
    
    This service demonstrates best practices for:
    - Type annotations
    - Comprehensive docstrings  
    - Error handling
    - Logging
    
    Attributes:
        _initialized (bool): Service initialization status
        _model: The loaded model instance
    """
    
    def __init__(self) -> None:
        """Initialize the service with required dependencies."""
        self._initialized = False
        self._model = None
        self._setup_service()
    
    def _setup_service(self) -> None:
        """Setup service dependencies and configuration."""
        try:
            # Initialize components
            self._model = self._load_model()
            self._initialized = True
            logger.info("Service initialized successfully")
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise
    
    @bentoml.api
    def process(self, request: YourServiceRequest) -> YourServiceResponse:
        """
        Process the input request and return results.
        
        Args:
            request: The input request containing data to process
            
        Returns:
            YourServiceResponse: Processing results and metadata
            
        Raises:
            ServiceError: When processing fails
            
        Example:
            >>> request = YourServiceRequest(input_field="test data")
            >>> response = service.process(request)
            >>> assert response.status == "success"
        """
        if not self._initialized:
            raise ServiceError(
                ErrorType.SYSTEM_ERROR,
                "Service not properly initialized"
            )
        
        # Implementation...
        return YourServiceResponse(result="processed")
```

## Next Steps

After creating your service implementation:

1. **Configuration**: Set up Bento build configuration (see [Configuration Guide](configuration.md))
2. **Testing**: Create comprehensive tests (see [Testing Guide](testing.md))
3. **Integration**: Update scripts and multi-service (see [Scripts Integration Guide](scripts-integration.md))
4. **Documentation**: Update project documentation with your new service

## Example Services Reference

Study these existing services for patterns:

- **`example_service.py`**: Basic service structure
- **`stable_diffusion_service.py`**: GPU-intensive model service
- **`llava_service.py`**: Image processing with multiple endpoints
- **`whisper_service.py`**: Audio file processing
- **`upscaler_service.py`**: Image enhancement with file uploads
- **`rag_service.py`**: Vector database integration with complex workflows