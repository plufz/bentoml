# Service Development Best Practices

This guide covers best practices for developing robust, maintainable, and scalable BentoML services in this project.

## Service Design Principles

### 1. Single Responsibility Principle

Each service should have a clear, focused purpose:

```python
# ✅ Good - focused service
@bentoml.service()
class ImageClassificationService:
    """Service focused on image classification only"""
    
    @bentoml.api
    def classify_image(self, request: ClassificationRequest) -> ClassificationResponse:
        return self._classify(request.image)

# ❌ Avoid - service doing too much
@bentoml.service()
class AIService:
    """Service trying to do everything"""
    
    @bentoml.api
    def classify_image(self, request): pass
    
    @bentoml.api
    def generate_text(self, request): pass
    
    @bentoml.api
    def translate_text(self, request): pass
```

### 2. Clear API Contracts

Use Pydantic models for strong typing and validation:

```python
# ✅ Good - clear, typed interfaces
class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis"""
    image: str = Field(..., description="Base64 encoded image")
    analysis_type: Literal["objects", "scene", "text"] = Field(
        default="objects",
        description="Type of analysis to perform"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )

class ImageAnalysisResponse(BaseModel):
    """Response model for image analysis"""
    results: List[AnalysisResult]
    confidence: float
    processing_time_ms: int
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None
```

### 3. Error Handling Strategy

Implement comprehensive error handling:

```python
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ServiceErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    MODEL_ERROR = "model_error"
    RESOURCE_ERROR = "resource_error"
    EXTERNAL_API_ERROR = "external_api_error"

class ServiceError(Exception):
    def __init__(self, error_type: ServiceErrorType, message: str, details: Optional[Dict] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

@bentoml.service()
class RobustService:
    @bentoml.api
    def process_request(self, request: ProcessRequest) -> ProcessResponse:
        try:
            # Input validation
            self._validate_input(request)
            
            # Core processing
            result = self._process_data(request)
            
            return ProcessResponse(
                result=result,
                status="success",
                metadata={"processed_at": datetime.utcnow().isoformat()}
            )
            
        except ServiceError as e:
            logger.error(f"Service error: {e.error_type.value} - {e.message}", extra=e.details)
            return ProcessResponse(
                result="",
                status="error",
                error_type=e.error_type.value,
                error_message=e.message,
                metadata=e.details
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return ProcessResponse(
                result="",
                status="error",
                error_type=ServiceErrorType.RESOURCE_ERROR.value,
                error_message="Internal server error"
            )
    
    def _validate_input(self, request: ProcessRequest) -> None:
        """Validate request input"""
        if not request.input_data or len(request.input_data.strip()) == 0:
            raise ServiceError(
                ServiceErrorType.VALIDATION_ERROR,
                "Input data cannot be empty"
            )
```

## Configuration Best Practices

### 1. Environment-Based Configuration

```python
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings

class ServiceConfig(BaseSettings):
    """Service configuration with environment variable support"""
    
    # Model configuration
    model_name: str = "default-model"
    model_path: Optional[Path] = None
    model_revision: str = "main"
    
    # Processing configuration
    max_batch_size: int = 10
    timeout_seconds: int = 30
    
    # Resource configuration
    device: str = "auto"  # auto, cpu, cuda
    precision: str = "float16"  # float32, float16
    
    # Cache configuration
    cache_dir: Path = Path.home() / ".cache" / "service"
    enable_cache: bool = True
    
    class Config:
        env_prefix = "SERVICE_"  # Environment variables like SERVICE_MODEL_NAME
        case_sensitive = False

# Usage in service
@bentoml.service()
class ConfiguredService:
    def __init__(self):
        self.config = ServiceConfig()
        self.model = self._load_model()
    
    def _load_model(self):
        """Load model based on configuration"""
        model_path = self.config.model_path or self._download_model()
        # Load model logic
        return model
```

### 2. Resource Management

```python
@bentoml.service(
    resources={
        "cpu": "2000m",      # 2 CPU cores
        "memory": "8Gi",     # 8GB memory
        "gpu": "1",          # 1 GPU if available
    },
    timeout=300,             # 5 minute timeout
)
class ResourceManagedService:
    def __init__(self):
        # Initialize with resource awareness
        self.device = self._detect_device()
        self.model = self._load_model_for_device()
    
    def _detect_device(self) -> str:
        """Detect optimal device based on available resources"""
        try:
            import torch
            if torch.cuda.is_available() and self._gpu_memory_sufficient():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _gpu_memory_sufficient(self) -> bool:
        """Check if GPU has sufficient memory"""
        import torch
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return memory_gb >= 6.0  # Require at least 6GB
        return False
```

## Code Organization

### 1. Service Structure

```python
# services/your_service.py
import bentoml
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime

# Import utilities
from utils.your_service_utils import preprocess_data, postprocess_result
from utils.common_utils import timing_decorator

logger = logging.getLogger(__name__)

# Models at the top
class YourServiceRequest(BaseModel):
    """Request model with comprehensive validation"""
    # Implementation

class YourServiceResponse(BaseModel):
    """Response model with metadata"""
    # Implementation

@bentoml.service(
    name="your-service",
    resources={"cpu": "2", "memory": "4Gi"}
)
class YourService:
    """
    Your service with clear documentation.
    
    This service provides [functionality description].
    Supports [input types] and returns [output types].
    """
    
    def __init__(self):
        """Initialize service with proper setup"""
        logger.info("Initializing YourService...")
        self._setup_service()
        logger.info("YourService initialized successfully")
    
    def _setup_service(self) -> None:
        """Private setup method"""
        # Setup logic
        pass
    
    @bentoml.api
    @timing_decorator
    def your_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        """
        Main API endpoint with comprehensive documentation.
        
        Args:
            request: Input request with validated data
            
        Returns:
            Response with results and metadata
            
        Raises:
            ServiceError: When processing fails
        """
        # Implementation
        pass
    
    def _private_method(self, data: Any) -> Any:
        """Private methods for internal logic"""
        pass
```

### 2. Utility Organization

```python
# utils/your_service_utils.py
"""Utility functions for YourService"""
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def preprocess_data(data: str, config: Optional[Dict] = None) -> str:
    """
    Preprocess input data for service.
    
    Args:
        data: Raw input data
        config: Optional preprocessing configuration
        
    Returns:
        Preprocessed data ready for model
    """
    config = config or {}
    
    # Preprocessing logic
    processed = data.strip().lower()
    
    logger.debug(f"Preprocessed data: {len(processed)} characters")
    return processed

class YourServiceHelper:
    """Helper class for complex operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
        self._setup()
    
    def _setup(self) -> None:
        """Setup helper with configuration"""
        # Setup logic
        self._initialized = True
    
    def complex_operation(self, input_data: Any) -> Any:
        """Perform complex data transformation"""
        if not self._initialized:
            raise RuntimeError("Helper not properly initialized")
        
        # Complex operation logic
        return input_data
```

## Performance Optimization

### 1. Caching Strategy

```python
from functools import lru_cache
import pickle
from pathlib import Path

@bentoml.service()
class OptimizedService:
    def __init__(self):
        self.cache_dir = Path(".cache/service")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @lru_cache(maxsize=128)
    def _cached_preprocessing(self, data_hash: str) -> str:
        """Cache expensive preprocessing operations"""
        # Expensive preprocessing
        return processed_data
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if available"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
```

### 2. Batch Processing

```python
from typing import List
import asyncio

@bentoml.service()
class BatchProcessingService:
    def __init__(self):
        self.max_batch_size = 10
        self.batch_timeout = 0.1  # 100ms
    
    @bentoml.api
    async def process_batch(self, requests: List[ProcessRequest]) -> List[ProcessResponse]:
        """Process multiple requests efficiently"""
        if len(requests) > self.max_batch_size:
            # Split into smaller batches
            batches = [
                requests[i:i + self.max_batch_size] 
                for i in range(0, len(requests), self.max_batch_size)
            ]
            
            # Process batches concurrently
            results = []
            for batch in batches:
                batch_results = await self._process_single_batch(batch)
                results.extend(batch_results)
            
            return results
        else:
            return await self._process_single_batch(requests)
    
    async def _process_single_batch(self, batch: List[ProcessRequest]) -> List[ProcessResponse]:
        """Process a single batch of requests"""
        # Batch processing logic
        tasks = [self._process_single(req) for req in batch]
        return await asyncio.gather(*tasks)
```

## Security Best Practices

### 1. Input Validation

```python
import re
from typing import Any
from pydantic import validator

class SecureRequest(BaseModel):
    """Request model with security validations"""
    
    user_input: str = Field(..., max_length=1000)
    file_path: Optional[str] = None
    
    @validator('user_input')
    def validate_user_input(cls, v):
        """Validate and sanitize user input"""
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>"\']', '', v)
        
        # Check for common injection patterns
        dangerous_patterns = ['script', 'javascript:', 'data:', '../']
        for pattern in dangerous_patterns:
            if pattern in sanitized.lower():
                raise ValueError(f"Potentially dangerous input detected")
        
        return sanitized
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path to prevent directory traversal"""
        if v is None:
            return v
        
        # Resolve and validate path
        from pathlib import Path
        try:
            path = Path(v).resolve()
            # Ensure path is within allowed directory
            allowed_dir = Path("/allowed/upload/dir").resolve()
            path.relative_to(allowed_dir)  # Raises ValueError if outside
            return str(path)
        except (ValueError, OSError):
            raise ValueError("Invalid or unauthorized file path")
```

### 2. Secrets Management

```python
import os
from typing import Optional

class SecureService:
    def __init__(self):
        # Never hardcode secrets
        self.api_key = self._get_secret("API_KEY")
        self.db_password = self._get_secret("DB_PASSWORD")
    
    def _get_secret(self, key: str) -> str:
        """Securely retrieve secrets from environment"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required secret {key} not found in environment")
        return value
    
    def _log_safely(self, message: str, sensitive_data: Optional[str] = None) -> None:
        """Log messages without exposing sensitive data"""
        if sensitive_data:
            # Hash or truncate sensitive data for logging
            safe_data = f"{sensitive_data[:4]}***{sensitive_data[-4:]}" if len(sensitive_data) > 8 else "***"
            message = message.replace(sensitive_data, safe_data)
        
        logger.info(message)
```

## Testing Best Practices

### 1. Test Organization

```python
# tests/test_your_service.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.your_service import YourService, YourServiceRequest

class TestYourServiceUnit:
    """Unit tests - fast tests with mocked dependencies"""
    
    @pytest.fixture
    def service(self):
        """Service fixture with mocked dependencies"""
        with patch('services.your_service.heavy_dependency'):
            return YourService()
    
    @pytest.fixture
    def sample_request(self):
        """Sample request fixture"""
        return YourServiceRequest(
            input_field="test data",
            optional_field="test value"
        )
    
    def test_happy_path(self, service, sample_request):
        """Test successful processing"""
        response = service.your_endpoint(sample_request)
        
        assert response.status == "success"
        assert "test data" in response.result
    
    @pytest.mark.parametrize("input_data,expected", [
        ("hello", "HELLO"),
        ("world", "WORLD"),
        ("123", "123"),
    ])
    def test_data_variations(self, service, input_data, expected):
        """Test with various input data"""
        request = YourServiceRequest(input_field=input_data)
        response = service.your_endpoint(request)
        
        assert expected in response.result
    
    def test_error_handling(self, service):
        """Test error scenarios"""
        with patch.object(service, '_process_data', side_effect=Exception("Test error")):
            request = YourServiceRequest(input_field="test")
            response = service.your_endpoint(request)
            
            assert response.status == "error"
            assert response.error_message is not None

@pytest.mark.slow
class TestYourServiceIntegration:
    """Integration tests - slower tests with actual service"""
    
    @pytest.fixture(scope="class")
    def service_client(self):
        """HTTP client for integration testing"""
        from bentoml.testing import get_client
        
        with get_client(YourService) as client:
            yield client
    
    @pytest.mark.timeout(30)
    def test_endpoint_integration(self, service_client):
        """Test actual HTTP endpoint"""
        payload = {"request": {"input_field": "integration test"}}
        
        response = service_client.post("/your_endpoint", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
```

### 2. Mock Strategies

```python
# Effective mocking patterns
class TestServiceWithMocks:
    def test_with_external_api_mock(self):
        """Mock external API calls"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {"result": "mocked"}
            mock_post.return_value.status_code = 200
            
            # Test service method that uses external API
            service = YourService()
            result = service.call_external_api("test")
            
            assert result == {"result": "mocked"}
            mock_post.assert_called_once()
    
    def test_with_model_mock(self):
        """Mock heavy model operations"""
        with patch('torch.load') as mock_load, \
             patch('transformers.AutoModel.from_pretrained') as mock_model:
            
            mock_model.return_value = Mock()
            mock_model.return_value.predict.return_value = "mocked prediction"
            
            service = YourService()
            result = service.predict("test input")
            
            assert result == "mocked prediction"
```

## Documentation Standards

### 1. Code Documentation

```python
def complex_function(
    input_data: str,
    config: Dict[str, Any],
    timeout: Optional[int] = None
) -> ProcessResult:
    """
    Process input data with given configuration.
    
    This function performs complex data processing including preprocessing,
    model inference, and postprocessing steps.
    
    Args:
        input_data: Raw input data to process. Must be non-empty string.
        config: Processing configuration dictionary. Required keys:
            - model_name (str): Name of the model to use
            - threshold (float): Confidence threshold (0.0-1.0)
            Optional keys:
            - batch_size (int): Batch size for processing (default: 1)
        timeout: Maximum processing time in seconds. None for no timeout.
        
    Returns:
        ProcessResult containing:
            - result: Processed output string
            - confidence: Processing confidence score (0.0-1.0)
            - processing_time: Time taken in seconds
            - metadata: Additional processing information
    
    Raises:
        ValueError: If input_data is empty or config is invalid
        TimeoutError: If processing exceeds timeout
        ModelError: If model processing fails
        
    Example:
        >>> config = {"model_name": "bert-base", "threshold": 0.8}
        >>> result = complex_function("Hello world", config, timeout=30)
        >>> print(result.result)
        "Processed: Hello world"
    """
    # Implementation
```

### 2. Service Documentation

Create comprehensive service documentation:

**File**: `docs/services/your-service.md`

```markdown
# Your Service Documentation

## Overview
Brief description of what the service does and its primary use cases.

## API Reference

### Endpoints

#### POST /your_endpoint
Process input data and return results.

**Request:**
```json
{
  "request": {
    "input_field": "string",
    "optional_field": "string"
  }
}
```

**Response:**
```json
{
  "result": "string",
  "status": "success|error",
  "metadata": {}
}
```

## Configuration
Service configuration options and environment variables.

## Examples
Practical examples of using the service.

## Troubleshooting
Common issues and solutions.
```

This comprehensive approach to best practices ensures services are robust, maintainable, secure, and well-documented.