# BentoML Error Handling

Comprehensive guide to implementing robust error handling patterns in BentoML services.

## Overview

Effective error handling is crucial for building user-friendly AI applications. BentoML provides flexible mechanisms for:
- **Custom exception classes** with specific error codes
- **Structured error responses** for API consistency  
- **Graceful degradation** when services encounter issues
- **Detailed logging** for debugging and monitoring

## Custom Exception Classes

### Basic Custom Exception

```python
from http import HTTPStatus
from bentoml import BentoMLException
import bentoml

class ValidationError(BentoMLException):
    error_code = HTTPStatus.BAD_REQUEST

class ModelError(BentoMLException):
    error_code = HTTPStatus.INTERNAL_SERVER_ERROR

class NotFoundError(BentoMLException):
    error_code = HTTPStatus.NOT_FOUND

@bentoml.service
class RobustService:
    @bentoml.api
    def validate_and_process(self, text: str) -> dict:
        # Input validation
        if not text or len(text.strip()) == 0:
            raise ValidationError("Input text cannot be empty")
        
        if len(text) > 10000:
            raise ValidationError("Input text too long (max 10,000 characters)")
        
        try:
            result = self.process_text(text)
            return {"success": True, "result": result}
        except Exception as e:
            raise ModelError(f"Processing failed: {str(e)}")
```

### Hierarchical Exception Structure

```python
from http import HTTPStatus
from bentoml import BentoMLException

class ServiceException(BentoMLException):
    """Base exception for service-specific errors"""
    pass

class InputException(ServiceException):
    """Exceptions related to input validation"""
    error_code = HTTPStatus.BAD_REQUEST

class ProcessingException(ServiceException):
    """Exceptions during model processing"""
    error_code = HTTPStatus.INTERNAL_SERVER_ERROR

class ResourceException(ServiceException):
    """Exceptions related to resource availability"""
    error_code = HTTPStatus.SERVICE_UNAVAILABLE

# Specific exception types
class InvalidFormatError(InputException):
    def __init__(self, format_type: str):
        super().__init__(f"Invalid {format_type} format provided")

class ModelTimeoutError(ProcessingException):
    def __init__(self, timeout: int):
        super().__init__(f"Model processing timed out after {timeout} seconds")

class ResourceExhaustedError(ResourceException):
    def __init__(self, resource: str):
        super().__init__(f"{resource} resource exhausted")
```

## Error Handling Patterns

### Input Validation

```python
import bentoml
from pydantic import BaseModel, Field, validator
from typing import Optional

class ProcessingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field("en", regex=r'^[a-z]{2}$')
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    
    @validator('text')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be only whitespace")
        return v

@bentoml.service
class ValidatedService:
    @bentoml.api
    def process_with_validation(self, request: ProcessingRequest) -> dict:
        """Automatic validation through Pydantic"""
        try:
            result = self.model.process(
                request.text, 
                language=request.language,
                threshold=request.confidence_threshold
            )
            return {"success": True, "result": result}
        except ValueError as e:
            raise ValidationError(f"Invalid input: {str(e)}")
        except Exception as e:
            raise ProcessingException(f"Processing error: {str(e)}")
```

### Resource Management

```python
import bentoml
import time
from contextlib import contextmanager

class ResourceManager:
    def __init__(self, max_concurrent=10):
        self.active_requests = 0
        self.max_concurrent = max_concurrent
    
    @contextmanager
    def acquire_resource(self):
        if self.active_requests >= self.max_concurrent:
            raise ResourceExhaustedError("concurrent_requests")
        
        self.active_requests += 1
        try:
            yield
        finally:
            self.active_requests -= 1

@bentoml.service
class ResourceAwareService:
    def __init__(self):
        self.resource_manager = ResourceManager(max_concurrent=5)
        self.model = self.load_model()
    
    @bentoml.api
    def controlled_processing(self, data: str) -> dict:
        with self.resource_manager.acquire_resource():
            try:
                # Simulate time-intensive processing
                start_time = time.time()
                result = self.model.process(data)
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "result": result,
                    "processing_time": processing_time
                }
            except TimeoutError:
                raise ModelTimeoutError(30)
            except Exception as e:
                raise ProcessingException(str(e))
```

### Graceful Degradation

```python
import bentoml
from typing import Optional, Union

@bentoml.service
class ResilientService:
    def __init__(self):
        self.primary_model = self.load_primary_model()
        self.fallback_model = self.load_fallback_model()
        self.cache = {}
    
    @bentoml.api
    def resilient_prediction(self, input_data: str) -> dict:
        """Service with multiple fallback strategies"""
        
        # Strategy 1: Check cache first
        cache_key = hash(input_data)
        if cache_key in self.cache:
            return {
                "success": True,
                "result": self.cache[cache_key],
                "source": "cache"
            }
        
        # Strategy 2: Try primary model
        try:
            result = self.primary_model.predict(input_data)
            self.cache[cache_key] = result
            return {
                "success": True,
                "result": result,
                "source": "primary_model"
            }
        except Exception as primary_error:
            print(f"Primary model failed: {primary_error}")
        
        # Strategy 3: Fallback to secondary model
        try:
            result = self.fallback_model.predict(input_data)
            return {
                "success": True,
                "result": result,
                "source": "fallback_model",
                "warning": "Primary model unavailable"
            }
        except Exception as fallback_error:
            print(f"Fallback model failed: {fallback_error}")
        
        # Strategy 4: Return error with helpful information
        raise ProcessingException(
            "All prediction methods failed. Please try again later."
        )
```

## Structured Error Responses

### Consistent Error Format

```python
import bentoml
from typing import Dict, Any, Optional
import traceback
import logging

logger = logging.getLogger(__name__)

class ErrorResponse:
    def __init__(
        self, 
        error_type: str, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.request_id = request_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": False,
            "error": {
                "type": self.error_type,
                "message": self.message,
                "details": self.details,
                "request_id": self.request_id
            }
        }

@bentoml.service
class StructuredErrorService:
    @bentoml.api
    def process_with_structured_errors(self, text: str) -> dict:
        request_id = f"req_{int(time.time())}"
        
        try:
            # Validate input
            if not text:
                error = ErrorResponse(
                    error_type="VALIDATION_ERROR",
                    message="Input text is required",
                    details={"field": "text", "constraint": "non_empty"},
                    request_id=request_id
                )
                return error.to_dict()
            
            # Process input
            result = self.model.process(text)
            
            return {
                "success": True,
                "result": result,
                "request_id": request_id
            }
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            error = ErrorResponse(
                error_type="VALIDATION_ERROR",
                message=str(e),
                request_id=request_id
            )
            return error.to_dict()
            
        except ProcessingException as e:
            logger.error(f"Processing error: {e}")
            error = ErrorResponse(
                error_type="PROCESSING_ERROR",
                message=str(e),
                details={"traceback": traceback.format_exc()},
                request_id=request_id
            )
            return error.to_dict()
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            error = ErrorResponse(
                error_type="INTERNAL_ERROR",
                message="An unexpected error occurred",
                details={"original_error": str(e)},
                request_id=request_id
            )
            return error.to_dict()
```

### Health Check Integration

```python
@bentoml.service
class HealthAwareService:
    def __init__(self):
        self.model = None
        self.model_healthy = False
        self.load_model()
    
    def load_model(self):
        try:
            self.model = self.initialize_model()
            self.model_healthy = True
            logger.info("Model loaded successfully")
        except Exception as e:
            self.model_healthy = False
            logger.error(f"Failed to load model: {e}")
    
    @bentoml.api
    def health_check(self) -> dict:
        """Comprehensive health check"""
        checks = {
            "model_loaded": self.model is not None,
            "model_healthy": self.model_healthy,
            "service_ready": self.model_healthy
        }
        
        if not checks["service_ready"]:
            raise ResourceException("Service not ready - model unavailable")
        
        return {
            "success": True,
            "status": "healthy",
            "checks": checks,
            "timestamp": time.time()
        }
    
    @bentoml.api
    def predict(self, input_data: str) -> dict:
        # Check service health before processing
        if not self.model_healthy:
            raise ResourceException("Service unhealthy - model not available")
        
        try:
            result = self.model.predict(input_data)
            return {"success": True, "result": result}
        except Exception as e:
            # Mark model as unhealthy if prediction fails
            self.model_healthy = False
            raise ProcessingException(f"Prediction failed: {str(e)}")
```

## Advanced Error Handling

### Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise ResourceException("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

@bentoml.service
class CircuitBreakerService:
    def __init__(self):
        self.model = self.load_model()
        self.circuit_breaker = CircuitBreaker()
    
    @bentoml.api
    def protected_prediction(self, input_data: str) -> dict:
        try:
            result = self.circuit_breaker.call(
                self.model.predict, 
                input_data
            )
            return {"success": True, "result": result}
        except ResourceException as e:
            return {
                "success": False,
                "error": str(e),
                "circuit_state": self.circuit_breaker.state.value
            }
```

## Best Practices

### 1. Exception Hierarchy

```python
# Create clear exception hierarchies
class ServiceException(BentoMLException):
    """Base for all service exceptions"""
    pass

class UserException(ServiceException):
    """User-caused errors (4xx)"""
    pass

class SystemException(ServiceException):
    """System/server errors (5xx)"""
    pass
```

### 2. Logging Strategy

```python
import logging
import json

def setup_error_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

@bentoml.service
class WellLoggedService:
    def __init__(self):
        self.logger = setup_error_logging()
    
    @bentoml.api
    def logged_processing(self, data: str) -> dict:
        self.logger.info(f"Processing request: {len(data)} characters")
        
        try:
            result = self.process(data)
            self.logger.info("Processing completed successfully")
            return {"success": True, "result": result}
            
        except ValidationError as e:
            self.logger.warning(f"Validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ProcessingException("Internal processing error")
```

### 3. Error Monitoring

```python
from prometheus_client import Counter, Histogram

# Metrics for error tracking
ERROR_COUNTER = Counter('service_errors_total', 'Total errors', ['error_type'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@bentoml.service
class MonitoredService:
    @bentoml.api
    def monitored_endpoint(self, data: str) -> dict:
        with REQUEST_DURATION.time():
            try:
                result = self.process(data)
                return {"success": True, "result": result}
            except ValidationError as e:
                ERROR_COUNTER.labels(error_type='validation').inc()
                raise
            except ProcessingException as e:
                ERROR_COUNTER.labels(error_type='processing').inc()
                raise
            except Exception as e:
                ERROR_COUNTER.labels(error_type='unknown').inc()
                raise ProcessingException("Unexpected error occurred")
```

## Reserved Error Codes

**Avoid using these HTTP status codes (reserved by BentoML):**
- `401` - Unauthorized
- `403` - Forbidden  
- `500+` - Internal Server Errors

**Safe to use:**
- `400` - Bad Request
- `404` - Not Found
- `405` - Method Not Allowed
- `422` - Unprocessable Entity
- `429` - Too Many Requests

## Complete Example

```python
import bentoml
from http import HTTPStatus
from bentoml import BentoMLException
import logging
import time
from typing import Dict, Any

# Custom exceptions
class ServiceError(BentoMLException):
    pass

class ValidationError(ServiceError):
    error_code = HTTPStatus.BAD_REQUEST

class ProcessingError(ServiceError):
    error_code = HTTPStatus.INTERNAL_SERVER_ERROR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"}
)
class RobustAIService:
    def __init__(self):
        self.model = self.load_model_safely()
        self.request_count = 0
    
    def load_model_safely(self):
        try:
            logger.info("Loading model...")
            # model = load_your_model()
            logger.info("Model loaded successfully")
            return None  # Replace with actual model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ProcessingError("Service initialization failed")
    
    @bentoml.api
    def robust_predict(self, text: str) -> Dict[str, Any]:
        """Prediction endpoint with comprehensive error handling"""
        request_id = f"req_{int(time.time())}_{self.request_count}"
        self.request_count += 1
        
        logger.info(f"Processing request {request_id}")
        
        try:
            # Input validation
            if not isinstance(text, str):
                raise ValidationError("Input must be a string")
            
            if not text.strip():
                raise ValidationError("Input text cannot be empty")
            
            if len(text) > 10000:
                raise ValidationError("Input text too long (max 10,000 characters)")
            
            # Processing with timeout
            start_time = time.time()
            result = self.safe_model_predict(text)
            processing_time = time.time() - start_time
            
            logger.info(f"Request {request_id} completed in {processing_time:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "request_id": request_id,
                "processing_time": processing_time
            }
            
        except ValidationError as e:
            logger.warning(f"Validation error for {request_id}: {e}")
            return {
                "success": False,
                "error": {
                    "type": "VALIDATION_ERROR",
                    "message": str(e),
                    "request_id": request_id
                }
            }
        
        except ProcessingError as e:
            logger.error(f"Processing error for {request_id}: {e}")
            return {
                "success": False,
                "error": {
                    "type": "PROCESSING_ERROR", 
                    "message": str(e),
                    "request_id": request_id
                }
            }
        
        except Exception as e:
            logger.error(f"Unexpected error for {request_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": {
                    "type": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id
                }
            }
    
    def safe_model_predict(self, text: str) -> Dict[str, Any]:
        """Model prediction with error handling"""
        try:
            if self.model is None:
                raise ProcessingError("Model not available")
            
            # Simulate model prediction
            # result = self.model.predict(text)
            result = {
                "prediction": f"Processed: {text[:50]}...",
                "confidence": 0.95
            }
            
            return result
            
        except Exception as e:
            raise ProcessingError(f"Model prediction failed: {str(e)}")
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy" if self.model else "unhealthy",
            "model_loaded": self.model is not None,
            "requests_processed": self.request_count,
            "timestamp": time.time()
        }
```

This comprehensive approach to error handling ensures your BentoML services are robust, user-friendly, and maintainable in production environments.