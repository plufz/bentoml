# BentoML Deployment Guide

Comprehensive guide to deploying BentoML services from local development to production.

## Overview

BentoML provides a **Unified Inference Platform** for deploying AI models with production-grade reliability across different cloud environments. It supports autoscaling, GPU inference, and distributed services.

## Deployment Options

### 1. Local Development

Start with local serving for development and testing:

```bash
# Serve directly from Python file
bentoml serve service.py:MyService

# Serve with custom port
bentoml serve service.py:MyService --port 8000

# Development mode with auto-reload
bentoml serve service.py:MyService --reload
```

### 2. Containerization

Create Docker containers for consistent deployment:

```bash
# Build Bento (creates deployable package)
bentoml build

# Generate Dockerfile
bentoml containerize my-service:latest

# Build Docker image
docker build -t my-service:latest .

# Run container
docker run -p 3000:3000 my-service:latest
```

### 3. Cloud Deployment

Deploy to various cloud platforms:

#### BentoCloud (Managed)
```bash
# Login to BentoCloud
bentoml cloud login

# Deploy to BentoCloud
bentoml deploy my-service:latest
```

#### Kubernetes
```bash
# Generate Kubernetes manifests
bentoml generate kubernetes my-service:latest

# Apply to cluster
kubectl apply -f ./
```

#### AWS, GCP, Azure
- Use container deployment options
- Leverage cloud-specific AI/ML services
- Integrate with managed Kubernetes services

## Production Considerations

### Resource Configuration

```python
@bentoml.service(
    resources={
        "cpu": "4",           # CPU cores
        "memory": "8Gi",      # RAM allocation  
        "gpu": "1",           # GPU count
        "gpu_type": "nvidia-t4"
    },
    traffic={
        "timeout": 60,        # Request timeout (seconds)
        "concurrency": 100    # Max concurrent requests
    }
)
class ProductionService:
    # Service implementation
```

### Scaling Configuration

```python
@bentoml.service(
    scaling={
        "min_replicas": 2,    # Minimum instances
        "max_replicas": 10,   # Maximum instances  
        "target_cpu_percent": 70,  # CPU threshold for scaling
        "scale_down_delay": 300    # Cool-down period
    }
)
class ScalableService:
    # Service implementation
```

### Health Checks

```python
@bentoml.service
class HealthAwareService:
    @bentoml.api
    def health(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": time.time()
        }
    
    def ready(self) -> bool:
        """Readiness check"""
        return self.model is not None
```

## Performance Optimization

### GPU Inference

```python
import torch

@bentoml.service(
    resources={"gpu": "1"}
)
class GPUService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model().to(self.device)
    
    @bentoml.api
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        with torch.no_grad():
            return self.model(data)
```

### Batch Processing

```python
@bentoml.service
class BatchService:
    @bentoml.api(
        batch=True,
        max_batch_size=32,
        max_latency_ms=100
    )
    def batch_predict(self, inputs: list[np.ndarray]) -> list[dict]:
        # Process batch efficiently
        batch = np.stack(inputs)
        results = self.model.predict(batch)
        return [{"prediction": r} for r in results]
```

### Caching

```python
from functools import lru_cache

@bentoml.service
class CachedService:
    @lru_cache(maxsize=1000)
    def cached_computation(self, key: str) -> str:
        # Expensive computation
        return self.heavy_processing(key)
    
    @bentoml.api
    def predict(self, input: str) -> dict:
        result = self.cached_computation(input)
        return {"result": result}
```

## Monitoring and Observability

### Logging

```python
import logging

@bentoml.service
class LoggingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    @bentoml.api
    def predict(self, data: str) -> dict:
        self.logger.info(f"Processing request: {data[:50]}...")
        
        try:
            result = self.process(data)
            self.logger.info("Request processed successfully")
            return {"result": result}
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise
```

### Metrics

```python
from prometheus_client import Counter, Histogram

# Metrics collection
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@bentoml.service
class MetricsService:
    @bentoml.api
    def predict(self, data: str) -> dict:
        REQUEST_COUNT.inc()
        
        with REQUEST_DURATION.time():
            result = self.process(data)
        
        return {"result": result}
```

### Distributed Tracing

```python
from opentelemetry import trace

@bentoml.service
class TracedService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
    
    @bentoml.api
    def predict(self, data: str) -> dict:
        with self.tracer.start_as_current_span("prediction"):
            with self.tracer.start_as_current_span("preprocessing"):
                processed = self.preprocess(data)
            
            with self.tracer.start_as_current_span("inference"):
                result = self.model.predict(processed)
            
            return {"result": result}
```

## Security

### Authentication

```python
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not validate_token(auth_header):
            raise bentoml.exceptions.Unauthorized("Invalid authentication")
        return f(*args, **kwargs)
    return decorated_function

@bentoml.service
class SecureService:
    @bentoml.api
    @require_auth
    def secure_predict(self, data: str) -> dict:
        return {"result": self.process(data)}
```

### Input Validation

```python
from pydantic import BaseModel, validator

class SecureInput(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        # Sanitize input
        if len(v) > 10000:
            raise ValueError("Text too long")
        if any(char in v for char in ['<', '>', '&']):
            raise ValueError("Invalid characters")
        return v

@bentoml.service
class ValidatedService:
    @bentoml.api
    def predict(self, input: SecureInput) -> dict:
        return {"result": self.process(input.text)}
```

## Environment Configuration

### Environment Variables

```python
import os

@bentoml.service
class ConfigurableService:
    def __init__(self):
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.model_path = os.getenv('MODEL_PATH', './models')
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
```

### Configuration Files

```yaml
# bentoml-config.yaml
services:
  my-service:
    resources:
      cpu: "4"
      memory: "8Gi" 
    scaling:
      min_replicas: 2
      max_replicas: 10
    environment:
      DEBUG: "false"
      MODEL_PATH: "/opt/models"
```

## Deployment Best Practices

1. **Version Control**: Tag and version your models and services
2. **Testing**: Implement comprehensive testing before deployment
3. **Gradual Rollout**: Use canary deployments for production updates
4. **Monitoring**: Set up comprehensive monitoring and alerting
5. **Resource Management**: Right-size resources based on actual usage
6. **Security**: Implement proper authentication and input validation
7. **Documentation**: Maintain deployment runbooks and procedures

## Example: Complete Production Service

```python
import os
import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import bentoml
from prometheus_client import Counter, Histogram

# Metrics
REQUESTS = Counter('http_requests_total', 'Total HTTP requests')
LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

class PredictionRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    options: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    model_version: str

@bentoml.service(
    resources={
        "cpu": "2",
        "memory": "4Gi",
        "gpu": "1" if os.getenv("GPU_ENABLED") == "true" else "0"
    },
    traffic={
        "timeout": 30,
        "concurrency": 50
    }
)
class ProductionMLService:
    def __init__(self):
        self.setup_logging()
        self.model = self.load_model()
        self.model_version = "1.0.0"
        
    def setup_logging(self):
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        model_path = os.getenv('MODEL_PATH', './model')
        self.logger.info(f"Loading model from {model_path}")
        # Load your model here
        return None  # Replace with actual model
        
    @bentoml.api
    async def predict(
        self, 
        request: PredictionRequest
    ) -> PredictionResponse:
        start_time = time.time()
        REQUESTS.inc()
        
        try:
            with LATENCY.time():
                # Model inference
                prediction = await self.run_prediction(request.text, request.options)
                
                processing_time = time.time() - start_time
                
                return PredictionResponse(
                    prediction=prediction["text"],
                    confidence=prediction["confidence"],
                    processing_time=processing_time,
                    model_version=self.model_version
                )
                
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise bentoml.exceptions.BentoMLException(f"Prediction failed: {str(e)}")
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "version": self.model_version,
            "timestamp": time.time()
        }
    
    async def run_prediction(self, text: str, options: Optional[Dict] = None):
        # Actual prediction logic
        self.logger.info(f"Processing text of length {len(text)}")
        
        # Simulate prediction
        await asyncio.sleep(0.1)  # Replace with actual model call
        
        return {
            "text": f"Processed: {text[:50]}...",
            "confidence": 0.95
        }
```

This production-ready example includes metrics, logging, error handling, health checks, and proper configuration management.