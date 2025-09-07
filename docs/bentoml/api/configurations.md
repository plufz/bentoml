# BentoML Configurations

BentoML provides a flexible configuration interface to customize runtime behavior for individual Services within a Bento. Configurations can be set using the `@bentoml.service` decorator, allowing granular control over various aspects of service deployment and performance.

## Service Configuration Overview

### Basic Service Configuration
```python
import bentoml

@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"},
    workers=4,
    timeout=120
)
class MyService:
    @bentoml.api
    def predict(self, data):
        return {"result": "prediction"}
```

## Configuration Fields

### Resources
Specifies resource allocation for a Service, particularly important for BentoCloud deployments.

```python
@bentoml.service(
    resources={
        "cpu": "2",           # Number of CPU cores
        "memory": "4Gi",      # Memory allocation
        "gpu": 1,             # Number of GPUs
        "gpu_type": "nvidia-tesla-a100"  # Specific GPU type
    }
)
class ResourceManagedService:
    pass
```

**CPU Options:**
- String format: `"1"`, `"2"`, `"0.5"`
- Millicores: `"500m"` (500 millicores = 0.5 cores)

**Memory Options:**
- String format with units: `"1Gi"`, `"512Mi"`, `"2G"`
- Without units defaults to bytes

**GPU Types:**
- `nvidia-tesla-k80`
- `nvidia-tesla-v100` 
- `nvidia-tesla-a100`
- `nvidia-l4`
- Custom GPU specifications

### Workers
Defines process-level parallelism within a Service.

```python
# Fixed number of workers
@bentoml.service(workers=4)
class FixedWorkerService:
    pass

# Dynamic worker allocation based on CPU count
@bentoml.service(workers="cpu_count")
class DynamicWorkerService:
    pass
```

**Options:**
- Integer: Fixed number of workers
- `"cpu_count"`: Match number of available CPUs
- Default: 1 worker

### Traffic Management
Control request handling, concurrency, and timeout behavior.

```python
@bentoml.service(
    traffic={
        "timeout": 300,           # Maximum response time (seconds)
        "max_concurrency": 100,   # Maximum simultaneous requests
        "concurrency": 32,        # Ideal simultaneous requests
        "external_queue": True    # Handle request overflow
    }
)
class TrafficManagedService:
    pass
```

**Traffic Parameters:**
- `timeout`: Request timeout in seconds (default: 60)
- `max_concurrency`: Hard limit on concurrent requests
- `concurrency`: Target concurrency level
- `external_queue`: Enable external queueing for overflow requests

### Environment Variables
Set environment variables for service configuration.

```python
@bentoml.service(
    envs=[
        {"name": "HF_TOKEN"},           # Reference from environment
        {"name": "API_KEY", "value": "secret"},  # Set explicit value
        {"name": "DEBUG", "value": "true"},
        {"name": "MODEL_PATH", "value": "/opt/models"}
    ]
)
class ConfiguredService:
    pass
```

**Environment Variable Formats:**
- `{"name": "VAR_NAME"}`: Reference existing environment variable
- `{"name": "VAR_NAME", "value": "value"}`: Set explicit value

### SSL Configuration
Enable secure communication for production deployments.

```python
@bentoml.service(
    ssl={
        "enabled": True,
        "cert_file": "/path/to/cert.pem",
        "key_file": "/path/to/key.pem",
        "ca_certs": "/path/to/ca.pem"
    }
)
class SecureService:
    pass
```

### HTTP Configuration
Customize HTTP server behavior.

```python
@bentoml.service(
    http={
        "host": "0.0.0.0",
        "port": 8080,
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "POST"]
        }
    }
)
class HTTPConfiguredService:
    pass
```

### Image Specifications
Define container image runtime specifications.

```python
@bentoml.service(
    image={
        "base_image": "python:3.9-slim",
        "dockerfile_template": "custom.Dockerfile",
        "cuda_version": "11.8",
        "python_version": "3.9"
    }
)
class CustomImageService:
    pass
```

## Complete Configuration Example

```python
import bentoml
from typing import Dict, Any

@bentoml.service(
    name="production_ml_service",
    resources={
        "cpu": "4",
        "memory": "8Gi", 
        "gpu": 2,
        "gpu_type": "nvidia-tesla-a100"
    },
    workers="cpu_count",
    traffic={
        "timeout": 180,
        "max_concurrency": 200,
        "concurrency": 64,
        "external_queue": True
    },
    envs=[
        {"name": "MODEL_CACHE_DIR", "value": "/tmp/model_cache"},
        {"name": "BATCH_SIZE", "value": "32"},
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "HUGGING_FACE_TOKEN"}  # From environment
    ],
    http={
        "host": "0.0.0.0",
        "port": 3000,
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["https://myapp.com"],
            "access_control_allow_methods": ["GET", "POST", "OPTIONS"]
        }
    },
    ssl={
        "enabled": True,
        "cert_file": "/etc/ssl/certs/service.crt",
        "key_file": "/etc/ssl/private/service.key"
    }
)
class ProductionMLService:
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        # Model loading logic
        pass
    
    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        predictions = self.model.predict(input_data)
        return {"predictions": predictions}
    
    @bentoml.api
    def batch_predict(self, batch_data: list) -> list:
        results = []
        for item in batch_data:
            result = self.model.predict(item)
            results.append(result)
        return results
```

## Configuration Best Practices

### 1. Environment-Specific Configuration
Use different configurations for development and production:

```python
import os

# Development configuration
if os.getenv("ENV") == "development":
    service_config = {
        "workers": 1,
        "traffic": {"timeout": 300},
        "envs": [{"name": "DEBUG", "value": "true"}]
    }
else:
    # Production configuration
    service_config = {
        "workers": "cpu_count",
        "resources": {"cpu": "4", "memory": "8Gi"},
        "traffic": {"timeout": 60, "max_concurrency": 100},
        "envs": [{"name": "DEBUG", "value": "false"}]
    }

@bentoml.service(**service_config)
class AdaptiveService:
    pass
```

### 2. Resource Planning
Plan resources based on model requirements:

```python
# For CPU-intensive models
@bentoml.service(
    resources={"cpu": "8", "memory": "16Gi"},
    workers=4
)
class CPUIntensiveService:
    pass

# For GPU-accelerated models
@bentoml.service(
    resources={"gpu": 1, "gpu_type": "nvidia-tesla-v100"},
    workers=1  # Usually 1 worker per GPU
)
class GPUService:
    pass
```

### 3. Security Configuration
Always configure security for production:

```python
@bentoml.service(
    ssl={"enabled": True},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["https://trusted-domain.com"]
        }
    },
    envs=[
        {"name": "API_SECRET_KEY"}  # Never hardcode secrets
    ]
)
class SecureProductionService:
    pass
```

### 4. Monitoring and Observability
Configure for monitoring and debugging:

```python
@bentoml.service(
    envs=[
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "METRICS_ENABLED", "value": "true"},
        {"name": "TRACE_SAMPLING_RATE", "value": "0.1"}
    ],
    traffic={"timeout": 120}  # Allow time for debugging
)
class MonitoredService:
    pass
```