# BentoML Services

Complete guide to defining and configuring BentoML services.

## Service Definition

BentoML services are defined using the `@bentoml.service` decorator on Python classes. This creates a deployable service that can serve machine learning models through API endpoints.

### Basic Service Structure

```python
import bentoml

@bentoml.service
class MyService:
    def __init__(self):
        # Initialize models, resources, etc.
        pass
    
    @bentoml.api
    def predict(self, input_data: str) -> str:
        # API endpoint implementation
        return "processed: " + input_data
```

## Service Configuration

The `@bentoml.service` decorator accepts configuration options for resources and behavior:

```python
@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 30},
    workers=4
)
class ConfiguredService:
    # Service implementation
```

### Configuration Options

- **resources**: Hardware resource allocation
  - `cpu`: CPU cores (e.g., "2", "0.5")  
  - `memory`: RAM allocation (e.g., "4Gi", "512Mi")
  - `gpu`: GPU specification (e.g., "1", "nvidia.com/gpu=2")

- **traffic**: Request handling configuration
  - `timeout`: Request timeout in seconds
  - `max_concurrency`: Maximum concurrent requests

- **workers**: Number of worker processes

## API Endpoints

Define API endpoints using the `@bentoml.api` decorator:

### Synchronous APIs

```python
@bentoml.api
def sync_predict(self, data: str) -> str:
    # Synchronous processing
    return self.model.predict(data)
```

### Asynchronous APIs

```python
@bentoml.api
async def async_predict(self, data: str) -> str:
    # Asynchronous processing
    result = await self.async_model.predict(data)
    return result
```

### Custom Route Paths

```python
@bentoml.api(route="/custom-endpoint")
def custom_endpoint(self, input: str) -> str:
    return self.process(input)
```

## Service Lifecycle

### Initialization

The `__init__` method runs when the service starts:

```python
@bentoml.service
class ServiceWithInit:
    def __init__(self):
        # Load models
        self.model = self.load_model()
        
        # Initialize resources
        self.cache = {}
        
        # Setup configuration
        self.setup_logging()
```

### Lifecycle Hooks

BentoML provides hooks for service lifecycle management:

```python
@bentoml.service
class LifecycleService:
    def __init__(self):
        self.setup()
    
    async def startup(self):
        """Called when service starts"""
        print("Service starting up...")
    
    async def shutdown(self):
        """Called when service shuts down"""  
        print("Service shutting down...")
        self.cleanup_resources()
```

## Advanced Features

### Task Queues

For background processing:

```python
@bentoml.service
class TaskService:
    @bentoml.api
    def submit_task(self, data: str) -> str:
        # Submit to background queue
        task_id = self.queue.submit(self.process_task, data)
        return f"Task submitted: {task_id}"
    
    def process_task(self, data):
        # Background processing
        pass
```

### Environment Variables

Configure services with environment variables:

```python
import os

@bentoml.service
class ConfigurableService:
    def __init__(self):
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.model_path = os.getenv('MODEL_PATH', './model')
```

### Custom Start Commands

Override default service behavior:

```python
@bentoml.service
class CustomStartService:
    def start(self):
        # Custom startup logic
        self.initialize_custom_resources()
        super().start()
```

## Best Practices

1. **Resource Management**: Specify appropriate resource limits
2. **Error Handling**: Implement proper exception handling
3. **Async When Appropriate**: Use async APIs for I/O-bound operations  
4. **Stateless Design**: Keep services stateless for better scaling
5. **Logging**: Implement proper logging for debugging and monitoring
6. **Testing**: Write unit tests for service methods

## Example: Complete Service

```python
import bentoml
import logging
from typing import Dict, Any

@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 60}
)
class ProductionService:
    def __init__(self):
        self.setup_logging()
        self.model = self.load_model()
        self.cache = {}
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        # Model loading logic
        self.logger.info("Loading model...")
        return None  # Replace with actual model loading
        
    @bentoml.api
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info(f"Processing request: {input_data}")
            
            # Model inference
            result = await self.run_inference(input_data)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_inference(self, data):
        # Actual inference logic
        return {"prediction": "example"}
```

This comprehensive service example demonstrates resource configuration, proper logging, error handling, and structured responses.