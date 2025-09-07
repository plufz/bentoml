# Example Service Documentation

👋 **Simple Hello World service** for testing BentoML setup and API functionality.

## Overview

The Example Service provides a basic greeting endpoint that demonstrates BentoML service patterns, request/response handling, and serves as a starting point for building custom services.

## Features

- **🎯 Simple API**: Easy-to-understand greeting endpoint
- **🔧 Minimal Dependencies**: No AI models or complex setup required
- **⚡ Fast Startup**: Instant service availability for testing
- **📊 Request Validation**: Demonstrates Pydantic model usage
- **🛠️ Testing Framework**: Comprehensive test coverage examples

## Quick Start

### Start the Service

```bash
# Individual service
./scripts/start.sh example

# Or as part of multi-service (recommended)
./scripts/start.sh
```

### Test the Service

```bash
# Simple greeting
./scripts/endpoint.sh hello '{"name": "World"}'

# Custom greeting
./scripts/endpoint.sh hello '{"name": "BentoML Developer"}'

# Default greeting (empty request)
./scripts/endpoint.sh hello '{}'
```

## API Endpoints

### 👋 `/hello` - Simple Greeting

Generate a personalized greeting message.

**Request:**
```json
{
  "request": {
    "name": "World"
  }
}
```

**Parameters:**
- `name` (string, optional): Name to include in greeting (default: "World")

**Response:**
```json
{
  "message": "Hello, World!",
  "timestamp": "2024-01-07T14:30:22.123456",
  "service": "ExampleService"
}
```

## Use Cases

### 🧪 **Testing & Development**
- Verify BentoML setup and configuration
- Test endpoint.sh script functionality
- Practice API request/response patterns

### 📚 **Learning & Training**
- Understand BentoML service structure
- Learn Pydantic request model patterns
- Study modern BentoML API usage

### 🔧 **Template & Starting Point**
- Base template for new services
- Reference implementation patterns
- Copy for custom service development

## Service Structure

```python
import bentoml
from pydantic import BaseModel
from datetime import datetime

class HelloRequest(BaseModel):
    name: str = "World"

@bentoml.service()
class ExampleService:
    @bentoml.api
    def hello(self, request: HelloRequest) -> dict:
        return {
            "message": f"Hello, {request.name}!",
            "timestamp": datetime.now().isoformat(),
            "service": "ExampleService"
        }
```

## Testing

Run example service tests:

```bash
# Run all example tests
./scripts/test.sh --service example

# Run specific test classes
uv run pytest tests/test_example_service.py
```

## Configuration

### Environment Variables

```bash
# Service port (default: 3002)
EXAMPLE_SERVICE_PORT=3002

# Server configuration
BENTOML_HOST=127.0.0.1
BENTOML_PROTOCOL=http
```

## Building Custom Services

Use the Example Service as a template:

1. Copy `services/example_service.py`
2. Modify the request/response models
3. Implement your business logic
4. Create corresponding tests
5. Update documentation

## Related Services

- **[Stable Diffusion](stable-diffusion.md)** - AI image generation
- **[LLaVA Vision](llava-service.md)** - AI image analysis  
- **[Photo Upscaler](photo-upscaler.md)** - AI image enhancement
- **[Whisper](whisper-service.md)** - AI audio transcription

---

💡 **Development Tip**: The Example Service is perfect for testing new BentoML features and configurations before adding complexity with AI models.