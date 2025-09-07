# BentoML Documentation

Comprehensive documentation for BentoML scraped from the official documentation website.

## Overview

This directory contains documentation scraped from the official BentoML documentation at https://docs.bentoml.com to provide easy access to reference materials during development.

## Documentation Files

### Core Concepts
- **[overview.md](overview.md)** - BentoML platform overview, key concepts, and value proposition
- **[quickstart.md](quickstart.md)** - Complete quickstart guide from installation to deployment

### Development Guides  
- **[services.md](services.md)** - Service definition, configuration, and lifecycle management
- **[input-output-types.md](input-output-types.md)** - Comprehensive guide to data types, validation, and serialization
- **[deployment.md](deployment.md)** - Production deployment, scaling, monitoring, and best practices

## Quick Reference

### Basic Service Structure
```python
import bentoml

@bentoml.service
class MyService:
    def __init__(self):
        self.model = self.load_model()
    
    @bentoml.api
    def predict(self, input_data: str) -> dict:
        result = self.model.predict(input_data)
        return {"prediction": result}
```

### File Upload Handling
```python
from pathlib import Path
from typing import Annotated
from bentoml._internal.types import ContentType

@bentoml.api  
def process_file(
    self,
    file: Annotated[Path, ContentType('audio/wav')]
) -> dict:
    # Process uploaded file
    return {"processed": True}
```

### Production Configuration
```python
@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 30, "concurrency": 50}
)
class ProductionService:
    # Service implementation
```

## Key Takeaways

1. **Modern API**: BentoML v1.4+ uses `@bentoml.service` and `@bentoml.api` decorators
2. **Type Safety**: Use Python type annotations and Pydantic models for validation  
3. **File Handling**: Use `Path` with `ContentType` annotations for file uploads
4. **Error Handling**: Return structured error responses instead of raising exceptions
5. **Production Ready**: Built-in support for scaling, monitoring, and deployment

## Official Documentation

For the most up-to-date information, always refer to the official documentation at:
- **Main Site**: https://docs.bentoml.com
- **GitHub**: https://github.com/bentoml/bentoml
- **Community**: Join the Slack community for support

---

*Documentation scraped on 2025-09-07 for offline reference during development.*