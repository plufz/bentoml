# BentoML Service Structure Guide

This guide explains how BentoML services are structured in this project and which files need to be created or updated when adding or modifying services.

## Service Architecture Overview

```
bentoml/
├── services/           # Main service implementations
├── utils/             # Shared utilities and helpers
├── config/            # Service configurations
│   └── bentofiles/    # Bento build configurations
├── tests/             # Test files for all services
├── scripts/           # Development and deployment scripts
└── docs/              # Documentation files
```

## Creating a New Service

### 1. Service Implementation (`services/`)

Create your main service file in the `services/` directory:

**File**: `services/your_service.py`

```python
import bentoml
from pydantic import BaseModel
from typing import Optional

# Request/Response models
class YourServiceRequest(BaseModel):
    input_field: str
    optional_field: Optional[str] = "default"

class YourServiceResponse(BaseModel):
    result: str
    metadata: dict = {}

@bentoml.service(
    name="your-service",
    resources={"cpu": "2", "memory": "4Gi"}
)
class YourService:
    def __init__(self):
        # Initialize models, load dependencies
        pass
    
    @bentoml.api
    def your_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        # Implementation
        return YourServiceResponse(
            result=f"Processed: {request.input_field}",
            metadata={"status": "success"}
        )
```

### 2. Utility Functions (`utils/`)

If your service needs shared utilities, create helper modules:

**File**: `utils/your_service_utils.py`

```python
"""Utility functions for YourService"""

def preprocess_data(data: str) -> str:
    """Preprocess input data"""
    return data.strip().lower()

def postprocess_result(result: dict) -> dict:
    """Postprocess service result"""
    return {**result, "processed": True}
```

### 3. Configuration (`config/`)

#### Bento Build Configuration

**File**: `config/bentofiles/your-service.yaml`

```yaml
service: "services.your_service:YourService"
name: "your-service"
labels:
  owner: bentoml-team
  stage: dev
include:
  - "services/your_service.py"
  - "utils/your_service_utils.py"
python:
  requirements_txt: |
    bentoml[io]>=1.4.0
    pydantic>=2.5.0
    # Add service-specific dependencies
```

#### BentoML Configuration (if needed)

Update `config/bentoml.yaml` if your service needs specific configurations:

```yaml
api_server:
  port: 3000
  host: 127.0.0.1
  cors:
    enabled: true
```

### 4. Testing (`tests/`)

Create comprehensive tests for your service:

**File**: `tests/test_your_service.py`

```python
import pytest
import asyncio
from unittest.mock import Mock, patch
from services.your_service import YourService, YourServiceRequest

class TestYourServiceUnit:
    """Unit tests for YourService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = YourService()
    
    def test_your_endpoint_success(self):
        """Test successful endpoint execution"""
        request = YourServiceRequest(input_field="test input")
        response = self.service.your_endpoint(request)
        
        assert response.result == "Processed: test input"
        assert response.metadata["status"] == "success"
    
    def test_your_endpoint_with_optional_field(self):
        """Test endpoint with optional parameters"""
        request = YourServiceRequest(
            input_field="test", 
            optional_field="custom"
        )
        response = self.service.your_endpoint(request)
        
        assert "test" in response.result

@pytest.mark.slow
class TestYourServiceIntegration:
    """Integration tests for YourService"""
    
    @pytest.fixture(scope="class")
    def service_client(self):
        """Start service for integration testing"""
        # Implementation for service startup
        pass
    
    def test_api_endpoint_integration(self, service_client):
        """Test actual API endpoint"""
        payload = {"request": {"input_field": "integration test"}}
        response = service_client.post("/your_endpoint", json=payload)
        
        assert response.status_code == 200
        assert "result" in response.json()
```

### 5. Scripts Integration (`scripts/`)

Update relevant scripts to include your service:

#### Build Script (`scripts/build_services.sh`)

Add your service to the build process:

```bash
# Add to build_services.sh
echo "Building Your Service..."
BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py
```

#### Test Script (`scripts/test.sh`)

Add service-specific testing option:

```bash
# Add to test.sh case statement
your_service)
    echo "Testing Your Service..."
    uv run pytest tests/test_your_service.py -v
    ;;
```

#### Multi-Service Integration (`services/multi_service.py`)

If integrating with the multi-service architecture:

```python
from services.your_service import YourService

@bentoml.service()
class MultiService:
    def __init__(self):
        self.your_service = YourService()
    
    @bentoml.api
    def your_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        return self.your_service.your_endpoint(request)
```

## Updating Existing Services

### Code Changes

1. **Service Logic**: Update `services/your_service.py`
2. **Utilities**: Modify `utils/your_service_utils.py`
3. **Models**: Update Pydantic models for request/response
4. **Dependencies**: Update `config/bentofiles/your-service.yaml`

### Testing Updates

1. **Unit Tests**: Update `tests/test_your_service.py`
2. **Integration Tests**: Add new test scenarios
3. **Test Data**: Update test fixtures if needed

### Documentation Updates

1. **Service Documentation**: Update `docs/services/your-service.md`
2. **Main README**: Update `README.md` with new features
3. **Claude Instructions**: Update `CLAUDE.md` with new endpoints/usage
4. **API Documentation**: Update endpoint examples

### Script Updates

1. **Endpoint Testing**: Update `scripts/endpoint.sh` examples
2. **Health Checks**: Update `scripts/health.sh` if needed  
3. **Test Scripts**: Update `scripts/test.sh` with new test cases

## File Checklist for New Services

### Required Files
- [ ] `services/your_service.py` - Main service implementation
- [ ] `config/bentofiles/your-service.yaml` - Bento build configuration
- [ ] `tests/test_your_service.py` - Unit and integration tests

### Optional Files (as needed)
- [ ] `utils/your_service_utils.py` - Utility functions
- [ ] `docs/services/your-service.md` - Service-specific documentation

### Files to Update
- [ ] `README.md` - Add service to main documentation
- [ ] `CLAUDE.md` - Update with new endpoints and usage examples
- [ ] `scripts/build_services.sh` - Add build command
- [ ] `scripts/test.sh` - Add test option
- [ ] `services/multi_service.py` - Integrate with multi-service (if applicable)
- [ ] `pyproject.toml` - Add new dependencies

### Documentation Files to Update/Create
- [ ] `docs/services/your-service.md` - Detailed service guide
- [ ] `docs/README.md` - Update documentation index
- [ ] Service-specific configuration or troubleshooting docs

## Best Practices

### Service Design
- Use Pydantic models for request/response validation
- Include proper error handling and logging
- Follow BentoML 1.4+ modern API patterns
- Design for scalability and resource management

### Testing Strategy
- Write both unit and integration tests
- Mock external dependencies in unit tests
- Test actual service startup in integration tests
- Use pytest markers (`@pytest.mark.slow`) for long-running tests

### Documentation
- Document all API endpoints with examples
- Include troubleshooting common issues
- Update main project documentation
- Keep Claude instructions current for AI assistance

### Configuration Management
- Use YAML-based Bento configurations
- Specify resource requirements clearly
- Include all necessary dependencies
- Use environment variables for secrets/config

## Example Service Templates

See existing services for patterns:
- `services/example_service.py` - Basic service template
- `services/stable_diffusion_service.py` - Model-heavy service
- `services/llava_service.py` - Vision-language service
- `services/whisper_service.py` - Audio processing service
- `services/upscaler_service.py` - Image processing service

## Integration with Development Workflow

1. **Development**: Use `./scripts/start.sh` for local development
2. **Testing**: Use `./scripts/test.sh --service your_service`
3. **Health Check**: Use `./scripts/health.sh` to verify service status
4. **Endpoint Testing**: Use `./scripts/endpoint.sh your_endpoint '{}'`
5. **Building**: Use `./scripts/build_services.sh` for production builds

This structure ensures consistency across services and makes it easy to maintain, test, and deploy BentoML services in this project.