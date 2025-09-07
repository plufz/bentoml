# Testing Guide

This guide covers comprehensive testing strategies for BentoML services in this project using pytest and official BentoML testing patterns.

## Testing Architecture

Tests are organized in the `tests/` directory with the following structure:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_example_service.py  # Example service tests
├── test_llava_service.py    # LLaVA service tests
├── test_stable_diffusion_service.py
├── test_whisper_service.py
├── test_upscaler_service.py
├── test_rag_service.py
└── test_multi_service.py    # Multi-service integration tests
```

## Test Types

### 1. Unit Tests
Test individual service methods with mocked dependencies.

### 2. Integration Tests  
Test actual service startup and API endpoints (marked as `@pytest.mark.slow`).

### 3. HTTP Behavior Tests
Test API response formats and error handling.

### 4. End-to-End Tests
Test full service workflows including model loading.

## Creating Service Tests

### Basic Test Structure

**File**: `tests/test_your_service.py`

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from services.your_service import YourService, YourServiceRequest, YourServiceResponse

class TestYourServiceUnit:
    """Unit tests for YourService - fast tests with mocked dependencies"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        # Mock heavy dependencies to avoid loading
        with patch('services.your_service.YourService._load_model'):
            self.service = YourService()
    
    def test_service_initialization(self):
        """Test service initializes correctly"""
        assert self.service is not None
        assert hasattr(self.service, '_initialized')
    
    def test_your_endpoint_success(self):
        """Test successful endpoint execution"""
        # Arrange
        request = YourServiceRequest(
            input_field="test input",
            optional_field="custom value"
        )
        
        # Act
        response = self.service.your_endpoint(request)
        
        # Assert
        assert isinstance(response, YourServiceResponse)
        assert response.status == "success"
        assert "test input" in response.result
        assert response.metadata is not None
    
    def test_your_endpoint_with_defaults(self):
        """Test endpoint with default parameters"""
        request = YourServiceRequest(input_field="test")
        response = self.service.your_endpoint(request)
        
        assert response.status == "success"
        assert response.result is not None
    
    def test_your_endpoint_error_handling(self):
        """Test endpoint error handling"""
        # Mock an exception in processing
        with patch.object(self.service, '_process_data', side_effect=Exception("Test error")):
            request = YourServiceRequest(input_field="test")
            response = self.service.your_endpoint(request)
            
            assert response.status == "error"
            assert "error" in response.metadata
    
    def test_input_validation(self):
        """Test input validation logic"""
        # Test empty input
        request = YourServiceRequest(input_field="")
        response = self.service.your_endpoint(request)
        
        # Should handle gracefully
        assert response is not None
    
    @pytest.mark.parametrize("input_data,expected", [
        ("hello", "Processed: hello"),
        ("WORLD", "Processed: WORLD"),
        ("123", "Processed: 123"),
    ])
    def test_process_data_variations(self, input_data, expected):
        """Test processing with various inputs"""
        result = self.service._process_data(input_data, "default")
        assert expected in result

@pytest.mark.slow
class TestYourServiceIntegration:
    """Integration tests for YourService - slower tests with actual service startup"""
    
    @pytest.fixture(scope="class")
    def service_client(self):
        """Start actual service for integration testing"""
        import bentoml
        from bentoml.testing import get_client
        
        # This will start the actual service
        with get_client(YourService) as client:
            yield client
    
    @pytest.mark.timeout(30)
    def test_service_health(self, service_client):
        """Test service health endpoint"""
        response = service_client.get("/health")
        assert response.status_code == 200
    
    @pytest.mark.timeout(30)
    def test_api_endpoint_integration(self, service_client):
        """Test actual API endpoint over HTTP"""
        payload = {
            "request": {
                "input_field": "integration test",
                "optional_field": "test_value"
            }
        }
        
        response = service_client.post("/your_endpoint", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "metadata" in data
        assert data.get("status") == "success"
    
    @pytest.mark.timeout(30)
    def test_api_error_handling_integration(self, service_client):
        """Test API error handling over HTTP"""
        # Send malformed payload
        payload = {"invalid": "payload"}
        
        response = service_client.post("/your_endpoint", json=payload)
        
        # Should return error but not crash
        assert response.status_code in [400, 422, 500]
    
    @pytest.mark.timeout(60)
    def test_concurrent_requests(self, service_client):
        """Test service handles concurrent requests"""
        import concurrent.futures
        import requests
        
        def make_request():
            payload = {
                "request": {
                    "input_field": "concurrent test"
                }
            }
            return service_client.post("/your_endpoint", json=payload)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        for result in results:
            assert result.status_code == 200
```

### Advanced Testing Patterns

#### File Upload Testing

```python
import io
from PIL import Image

class TestFileProcessingService:
    def test_file_upload_endpoint(self, service_client):
        """Test file upload functionality"""
        # Create test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Upload file
        files = {"file": ("test.png", img_bytes, "image/png")}
        response = service_client.post("/process_file", files=files)
        
        assert response.status_code == 200
        assert "result" in response.json()
```

#### Async Service Testing

```python
class TestAsyncService:
    @pytest.mark.asyncio
    async def test_async_endpoint(self):
        """Test async service endpoint"""
        service = AsyncService()
        request = AsyncRequest(data="test")
        
        response = await service.async_endpoint(request)
        
        assert response.result is not None
    
    @pytest.mark.asyncio
    async def test_streaming_endpoint(self):
        """Test streaming endpoint"""
        service = AsyncService()
        request = AsyncRequest(data="stream test")
        
        results = []
        async for chunk in service.stream_endpoint(request):
            results.append(chunk)
        
        assert len(results) > 0
        assert "stream test" in results[0]
```

#### Model-Heavy Service Testing

```python
class TestModelService:
    @pytest.fixture(autouse=True)
    def mock_heavy_dependencies(self):
        """Mock heavy model loading for all tests"""
        with patch('torch.load'), \
             patch('transformers.AutoModel.from_pretrained'), \
             patch('diffusers.DiffusionPipeline.from_pretrained'):
            yield
    
    def test_model_service_without_loading(self):
        """Test service logic without loading actual models"""
        service = ModelService()
        # Test service methods that don't require actual model
```

## Test Configuration

### pytest Configuration

**File**: `pyproject.toml` (testing section)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "-m not slow"  # Skip slow tests by default
]
markers = [
    "slow: marks tests as slow (integration tests)",
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests", 
    "gpu: marks tests requiring GPU",
    "network: marks tests requiring network access"
]
timeout = 30  # Global timeout for tests
```

### Test Fixtures

**File**: `tests/conftest.py`

```python
import pytest
import os
import tempfile
from unittest.mock import patch

@pytest.fixture(scope="session")
def test_config():
    """Test configuration shared across all tests"""
    return {
        "test_mode": True,
        "mock_models": True,
        "timeout": 30
    }

@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_model_loading():
    """Mock heavy model loading operations"""
    with patch('torch.load'), \
         patch('transformers.AutoModel.from_pretrained'), \
         patch('diffusers.DiffusionPipeline.from_pretrained'):
        yield

@pytest.fixture(scope="class")
def example_service():
    """Example service instance for testing"""
    from services.example_service import ExampleService
    return ExampleService()

# Auto-use fixture to ensure test isolation
@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment before each test"""
    # Save original env vars
    original_env = dict(os.environ)
    
    yield
    
    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)
```

## Running Tests

### Using Test Script (Recommended)

```bash
# Run fast tests only (default)
./scripts/test.sh

# Run all tests including slow integration tests
./scripts/test.sh --all

# Run tests with coverage report
./scripts/test.sh --coverage

# Run tests for specific service
./scripts/test.sh --service your_service

# Run only unit tests
./scripts/test.sh --unit

# Show all test options
./scripts/test.sh --help
```

### Direct pytest Commands

```bash
# Run all fast tests
uv run pytest

# Run specific test file
uv run pytest tests/test_your_service.py

# Run specific test class
uv run pytest tests/test_your_service.py::TestYourServiceUnit

# Run specific test method
uv run pytest tests/test_your_service.py::TestYourServiceUnit::test_your_endpoint_success

# Run with verbose output
uv run pytest -v

# Run integration tests only
uv run pytest -m "slow"

# Run with coverage
uv run pytest --cov=services --cov-report=term-missing

# Run with custom timeout
uv run pytest --timeout=60
```

## Test Timeouts

### Configuration

- **Global default**: 30 seconds (set in `pyproject.toml`)
- **Custom timeouts**: Use `@pytest.mark.timeout(seconds)` decorator
- **Integration tests**: Typically 60-180 seconds for service startup
- **API tests**: 10-30 seconds for endpoint responses
- **Heavy processing**: Up to 120 seconds for model operations

### Timeout Examples

```python
@pytest.mark.timeout(60)
def test_slow_integration(self, service_client):
    """Test with custom timeout"""
    # Long-running integration test
    pass

@pytest.mark.timeout(120)
def test_model_inference(self):
    """Test requiring longer timeout for model operations"""
    # Model inference test
    pass
```

## Best Practices

### Test Organization

1. **Separate unit and integration tests** into different classes
2. **Use descriptive test names** that explain what is being tested
3. **Group related tests** using test classes
4. **Use fixtures** for common setup and teardown

### Mocking Strategy

1. **Mock heavy dependencies** (models, external APIs) in unit tests
2. **Use actual services** in integration tests
3. **Mock at the right level** - not too deep, not too shallow
4. **Verify mock interactions** when testing side effects

### Test Data

1. **Use small, focused test data** for fast tests
2. **Create realistic test scenarios** for integration tests  
3. **Test edge cases** and error conditions
4. **Use parameterized tests** for multiple input scenarios

### Error Testing

1. **Test expected errors** and their handling
2. **Test malformed inputs** and API misuse
3. **Verify error messages** are helpful
4. **Test recovery scenarios** where applicable

## Troubleshooting Tests

### Common Issues

1. **Port conflicts**: Services trying to bind to occupied ports
2. **Model loading timeouts**: Heavy models taking too long to load
3. **Resource constraints**: Tests running out of memory/CPU
4. **Flaky tests**: Tests that sometimes pass/fail

### Solutions

1. **Use test-specific ports**: Configure services with random ports
2. **Mock heavy operations**: Avoid loading actual models in tests
3. **Increase timeouts**: For legitimate slow operations
4. **Retry flaky tests**: Use pytest-rerunfailures plugin

### Debugging

```bash
# Run with debugging output
uv run pytest -s -vvv

# Run single test for debugging
uv run pytest tests/test_your_service.py::test_specific_method -s

# Use pdb for interactive debugging
uv run pytest --pdb

# Show test coverage gaps
uv run pytest --cov=services --cov-report=html
```

This comprehensive testing approach ensures service reliability and makes it easy to catch regressions during development.