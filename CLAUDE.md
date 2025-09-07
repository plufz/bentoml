# BentoML Local Setup - Project Context

This is a BentoML local development setup configured to run on macOS without Docker, using UV for fast Python package management.

## Project Structure

- **Language**: Python 3.8+
- **Package Manager**: UV (replaces pip/venv)
- **Framework**: BentoML 1.4+ with modern API
- **Configuration**: YAML-based configuration for local development

## Key Files

- `scripts/setup_env.sh` - Installs UV and sets up environment
- `scripts/run_bentoml.sh` - Script to build/serve BentoML services
- `scripts/start.sh` - Quick start script for multi-service
- `scripts/build_services.sh` - Build all services script
- `scripts/health.sh` - Check health of running BentoML service
- `scripts/endpoint.sh` - Test any service endpoint with JSON payloads
- `scripts/check_setup.sh` - Verifies installation and configuration
- `scripts/test_service.sh` - Automated testing script
- `pyproject.toml` - UV project configuration with dependencies  
- `config/bentoml.yaml` - BentoML server configuration
- `config/bentofiles/` - Service-specific Bento build configurations
- `services/example_service.py` - Example service using modern BentoML API
- `services/stable_diffusion_service.py` - Stable Diffusion image generation service
- `services/llava_service.py` - LLaVA vision-language service using llama-cpp-python
- `services/whisper_service.py` - Whisper audio transcription service
- `services/multi_service.py` - Multi-service composition with unified endpoints
- `scripts/test_llava.sh` - LLaVA service testing script
- `scripts/test_whisper.sh` - Whisper service testing script
- `scripts/test_multi_service.sh` - Multi-service comprehensive testing script

## Development Workflow

1. **Setup**: Run `./scripts/setup_env.sh` to install UV and dependencies
2. **Verify**: Run `./scripts/check_setup.sh` to confirm setup
3. **Build**: Use `./scripts/build_services.sh` to build all services
4. **Serve**: Use `./scripts/start.sh` to start the multi-service
5. **Health Check**: Use `./scripts/health.sh` to check service health
6. **Test**: Use service-specific test scripts for automated testing

### Quick Start Scripts

For convenience, use these scripts for common operations:
```bash
./scripts/start.sh           # Start the multi-service 
./scripts/build_services.sh  # Build all services
./scripts/health.sh          # Check health of running service
./scripts/endpoint.sh <endpoint> <json>  # Test any endpoint
```

### Testing Endpoints

Use the endpoint testing script for interactive API testing:

```bash
# Test health check
./scripts/endpoint.sh health '{}'

# Test hello service with custom name
./scripts/endpoint.sh hello '{"name": "BentoML"}'

# Test with empty payload (uses defaults)
./scripts/endpoint.sh hello '{}'

# Test Stable Diffusion image generation
./scripts/endpoint.sh generate_image '{"prompt": "A beautiful sunset", "width": 512, "height": 512}'

# Test LLaVA image analysis
./scripts/endpoint.sh analyze_image '{"image_data": "base64...", "query": "What is in this image?"}'

# Test Whisper audio transcription
./scripts/endpoint.sh transcribe_url '{"url": "https://example.com/audio.mp3"}'

# Use custom host/port and verbose output
./scripts/endpoint.sh health '{}' --host localhost --port 3001 --verbose

# Get help with available endpoints
./scripts/endpoint.sh --help
```

**Note**: The script automatically wraps payloads in BentoML's expected `{"request": {...}}` format for service endpoints, while system endpoints (health, info) use direct payloads.

### Running Services

Services are served using the module path format:
```bash
# Stable Diffusion service
./scripts/run_bentoml.sh serve services.stable_diffusion_service:StableDiffusionService

# LLaVA service  
./scripts/run_bentoml.sh serve services.llava_service:LLaVAService

# Example service
./scripts/run_bentoml.sh serve services.example_service:ExampleService
```

### Building Services

Build services into Bento packages:
```bash
# Build with default bentofile
./scripts/run_bentoml.sh build services/service_name.py

# Build with custom bentofile
BENTOFILE=config/bentofiles/stable-diffusion.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py
BENTOFILE=config/bentofiles/whisper.yaml ./scripts/run_bentoml.sh build services/whisper_service.py
BENTOFILE=config/bentofiles/llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py
```

### Multi-Service Architecture

For unified deployment, use the multi-service composition:
```bash
# Serve all services in a single unified endpoint
BENTOFILE=config/bentofiles/multi-service.yaml ./scripts/run_bentoml.sh serve services.multi_service:MultiService

# Test all services in the multi-service
./scripts/test_multi_service.sh
```

**Multi-Service Endpoints:**
- System: `/health`, `/info`
- Hello Service: `/hello`
- Stable Diffusion: `/generate_image`
- LLaVA: `/analyze_image`, `/analyze_structured`, `/analyze_url`, `/example_schemas`
- Whisper: `/transcribe_file`, `/transcribe_url`

**Total: 10 endpoints in a single service** 🚀

### Testing Services

Each service has its own test script:
```bash
./scripts/test_llava.sh          # Test LLaVA service
./scripts/test_service.sh        # Test general services
./scripts/test_multi_service.sh  # Test all services in multi-service
```

## BentoML Service Pattern

Services use modern BentoML API with Pydantic models:

```python
import bentoml
from pydantic import BaseModel

class RequestModel(BaseModel):
    field: str = "default"

@bentoml.service()
class MyService:
    @bentoml.api
    def endpoint(self, request: RequestModel) -> dict:
        return {"result": request.field}
```

## API Testing

Services expect nested JSON payloads:
- Endpoint: `POST /endpoint`
- Payload: `{"request": {"field": "value"}}`

## Configuration

- **Server**: Runs on `127.0.0.1:3000` by default
- **Storage**: Uses local filesystem (`./bentos`, `./models`)
- **Development**: Auto-reload enabled, web interface at root `/`
- **HuggingFace Cache**: Custom location at `/Volumes/Second/huggingface` (not default `~/.cache/huggingface`)

## Dependencies

Core dependencies managed by UV:
- `bentoml[io]>=1.4.0` - Main framework with modern API
- `fastapi>=0.100.0` - API framework  
- `pandas>=1.3.0`, `numpy>=1.21.0`, `scikit-learn>=1.0.0` - Data science libraries
- `uvicorn[standard]>=0.18.0` - ASGI server
- `diffusers>=0.25.0` - Hugging Face Diffusers for Stable Diffusion
- `transformers>=4.30.0` - Transformers library with latest models
- `torch>=2.1.0`, `torchvision>=0.16.0` - PyTorch with MPS support
- `accelerate>=0.25.0` - Model acceleration
- `pillow>=10.0.0` - Image processing
- `llama-cpp-python>=0.2.27` - Fast GGUF model inference for LLaVA
- `pydantic>=2.5.0` - Data validation with modern API
- `jsonschema>=4.17.0` - JSON schema validation
- `vllm>=0.3.0` - High-performance LLM serving (optional)
- `openai>=1.0.0` - OpenAI API client (for examples)
- `requests>=2.28.0` - HTTP client library

## Testing

### Pytest (Recommended - Official BentoML Testing)

**Framework**: `pytest` with comprehensive test coverage using official BentoML testing patterns.

**Test Structure**:
- `tests/` - Main test directory
- `tests/conftest.py` - Shared fixtures and configuration
- `tests/test_*.py` - Individual service test files

**Test Types**:
1. **Unit Tests** - Test individual service methods with mocked dependencies
2. **Integration Tests** - Test actual service startup and API endpoints  
3. **HTTP Behavior Tests** - Test API response formats and error handling
4. **End-to-End Tests** - Test full service workflows (marked as slow)

**Running Tests**:

*Using the test script (recommended):*
```bash
./scripts/test.sh                    # Fast tests only
./scripts/test.sh --all              # All tests including slow integration  
./scripts/test.sh --coverage         # Fast tests with coverage
./scripts/test.sh --service example  # Test specific service
./scripts/test.sh --unit             # Unit tests only
./scripts/test.sh --help             # Show all options
```

*Direct UV commands:*
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_example_service.py

# Run only fast tests (exclude slow integration tests)
uv run pytest -m "not slow"

# Run with coverage report
uv run pytest --cov=. --cov-report=term-missing

# Run specific test types
uv run pytest tests/test_example_service.py::TestHelloServiceUnit
```

**Test Configuration**:
- Configured in `pyproject.toml` with test discovery patterns
- Coverage reporting to console and HTML
- Slow tests marked with `@pytest.mark.slow` for optional skipping
- Automatic mock setup for services to avoid model loading during unit tests

**Timeout Configuration**:
- **Global default timeout**: 30 seconds per test (configured in `pyproject.toml`)
- **Custom timeouts**: Individual tests can override with `@pytest.mark.timeout(seconds)`
- **Integration test timeouts**: 
  - Service fixtures: 60-180 seconds (for model loading/startup)
  - API endpoint tests: 10-30 seconds 
  - Image processing tests: 120 seconds
- **Unit test timeouts**: Use global default (30 seconds)

**Important**: Integration tests may timeout when trying to start actual BentoML services. **Timeouts are test failures, not passing tests.** If integration tests consistently timeout:
- Check for port conflicts (services try to bind to specific ports)
- Verify BentoML dependencies are properly installed
- Consider running unit tests only with `-m "not slow"` flag for faster CI/development cycles
- Timeout errors indicate the service failed to start within the expected time window

**Available Test Files**:
- `test_example_service.py` - Hello service unit and integration tests
- `test_llava_service.py` - Vision-language service tests with image processing
- `test_stable_diffusion_service.py` - Image generation service tests
- `test_whisper_service.py` - Audio transcription service tests
- `test_multi_service.py` - Multi-service composition tests

### Legacy Bash Scripts (Deprecated)

- `./scripts/test_service.sh` - Basic service testing script
- `./scripts/test_llava.sh` - LLaVA service testing script  
- `./scripts/test_multi_service.sh` - Multi-service testing script
- Health check: Use `./scripts/health.sh` (instead of manual curl commands)
- Example endpoint: `POST /hello` with `{"request": {"name": "value"}}`
- Web interface available at service root URL

**Migration Note**: Bash scripts are kept for development convenience but pytest is the recommended testing approach following BentoML best practices.

## Documentation Index

Complete reference to all documentation files in the project:

### Root Documentation
- [docs/README.md](docs/README.md) - Documentation overview
- [docs/installation.md](docs/installation.md) - Installation guide
- [docs/quick-start.md](docs/quick-start.md) - Quick start guide
- [docs/testing.md](docs/testing.md) - Testing documentation
- [docs/troubleshooting.md](docs/troubleshooting.md) - Troubleshooting guide
- [docs/stable-diffusion.md](docs/stable-diffusion.md) - Stable Diffusion service guide
- [docs/llava-service.md](docs/llava-service.md) - LLaVA service guide

### BentoML Core Documentation
- [docs/bentoml/README.md](docs/bentoml/README.md) - BentoML documentation overview
- [docs/bentoml/overview.md](docs/bentoml/overview.md) - BentoML overview
- [docs/bentoml/quickstart.md](docs/bentoml/quickstart.md) - BentoML quickstart
- [docs/bentoml/services.md](docs/bentoml/services.md) - Services documentation
- [docs/bentoml/input-output-types.md](docs/bentoml/input-output-types.md) - I/O types
- [docs/bentoml/deployment.md](docs/bentoml/deployment.md) - Deployment guide

### Getting Started
- [docs/bentoml/getting-started/model-composition.md](docs/bentoml/getting-started/model-composition.md) - Model composition
- [docs/bentoml/getting-started/async-task-queues.md](docs/bentoml/getting-started/async-task-queues.md) - Async task queues

### Guides
- [docs/bentoml/guides/error-handling.md](docs/bentoml/guides/error-handling.md) - Error handling
- [docs/bentoml/guides/gradio-integration.md](docs/bentoml/guides/gradio-integration.md) - Gradio integration

### API Reference
- [docs/bentoml/api/client.md](docs/bentoml/api/client.md) - Client API
- [docs/bentoml/api/cli.md](docs/bentoml/api/cli.md) - CLI reference
- [docs/bentoml/api/types.md](docs/bentoml/api/types.md) - Types and signatures
- [docs/bentoml/api/exceptions.md](docs/bentoml/api/exceptions.md) - Exception handling
- [docs/bentoml/api/configurations.md](docs/bentoml/api/configurations.md) - Service configurations
- [docs/bentoml/api/bento-build-options.md](docs/bentoml/api/bento-build-options.md) - Build options

### Framework Integrations
- [docs/bentoml/api/frameworks/README.md](docs/bentoml/api/frameworks/README.md) - Frameworks overview
- [docs/bentoml/api/frameworks/pytorch.md](docs/bentoml/api/frameworks/pytorch.md) - PyTorch integration
- [docs/bentoml/api/frameworks/tensorflow.md](docs/bentoml/api/frameworks/tensorflow.md) - TensorFlow integration
- [docs/bentoml/api/frameworks/sklearn.md](docs/bentoml/api/frameworks/sklearn.md) - Scikit-learn integration
- [docs/bentoml/api/frameworks/transformers.md](docs/bentoml/api/frameworks/transformers.md) - Transformers integration
- [docs/bentoml/api/frameworks/diffusers.md](docs/bentoml/api/frameworks/diffusers.md) - Diffusers integration
- [docs/bentoml/api/frameworks/onnx.md](docs/bentoml/api/frameworks/onnx.md) - ONNX integration
- [docs/bentoml/api/frameworks/ray.md](docs/bentoml/api/frameworks/ray.md) - Ray integration
- [docs/bentoml/api/frameworks/easyocr.md](docs/bentoml/api/frameworks/easyocr.md) - EasyOCR integration
- [docs/bentoml/api/frameworks/detectron.md](docs/bentoml/api/frameworks/detectron.md) - Detectron2 integration

### Examples
- [docs/bentoml/examples/README.md](docs/bentoml/examples/README.md) - Examples overview
- [docs/bentoml/examples/overview.md](docs/bentoml/examples/overview.md) - Complete examples catalog
- [docs/bentoml/examples/vllm.md](docs/bentoml/examples/vllm.md) - vLLM serving
- [docs/bentoml/examples/function-calling.md](docs/bentoml/examples/function-calling.md) - Function calling agents
- [docs/bentoml/examples/langgraph.md](docs/bentoml/examples/langgraph.md) - LangGraph agents
- [docs/bentoml/examples/shieldgemma.md](docs/bentoml/examples/shieldgemma.md) - AI safety with ShieldGemma
- [docs/bentoml/examples/rag.md](docs/bentoml/examples/rag.md) - RAG implementation
- [docs/bentoml/examples/sdxl-turbo.md](docs/bentoml/examples/sdxl-turbo.md) - SDXL Turbo image generation
- [docs/bentoml/examples/comfyui.md](docs/bentoml/examples/comfyui.md) - ComfyUI workflows
- [docs/bentoml/examples/controlnet.md](docs/bentoml/examples/controlnet.md) - ControlNet integration

### Example Code
- [docs/bentoml/example-code/rag/README.md](docs/bentoml/example-code/rag/README.md) - RAG examples overview
- [docs/bentoml/example-code/rag/00-simple-local-rag/README.md](docs/bentoml/example-code/rag/00-simple-local-rag/README.md) - Simple local RAG
- [docs/bentoml/example-code/rag/01-simple-rag/README.md](docs/bentoml/example-code/rag/01-simple-rag/README.md) - Basic RAG
- [docs/bentoml/example-code/rag/02-custom-embedding/README.md](docs/bentoml/example-code/rag/02-custom-embedding/README.md) - Custom embeddings
- [docs/bentoml/example-code/rag/03-custom-llm/README.md](docs/bentoml/example-code/rag/03-custom-llm/README.md) - Custom LLM
- [docs/bentoml/example-code/rag/04a-vector-store-milvus/README.md](docs/bentoml/example-code/rag/04a-vector-store-milvus/README.md) - Milvus vector store

## Git Workflow

When making commits, use combined add && commit commands:
```bash
git add file1 file2 && git commit -m "commit message"
```

For commit attribution, use only: "Generated with Claude"

## Notes

- UV is installed via official installer (curl method)
- Scripts include PATH setup automatically - no manual export needed
- Services built with organized configuration files in `config/bentofiles/`
- Modern Pydantic-based API (no deprecated `bentoml.io` imports)

## Claude Instructions

**IMPORTANT**: When checking BentoML service health, always use `./scripts/health.sh` instead of manual curl commands. This ensures consistent health check behavior and proper formatting.