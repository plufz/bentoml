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
- `scripts/check_setup.sh` - Verifies installation and configuration
- `scripts/test_service.sh` - Automated testing script
- `pyproject.toml` - UV project configuration with dependencies  
- `bentoml_config.yaml` - BentoML server configuration
- `services/example_service.py` - Example service using modern BentoML API
- `services/stable_diffusion_service.py` - Stable Diffusion image generation service
- `services/llava_service.py` - LLaVA vision-language service using llama-cpp-python
- `scripts/test_llava.sh` - LLaVA service testing script
- `bentofile_sd.yaml` - Configuration for Stable Diffusion service

## Development Workflow

1. **Setup**: Run `./scripts/setup_env.sh` to install UV and dependencies
2. **Verify**: Run `./scripts/check_setup.sh` to confirm setup
3. **Build**: Use `./scripts/run_bentoml.sh build <service.py>` to build services
4. **Serve**: Use `./scripts/run_bentoml.sh serve <module.service:ServiceClass>` to run services
5. **Test**: Use service-specific test scripts for automated testing

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
BENTOFILE=bentofile_custom.yaml ./scripts/run_bentoml.sh build services/service_name.py
```

### Testing Services

Each service has its own test script:
```bash
./scripts/test_llava.sh        # Test LLaVA service
./scripts/test_service.sh      # Test general services
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

- All scripts automatically set PATH for UV
- Health check: `POST /health` with `{}`
- Example endpoint: `POST /hello` with `{"request": {"name": "value"}}`
- Web interface available at service root URL

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
- Services built with `bentofile.yaml` configuration
- Modern Pydantic-based API (no deprecated `bentoml.io` imports)