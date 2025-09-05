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
- `bentoml[io]>=1.2.0` - Main framework
- `fastapi>=0.100.0` - API framework  
- `pandas`, `numpy`, `scikit-learn` - Data science libraries
- `uvicorn[standard]` - ASGI server
- `diffusers>=0.21.0` - Hugging Face Diffusers for Stable Diffusion
- `transformers>=4.25.0` - Transformers library
- `torch>=2.0.0`, `torchvision>=0.15.0` - PyTorch with MPS support
- `accelerate>=0.20.0` - Model acceleration
- `pillow>=9.0.0` - Image processing
- `llama-cpp-python>=0.2.0` - Fast GGUF model inference for LLaVA
- `pydantic>=2.0.0` - Data validation
- `jsonschema>=4.0.0` - JSON schema validation

## Testing

- All scripts automatically set PATH for UV
- Health check: `POST /health` with `{}`
- Example endpoint: `POST /hello` with `{"request": {"name": "value"}}`
- Web interface available at service root URL

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