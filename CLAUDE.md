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
- `bentofile_sd.yaml` - Configuration for Stable Diffusion service

## Development Workflow

1. **Setup**: Run `./scripts/setup_env.sh` to install UV and dependencies
2. **Verify**: Run `./scripts/check_setup.sh` to confirm setup
3. **Build**: Use `./scripts/run_bentoml.sh build <service.py>` to build services
4. **Serve**: Use `./scripts/run_bentoml.sh serve <service:tag>` to run services
5. **Test**: Use `./scripts/test_service.sh` for automated testing

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

## Testing

- All scripts automatically set PATH for UV
- Health check: `POST /health` with `{}`
- Example endpoint: `POST /hello` with `{"request": {"name": "value"}}`
- Web interface available at service root URL

## Notes

- UV is installed via official installer (curl method)
- Scripts include PATH setup automatically - no manual export needed
- Services built with `bentofile.yaml` configuration
- Modern Pydantic-based API (no deprecated `bentoml.io` imports)