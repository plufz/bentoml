# BentoML Local Setup - Project Context

This is a BentoML local development setup configured to run on macOS without Docker, using UV for fast Python package management.

## Project Structure

- **Language**: Python 3.8+
- **Package Manager**: UV (replaces pip/venv)
- **Framework**: BentoML 1.4+ with modern API
- **Configuration**: YAML-based configuration for local development

## Key Files

- `setup_env.sh` - Installs UV and sets up environment
- `pyproject.toml` - UV project configuration with dependencies  
- `bentoml_config.yaml` - BentoML server configuration
- `run_bentoml.sh` - Script to build/serve BentoML services
- `check_setup.sh` - Verifies installation and configuration
- `test_service.sh` - Automated testing script
- `example_service.py` - Example service using modern BentoML API

## Development Workflow

1. **Setup**: Run `./setup_env.sh` to install UV and dependencies
2. **Verify**: Run `./check_setup.sh` to confirm setup
3. **Build**: Use `./run_bentoml.sh build <service.py>` to build services
4. **Serve**: Use `./run_bentoml.sh serve <service:tag>` to run services
5. **Test**: Use `./test_service.sh` for automated testing

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