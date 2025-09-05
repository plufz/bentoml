# BentoML Local Setup (macOS - No Docker) with UV

This repository contains a standard base setup of BentoML configured to run locally on macOS without Docker, using UV for fast Python package management.

## Quick Start

1. **Setup Environment**
   ```bash
   ./scripts/setup_env.sh
   ```

2. **Check Setup**
   ```bash
   ./scripts/check_setup.sh
   ```

3. **Build Example Service**
   ```bash
   ./scripts/run_bentoml.sh build services/example_service.py
   ```

4. **Run Service**
   ```bash
   ./scripts/run_bentoml.sh serve hello_service:latest
   ```

5. **Test Service**
   ```bash
   ./scripts/test_service.sh
   ```

## Files Overview

| File | Purpose |
|------|---------|
| `scripts/setup_env.sh` | Installs UV and sets up Python environment with dependencies |
| `scripts/run_bentoml.sh` | Script to build and serve BentoML services using UV |
| `scripts/check_setup.sh` | Verifies setup is working correctly |
| `scripts/test_service.sh` | Automated testing script |
| `pyproject.toml` | UV project configuration with dependencies |
| `bentoml_config.yaml` | BentoML configuration optimized for local development |
| `.env.example` | Environment variables template |
| `services/example_service.py` | Simple example service for testing |

## Usage Examples

### Building a Service
```bash
./scripts/run_bentoml.sh build my_service.py
```

### Serving a Service
```bash
./scripts/run_bentoml.sh serve my_service:latest
```

### Listing Available Services
```bash
./scripts/run_bentoml.sh list
```

### Running Tests
```bash
./scripts/test_service.sh test
./scripts/test_service.sh load 20  # Load test with 20 requests
```

## Configuration

The setup uses UV for dependency management and `bentoml_config.yaml` for BentoML configuration:

### UV Configuration (`pyproject.toml`)
- **Dependencies**: BentoML, FastAPI, Pandas, NumPy, Scikit-learn
- **Dev Dependencies**: pytest, black, isort, ruff, jupyter
- **Python**: 3.8+ compatible

### BentoML Configuration
- **Server**: Runs on `127.0.0.1:3000`
- **Workers**: 1 (suitable for local testing)
- **Storage**: Local filesystem in `./bentos` and `./models`
- **Development**: Auto-reload enabled, Swagger UI enabled

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Accessing Your Service

Once running:
- **API**: http://127.0.0.1:3000
- **Swagger UI**: http://127.0.0.1:3000/docs
- **Health Check**: http://127.0.0.1:3000/healthz
- **Metrics**: http://127.0.0.1:3000/metrics

## UV Commands

Common UV commands for development:

```bash
uv add <package>        # Add a dependency
uv add --dev <package>  # Add a dev dependency  
uv remove <package>     # Remove a dependency
uv sync                 # Sync environment with lockfile
uv run <command>        # Run command in UV environment
uv shell                # Activate UV shell (optional)
uv lock                 # Update lockfile
```

## Troubleshooting

### UV Not Found
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Port 3000 in Use
```bash
lsof -i :3000  # Find what's using the port
# Kill the process or change port in bentoml_config.yaml
```

### Missing Dependencies
```bash
./scripts/setup_env.sh  # Reinstall dependencies
./scripts/check_setup.sh  # Verify installation
```

### Service Won't Start
```bash
uv sync  # Sync dependencies
uv run bentoml list  # Check if service is built
./scripts/check_setup.sh  # Verify setup
```