# Quick Start Guide

Get up and running with BentoML services in 5 minutes.

## Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.8+
- Internet connection for model downloads

## 1. Setup Environment

```bash
./scripts/setup_env.sh
```

This installs UV package manager and all dependencies.

## 2. Verify Setup

```bash
./scripts/check_setup.sh
```

Should show all green checkmarks ✅.

## 3. Build and Run a Service

### Option A: Simple Example Service
```bash
# Build
./scripts/run_bentoml.sh build services/example_service.py

# Serve
./scripts/run_bentoml.sh serve hello_service:latest

# Test
./scripts/test_service.sh
```

### Option B: Stable Diffusion (Text → Image)
```bash
# Build
BENTOFILE=bentofile_sd.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py

# Serve (will download ~4GB model on first run)
./scripts/run_bentoml.sh serve stable_diffusion_service:latest

# Test
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"request": {"prompt": "a cute cat"}}' \
  | jq '.success'
```

### Option C: Photo Upscaler (AI Image Enhancement)
```bash
# Build
./scripts/run_bentoml.sh build services/upscaler_service.py

# Serve (will download ~100MB models on first run)
./scripts/run_bentoml.sh serve upscaler_service:latest

# Test
./scripts/endpoint.sh upscale_url '{"url": "https://plufz.com/test-assets/test-office.jpg", "scale_factor": 2.0}'
```

### Option D: LLaVA (Image Analysis)
```bash
# Build  
BENTOFILE=config/bentofiles/llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py

# Serve (will download ~13GB model on first run)
./scripts/run_bentoml.sh serve llava_service:latest

# Test
./scripts/test_llava.sh health
```

### 🚀 Recommended: Multi-Service (All Services at Once)
```bash
# Build all services
./scripts/build_services.sh

# Serve all services together
./scripts/start.sh

# Test all services
./scripts/test.sh --all
```

This gives you **12 endpoints in a single service**:
- Hello service, Stable Diffusion, LLaVA, Whisper, Photo Upscaler
- System endpoints (health, info)

## 4. Access Your Service

Once running:
- **API**: http://127.0.0.1:3000
- **Swagger UI**: http://127.0.0.1:3000/docs  
- **Health Check**: http://127.0.0.1:3000/health

## Next Steps

- **[Stable Diffusion Guide](stable-diffusion.md)** - Learn text-to-image generation
- **[LLaVA Guide](llava-service.md)** - Learn image analysis with structured output
- **[Photo Upscaler Guide](photo-upscaler.md)** - Learn AI-powered image upscaling
- **[Testing Guide](testing.md)** - Comprehensive testing approaches
- **[Configuration](configuration.md)** - Customize your setup

## Need Help?

Check the **[Troubleshooting Guide](troubleshooting.md)** for common issues and solutions.