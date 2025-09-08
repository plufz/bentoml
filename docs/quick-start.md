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

### Option A: Simple Example Service (Enhanced)
```bash
# Build with new enhanced build system
./scripts/build_services.sh example

# Or see all available services
./scripts/build_services.sh --list

# Serve
./scripts/run_bentoml.sh serve hello_service:latest

# Test with enhanced testing
./scripts/test.sh --service example
./scripts/endpoint.sh hello '{"name": "Quick Start"}'
```

### Option B: Stable Diffusion (Text → Image)
```bash
# Build with enhanced build system
./scripts/build_services.sh stable-diffusion

# Serve (will download ~4GB model on first run)
./scripts/run_bentoml.sh serve stable_diffusion_service:latest

# Test with enhanced endpoint script
./scripts/endpoint.sh generate_image '{"prompt": "a cute cat", "width": 512, "height": 512}'

# Or test with service-specific tests
./scripts/test.sh --service stable_diffusion
```

### Option C: Photo Upscaler (AI Image Enhancement)
```bash
# Build with enhanced build system
./scripts/build_services.sh upscaler

# Serve (will download ~100MB models on first run)
./scripts/run_bentoml.sh serve upscaler_service:latest

# Test with endpoint script
./scripts/endpoint.sh upscale_url '{"url": "https://plufz.com/test-assets/test-office.jpg", "scale_factor": 2.0}'

# Or run service tests
./scripts/test.sh --service upscaler
```

### Option D: LLaVA (Image Analysis)
```bash
# Build with enhanced build system
./scripts/build_services.sh llava

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

- **[Stable Diffusion Guide](services/stable-diffusion.md)** - Learn text-to-image generation
- **[LLaVA Guide](services/llava-service.md)** - Learn image analysis with structured output
- **[Photo Upscaler Guide](services/photo-upscaler.md)** - Learn AI-powered image upscaling
- **[Testing Guide](testing.md)** - Comprehensive testing approaches
- **[Configuration](configuration.md)** - Customize your setup

## Need Help?

Check the **[Troubleshooting Guide](troubleshooting.md)** for common issues and solutions.