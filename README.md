# BentoML Local Setup (macOS - No Docker) with UV

🚀 **Production-ready AI services** running locally on macOS with Apple Silicon optimization and UV package management.

## ✨ What This Gives You

- **🎨 Stable Diffusion Service** - Generate images from text prompts
- **👁️ LLaVA Vision Service** - Analyze images with structured JSON output
- **🎯 Whisper Audio Service** - Transcribe audio files and URLs
- **📸 Photo Upscaler Service** - AI-powered photo upscaling with Real-ESRGAN
- **🧠 RAG Service** - Document ingestion and question-answering with retrieval-augmented generation
- **⚡ Apple Silicon Optimized** - MPS acceleration for M1/M2/M3 Macs
- **📦 UV Package Management** - Lightning-fast dependency resolution
- **🔧 Zero Docker Required** - Pure Python with BentoML 1.4+

## 🏃‍♀️ Quick Start

```bash
# 1. Setup (installs UV + dependencies)
./scripts/setup_env.sh

# 2. Verify setup
./scripts/check_setup.sh

# 3. Build and start all services
./scripts/build_services.sh
./scripts/start.sh

# 4. Check service health
./scripts/health.sh
```

Then visit: **http://127.0.0.1:3000/docs** for the interactive API!

## 🍺 Homebrew Service (macOS)

To run as a system service using Homebrew:

```bash
# 1. Install as a Homebrew service
brew install --formula ./config/brew-bentoml-service.rb

# 2. Start the service
brew services start bentoml-multiservice

# 3. Check service status
brew services list | grep bentoml

# 4. Stop the service
brew services stop bentoml-multiservice

# 5. View service logs
tail -f /opt/homebrew/var/log/bentoml-multiservice.log
```

The service will automatically start on boot and restart if it crashes.

## 🎯 Available Services

| Service | What it does | Build & Serve |
|---------|-------------|---------------|
| **Stable Diffusion** | Text → Image generation | `BENTOFILE=config/bentofiles/stable-diffusion.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py` |
| **LLaVA Vision** | Image + Text → JSON analysis | `BENTOFILE=config/bentofiles/llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py` |
| **Whisper Audio** | Audio → Text transcription | `BENTOFILE=config/bentofiles/whisper.yaml ./scripts/run_bentoml.sh build services/whisper_service.py` |
| **Photo Upscaler** | Image → AI upscaled image | `BENTOFILE=config/bentofiles/upscaler.yaml ./scripts/run_bentoml.sh build services/upscaler_service.py` |
| **RAG Service** | Document ingestion + Q&A | `BENTOFILE=config/bentofiles/rag.yaml ./scripts/run_bentoml.sh build services/rag_service.py` |
| **Example** | Simple API for testing | `./scripts/run_bentoml.sh build services/example_service.py` |

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

# Test LLaVA image analysis (base64/bytes)
./scripts/endpoint.sh analyze_image '{"image": "base64...", "prompt": "What is in this image?"}'

# Test LLaVA image analysis from URL
./scripts/endpoint.sh analyze_image_url '{"image_url": "https://plufz.com/test-assets/test-office.jpg", "prompt": "What is in this image?"}'

# Test Whisper audio transcription from URL
./scripts/endpoint.sh transcribe_url '{"url": "https://plufz.com/test-assets/test-english.mp3"}'

# Test Whisper audio transcription from file (requires curl for file upload)
curl -X POST http://127.0.0.1:3000/transcribe_file \
  -F "audio_file=@./test-assets/test-english.mp3"

# Test Photo Upscaler from URL
./scripts/endpoint.sh upscale_url '{"url": "https://plufz.com/test-assets/test-office.jpg", "scale_factor": 2.0}'

# Test Photo Upscaler from file (requires curl for file upload)
curl -X POST http://127.0.0.1:3000/upscale_file \
  -F "image_file=@./test-assets/test-upscale.jpg" \
  -F "scale_factor=2.5" \
  -F "output_format=PNG"

# Test RAG document ingestion (text)
./scripts/endpoint.sh rag_ingest_text '{"text": "This is a test document about AI.", "metadata": {"source": "test"}}'

# Test RAG query
./scripts/endpoint.sh rag_query '{"query": "What is AI?", "max_tokens": 512}'

# Use custom host/port and verbose output
./scripts/endpoint.sh health '{}' --host localhost --port 3001 --verbose

# Get help with available endpoints
./scripts/endpoint.sh --help
```

**Note**: Image endpoints (generate_image, upscale_*) automatically extract base64 image data and save actual image files to `endpoint_images/` directory, providing clean JSON output.

## 📚 Complete Documentation

**📖 [Full Documentation in `docs/`](docs/README.md)**

### 🚀 Getting Started
- **[Quick Start Guide](docs/quick-start.md)** - Up and running in 5 minutes
- **[Installation & Setup](docs/installation.md)** - Detailed installation
- **[Configuration](docs/configuration.md)** - Customize your setup

### 🤖 AI Services  
- **[Stable Diffusion Service](docs/services/stable-diffusion.md)** - Text-to-image generation
- **[LLaVA Service](docs/services/llava-service.md)** - Vision-language analysis
- **[Testing Guide](docs/testing.md)** - pytest test suite and legacy scripts

### 🔧 Advanced
- **[API Reference](docs/api-reference.md)** - Complete API docs
- **[Utilities Documentation](docs/utilities.md)** - Reusable components
- **[Troubleshooting](docs/troubleshooting.md)** - Fix common issues

## 🏗️ Project Structure

```
bentoml-project/
├── docs/                 # 📚 Complete documentation
├── scripts/             # 🛠️ Management scripts  
│   ├── run_bentoml.sh   # Build & serve services
│   ├── check_setup.sh   # Verify installation
│   └── test_*.sh        # Legacy test scripts (deprecated)
├── tests/               # 🧪 Pytest test suite (recommended)
│   ├── conftest.py      # Shared test fixtures
│   ├── test_example_service.py
│   ├── test_llava_service.py
│   ├── test_stable_diffusion_service.py
│   ├── test_whisper_service.py
│   └── test_multi_service.py
├── services/            # 🤖 AI services
│   ├── stable_diffusion_service.py
│   ├── llava_service.py
│   ├── whisper_service.py
│   ├── multi_service.py
│   └── example_service.py
├── utils/               # 🔧 Reusable utilities
│   ├── stable_diffusion/ # SD pipeline utilities
│   └── llava/           # LLaVA utilities
└── config/              # ⚙️ Service configurations
    ├── bentoml.yaml     # Server config
    └── bentofiles/      # Service build configs
```

## 🎨 Example Usage

### Generate an Image
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"request": {"prompt": "a cute cat in a garden"}}' \
  | jq -r '.image' | base64 -d > cat.png
```

### Analyze an Image (PNG)
```bash  
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "What objects are in this image?",
      "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
      "json_schema": {
        "type": "object", 
        "properties": {
          "objects": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }'
```

### Analyze a JPEG Image
```bash
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "Describe what you see in this image",
      "image": "https://httpbin.org/image/jpeg"
    }
  }' | jq '.response'
```

## 🧪 Testing Your Services

### Recommended: pytest (Official BentoML Testing)

**Using the test script (easiest):**
```bash
./scripts/test.sh                    # Fast tests only
./scripts/test.sh --all              # All tests including slow integration  
./scripts/test.sh --coverage         # Fast tests with coverage
./scripts/test.sh --service example  # Test specific service
./scripts/test.sh --unit             # Unit tests only
./scripts/test.sh --help             # Show all options
```

**Direct UV commands:**
```bash
# Run all fast tests (unit + behavior)
uv run pytest -m "not slow"

# Run all tests including slow integration tests
uv run pytest

# Run specific service tests
uv run pytest tests/test_example_service.py

# Run with coverage report
uv run pytest --cov=. --cov-report=term-missing
```

### Legacy: Bash Scripts (Deprecated)
```bash
./scripts/test_service.sh          # Basic service test
./scripts/test_llava.sh           # LLaVA service test
./scripts/test_multi_service.sh   # Multi-service test
```

**✨ The pytest suite includes:**
- **Unit Tests** - Test individual methods with mocked dependencies
- **Integration Tests** - Test actual service startup and API endpoints
- **HTTP Behavior Tests** - Test response formats and error handling
- **62% Code Coverage** - Comprehensive test coverage across all services

## ⚡ Key Features

- **🍎 Apple Silicon Optimized** - Uses MPS backend with float32 for stability
- **💾 External Drive Support** - Custom HF_HOME for model storage
- **🔄 Auto Device Detection** - MPS → CUDA → CPU fallback
- **📊 Structured Output** - JSON schema validation for LLaVA
- **🧪 Professional Testing** - pytest-based test suite following BentoML best practices
- **📈 Modular Architecture** - Reusable utilities for easy extension

## 🛠️ Requirements

- **macOS** (Apple Silicon recommended)
- **Python 3.8+**  
- **8GB+ RAM** (16GB for LLaVA)
- **20GB+ storage** (for AI models)

## 🚨 Need Help?

- **Quick Issues**: Check **[Troubleshooting Guide](docs/troubleshooting.md)**
- **Service Docs**: See **[docs/](docs/README.md)** for complete guides  
- **Getting Started**: Follow **[Quick Start](docs/quick-start.md)**

---

**Built with** ❤️ **using BentoML, UV, and optimized for Apple Silicon**