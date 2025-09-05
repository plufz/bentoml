# BentoML Local Setup (macOS - No Docker) with UV

🚀 **Production-ready AI services** running locally on macOS with Apple Silicon optimization and UV package management.

## ✨ What This Gives You

- **🎨 Stable Diffusion Service** - Generate images from text prompts
- **👁️ LLaVA Vision Service** - Analyze images with structured JSON output  
- **⚡ Apple Silicon Optimized** - MPS acceleration for M1/M2/M3 Macs
- **📦 UV Package Management** - Lightning-fast dependency resolution
- **🔧 Zero Docker Required** - Pure Python with BentoML 1.4+

## 🏃‍♀️ Quick Start

```bash
# 1. Setup (installs UV + dependencies)
./scripts/setup_env.sh

# 2. Verify setup
./scripts/check_setup.sh

# 3. Generate your first image
BENTOFILE=bentofile_sd.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py
./scripts/run_bentoml.sh serve stable_diffusion_service:latest
```

Then visit: **http://127.0.0.1:3000/docs** for the interactive API!

## 🎯 Available Services

| Service | What it does | Build & Serve |
|---------|-------------|---------------|
| **Stable Diffusion** | Text → Image generation | `BENTOFILE=bentofile_sd.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py` |
| **LLaVA Vision** | Image + Text → JSON analysis | `BENTOFILE=bentofile_llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py` |
| **Example** | Simple API for testing | `./scripts/run_bentoml.sh build services/example_service.py` |

## 📚 Complete Documentation

**📖 [Full Documentation in `docs/`](docs/README.md)**

### 🚀 Getting Started
- **[Quick Start Guide](docs/quick-start.md)** - Up and running in 5 minutes
- **[Installation & Setup](docs/installation.md)** - Detailed installation
- **[Configuration](docs/configuration.md)** - Customize your setup

### 🤖 AI Services  
- **[Stable Diffusion Service](docs/stable-diffusion.md)** - Text-to-image generation
- **[LLaVA Service](docs/llava-service.md)** - Vision-language analysis
- **[Testing Guide](docs/testing.md)** - Test your services

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
│   └── test_llava.sh    # Test LLaVA service
├── services/            # 🤖 AI services
│   ├── stable_diffusion_service.py
│   ├── llava_service.py
│   └── example_service.py
├── utils/               # 🔧 Reusable utilities
│   ├── stable_diffusion/ # SD pipeline utilities
│   └── llava/           # LLaVA utilities
└── .claude/             # Claude Code settings
```

## 🎨 Example Usage

### Generate an Image
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"request": {"prompt": "a cute cat in a garden"}}' \
  | jq -r '.image' | base64 -d > cat.png
```

### Analyze an Image
```bash  
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "What objects are in this image?",
      "image": "https://example.com/image.jpg",
      "json_schema": {
        "type": "object", 
        "properties": {
          "objects": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }'
```

## ⚡ Key Features

- **🍎 Apple Silicon Optimized** - Uses MPS backend with float32 for stability
- **💾 External Drive Support** - Custom HF_HOME for model storage
- **🔄 Auto Device Detection** - MPS → CUDA → CPU fallback
- **📊 Structured Output** - JSON schema validation for LLaVA
- **🧪 Comprehensive Testing** - Dedicated test scripts for all services
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