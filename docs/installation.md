# Installation & Setup

Complete installation guide for BentoML local setup with UV package manager.

## Prerequisites

### System Requirements
- **macOS**: 10.15 (Catalina) or later
- **Python**: 3.8 or later
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 20GB free space (for models)
- **Network**: Internet connection for model downloads

### Hardware Support
- **Apple Silicon (M1/M2/M3)**: Optimal performance with MPS acceleration
- **Intel Mac**: Supported with CPU inference
- **GPU**: NVIDIA GPUs supported with CUDA (if available)

## Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
# Clone or navigate to the project directory
cd /path/to/bentoml-project

# Run the setup script
./scripts/setup_env.sh
```

This script will:
1. Install UV package manager
2. Create Python environment  
3. Install all dependencies
4. Configure BentoML

### Method 2: Manual Installation

#### Step 1: Install UV Package Manager
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to your PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

#### Step 2: Setup Project Environment
```bash
# Navigate to project directory
cd /path/to/bentoml-project

# Sync dependencies
uv sync

# Install additional development tools (optional)
uv sync --extra dev
```

#### Step 3: Verify Installation
```bash
# Check BentoML installation
uv run bentoml --version

# Verify Python environment
uv run python --version

# List installed packages
uv pip list
```

## Configuration

### Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your preferences:
```bash
# BentoML Configuration
BENTOML_HOME=./bentoml_home
BENTOML_CONFIG_FILE=bentoml_config.yaml

# Server Configuration
BENTOML_PORT=3000
BENTOML_HOST=127.0.0.1

# HuggingFace Configuration (optional)
# HF_HOME=/Volumes/External/huggingface  # For external drive storage
# TRANSFORMERS_CACHE=${HF_HOME}/hub
# HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
```

### HuggingFace Cache (Optional)

For external drive storage of large models:

1. **Set HF_HOME in your shell profile** (`.zshrc` or `.bash_profile`):
```bash
export HF_HOME="/Volumes/External/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
```

2. **Create cache directories**:
```bash
mkdir -p "$HF_HOME/hub"
```

3. **Verify setup**:
```bash
echo $HF_HOME  # Should show your custom path
```

## Verification

Run the setup verification script:
```bash
./scripts/check_setup.sh
```

Expected output:
```
🔍 BentoML Setup Verification
✅ UV is installed and accessible
✅ Python 3.11+ is available
✅ BentoML is installed correctly
✅ Dependencies are satisfied
✅ Storage directories created
✅ Configuration file is valid
✅ Server can start (dry run)
🎉 Setup verification complete!
```

## Directory Structure

After installation, your project should have:
```
bentoml-project/
├── scripts/              # Management scripts
├── services/             # BentoML services
├── utils/                # Utility modules
├── docs/                 # Documentation
├── bentoml_home/         # BentoML storage
├── pyproject.toml        # UV configuration
├── bentoml_config.yaml   # BentoML configuration
└── .env                  # Environment variables
```

## Initial Test

Build and run the example service:
```bash
# Build example service
./scripts/run_bentoml.sh build services/example_service.py

# Start service
./scripts/run_bentoml.sh serve hello_service:latest

# Test (in another terminal)
curl -X POST http://127.0.0.1:3000/hello \
  -H "Content-Type: application/json" \
  -d '{"request": {"name": "World"}}'
```

Expected response:
```json
{
  "message": "Hello, World!"
}
```

## Next Steps

- **[Quick Start Guide](quick-start.md)** - Build your first AI service
- **[Stable Diffusion Service](services/stable-diffusion.md)** - Text-to-image generation
- **[LLaVA Service](services/llava-service.md)** - Image analysis with AI
- **[Configuration Guide](configuration.md)** - Customize your setup

## Troubleshooting Installation

### UV Installation Issues
```bash
# Manual UV installation
curl -LsSf https://astral.sh/uv/0.4.0/install.sh | sh

# Add to PATH permanently
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Python Version Issues
```bash
# Check Python version
python3 --version

# UV uses system Python, ensure 3.8+
uv python install 3.11  # Install specific version if needed
```

### Permission Issues
```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix ownership if needed
sudo chown -R $(whoami) .
```

### Network/Proxy Issues
If behind a corporate firewall:
```bash
# Configure UV with proxy
export UV_HTTP_PROXY=http://proxy.company.com:8080
export UV_HTTPS_PROXY=http://proxy.company.com:8080

# Or use UV's built-in proxy support
uv --proxy http://proxy.company.com:8080 sync
```

### Disk Space Issues
- **Models require significant space**: Stable Diffusion (~4GB), LLaVA (~13GB)
- **Use external drive**: Set `HF_HOME` to external storage
- **Clean cache periodically**: `rm -rf ~/.cache/huggingface/transformers`

Need more help? Check the **[Troubleshooting Guide](troubleshooting.md)**.