# Troubleshooting Guide

Common issues and solutions for BentoML local setup.

## Installation Issues

### UV Not Found
**Problem**: `uv: command not found`

**Solution**:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Make permanent
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Python Version Compatibility
**Problem**: `Python 3.8+ required`

**Solution**:
```bash
# Check current version
python3 --version

# Install Python 3.11 with UV
uv python install 3.11

# Verify UV Python
uv run python --version
```

### Permission Denied on Scripts
**Problem**: `./scripts/setup_env.sh: Permission denied`

**Solution**:
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Or run with bash
bash scripts/setup_env.sh
```

## Service Issues

### Port 3000 Already in Use
**Problem**: `Address already in use: 127.0.0.1:3000`

**Solutions**:
```bash
# Option 1: Find and kill process using port 3000
lsof -ti:3000 | xargs kill -9

# Option 2: Use different port
# Edit bentoml_config.yaml and change port to 3001

# Option 3: Stop existing BentoML services
./scripts/run_bentoml.sh list
# Kill specific service process
```

### Service Won't Start
**Problem**: Service fails to initialize

**Diagnostic Steps**:
```bash
# 1. Check if service is built
./scripts/run_bentoml.sh list

# 2. Verify dependencies
uv sync

# 3. Check BentoML installation
uv run bentoml --version

# 4. Verify configuration
./scripts/check_setup.sh

# 5. Check logs for specific errors
tail -f bentoml_home/logs/bentoml.log
```

### Model Loading Failures

#### Stable Diffusion Model Issues
**Problem**: `Failed to load stable-diffusion-v1-5`

**Solutions**:
```bash
# Check available disk space (need >5GB)
df -h

# Clear model cache
rm -rf ~/.cache/huggingface/transformers
rm -rf $HF_HOME/hub  # If using custom HF_HOME

# Restart service to re-download
./scripts/run_bentoml.sh serve stable_diffusion_service:latest
```

#### LLaVA Model Issues  
**Problem**: `Failed to load llava-1.6-mistral-7b`

**Solutions**:
```bash
# Service should fallback automatically, but if not:
# Check available disk space (need >15GB)
df -h

# Verify HF_HOME path
echo $HF_HOME

# Clear and restart
rm -rf $HF_HOME/hub/models--llava-hf--llava-v1.6-mistral-7b-hf
./scripts/run_bentoml.sh serve l_la_va_service:latest
```

## Performance Issues

### Slow Model Loading
**Problem**: Models take very long to load

**Solutions**:
```bash
# 1. Use external SSD for model cache
export HF_HOME="/Volumes/FastSSD/huggingface"

# 2. Pre-download models
uv run python -c "
from transformers import pipeline
pipe = pipeline('image-text-to-text', model='llava-hf/llava-v1.6-mistral-7b-hf')
"

# 3. Check network speed
curl -o /dev/null -s -w "%{speed_download}\n" https://huggingface.co
```

### Memory Issues
**Problem**: `Out of memory` or system becomes unresponsive

**Solutions**:
```bash
# 1. Close other applications
# 2. Check memory usage
top -o MEM

# 3. Use CPU instead of MPS (if on Apple Silicon)
# Edit services to force device="cpu"

# 4. Reduce batch sizes or image dimensions
# In service requests, use smaller width/height values
```

### Apple Silicon Black Images (Stable Diffusion)
**Problem**: Generated images are completely black

**Status**: ✅ **Fixed in current version**

The current Stable Diffusion service uses float32 precision on MPS to prevent this issue. If you still see black images:

```bash
# Verify you're using the latest service
git pull  # Get latest code
BENTOFILE=bentofile_sd.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py

# Check response shows correct device
curl -X POST http://127.0.0.1:3000/health -d '{}' | jq '.device, .dtype'
# Should show: "mps", "torch.float32"
```

## Network Issues

### Model Download Fails
**Problem**: `Connection timeout` during model download

**Solutions**:
```bash
# 1. Check internet connection
ping huggingface.co

# 2. Use proxy if behind firewall
export UV_HTTP_PROXY=http://proxy.company.com:8080
export UV_HTTPS_PROXY=http://proxy.company.com:8080

# 3. Retry with timeout increase
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10 minutes

# 4. Manual download (as last resort)
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json
```

### API Requests Timeout
**Problem**: curl requests hang or timeout

**Solutions**:
```bash
# 1. Check if service is responding
curl -v http://127.0.0.1:3000/health

# 2. Increase timeout
curl --max-time 300 http://127.0.0.1:3000/endpoint

# 3. Check if model is still loading
# Look for loading progress in service logs
```

## Configuration Issues

### Invalid Configuration File
**Problem**: `Invalid bentoml_config.yaml`

**Solution**:
```bash
# Validate YAML syntax
uv run python -c "
import yaml
with open('bentoml_config.yaml') as f:
    yaml.safe_load(f)
print('Config is valid')
"

# Reset to default if needed
git checkout bentoml_config.yaml
```

### Environment Variables Not Loading
**Problem**: Custom settings not applied

**Solution**:
```bash
# Check if .env exists
ls -la .env

# Source manually if needed
source .env

# Verify variables are set
echo $BENTOML_HOME
echo $HF_HOME
```

## Development Issues

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'utils'`

**Solution**:
```bash
# Ensure utils directory is included in bentofile
# Check bentofile_*.yaml contains:
# include:
#   - "services/"
#   - "utils/"

# Rebuild service
BENTOFILE=bentofile_*.yaml ./scripts/run_bentoml.sh build services/your_service.py
```

### Code Changes Not Reflected
**Problem**: Service still runs old code after changes

**Solution**:
```bash
# 1. Rebuild service
./scripts/run_bentoml.sh build services/your_service.py

# 2. Restart service
./scripts/run_bentoml.sh serve your_service:latest

# 3. Clear Python cache
find . -name "__pycache__" -delete
find . -name "*.pyc" -delete
```

## Testing Issues

### Test Scripts Fail
**Problem**: `./scripts/test_llava.sh` returns errors

**Solutions**:
```bash
# 1. Ensure service is running
curl -I http://127.0.0.1:3000/health

# 2. Check script permissions  
chmod +x scripts/test_llava.sh

# 3. Install jq for JSON parsing
brew install jq  # macOS
# or
apt install jq   # Linux

# 4. Run individual tests
./scripts/test_llava.sh health
```

## Getting More Help

### Enable Debug Logging
```bash
# Set debug level in .env
echo "BENTOML_LOG_LEVEL=DEBUG" >> .env

# Check logs
tail -f bentoml_home/logs/bentoml.log
```

### Collect System Information
```bash
# Create debug report
echo "=== System Info ===" > debug.txt
uname -a >> debug.txt
python3 --version >> debug.txt
uv --version >> debug.txt
echo "=== BentoML Info ===" >> debug.txt
uv run bentoml --version >> debug.txt
echo "=== Memory Info ===" >> debug.txt
free -h >> debug.txt  # Linux
vm_stat >> debug.txt  # macOS
echo "=== Disk Space ===" >> debug.txt
df -h >> debug.txt
```

### Common Log Locations
- **BentoML logs**: `bentoml_home/logs/bentoml.log`
- **Service logs**: Check terminal output where service is running
- **UV logs**: Usually in terminal output during `uv sync`

### Still Need Help?
1. Check the specific service documentation:
   - [Stable Diffusion Service](stable-diffusion.md)
   - [LLaVA Service](llava-service.md)
2. Review [Configuration Guide](configuration.md)
3. Create an issue with your debug information