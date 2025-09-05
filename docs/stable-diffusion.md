# Stable Diffusion Service

Text-to-image generation service using Stable Diffusion v1.5, optimized for Apple Silicon.

## Overview

The Stable Diffusion service generates high-quality images from text prompts using the `runwayml/stable-diffusion-v1-5` model.

**Key Features:**
- Automatic device detection (MPS/CUDA/CPU)
- Apple Silicon MPS optimization with float32 precision
- Custom HuggingFace cache directory support
- Base64 image output for API integration
- Configurable generation parameters

## Quick Start

```bash
# Build service
BENTOFILE=bentofile_sd.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py

# Start service (downloads ~4GB model on first run)
./scripts/run_bentoml.sh serve stable_diffusion_service:latest
```

## API Reference

### Generate Image
**Endpoint**: `POST /generate_image`

**Request Format**:
```json
{
  "request": {
    "prompt": "a beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

**Parameters**:
- `prompt` (required): Text description of desired image
- `negative_prompt` (optional): What to avoid in the image
- `width`, `height` (optional): Image dimensions (256-1024, default: 512)
- `num_inference_steps` (optional): Quality vs speed (1-50, default: 20)
- `guidance_scale` (optional): Prompt adherence (1.0-20.0, default: 7.5)
- `seed` (optional): Random seed for reproducible results (-1 for random)

**Response**:
```json
{
  "success": true,
  "image": "iVBORw0KGgoAAAA...",  // Base64 encoded PNG
  "prompt": "a beautiful sunset over mountains",
  "width": 512,
  "height": 512,
  "device_used": "mps"
}
```

### Health Check
**Endpoint**: `POST /health`

Returns service status and device information.

## Testing Examples

### Basic Image Generation
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "a cute cat sitting in a garden",
      "negative_prompt": "blurry, low quality",
      "num_inference_steps": 15
    }
  }' | jq '.success, .device_used'
```

### Save Generated Image
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "a serene lake with mountains at sunset",
      "width": 512,
      "height": 512
    }
  }' | jq -r '.image' | base64 -d > generated_image.png
```

### Reproducible Generation
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "abstract art with vibrant colors",
      "seed": 12345,
      "guidance_scale": 10.0
    }
  }'
```

## Performance Notes

- **First Run**: Model downloads (~4GB) automatically to your HF_HOME
- **Generation Time**: 6-15 seconds on Apple Silicon M1/M2, 15-30 seconds on Intel
- **Memory Usage**: ~8GB recommended for smooth operation
- **Apple Silicon**: Uses MPS backend with float32 for stable results

## Configuration

The service uses `bentofile_sd.yaml` for configuration:

```yaml
service: "services.stable_diffusion_service:StableDiffusionService"
labels:
  owner: bentoml-local-setup
  stage: dev
  type: image-generation
include:
  - "services/"
  - "utils/"
python:
  requirements_txt: "./requirements.txt"
```

## Troubleshooting

### Black Images Generated
This is fixed in the current version by using float32 precision on MPS. If you see black images:
1. Ensure you're using the latest service version
2. Check that `device_used` shows "mps" in responses
3. Restart the service

### Model Download Issues
```bash
# Check your HF_HOME setting
echo $HF_HOME

# Ensure sufficient disk space (>5GB)
df -h

# Clear cache and retry
rm -rf ~/.cache/huggingface/transformers
```

### Memory Issues
- Close other applications
- Use smaller image dimensions (e.g., 256x256)
- Reduce `num_inference_steps` to 10-15

## Advanced Usage

See **[Utilities Documentation](utilities.md)** for information about:
- Using `BasePipelineManager` for custom services
- Device detection utilities
- Image processing helpers