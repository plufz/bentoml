# Photo Upscaler Service Documentation

🚀 **AI-powered photo upscaling** using Real-ESRGAN for high-quality image enhancement.

## Overview

The Photo Upscaler Service uses Real-ESRGAN (Real-Enhanced Super-Resolution Generative Adversarial Network) to intelligently upscale images while preserving and enhancing details. It supports both file uploads and URL-based image processing.

## Features

- **🎯 Real-ESRGAN Models**: Industry-leading AI upscaling technology
- **📱 Face Enhancement**: Optional GFPGAN integration for portrait photos
- **🔧 Flexible Scaling**: Scale factors from 1.0x to 4.0x
- **🎨 Multiple Formats**: PNG, JPEG, WEBP output support
- **⚡ Device Optimization**: Automatic CUDA/MPS/CPU detection
- **📊 Quality Control**: Adjustable JPEG quality (50-100)
- **🌐 URL Support**: Process images directly from URLs
- **📐 Custom Dimensions**: Handles non-square and irregular image sizes

## Quick Start

### Start the Service

```bash
# Individual service
./scripts/start.sh upscaler

# Or as part of multi-service (recommended)
./scripts/start.sh
```

### Test the Service

```bash
# Health check
./scripts/endpoint.sh health '{}'

# Upscale from URL
./scripts/endpoint.sh upscale_url '{"url": "https://example.com/photo.jpg", "scale_factor": 2.0}'

# Upscale from file upload
curl -X POST http://127.0.0.1:3000/upscale_file \
  -F "image_file=@./test-assets/test-upscale.jpg" \
  -F "scale_factor=2.5" \
  -F "output_format=PNG"
```

## API Endpoints

### 🌐 `/upscale_url` - Upscale from URL

Process an image from a URL.

**Request:**
```json
{
  "request": {
    "url": "https://example.com/image.jpg",
    "scale_factor": 2.0,
    "face_enhance": false,
    "output_format": "PNG",
    "quality": 95
  }
}
```

**Parameters:**
- `url` (string, required): Direct URL to the image
- `scale_factor` (float, optional): Scale multiplier (1.0-4.0, default: 2.0)
- `face_enhance` (boolean, optional): Enable GFPGAN face enhancement (default: false)
- `output_format` (string, optional): Output format - "PNG", "JPEG", "WEBP" (default: "PNG")
- `quality` (integer, optional): JPEG quality 50-100 (default: 95, ignored for PNG/WEBP)

### 📁 `/upscale_file` - Upscale from File Upload

Upload and process an image file.

**Request (multipart/form-data):**
```bash
curl -X POST http://127.0.0.1:3000/upscale_file \
  -F "image_file=@path/to/image.jpg" \
  -F "scale_factor=3.0" \
  -F "face_enhance=true" \
  -F "output_format=JPEG" \
  -F "quality=90"
```

**Form Fields:**
- `image_file` (file, required): Image file to upscale
- `scale_factor` (float, optional): Scale multiplier (1.0-4.0, default: 2.0)
- `face_enhance` (boolean, optional): Enable face enhancement (default: false)
- `output_format` (string, optional): Output format (default: "PNG")
- `quality` (integer, optional): JPEG quality (default: 95)

### 💚 `/health` - Service Health Check

Check service status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "service": "PhotoUpscalerService",
  "version": "1.0.0",
  "device": "mps",
  "model": "Real-ESRGAN x4plus",
  "capabilities": ["upscaling", "face_enhance"]
}
```

## Response Format

### Successful Response

```json
{
  "success": true,
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "upscaling_info": {
    "original_size": [423, 317],
    "upscaled_size": [846, 634],
    "scale_factor": 2.0,
    "face_enhance": false,
    "output_format": "PNG",
    "quality": null,
    "device": "mps",
    "model": "Real-ESRGAN x4plus",
    "source_url": "https://example.com/image.jpg"
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Failed to download image from URL: Invalid URL"
}
```

## Supported Image Formats

**Input:** JPEG, PNG, WEBP, BMP, TIFF, and most PIL-supported formats
**Output:** PNG, JPEG, WEBP

## Performance & Device Support

### Device Priority
1. **CUDA** - NVIDIA GPUs (fastest)
2. **MPS** - Apple Silicon M1/M2/M3 (fast)
3. **CPU** - Fallback (slower but reliable)

### Memory Requirements
- **Minimum:** 4GB RAM
- **Recommended:** 8GB+ RAM for large images
- **GPU Memory:** 2GB+ VRAM for optimal performance

### Processing Times (approximate)
- **512x512 → 1024x1024**: 2-5 seconds (GPU), 10-30 seconds (CPU)
- **1024x1024 → 2048x2048**: 5-15 seconds (GPU), 30-120 seconds (CPU)
- **Custom dimensions**: Varies based on total pixel count

## Configuration

### Environment Variables

```bash
# Service port (default: 3006)
UPSCALER_SERVICE_PORT=3006

# Server configuration
BENTOML_HOST=127.0.0.1
BENTOML_PROTOCOL=http
```

### Model Configuration

The service automatically downloads and caches Real-ESRGAN models:
- **Real-ESRGAN x4plus**: General-purpose upscaling
- **GFPGAN**: Face enhancement (optional)

Models are cached in the HuggingFace cache directory.

## Testing

Run comprehensive tests including custom dimensions:

```bash
# Run all upscaler tests
./scripts/test.sh --service upscaler

# Run specific test classes
uv run pytest tests/test_upscaler_service.py::TestPhotoUpscalerServiceUnit
uv run pytest tests/test_upscaler_service.py::TestPhotoUpscalerServiceIntegration
```

## Use Cases

### 📱 Photo Enhancement
- Upscale mobile photos for print quality
- Enhance old family photos
- Improve social media image quality

### 🎨 Digital Art
- Upscale artwork for high-resolution prints
- Enhance AI-generated images
- Prepare images for large displays

### 🏢 Business Applications
- Product photography enhancement
- Marketing material preparation
- Website image optimization

### 🎮 Gaming & Media
- Upscale game assets
- Enhance video thumbnails
- Improve streaming overlays

## Troubleshooting

### Common Issues

**"Service failed to start"**
- Ensure Real-ESRGAN dependencies are installed
- Check available GPU memory
- Try CPU-only mode if GPU issues persist

**"Model loading timeout"**
- First run downloads models (can take 5-10 minutes)
- Check internet connection
- Verify HuggingFace cache permissions

**"Out of memory errors"**
- Reduce image size before upscaling
- Use smaller scale factors (2.0x instead of 4.0x)
- Enable system swap if using CPU

**"Poor upscaling quality"**
- Try enabling face_enhance for portraits
- Use PNG format for better quality preservation
- Consider pre-processing images (noise reduction, etc.)

### Performance Optimization

```bash
# For Apple Silicon Macs
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For CUDA systems
export CUDA_VISIBLE_DEVICES=0

# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Architecture

```
Photo Upscaler Service
├── services/upscaler_service.py         # Main BentoML service
├── utils/upscaler/
│   ├── pipeline_manager.py              # Model management & upscaling
│   └── image_processing.py              # Image utilities & validation
└── tests/test_upscaler_service.py       # Comprehensive test suite
```

## Advanced Usage

### Custom Model Configuration

```python
# In pipeline_manager.py, you can customize:
# - Model selection (RealESRGAN_x4plus, RealESRGAN_x2plus)
# - Device selection
# - Memory optimization settings
# - Face enhancement models
```

### Batch Processing

While the service processes one image at a time, you can:
- Use multiple concurrent requests
- Implement client-side queuing
- Use the multi-service for mixed workloads

## Related Services

- **[Stable Diffusion](stable-diffusion.md)** - Generate images to upscale
- **[LLaVA Vision](llava-service.md)** - Analyze upscaled images
- **[Multi-Service](../CLAUDE.md#multi-service-architecture)** - Use all services together

---

📊 **Performance Tip**: For best results with photos containing faces, enable `face_enhance: true`. For artwork or landscapes, standard upscaling usually produces better results.