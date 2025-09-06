# LLaVA Vision-Language Service

Image analysis and understanding service using LLaVA-1.6-Mistral-7B with structured JSON output support.

## Overview

The LLaVA service analyzes images with text prompts and can return structured JSON responses based on provided schemas.

**Key Features:**
- Multiple image input formats (URL, base64, bytes)
- Optional JSON schema validation for structured output
- Visual Question Answering (VQA)
- Image captioning and object detection
- Automatic device detection (MPS/CUDA/CPU)
- Custom HuggingFace cache directory support

## Quick Start

```bash
# Build service
BENTOFILE=bentofile_llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py

# Start service (downloads ~13GB model on first run)
./scripts/run_bentoml.sh serve l_la_va_service:latest

# Test with dedicated script
./scripts/test_llava.sh all
```

## API Reference

### Analyze Image
**Endpoint**: `POST /analyze_image`

**Request Format**:
```json
{
  "request": {
    "prompt": "What objects do you see in this image?",
    "image": "data:image/jpeg;base64,/9j/4AAQ..." or "https://httpbin.org/image/jpeg",
    "json_schema": {
      "type": "object",
      "properties": {
        "objects": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "confidence": {"type": "number"}
      },
      "required": ["objects", "description"]
    },
    "include_raw_response": false,
    "temperature": 0.1,
    "max_new_tokens": 512
  }
}
```

**Parameters**:
- `prompt` (required): Question or instruction about the image
- `image` (required): Image data (URL, base64, or bytes)
- `json_schema` (optional): JSON schema for structured output
- `include_raw_response` (optional): Include raw model response
- `temperature` (optional): Generation randomness (0.0-2.0, default: 0.1)
- `max_new_tokens` (optional): Maximum response length (1-2048, default: 512)

**Image Input Formats**:
- **Base64 with header**: `"data:image/png;base64,iVBORw0KGgo..."` or `"data:image/jpeg;base64,/9j/4AAQ..."`
- **Base64 only**: `"iVBORw0KGgo..."` (PNG) or `"/9j/4AAQ..."` (JPEG)
- **URL**: `"https://httpbin.org/image/jpeg"` (JPEG) or `"https://httpbin.org/image/png"` (PNG)
- **Bytes**: Raw image bytes (programmatic use)

**Response with JSON Schema**:
```json
{
  "success": true,
  "response": {
    "objects": ["cat", "garden", "flowers"],
    "description": "A cute orange cat sitting in a colorful flower garden",
    "confidence": 0.95
  },
  "format": "structured_json",
  "device_used": "mps"
}
```

**Response without Schema**:
```json
{
  "success": true,
  "response": "I can see a cute orange cat sitting in a beautiful garden with colorful flowers around it.",
  "format": "raw_text",
  "device_used": "mps"
}
```

### Get Example Schemas
**Endpoint**: `POST /get_example_schemas`

Returns pre-built JSON schemas for common use cases:
- `image_description`: General image description
- `object_detection`: Object identification
- `image_qa`: Question answering
- `text_extraction`: Text detection in images

### Health Check
**Endpoint**: `POST /health`

Returns service status, capabilities, and device information.

## Testing Examples

### Simple Image Analysis
```bash
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "Describe this image",
      "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png"
    }
  }' | jq '.response'
```

### Structured Object Detection
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
          "objects": {"type": "array", "items": {"type": "string"}},
          "count": {"type": "integer"},
          "scene_type": {"type": "string"}
        },
        "required": ["objects", "count"]
      }
    }
  }' | jq '.response'
```

### Visual Question Answering
```bash
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "Is there a person in this image? What are they doing?",
      "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
      "json_schema": {
        "type": "object",
        "properties": {
          "person_present": {"type": "boolean"},
          "activity": {"type": "string"},
          "confidence": {"type": "number"}
        }
      }
    }
  }'
```

### JPEG Image Analysis
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

### Using Test Script
```bash
# Test all endpoints
./scripts/test_llava.sh all

# Individual tests
./scripts/test_llava.sh health    # Health check
./scripts/test_llava.sh schemas   # Get example schemas  
./scripts/test_llava.sh image     # Simple image analysis
./scripts/test_llava.sh json      # Structured output test
```

## Use Cases

### 1. Image Captioning
Generate detailed descriptions of images for accessibility or content management.

### 2. Visual Question Answering
Ask specific questions about image content for interactive applications.

### 3. Object Detection
Identify and list objects in images for inventory or analysis systems.

### 4. Text Extraction (OCR)
Extract and locate text within images for document processing.

### 5. Structured Analysis
Get formatted JSON responses for easy integration with other systems.

### 6. Content Moderation
Analyze images for policy compliance or content categorization.

## Performance Notes

- **First Run**: Model downloads (~13GB) automatically to your HF_HOME
- **Analysis Time**: 10-30 seconds depending on image size and complexity
- **Memory Usage**: ~16GB recommended for optimal performance
- **Image Size**: Automatically resized to max 1344px while maintaining aspect ratio
- **Supported Formats**: JPEG, PNG, GIF, BMP, WebP (all common image formats via PIL)

## Configuration

The service uses `bentofile_llava.yaml`:

```yaml
service: "services.llava_service:LLaVAService"
labels:
  owner: bentoml-local-setup
  stage: dev
  type: vision-language
include:
  - "services/"
  - "utils/"
python:
  requirements_txt: "./requirements.txt"
```

## JSON Schema Examples

### Image Description Schema
```json
{
  "type": "object",
  "properties": {
    "description": {"type": "string"},
    "objects": {"type": "array", "items": {"type": "string"}},
    "scene": {"type": "string"},
    "mood": {"type": "string"}
  },
  "required": ["description", "objects"]
}
```

### Object Detection Schema
```json
{
  "type": "object",
  "properties": {
    "objects": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "confidence": {"type": "number"},
          "description": {"type": "string"}
        },
        "required": ["name", "confidence"]
      }
    },
    "total_objects": {"type": "integer"}
  },
  "required": ["objects", "total_objects"]
}
```

## Troubleshooting

### Model Loading Issues
If the service fails to load the local model `llava-1.6-mistral-7b`:
1. It automatically falls back to `llava-hf/llava-v1.6-mistral-7b-hf`
2. Ensure sufficient disk space for model download
3. Check your HF_HOME path is accessible

### JSON Parsing Errors
- Ensure your JSON schema is valid
- Use simple schemas initially
- Check the `include_raw_response` flag to debug model output

### Memory Issues
- Close other applications  
- Use smaller images (<2MB recommended)
- Reduce `max_new_tokens` to 256 or less

### Slow Response Times
- Use lower `temperature` values (0.0-0.2)
- Reduce image size
- Simplify prompts and schemas

## Advanced Usage

See **[Utilities Documentation](utilities.md)** for information about:
- Using `LLaVAPipelineManager` for custom services
- Image processing utilities
- JSON schema validation helpers