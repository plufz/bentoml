# Testing Guide

Comprehensive testing approaches for all BentoML services with enhanced configuration-driven capabilities.

## Overview

This project provides multiple testing approaches:
- **Configuration-driven automated testing** with pytest
- **Dynamic endpoint discovery and testing**  
- **Service-specific test suites** with comprehensive coverage
- **Interactive API exploration** with generated examples

## 🚀 Enhanced Test Scripts (Recommended)

### Configuration-Driven Testing
```bash
# List all available test services and types
./scripts/test.sh --list

# Run tests for specific services  
./scripts/test.sh --service rag         # RAG service tests
./scripts/test.sh --service llava       # LLaVA service tests
./scripts/test.sh --service sd          # Stable Diffusion (alias support)
./scripts/test.sh --service multi       # Multi-service composition tests

# Run specific test types
./scripts/test.sh --unit                # Fast unit tests
./scripts/test.sh --behavior            # HTTP behavior tests  
./scripts/test.sh --integration         # Full integration tests (slow)

# Coverage and reporting
./scripts/test.sh --coverage            # Fast tests with coverage
./scripts/test.sh --coverage-all        # All tests with coverage
./scripts/test.sh --service rag --coverage --verbose  # Combined options

# Basic testing (default)
./scripts/test.sh                       # Run fast tests (unit + behavior)
./scripts/test.sh --all                 # Run all tests including integration
```

### Dynamic Endpoint Testing
```bash
# List all available endpoints with examples
./scripts/endpoint.sh --list

# Test endpoints with auto-generated examples
./scripts/endpoint.sh health '{}'
./scripts/endpoint.sh rag_query '{"query": "What is machine learning?", "max_tokens": 256}'
./scripts/endpoint.sh generate_image '{"prompt": "A sunset", "width": 512, "height": 512}'

# Get help and usage examples
./scripts/endpoint.sh --help
```

## Manual Testing

### Health Checks
Test if services are running and configured correctly:

```bash
# Any service health check
curl -X POST http://127.0.0.1:3000/health \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Stable Diffusion Testing

#### Basic Image Generation
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "a cute cat in a garden",
      "num_inference_steps": 10
    }
  }' | jq '.success, .device_used'
```

#### Save Generated Image
```bash
curl -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "abstract art with bright colors",
      "width": 256,
      "height": 256,
      "seed": 42
    }
  }' | jq -r '.image' | base64 -d > test_image.png
```

#### Parameter Testing
```bash
# Test different parameters
for steps in 5 10 15; do
  echo "Testing $steps inference steps"
  curl -s -X POST http://127.0.0.1:3000/generate_image \
    -H "Content-Type: application/json" \
    -d '{
      "request": {
        "prompt": "mountain landscape",
        "num_inference_steps": '$steps',
        "seed": 123
      }
    }' | jq '.success'
done
```

### LLaVA Testing

#### Basic Image Analysis
```bash
# Test with a simple base64 image (1x1 white pixel)
TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "What do you see in this image?",
      "image": "'$TEST_IMAGE'",
      "temperature": 0.1
    }
  }' | jq '.response'
```

#### Structured JSON Output
```bash
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "Analyze this image",
      "image": "'$TEST_IMAGE'",
      "json_schema": {
        "type": "object",
        "properties": {
          "colors": {"type": "array", "items": {"type": "string"}},
          "objects": {"type": "array", "items": {"type": "string"}},
          "description": {"type": "string"}
        },
        "required": ["description"]
      }
    }
  }' | jq '.response'
```

#### Test with Real Image URL
```bash
curl -X POST http://127.0.0.1:3000/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "prompt": "What objects can you identify in this image?",
      "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
      "json_schema": {
        "type": "object",
        "properties": {
          "objects": {"type": "array", "items": {"type": "string"}},
          "scene_type": {"type": "string"},
          "weather": {"type": "string"}
        }
      }
    }
  }'
```

## Interactive Testing

### Swagger UI
Visit http://127.0.0.1:3000/docs when your service is running for:
- **Interactive API explorer**
- **Request/response examples**  
- **Parameter documentation**
- **Try it out** functionality

### Example Schemas (LLaVA)
Get pre-built schemas for common use cases:
```bash
curl -X POST http://127.0.0.1:3000/get_example_schemas \
  -H "Content-Type: application/json" \
  -d '{}' | jq keys
```

## Performance Testing

### Load Testing
```bash
# Basic load test
./scripts/test_service.sh load 10

# Custom load test with curl
for i in {1..5}; do
  echo "Request $i"
  time curl -s -X POST http://127.0.0.1:3000/health -d '{}' > /dev/null
done
```

### Memory Usage Monitoring
```bash
# Monitor during testing
# Terminal 1: Run service
./scripts/run_bentoml.sh serve stable_diffusion_service:latest

# Terminal 2: Monitor memory
while true; do
  ps aux | grep bentoml | grep -v grep | awk '{print $4 "%", $6/1024 "MB"}'
  sleep 5
done
```

### Response Time Testing
```bash
# Test response times
echo "Testing response times..."
for i in {1..3}; do
  echo "Test $i:"
  time curl -s -X POST http://127.0.0.1:3000/generate_image \
    -H "Content-Type: application/json" \
    -d '{
      "request": {
        "prompt": "simple test image",
        "num_inference_steps": 5,
        "width": 256,
        "height": 256
      }
    }' | jq '.success'
done
```

## Test Data

### Test Images for LLaVA
```bash
# Create test images directory
mkdir -p test_images

# Download test images
curl -o test_images/nature.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

# Convert to base64 for API testing
base64 -i test_images/nature.jpg > test_images/nature_base64.txt
```

### Test Prompts
Create a file `test_prompts.txt`:
```text
a beautiful sunset over mountains
abstract art with vibrant colors  
a cute cat sitting in a garden
modern architecture building
vintage car on a country road
```

### Batch Testing Script
```bash
#!/bin/bash
# batch_test.sh

while IFS= read -r prompt; do
  echo "Testing prompt: $prompt"
  
  curl -s -X POST http://127.0.0.1:3000/generate_image \
    -H "Content-Type: application/json" \
    -d '{"request": {"prompt": "'$prompt'", "num_inference_steps": 10}}' \
    | jq '.success'
    
done < test_prompts.txt
```

## Validation

### Image Output Validation
```bash
# Function to validate generated images
validate_image() {
  local response=$1
  local size=$(echo "$response" | jq -r '.image' | base64 -d | wc -c)
  
  if [ $size -gt 1000 ]; then
    echo "✅ Valid image generated (${size} bytes)"
  else
    echo "❌ Invalid or empty image (${size} bytes)"
  fi
}

# Use in tests
response=$(curl -s -X POST http://127.0.0.1:3000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"request": {"prompt": "test image"}}')

validate_image "$response"
```

### JSON Schema Validation
```bash
# Validate LLaVA JSON responses
validate_json_response() {
  local response=$1
  local has_response=$(echo "$response" | jq 'has("response")')
  local format=$(echo "$response" | jq -r '.format')
  
  if [ "$has_response" = "true" ] && [ "$format" = "structured_json" ]; then
    echo "✅ Valid structured JSON response"
  else
    echo "❌ Invalid JSON response format"
  fi
}
```

## Continuous Testing

### Pre-commit Tests
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Basic functionality test before commit

echo "Running pre-commit tests..."

# Check if services build
./scripts/run_bentoml.sh build services/example_service.py > /dev/null
if [ $? -eq 0 ]; then
  echo "✅ Example service builds"
else
  echo "❌ Example service build failed"
  exit 1
fi

echo "✅ Pre-commit tests passed"
```

### Integration Tests
```bash
# integration_test.sh
#!/bin/bash

echo "🧪 Running integration tests..."

# 1. Build all services
echo "Building services..."
./scripts/run_bentoml.sh build services/example_service.py
BENTOFILE=bentofile_sd.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py
BENTOFILE=bentofile_llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py

# 2. Test example service
echo "Testing example service..."
./scripts/run_bentoml.sh serve hello_service:latest &
SERVICE_PID=$!
sleep 10

curl -s -X POST http://127.0.0.1:3000/hello -d '{"request": {"name": "Test"}}' | jq '.message'

kill $SERVICE_PID
sleep 5

echo "✅ Integration tests completed"
```

## Troubleshooting Tests

### Common Test Issues
1. **Port conflicts**: Use `lsof -i :3000` to check
2. **Service not ready**: Wait longer after starting
3. **Memory issues**: Monitor with `top -o MEM`
4. **Network timeouts**: Increase curl timeout with `--max-time 300`

### Debug Test Failures
```bash
# Enable verbose curl output
curl -v -X POST http://127.0.0.1:3000/health -d '{}'

# Check service logs
tail -f bentoml_home/logs/bentoml.log

# Validate JSON syntax
echo '{"request": {"prompt": "test"}}' | jq '.'
```

For more troubleshooting help, see the **[Troubleshooting Guide](troubleshooting.md)**.