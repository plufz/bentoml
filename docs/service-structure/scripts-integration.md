# Scripts Integration Guide

This guide covers integrating new services with the project's development workflow scripts and automation tools.

## Scripts Overview

The project includes comprehensive scripts for development, testing, and deployment workflows:

```
scripts/
├── setup_env.sh          # Environment setup and UV installation
├── check_setup.sh        # Verify installation and configuration
├── run_bentoml.sh        # Core BentoML service runner
├── start.sh              # Quick start multi-service
├── build_services.sh     # Build all services
├── health.sh             # Service health checks
├── endpoint.sh           # Endpoint testing utility
└── test.sh               # Comprehensive testing script
```

## Script Integration Checklist

When adding a new service, update these scripts:

- [ ] `build_services.sh` - Add service build command
- [ ] `test.sh` - Add service-specific test options
- [ ] `start.sh` - Update multi-service startup (if applicable)
- [ ] `endpoint.sh` - Add endpoint examples (documentation)

## Build Script Integration

### Adding to build_services.sh

**File**: `scripts/build_services.sh`

```bash
#!/bin/bash
# Build all BentoML services

set -e

echo "Building all BentoML services..."

# Existing services
echo "Building Example Service..."
./scripts/run_bentoml.sh build services/example_service.py

echo "Building Stable Diffusion Service..."
BENTOFILE=config/bentofiles/stable-diffusion.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py

echo "Building LLaVA Service..."
BENTOFILE=config/bentofiles/llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py

echo "Building Whisper Service..."
BENTOFILE=config/bentofiles/whisper.yaml ./scripts/run_bentoml.sh build services/whisper_service.py

echo "Building Photo Upscaler Service..."
BENTOFILE=config/bentofiles/upscaler.yaml ./scripts/run_bentoml.sh build services/upscaler_service.py

echo "Building RAG Service..."
BENTOFILE=config/bentofiles/rag.yaml ./scripts/run_bentoml.sh build services/rag_service.py

# ADD YOUR NEW SERVICE HERE:
echo "Building Your Service..."
BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py

# Multi-service (build last as it depends on others)
echo "Building Multi-Service..."
BENTOFILE=config/bentofiles/multi-service.yaml ./scripts/run_bentoml.sh build services/multi_service.py

echo "All services built successfully!"
```

### Custom Build Script for Complex Services

For services requiring special build steps, create dedicated build scripts:

**File**: `scripts/build_your_service.sh`

```bash
#!/bin/bash
# Build your service with custom steps

set -e

SERVICE_NAME="your-service"
SERVICE_FILE="services/your_service.py"
BENTOFILE="config/bentofiles/your-service.yaml"

echo "Building ${SERVICE_NAME}..."

# Pre-build steps (if needed)
echo "Running pre-build steps..."
# Download models, prepare assets, etc.
if [ ! -f "models/your_model.bin" ]; then
    echo "Downloading model..."
    wget -O models/your_model.bin https://example.com/model.bin
fi

# Build service
echo "Building BentoML service..."
BENTOFILE=${BENTOFILE} ./scripts/run_bentoml.sh build ${SERVICE_FILE}

# Post-build verification
echo "Verifying build..."
bentoml list | grep ${SERVICE_NAME} || {
    echo "Build verification failed"
    exit 1
}

echo "${SERVICE_NAME} built successfully!"
```

## Test Script Integration

### Adding to test.sh

**File**: `scripts/test.sh` (relevant sections)

```bash
#!/bin/bash
# Comprehensive testing script with your service integration

# ... existing code ...

# Service-specific testing function
test_service() {
    local service=$1
    echo "Testing ${service} service..."
    
    case $service in
        example)
            echo "Running Example Service tests..."
            uv run pytest tests/test_example_service.py -v
            ;;
        llava)
            echo "Running LLaVA Service tests..."
            uv run pytest tests/test_llava_service.py -v
            ;;
        stable_diffusion)
            echo "Running Stable Diffusion Service tests..."
            uv run pytest tests/test_stable_diffusion_service.py -v
            ;;
        whisper)
            echo "Running Whisper Service tests..."
            uv run pytest tests/test_whisper_service.py -v
            ;;
        upscaler)
            echo "Running Photo Upscaler Service tests..."
            uv run pytest tests/test_upscaler_service.py -v
            ;;
        rag)
            echo "Running RAG Service tests..."
            uv run pytest tests/test_rag_service.py -v
            ;;
        # ADD YOUR SERVICE HERE:
        your_service)
            echo "Running Your Service tests..."
            uv run pytest tests/test_your_service.py -v
            ;;
        multi)
            echo "Running Multi-Service tests..."
            uv run pytest tests/test_multi_service.py -v
            ;;
        *)
            echo "Unknown service: $service"
            echo "Available services: example, llava, stable_diffusion, whisper, upscaler, rag, your_service, multi"
            exit 1
            ;;
    esac
}

# ... rest of script ...

# Update help message
show_help() {
    echo "BentoML Testing Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Run all tests including slow integration tests"
    echo "  --fast          Run only fast tests (default)"
    echo "  --unit          Run only unit tests"
    echo "  --integration   Run only integration tests"
    echo "  --coverage      Run tests with coverage report"
    echo "  --service NAME  Run tests for specific service"
    echo "                  Available: example, llava, stable_diffusion, whisper, upscaler, rag, your_service, multi"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run fast tests"
    echo "  $0 --all                        # Run all tests"
    echo "  $0 --service your_service       # Test your service"
    echo "  $0 --coverage                   # Run with coverage"
}
```

## Health Check Integration

### Service Health Check

Add health check capabilities to your service:

```python
@bentoml.service()
class YourService:
    @bentoml.api
    def health(self) -> dict:
        """Health check endpoint"""
        try:
            # Verify service dependencies
            status = self._check_dependencies()
            
            return {
                "status": "healthy" if status else "unhealthy",
                "service": "your-service",
                "timestamp": datetime.utcnow().isoformat(),
                "dependencies": status
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "your-service",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _check_dependencies(self) -> dict:
        """Check service dependencies"""
        return {
            "model_loaded": hasattr(self, '_model') and self._model is not None,
            "config_valid": hasattr(self, 'config') and self.config is not None,
            "initialized": getattr(self, '_initialized', False)
        }
```

### Update health.sh Script

**File**: `scripts/health.sh`

```bash
#!/bin/bash
# Enhanced health check with your service

# ... existing code ...

# Service-specific health checks
check_service_specific() {
    local service_name=$1
    local port=$2
    
    case $service_name in
        "your-service")
            echo "Checking your service specific endpoints..."
            curl -s -f "${PROTOCOL}://${HOST}:${port}/your_endpoint_health" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "✓ Your service specific endpoints responding"
                return 0
            else
                echo "✗ Your service specific endpoints not responding"
                return 1
            fi
            ;;
        *)
            return 0
            ;;
    esac
}
```

## Endpoint Testing Integration

### Update endpoint.sh Documentation

Add examples for your service endpoints:

**File**: `scripts/endpoint.sh` (documentation comments)

```bash
#!/bin/bash
# Enhanced endpoint testing with your service examples

# ... existing code ...

show_examples() {
    echo "Endpoint Testing Examples:"
    echo ""
    echo "System endpoints:"
    echo "  $0 health '{}'"
    echo "  $0 info '{}'"
    echo ""
    
    # ... existing examples ...
    
    echo "Your Service endpoints:"
    echo "  $0 your_endpoint '{\"input_field\": \"test data\", \"optional_field\": \"custom\"}'"
    echo "  $0 your_file_endpoint --file path/to/file.txt"
    echo ""
    
    echo "Advanced usage:"
    echo "  $0 your_endpoint '{}' --host localhost --port 3007 --verbose"
    echo "  $0 your_endpoint '{\"parameters\": {\"key\": \"value\"}}' --timeout 60"
}
```

## Multi-Service Integration

### Adding to Multi-Service Composition

**File**: `services/multi_service.py`

```python
from services.your_service import YourService, YourServiceRequest, YourServiceResponse

@bentoml.service(
    name="multi-service",
    resources={"cpu": "4", "memory": "16Gi"}
)
class MultiService:
    def __init__(self):
        # Initialize all services
        self.example_service = ExampleService()
        self.stable_diffusion_service = StableDiffusionService()
        self.llava_service = LLaVAService()
        self.whisper_service = WhisperService()
        self.upscaler_service = PhotoUpscalerService()
        self.rag_service = RAGService()
        
        # ADD YOUR SERVICE HERE:
        self.your_service = YourService()
    
    # ... existing endpoints ...
    
    # ADD YOUR SERVICE ENDPOINTS:
    @bentoml.api
    def your_endpoint(self, request: YourServiceRequest) -> YourServiceResponse:
        """Your service endpoint in multi-service"""
        return self.your_service.your_endpoint(request)
    
    # Add additional endpoints as needed
    @bentoml.api
    def your_other_endpoint(self, request: YourOtherRequest) -> YourOtherResponse:
        """Another endpoint from your service"""
        return self.your_service.your_other_endpoint(request)
```

### Update Multi-Service Build Configuration

**File**: `config/bentofiles/multi-service.yaml`

```yaml
# ... existing configuration ...

include:
  - "services/"
  - "utils/"
  - "config/bentoml.yaml"

python:
  requirements_txt: |
    # ... existing dependencies ...
    
    # Your service dependencies
    your-service-dependency>=1.0.0
    another-dependency>=2.0.0
```

## Environment Script Integration

### Service-Specific Environment Variables

Add your service's environment variables to setup scripts:

**File**: `scripts/setup_env.sh`

```bash
#!/bin/bash
# ... existing setup code ...

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# ... existing variables ...

# Your Service Configuration
YOUR_SERVICE_PORT=3007
YOUR_SERVICE_MODEL_PATH=/path/to/model
YOUR_SERVICE_CONFIG_FILE=config/your_service_config.yaml

EOF
    echo ".env file created with default values"
fi
```

## Development Workflow Scripts

### Service Development Script

Create a dedicated development script for your service:

**File**: `scripts/dev_your_service.sh`

```bash
#!/bin/bash
# Development workflow for your service

set -e

SERVICE_NAME="your-service"
SERVICE_PORT="${YOUR_SERVICE_PORT:-3007}"

echo "Starting development workflow for ${SERVICE_NAME}..."

# Check prerequisites
echo "Checking prerequisites..."
./scripts/check_setup.sh

# Run tests
echo "Running tests..."
./scripts/test.sh --service your_service

# Start service in development mode
echo "Starting service on port ${SERVICE_PORT}..."
BENTOML_PORT=${SERVICE_PORT} ./scripts/run_bentoml.sh serve services.your_service:YourService --reload

echo "Development workflow completed!"
```

### Hot Reload Development

For services requiring hot reload during development:

**File**: `scripts/dev_hot_reload.sh`

```bash
#!/bin/bash
# Hot reload development for multiple services

# Start your service with reload
BENTOML_PORT=3007 ./scripts/run_bentoml.sh serve services.your_service:YourService --reload &
YOUR_SERVICE_PID=$!

# Cleanup function
cleanup() {
    echo "Shutting down services..."
    kill $YOUR_SERVICE_PID 2>/dev/null || true
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

echo "Services running with hot reload enabled"
echo "Your Service: http://localhost:3007"
echo "Press Ctrl+C to stop all services"

# Wait for services
wait
```

## Production Deployment Scripts

### Production Build Script

**File**: `scripts/build_production.sh`

```bash
#!/bin/bash
# Production build for your service

set -e

SERVICE_NAME="your-service"
VERSION="${1:-latest}"

echo "Building ${SERVICE_NAME} for production..."

# Set production environment
export BENTOML_DEBUG=false
export BENTOML_RELOAD=false

# Build with production settings
BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py

# Tag for production
bentoml models tag ${SERVICE_NAME}:latest ${SERVICE_NAME}:${VERSION}

# Verify build
echo "Verifying production build..."
bentoml list | grep ${SERVICE_NAME}:${VERSION}

echo "Production build completed: ${SERVICE_NAME}:${VERSION}"
```

## Script Best Practices

### 1. Error Handling

```bash
#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Fail if any command in pipe fails

# Function for error handling
handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Check prerequisites
command -v uv >/dev/null 2>&1 || handle_error "UV not installed"
```

### 2. Logging and Output

```bash
# Consistent logging
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Example usage
log_info "Starting service build..."
log_error "Build failed"
```

### 3. Configuration Management

```bash
# Load configuration from environment
SERVICE_PORT="${YOUR_SERVICE_PORT:-3007}"
SERVICE_HOST="${BENTOML_HOST:-127.0.0.1}"
SERVICE_CONFIG="${YOUR_SERVICE_CONFIG:-config/your_service.yaml}"

# Validate configuration
if [ ! -f "$SERVICE_CONFIG" ]; then
    log_error "Configuration file not found: $SERVICE_CONFIG"
fi
```

This integration approach ensures your service works seamlessly with the existing development workflow and automation tools.