# Configuration Guide

This guide covers service configuration, Bento build setup, resource management, and environment configuration for BentoML services.

## Configuration Architecture

```
config/
├── bentoml.yaml           # Global BentoML configuration
├── bentofiles/            # Service-specific build configurations
│   ├── stable-diffusion.yaml
│   ├── llava.yaml
│   ├── whisper.yaml
│   ├── upscaler.yaml
│   ├── rag.yaml
│   └── multi-service.yaml
└── .env                   # Environment variables (local)
```

## Bento Build Configuration

### Basic Bentofile

Every service needs a Bentofile for building deployable packages.

**File**: `config/bentofiles/your-service.yaml`

```yaml
service: "services.your_service:YourService"
name: "your-service"
version: "latest"

labels:
  owner: "bentoml-team"
  stage: "development"
  service_type: "inference"

description: "Your service description and purpose"

include:
  - "services/your_service.py"
  - "utils/your_service_utils.py"
  - "config/your_service_config.yaml"

exclude:
  - "tests/"
  - "docs/"
  - "__pycache__/"
  - "*.pyc"

python:
  requirements_txt: |
    bentoml[io]>=1.4.0
    pydantic>=2.5.0
    # Add your service-specific dependencies here
    transformers>=4.30.0
    torch>=2.1.0
  
  lock_packages: true
  index_url: "https://pypi.org/simple"
  no_index: false
  trusted_host: []
  find_links: []
  extra_index_url: []

models: []

docker:
  distro: "debian"
  python_version: "3.11"
  cuda_version: null  # Set to "11.8" if GPU needed
  system_packages: []
  run_as_user: "bentoml"
```

### Model-Heavy Service Configuration

**File**: `config/bentofiles/model-service.yaml`

```yaml
service: "services.model_service:ModelService"
name: "model-service"

labels:
  owner: "ai-team"
  stage: "development"
  gpu_required: "true"
  model_size: "large"

include:
  - "services/model_service.py"
  - "utils/model_utils.py"
  - "models/"  # Include model files if stored locally

python:
  requirements_txt: |
    bentoml[io]>=1.4.0
    torch>=2.1.0
    torchvision>=0.16.0
    transformers>=4.30.0
    accelerate>=0.25.0
    diffusers>=0.25.0
    xformers>=0.0.22  # For memory optimization

models:
  - tag: "stable-diffusion:latest"
    filter: "latest"
  - tag: "your-model:v1.0"

docker:
  distro: "debian"
  python_version: "3.11"
  cuda_version: "11.8"  # Enable CUDA
  system_packages:
    - "git"
    - "wget"
    - "curl"
  run_as_user: "bentoml"
  dockerfile_template: |
    # Custom Dockerfile additions
    ENV HF_HOME=/tmp/huggingface
    ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Multi-Service Configuration

**File**: `config/bentofiles/multi-service.yaml`

```yaml
service: "services.multi_service:MultiService"
name: "multi-service"

labels:
  owner: "platform-team"
  stage: "production"
  service_type: "composition"
  endpoints: "17"

description: "Unified multi-service with all available endpoints"

include:
  - "services/"
  - "utils/"
  - "config/bentoml.yaml"

exclude:
  - "tests/"
  - "docs/"
  - "scripts/"
  - "__pycache__/"

python:
  requirements_txt: |
    # Core BentoML
    bentoml[io]>=1.4.0
    pydantic>=2.5.0
    
    # Image generation
    diffusers>=0.25.0
    transformers>=4.30.0
    torch>=2.1.0
    torchvision>=0.16.0
    accelerate>=0.25.0
    
    # Vision-language
    llama-cpp-python>=0.2.27
    
    # Audio processing
    whisper>=1.0.0
    
    # Image upscaling
    realesrgan>=0.3.0
    gfpgan>=1.3.8
    
    # RAG capabilities
    llama-index-core>=0.10.0
    sentence-transformers>=2.2.0
    pymilvus>=2.3.0
    pypdf>=3.0.0
    
    # Common utilities
    pillow>=10.0.0
    requests>=2.28.0
    numpy>=1.21.0

docker:
  distro: "debian"
  python_version: "3.11"
  cuda_version: "11.8"
  system_packages:
    - "git"
    - "wget"
    - "curl"
    - "ffmpeg"  # For audio processing
  run_as_user: "bentoml"
```

## Global BentoML Configuration

**File**: `config/bentoml.yaml`

```yaml
# Global BentoML configuration
version: 1

api_server:
  host: "127.0.0.1"
  port: 3000
  backlog: 2048
  timeout: 60
  max_request_size: 20971520  # 20MB
  workers: 1  # For development
  
  cors:
    enabled: true
    access_control_allow_origin: "*"
    access_control_allow_methods: ["GET", "POST", "HEAD", "OPTIONS"]
    access_control_allow_headers: ["Content-Type", "Authorization"]
    access_control_allow_credentials: false
    access_control_max_age: 1200
  
  ssl:
    enabled: false
    certfile: null
    keyfile: null
    keyfile_password: null
    ca_certs: null
    ssl_version: null
    cert_reqs: null
    ciphers: null

tracing:
  sample_rate: 1.0
  type: "jaeger"
  jaeger:
    protocol: "thrift"
    collector_endpoint: "http://localhost:14268/api/traces"
    
logging:
  level: "INFO"
  format: "text"
  
yatai:
  default_server: "local"
  
runners:
  batching:
    enabled: true
    max_batch_size: 100
    max_latency_ms: 500
```

## Environment Variables

### Local Development (.env)

**File**: `.env` (root directory)

```bash
# BentoML Configuration
BENTOML_HOST=127.0.0.1
BENTOML_PORT=3000
BENTOML_PROTOCOL=http

# Service-specific ports for individual testing
EXAMPLE_SERVICE_PORT=3001
STABLE_DIFFUSION_PORT=3002
LLAVA_SERVICE_PORT=3003
WHISPER_SERVICE_PORT=3004
UPSCALER_SERVICE_PORT=3005
RAG_SERVICE_PORT=3006

# Model and Cache Directories
HUGGINGFACE_HUB_CACHE=/Volumes/Second/huggingface
HF_HOME=/Volumes/Second/huggingface
TORCH_HOME=/Volumes/Second/torch
TRANSFORMERS_CACHE=/Volumes/Second/transformers

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# RAG Configuration
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_DB_NAME=rag_db

# Development Settings
BENTOML_DEBUG=true
BENTOML_RELOAD=true
BENTOML_ACCESS_LOG=true

# Security (development only - use secrets management in production)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Production Environment Variables

```bash
# Production BentoML Configuration
BENTOML_HOST=0.0.0.0
BENTOML_PORT=3000
BENTOML_WORKERS=4

# Security
BENTOML_DEBUG=false
BENTOML_RELOAD=false

# Resource Management
BENTOML_RUNNER_MAX_CONCURRENT_REQUESTS=10
BENTOML_API_SERVER_MAX_REQUEST_SIZE=52428800  # 50MB

# Monitoring
BENTOML_MONITORING_ENABLED=true
OPENTELEMETRY_ENDPOINT=http://otel-collector:4317

# External Services
DATABASE_URL=postgresql://user:pass@db:5432/bentoml
REDIS_URL=redis://redis:6379/0
MILVUS_HOST=milvus
MILVUS_PORT=19530
```

## Resource Configuration

### Service Resource Specifications

```yaml
# In your Bentofile or service decorator
@bentoml.service(
    name="resource-managed-service",
    resources={
        "cpu": "2000m",      # 2 CPU cores
        "memory": "4Gi",     # 4GB RAM
        "gpu": "1",          # 1 GPU
        "gpu_type": "nvidia-tesla-v100"
    },
    timeout=300,             # 5 minute timeout
    workers=2                # Number of worker processes
)
```

### Resource Configuration Examples

#### CPU-Intensive Service
```yaml
resources:
  cpu: "4000m"      # 4 CPU cores
  memory: "8Gi"     # 8GB RAM
  gpu: "0"          # No GPU
```

#### GPU-Intensive Service
```yaml
resources:
  cpu: "2000m"      # 2 CPU cores
  memory: "16Gi"    # 16GB RAM
  gpu: "1"          # 1 GPU
  gpu_type: "nvidia-tesla-v100"
```

#### Memory-Intensive Service
```yaml
resources:
  cpu: "1000m"      # 1 CPU core
  memory: "32Gi"    # 32GB RAM
  gpu: "0"          # No GPU
```

#### Lightweight Service
```yaml
resources:
  cpu: "500m"       # 0.5 CPU cores
  memory: "1Gi"     # 1GB RAM
  gpu: "0"          # No GPU
```

## Service-Specific Configuration

### Configuration Files

For services requiring additional configuration, create dedicated config files:

**File**: `config/stable_diffusion_config.yaml`

```yaml
model:
  name: "runwayml/stable-diffusion-v1-5"
  revision: "main"
  torch_dtype: "float16"
  
generation:
  default_steps: 20
  default_guidance_scale: 7.5
  max_width: 1024
  max_height: 1024
  
scheduler:
  type: "DPMSolverMultistepScheduler"
  
safety:
  safety_checker: true
  requires_safety_checker: true
```

**File**: `config/rag_config.yaml`

```yaml
vector_store:
  type: "milvus"
  host: "127.0.0.1"
  port: 19530
  collection_name: "documents"
  
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  
llm:
  model_path: "/path/to/model.gguf"
  max_tokens: 2048
  temperature: 0.1
  
indexing:
  chunk_size: 1000
  chunk_overlap: 200
```

### Loading Configuration in Services

```python
import yaml
import os
from pathlib import Path

@bentoml.service()
class ConfiguredService:
    def __init__(self):
        self.config = self._load_config()
        self.model = self._initialize_model()
    
    def _load_config(self) -> dict:
        """Load service configuration"""
        config_path = Path("config/your_service_config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._default_config()
        
        # Override with environment variables
        config = self._override_with_env(config)
        return config
    
    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            "model": {"name": "default-model"},
            "generation": {"steps": 20}
        }
    
    def _override_with_env(self, config: dict) -> dict:
        """Override config with environment variables"""
        # Example: MODEL_NAME env var overrides config['model']['name']
        if model_name := os.getenv('MODEL_NAME'):
            config['model']['name'] = model_name
        
        return config
```

## Docker Configuration

### Custom Dockerfile Template

```dockerfile
# Custom additions to Bentofile docker section
dockerfile_template: |
  # Install system dependencies
  RUN apt-get update && apt-get install -y \\
      ffmpeg \\
      libsm6 \\
      libxext6 \\
      libxrender-dev \\
      libglib2.0-0 \\
      && rm -rf /var/lib/apt/lists/*
  
  # Set up model cache directories
  ENV HF_HOME=/tmp/huggingface
  ENV TORCH_HOME=/tmp/torch
  ENV TRANSFORMERS_CACHE=/tmp/transformers
  
  # GPU memory management
  ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  
  # Create cache directories
  RUN mkdir -p /tmp/huggingface /tmp/torch /tmp/transformers
  
  # Set permissions
  RUN chown -R bentoml:bentoml /tmp/huggingface /tmp/torch /tmp/transformers
```

## Configuration Best Practices

### 1. Environment-Specific Configuration

- Use `.env` for local development
- Use environment variables for production
- Never commit secrets to version control
- Use separate configs for dev/staging/production

### 2. Resource Management

- Set appropriate CPU/memory limits
- Use GPU only when necessary
- Monitor resource usage and adjust
- Consider auto-scaling requirements

### 3. Model Configuration

- Pin model versions for reproducibility
- Use appropriate data types (float16 for GPU memory)
- Configure model caching appropriately
- Set reasonable timeout values

### 4. Security Configuration

- Use secrets management in production
- Enable SSL/TLS for production
- Configure CORS appropriately
- Set secure headers

### 5. Monitoring and Observability

- Enable tracing and metrics
- Configure appropriate log levels
- Set up health checks
- Monitor resource usage

## Troubleshooting Configuration

### Common Issues

1. **Port conflicts**: Multiple services trying to use same port
2. **Resource limits**: Services exceeding memory/CPU limits
3. **Model loading failures**: Incorrect model paths or permissions
4. **Dependency conflicts**: Incompatible package versions

### Solutions

1. **Use environment-specific ports**: Configure different ports per environment
2. **Monitor resource usage**: Use appropriate resource limits
3. **Verify model paths**: Ensure models are accessible
4. **Pin dependency versions**: Use specific versions in requirements

### Configuration Validation

```python
def validate_config(config: dict) -> None:
    """Validate service configuration"""
    required_keys = ['model', 'generation']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if config['model']['name'] is None:
        raise ValueError("Model name must be specified")
    
    # Additional validation logic
```

This configuration approach ensures services are properly configured for different environments while maintaining flexibility and security.