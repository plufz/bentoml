# Service Structure Troubleshooting Guide

This guide helps resolve common issues when creating, updating, and maintaining BentoML services in this project.

## Service Creation Issues

### 1. Service Won't Build

#### Problem: Bentofile Configuration Errors
```bash
Error: Failed to build Bento: Invalid service path
```

**Solutions:**
```yaml
# ❌ Incorrect service path
service: "your_service:YourService"

# ✅ Correct service path
service: "services.your_service:YourService"

# ❌ Missing required fields
service: "services.your_service:YourService"

# ✅ Complete configuration
service: "services.your_service:YourService"
name: "your-service"
python:
  requirements_txt: |
    bentoml[io]>=1.4.0
```

#### Problem: Dependency Installation Failures
```bash
Error: Could not install packages due to an EnvironmentError
```

**Solutions:**
1. **Check dependency versions:**
   ```yaml
   python:
     requirements_txt: |
       # ❌ Incompatible versions
       torch>=2.1.0
       torchvision>=0.15.0  # Too old for torch 2.1
       
       # ✅ Compatible versions
       torch>=2.1.0
       torchvision>=0.16.0
   ```

2. **Add system dependencies:**
   ```yaml
   docker:
     system_packages:
       - "ffmpeg"       # For audio processing
       - "libgl1-mesa-glx"  # For OpenCV
       - "libglib2.0-0"     # For some ML libraries
   ```

3. **Use UV for consistent builds:**
   ```bash
   # Build with UV environment
   uv run ./scripts/run_bentoml.sh build services/your_service.py
   ```

### 2. Service Won't Start

#### Problem: Import Errors
```python
ImportError: No module named 'services.your_service'
```

**Solutions:**
1. **Check Python path:**
   ```bash
   # Ensure you're running from project root
   pwd  # Should show /path/to/bentoml
   ls services/  # Should show your_service.py
   ```

2. **Verify service syntax:**
   ```python
   # ❌ Missing decorator
   class YourService:
       pass
   
   # ✅ Proper service definition
   @bentoml.service()
   class YourService:
       pass
   ```

#### Problem: Model Loading Failures
```python
FileNotFoundError: Model file not found
```

**Solutions:**
1. **Check model paths:**
   ```python
   # ❌ Hardcoded absolute path
   model_path = "/absolute/path/to/model"
   
   # ✅ Relative or configurable path
   model_path = os.getenv("MODEL_PATH", "models/your_model.bin")
   ```

2. **Implement graceful model loading:**
   ```python
   def __init__(self):
       try:
           self.model = self._load_model()
           logger.info("Model loaded successfully")
       except Exception as e:
           logger.error(f"Model loading failed: {e}")
           self.model = None  # Graceful degradation
   ```

## Testing Issues

### 3. Tests Fail to Run

#### Problem: Mock Import Errors
```python
ImportError: cannot import name 'your_service_function' from 'services.your_service'
```

**Solutions:**
1. **Check import paths:**
   ```python
   # ❌ Wrong import
   from your_service import YourService
   
   # ✅ Correct import
   from services.your_service import YourService
   ```

2. **Mock at correct level:**
   ```python
   # ❌ Mocking too deep
   @patch('torch.nn.Module.forward')
   
   # ✅ Mock at service level
   @patch('services.your_service.YourService._load_model')
   ```

#### Problem: Integration Tests Timeout
```bash
FAILED tests/test_your_service.py::TestIntegration::test_service_startup - TimeoutError
```

**Solutions:**
1. **Increase timeout for heavy services:**
   ```python
   @pytest.mark.timeout(180)  # 3 minutes for GPU services
   def test_service_startup(self, service_client):
       # Test implementation
   ```

2. **Mock heavy dependencies in integration tests:**
   ```python
   @pytest.fixture(scope="class")
   def service_client(self):
       # Mock heavy model loading even in integration tests
       with patch('services.your_service.load_heavy_model'):
           with get_client(YourService) as client:
               yield client
   ```

### 4. Test Coverage Issues

#### Problem: Low Coverage on Service Initialization
```bash
Missing coverage on __init__ method
```

**Solutions:**
1. **Test initialization separately:**
   ```python
   def test_service_initialization(self):
       """Test service initializes correctly"""
       with patch('services.your_service.load_model') as mock_load:
           service = YourService()
           assert service is not None
           mock_load.assert_called_once()
   ```

2. **Test initialization failures:**
   ```python
   def test_service_initialization_failure(self):
       """Test service handles initialization failure"""
       with patch('services.your_service.load_model', side_effect=Exception("Load failed")):
           with pytest.raises(Exception):
               YourService()
   ```

## Configuration Issues

### 5. Resource Configuration Problems

#### Problem: Out of Memory Errors
```bash
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Adjust service resources:**
   ```python
   @bentoml.service(
       resources={
           "memory": "16Gi",    # Increase memory
           "gpu": "1",
           "gpu_type": "nvidia-tesla-v100"
       }
   )
   ```

2. **Implement memory management:**
   ```python
   def __init__(self):
       # Enable memory optimization
       os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
       self.model = self._load_model_optimized()
   
   def _load_model_optimized(self):
       """Load model with memory optimization"""
       import torch
       if torch.cuda.is_available():
           return model.half()  # Use float16 to save memory
       return model
   ```

#### Problem: Port Conflicts
```bash
Error: Address already in use (port 3000)
```

**Solutions:**
1. **Use environment-specific ports:**
   ```bash
   # In .env file
   YOUR_SERVICE_PORT=3007
   
   # In script
   BENTOML_PORT=${YOUR_SERVICE_PORT} ./scripts/run_bentoml.sh serve services.your_service:YourService
   ```

2. **Check for running services:**
   ```bash
   # Check what's using port 3000
   lsof -i :3000
   
   # Kill conflicting processes if needed
   kill -9 <PID>
   ```

### 6. Environment Variable Issues

#### Problem: Configuration Not Loading
```bash
Service using default config instead of environment variables
```

**Solutions:**
1. **Verify environment loading:**
   ```python
   # Debug environment loading
   def __init__(self):
       print(f"MODEL_NAME env: {os.getenv('MODEL_NAME')}")
       print(f"Current working dir: {os.getcwd()}")
       self.config = self._load_config()
   ```

2. **Use explicit environment loading:**
   ```python
   from dotenv import load_dotenv
   
   def __init__(self):
       load_dotenv()  # Explicitly load .env file
       self.config = self._load_config()
   ```

## Script Integration Issues

### 7. Build Script Problems

#### Problem: Service Not Building in Batch
```bash
Building Your Service...
Error: Bentofile not found
```

**Solutions:**
1. **Check build script syntax:**
   ```bash
   # ❌ Missing BENTOFILE environment variable
   ./scripts/run_bentoml.sh build services/your_service.py
   
   # ✅ Correct with Bentofile
   BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py
   ```

2. **Verify file paths in build script:**
   ```bash
   # Add debugging to build script
   echo "Building Your Service..."
   echo "Bentofile: ${BENTOFILE}"
   echo "Service file: services/your_service.py"
   ls -la config/bentofiles/your-service.yaml  # Verify file exists
   BENTOFILE=config/bentofiles/your-service.yaml ./scripts/run_bentoml.sh build services/your_service.py
   ```

### 8. Test Script Integration

#### Problem: Service Not Found in Test Script
```bash
Unknown service: your_service
Available services: example, llava, ...
```

**Solutions:**
1. **Add service to test script:**
   ```bash
   # In scripts/test.sh, add to case statement:
   your_service)
       echo "Running Your Service tests..."
       uv run pytest tests/test_your_service.py -v
       ;;
   ```

2. **Update help message:**
   ```bash
   # In show_help() function:
   echo "  --service NAME  Run tests for specific service"
   echo "                  Available: example, llava, ..., your_service"
   ```

## Multi-Service Integration Issues

### 9. Multi-Service Import Problems

#### Problem: Circular Import Errors
```python
ImportError: cannot import name 'YourService' from partially initialized module
```

**Solutions:**
1. **Use lazy imports in multi-service:**
   ```python
   @bentoml.service()
   class MultiService:
       def __init__(self):
           # Lazy import to avoid circular dependencies
           from services.your_service import YourService
           self.your_service = YourService()
   ```

2. **Check import order:**
   ```python
   # ❌ Problematic import order
   from services.multi_service import MultiService
   from services.your_service import YourService
   
   # ✅ Import only what you need when you need it
   ```

### 10. Multi-Service Resource Conflicts

#### Problem: Services Competing for Resources
```bash
RuntimeError: Resource exhausted
```

**Solutions:**
1. **Optimize service initialization order:**
   ```python
   def __init__(self):
       # Initialize lightweight services first
       self.example_service = ExampleService()
       
       # Then resource-intensive services
       if self._sufficient_resources():
           self.your_heavy_service = YourHeavyService()
       else:
           logger.warning("Insufficient resources for heavy service")
           self.your_heavy_service = None
   ```

2. **Implement resource sharing:**
   ```python
   def __init__(self):
       # Share GPU context between services
       self._gpu_context = self._initialize_gpu()
       
       self.service_a = ServiceA(gpu_context=self._gpu_context)
       self.service_b = ServiceB(gpu_context=self._gpu_context)
   ```

## Performance Issues

### 11. Slow Service Startup

#### Problem: Service Takes Too Long to Initialize
```bash
Service startup timeout after 60 seconds
```

**Solutions:**
1. **Implement lazy loading:**
   ```python
   def __init__(self):
       # Don't load model in __init__
       self._model = None
       self._model_loaded = False
   
   @property
   def model(self):
       """Lazy load model on first access"""
       if not self._model_loaded:
           self._model = self._load_model()
           self._model_loaded = True
       return self._model
   ```

2. **Use background model loading:**
   ```python
   import threading
   
   def __init__(self):
       self._model = None
       self._model_ready = threading.Event()
       
       # Start loading in background
       threading.Thread(target=self._load_model_async, daemon=True).start()
   
   def _load_model_async(self):
       """Load model asynchronously"""
       self._model = self._load_model()
       self._model_ready.set()
   
   @bentoml.api
   def your_endpoint(self, request):
       # Wait for model if needed
       self._model_ready.wait(timeout=120)
       # Process request
   ```

### 12. High Memory Usage

#### Problem: Service Uses Too Much Memory
```bash
Service killed due to memory limit
```

**Solutions:**
1. **Implement model optimization:**
   ```python
   def _load_model_optimized(self):
       """Load model with memory optimization"""
       model = self._load_base_model()
       
       # Use half precision
       if torch.cuda.is_available():
           model = model.half()
       
       # Enable memory-efficient attention
       model.enable_memory_efficient_attention()
       
       return model
   ```

2. **Clean up after processing:**
   ```python
   @bentoml.api
   def process_large_data(self, request):
       """Process with memory cleanup"""
       try:
           result = self._process(request)
           return result
       finally:
           # Explicit cleanup
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
           import gc
           gc.collect()
   ```

## Debugging Tips

### 13. Enable Debug Logging

```python
import logging
import os

# Enable debug logging
if os.getenv('DEBUG', 'false').lower() == 'true':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
```

### 14. Add Health Checks

```python
@bentoml.api
def health(self) -> dict:
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check model loading
    health_status["checks"]["model_loaded"] = self._model is not None
    
    # Check dependencies
    try:
        import torch
        health_status["checks"]["torch_available"] = torch.cuda.is_available()
    except ImportError:
        health_status["checks"]["torch_available"] = False
    
    # Check disk space
    import shutil
    free_space = shutil.disk_usage("/").free / (1024**3)  # GB
    health_status["checks"]["disk_space_gb"] = free_space
    
    # Overall status
    if not all(health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    return health_status
```

### 15. Test Service Manually

```bash
# Test service startup
./scripts/run_bentoml.sh serve services.your_service:YourService

# Test endpoints manually
curl -X POST http://localhost:3000/your_endpoint \
  -H "Content-Type: application/json" \
  -d '{"request": {"input_field": "test"}}'

# Check service health
./scripts/health.sh

# Test with endpoint script
./scripts/endpoint.sh your_endpoint '{"input_field": "debug test"}' --verbose
```

This troubleshooting guide covers the most common issues you'll encounter when working with BentoML services in this project. Always check logs for detailed error messages and use the debugging techniques provided.