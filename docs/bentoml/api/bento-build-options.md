# Bento Build Options

BentoML build options allow you to configure runtime specifications for building a project into a Bento. These can be defined in multiple locations with different formats.

## Configuration Locations

### 1. pyproject.toml
Define build options under the `[tool.bentoml.build]` section:

```toml
[tool.bentoml.build]
service = "service:MyService"
description = "My ML service"
labels = { owner = "ml-team", stage = "production" }
```

### 2. YAML File (bentofile.yaml)
The traditional approach using YAML configuration:

```yaml
service: "service:MyService"
description: |
  ## My ML Service 🚀
  Production-ready model serving
labels:
  owner: ml-team
  stage: production
```

### 3. Python SDK (Recommended since v1.3.20)
Use the new Python SDK for runtime specifications:

```python
import bentoml

@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 60}
)
class MyService:
    # Service implementation
```

## Key Configuration Fields

### Service (Required)
Points to the Service object that defines your ML service.

```yaml
service: "service:MyService"
```

### Description
Annotate your Bento with documentation. Supports Markdown formatting.

**Inline description:**
```yaml
description: "A simple fraud detection service"
```

**Multiline description:**
```yaml
description: |
  ## Fraud Detection Service 🛡️
  
  This service provides real-time fraud detection using:
  - **Random Forest** for feature-based detection
  - **Neural Networks** for pattern recognition
  - **Rule Engine** for business logic
```

**External file reference:**
```yaml
description: "file:README.md"
```

### Labels
Key-value pairs for identifying or categorizing Bentos and models.

```yaml
labels:
  owner: bentoml-team
  stage: development
  version: "1.2.3"
  model_type: classification
```

### File Inclusion/Exclusion

#### Include Files
Specify files and directories to package in the Bento:

```yaml
include:
  - "data/"
  - "models/"
  - "**/*.py"
  - "config.json"
```

#### Exclude Files
Specify files to ignore (supports gitignore-style patterns):

```yaml
exclude:
  - "tests/"
  - "*.pyc"
  - "__pycache__/"
  - ".env"
  - "secrets.key"
```

### Python Dependencies

#### Basic Package Specification
```yaml
python:
  packages:
    - "numpy"
    - "pandas>=1.3.0"
    - "scikit-learn==1.1.0"
    - "git+https://github.com/username/mylib.git@main"
```

#### Advanced Python Configuration
```yaml
python:
  requirements_txt: "requirements.txt"
  packages:
    - "torch>=1.12.0"
    - "torchvision"
  lock_packages: true  # Default: true
  index_url: "https://pypi.org/simple"
  trusted_host: ["pypi.org"]
  find_links: ["https://download.pytorch.org/whl/cpu"]
  extra_index_url: ["https://pypi.anaconda.org/simple"]
```

#### Package Locking
BentoML automatically locks package versions for reproducibility:

```yaml
python:
  packages:
    - "numpy"  # Will be locked to specific version
  lock_packages: false  # Disable version locking
```

### Environment Variables
Set environment variables for configuration and secrets:

```yaml
envs:
  - name: "API_KEY"
    value: "your_api_key_here"
  - name: "MODEL_PATH"
    value: "/opt/ml/models"
  - name: "DEBUG"
    value: "false"
```

### Docker Configuration
Customize Docker image generation:

```yaml
docker:
  distro: debian  # Options: debian, alpine, ubi8
  python_version: "3.9"
  cuda_version: "11.6"
  env:
    - "PYTHONPATH=/opt/ml"
  dockerfile_template: "Dockerfile.template"
```

#### Available Docker Distributions
- `debian`: Default, full-featured
- `alpine`: Smaller image size
- `ubi8`: Red Hat Universal Base Image

### Conda Configuration
Use Conda for environment management:

```yaml
conda:
  environment_yml: "environment.yml"
  channels:
    - conda-forge
    - pytorch
  dependencies:
    - python=3.9
    - numpy
    - pytorch
```

## Complete Example

```yaml
# bentofile.yaml
service: "fraud_service:FraudDetector"

description: |
  ## Fraud Detection Service 🛡️
  
  Real-time fraud detection with 99.8% accuracy
  
  ### Features
  - Real-time scoring
  - Batch processing
  - Model explainability

labels:
  owner: ml-team
  stage: production
  model_version: "2.1.0"

include:
  - "models/"
  - "data/reference/"
  - "config.json"

exclude:
  - "tests/"
  - "notebooks/"
  - "*.log"

python:
  packages:
    - "scikit-learn==1.1.2"
    - "pandas>=1.4.0"
    - "numpy>=1.21.0"
    - "joblib"
  requirements_txt: "requirements.txt"
  lock_packages: true

envs:
  - name: "MODEL_THRESHOLD"
    value: "0.85"
  - name: "ENABLE_MONITORING"
    value: "true"

docker:
  distro: debian
  python_version: "3.9"
  cuda_version: "11.8"
```

## Best Practices

### 1. Version Pinning
Always pin critical dependencies to specific versions in production:

```yaml
python:
  packages:
    - "scikit-learn==1.1.2"  # Pin exact version
    - "pandas>=1.4.0,<2.0.0"  # Pin range
```

### 2. Minimal Inclusions
Only include necessary files to keep Bento size small:

```yaml
include:
  - "models/"
  - "config/"
exclude:
  - "tests/"
  - "docs/"
  - "*.ipynb"
```

### 3. Environment Configuration
Use environment variables for configuration that varies between deployments:

```yaml
envs:
  - name: "LOG_LEVEL"
    value: "INFO"
  - name: "MODEL_CACHE_SIZE"
    value: "1000"
```

### 4. Documentation
Always include comprehensive documentation:

```yaml
description: |
  ## Service Overview
  Brief description of what the service does
  
  ## API Endpoints
  - `/predict`: Main prediction endpoint
  - `/health`: Health check
  
  ## Model Information
  - Algorithm: Random Forest
  - Training Date: 2024-01-15
  - Accuracy: 94.2%
```