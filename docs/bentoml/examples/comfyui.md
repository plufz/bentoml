# ComfyUI: Deploy Workflows as APIs

ComfyUI workflows can be transformed into production-grade APIs using comfy-pack, enabling complex diffusion workflows to be served as HTTP endpoints with standardized interfaces.

## Overview

ComfyUI integration provides:
- **Workflow API Conversion**: Transform visual workflows into HTTP APIs
- **Standardized Schemas**: Define input/output specifications
- **Production Deployment**: Enterprise-grade serving capabilities
- **Portable Workflows**: Package workflows as reusable artifacts

## Installation

### ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for `comfy-pack`
3. Click **Install**
4. Restart and refresh ComfyUI

### Git Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bentoml/comfy-pack.git
```

## Workflow Configuration

### Input Nodes
Define workflow inputs using specialized nodes:
- **Image Input**: Accept image uploads
- **String Input**: Text prompts and parameters
- **Int Input**: Numeric parameters
- **File Input**: File uploads
- **Any Input**: Generic input types

### Output Nodes
Specify workflow outputs:
- **File Output**: Generated files
- **Image Output**: Generated images

## Serving Workflows

### Start Local Server
1. Click **Serve** on the toolbar
2. Set port (default: 3000)
3. Click **Start**

### API Usage Examples

#### cURL Request
```bash
curl -X 'POST' \
    'http://127.0.0.1:3000/generate' \
    -H 'accept: application/octet-stream' \
    -H 'Content-Type: application/json' \
    --output output.png \
    -d '{
    "prompt": "rocks in a bottle",
    "width": 512,
    "height": 512,
    "seed": 1
}'
```

#### Python Client
```python
import bentoml

with bentoml.SyncHTTPClient("http://127.0.0.1:3000") as client:
    result = client.generate(
        prompt="rocks in a bottle",
        width=512,
        height=512,
        seed=1
    )
    
    # Save generated image
    with open("output.png", "wb") as f:
        f.write(result)
```

## BentoCloud Deployment

### Deployment Process
1. Click **Deploy** on toolbar
2. Set deployment name
3. Select models and packages
4. Configure resources
5. Deploy to BentoCloud

### Advanced Features
- **Auto-scaling**: Handle variable workloads
- **Monitoring**: Track performance metrics
- **Version Control**: Manage workflow versions
- **Security**: Authentication and authorization

## Workflow Examples

### Text-to-Image Generation
- Input: Text prompt, dimensions, seed
- Processing: Diffusion model pipeline
- Output: Generated image

### Image-to-Image Translation
- Input: Source image, style prompt
- Processing: Style transfer pipeline
- Output: Stylized image

### Complex Multi-Stage Workflows
- Multiple model chaining
- Conditional processing
- Batch operations

## Production Considerations

### Performance Optimization
- GPU acceleration
- Model caching
- Batch processing
- Memory management

### Scalability
- Horizontal scaling
- Load balancing
- Resource allocation
- Queue management

### Monitoring
- Request metrics
- Processing times
- Error rates
- Resource utilization