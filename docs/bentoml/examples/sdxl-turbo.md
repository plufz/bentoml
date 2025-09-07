# Stable Diffusion XL Turbo

Stable Diffusion XL Turbo is a single-step image generation model capable of creating high-quality images quickly with minimal inference steps.

## Overview

SDXL Turbo provides:
- **Fast Generation**: Single-step image creation
- **High Quality**: Maintains image fidelity with reduced steps
- **Customizable Parameters**: Control generation with various settings
- **Production Ready**: Optimized for serving at scale

## Implementation

### Model Configuration
```python
MODEL_ID = "stabilityai/sdxl-turbo"

@bentoml.service(
    traffic={"timeout": 300},
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class SDXLTurbo:
    model_path = bentoml.models.HuggingFaceModel(MODEL_ID)
    
    def __init__(self):
        from diffusers import AutoPipelineForText2Image
        
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_path.path,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
```

### Text-to-Image API
```python
@bentoml.api
def txt2img(
    self,
    prompt: str = "A beautiful landscape with mountains",
    num_inference_steps: Annotated[int, Ge(1), Le(10)] = 1,
    guidance_scale: float = 0.0,
    width: int = 512,
    height: int = 512,
) -> Image:
    """Generate image from text prompt"""
    
    image = self.pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
    ).images[0]
    
    return image
```

## Deployment

### BentoCloud
```bash
pip install bentoml
bentoml cloud login
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sdxl-turbo
bentoml deploy
```

### Local Serving
```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sdxl-turbo
pip install -r requirements.txt
bentoml serve
```

## Usage

### Python Client
```python
import bentoml
from PIL import Image

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    image = client.txt2img(
        prompt="A futuristic city skyline at sunset",
        num_inference_steps=2,
        guidance_scale=0.0
    )
    image.save("generated_image.png")
```

### HTTP API
```bash
curl -X POST "http://localhost:3000/txt2img" \
     -H "Content-Type: application/json" \
     --output image.png \
     -d '{
       "prompt": "A serene lake surrounded by mountains",
       "num_inference_steps": 1,
       "guidance_scale": 0.0,
       "width": 768,
       "height": 768
     }'
```

## Performance Features

### Optimizations
- Single-step generation for speed
- FP16 precision for memory efficiency
- GPU acceleration support
- Batch processing capabilities

### Customization Options
- **Inference Steps**: 1-10 steps (1 recommended for turbo)
- **Guidance Scale**: Usually 0.0 for turbo models
- **Resolution**: Standard or custom dimensions
- **Batch Size**: Multiple images per request