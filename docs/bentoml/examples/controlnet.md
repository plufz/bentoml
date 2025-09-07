# ControlNet: Precision Image Generation

ControlNet enhances image generation by using text and image prompts together, providing precise control over composition, pose, and spatial consistency in generated images.

## Overview

ControlNet enables:
- **Pose Control**: Replicate poses from reference images
- **Spatial Consistency**: Maintain structural elements
- **Style Transfer**: Apply new contexts to existing compositions
- **Fine-grained Control**: Precise generation parameters

## Model Configuration

### Components Used
- **ControlNet Model**: `diffusers/controlnet-canny-sdxl-1.0`
- **VAE Model**: `madebyollin/sdxl-vae-fp16-fix`
- **Base Model**: `stabilityai/stable-diffusion-xl-base-1.0`

### Service Implementation
```python
CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

@bentoml.service(
    traffic={"timeout": 600},
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    }
)
class ControlNet:
    controlnet_path = bentoml.models.HuggingFaceModel(CONTROLNET_MODEL_ID)
    vae_path = bentoml.models.HuggingFaceModel(VAE_MODEL_ID)
    
    def __init__(self):
        from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
        import cv2
        
        # Load models
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path.path,
            torch_dtype=torch.float16
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            self.vae_path.path,
            torch_dtype=torch.float16
        )
        
        # Create pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
        ).to("cuda")
```

## API Implementation

### Image Generation Endpoint
```python
@bentoml.api
def generate_controlled_image(
    self,
    prompt: str,
    image: Image,
    negative_prompt: str = "ugly, disfigured, low resolution",
    controlnet_conditioning_scale: float = 0.5,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
) -> Image:
    """Generate image with ControlNet conditioning"""
    
    # Preprocess control image
    control_image = self.preprocess_image(image)
    
    # Generate image
    result = self.pipe(
        prompt=prompt,
        image=control_image,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    return result

def preprocess_image(self, image: Image) -> Image:
    """Apply Canny edge detection for control"""
    import cv2
    import numpy as np
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Apply Canny edge detection
    canny = cv2.Canny(image_array, 100, 200)
    canny_image = Image.fromarray(canny)
    
    return canny_image
```

## Deployment

### BentoCloud Deployment
```bash
pip install bentoml
bentoml cloud login
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/controlnet
bentoml deploy
```

### Local Serving
```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/controlnet
pip install -r requirements.txt
bentoml serve
```

## Usage Examples

### Example Request
```json
{
   "prompt": "A young man walking in a park, wearing jeans",
   "negative_prompt": "ugly, disfigured, ill-structured, low resolution",
   "controlnet_conditioning_scale": 0.5,
   "num_inference_steps": 25,
   "image": "reference-pose.png"
}
```

### Python Client
```python
import bentoml
from PIL import Image

# Load reference image
reference_image = Image.open("pose_reference.jpg")

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    generated_image = client.generate_controlled_image(
        prompt="A professional dancer in elegant attire",
        image=reference_image,
        controlnet_conditioning_scale=0.7,
        num_inference_steps=30
    )
    
    generated_image.save("controlled_output.png")
```

### Advanced Usage
```python
# High precision control
result = client.generate_controlled_image(
    prompt="A robot in the same pose",
    image=reference_image,
    controlnet_conditioning_scale=0.9,  # Strong control
    num_inference_steps=50,
    guidance_scale=8.0
)

# Loose style adaptation
result = client.generate_controlled_image(
    prompt="An abstract painting inspired by this pose",
    image=reference_image,
    controlnet_conditioning_scale=0.3,  # Loose control
    num_inference_steps=20,
    guidance_scale=6.0
)
```

## Use Cases

### Pose Replication
- Character animation
- Fashion modeling
- Fitness demonstrations
- Dance choreography

### Architectural Control
- Building design variations
- Interior layout planning
- Landscape architecture
- Urban planning

### Artistic Applications
- Style transfer with spatial control
- Concept art generation
- Illustration creation
- Digital art workflows

## Performance Optimization

### Memory Management
- FP16 precision for efficiency
- Model offloading strategies
- Batch processing support
- GPU memory optimization

### Quality Control
- Optimal conditioning scales
- Inference step tuning
- Guidance scale adjustment
- Preprocessing optimization