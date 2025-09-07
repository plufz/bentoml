# BentoML Diffusers Integration

Comprehensive API reference for integrating Hugging Face Diffusers models with BentoML for image generation and diffusion model deployment.

## Overview

BentoML provides robust integration with Hugging Face Diffusers, supporting stable diffusion pipelines, image generation models, and various diffusion-based tasks with optimized performance and configuration options.

## Core Functions

### import_model()

Import a Diffusion model from Hugging Face Hub or local directory.

```python
bentoml.diffusers.import_model(
    name: str,
    model_id: str,
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    metadata: dict = None,
    **kwargs
) -> bentoml.Tag
```

**Parameters:**
- `name` (str): Model name for identification in BentoML store
- `model_id` (str): Hugging Face model ID or local directory path
- `signatures` (dict, optional): Model signature configuration
- `labels` (dict, optional): Labels for model organization
- `custom_objects` (dict, optional): Additional objects to save
- `metadata` (dict, optional): Custom metadata
- `**kwargs`: Additional arguments passed to diffusers pipeline

**Returns:** `bentoml.Tag` with model name and version

#### Basic Import

```python
import bentoml

# Import Stable Diffusion 1.5
tag = bentoml.diffusers.import_model(
    name="stable_diffusion_v15",
    model_id="runwayml/stable-diffusion-v1-5"
)
print(f"Model imported: {tag}")
```

#### Advanced Import with Configuration

```python
# Import with specific variant and configuration
tag = bentoml.diffusers.import_model(
    name="stable_diffusion_v15_fp16",
    model_id="runwayml/stable-diffusion-v1-5",
    signatures={
        "__call__": {
            "batchable": False  # Disable batching for image generation
        }
    },
    labels={
        "model_type": "text_to_image",
        "precision": "fp16",
        "stage": "production"
    },
    metadata={
        "resolution": 512,
        "max_steps": 50,
        "guidance_scale": 7.5,
        "model_size": "4.3GB"
    },
    variant="fp16",  # Use fp16 variant for memory efficiency
    torch_dtype="float16",
    use_safetensors=True
)
```

### save_model()

Save a DiffusionPipeline to BentoML's model store.

```python
bentoml.diffusers.save_model(
    name: str,
    pipeline: diffusers.DiffusionPipeline,
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    metadata: dict = None
) -> bentoml.Tag
```

#### Save Custom Pipeline

```python
from diffusers import StableDiffusionPipeline
import torch

# Load and customize pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16"
)

# Apply optimizations
pipeline.enable_memory_efficient_attention()
pipeline.enable_vae_slicing()

# Save customized pipeline
tag = bentoml.diffusers.save_model(
    name="optimized_sd_pipeline",
    pipeline=pipeline,
    metadata={
        "optimizations": ["memory_efficient_attention", "vae_slicing"],
        "torch_dtype": "float16"
    }
)
```

### load_model()

Load a saved Diffusion model from BentoML's model store.

```python
bentoml.diffusers.load_model(
    tag: str,
    device: str = None,
    **kwargs
) -> diffusers.DiffusionPipeline
```

**Parameters:**
- `tag` (str): Model tag (name:version or name for latest)
- `device` (str, optional): Target device ("cuda", "cpu", etc.)
- `**kwargs`: Additional configuration options

#### Basic Loading

```python
# Load latest version
pipeline = bentoml.diffusers.load_model("stable_diffusion_v15:latest")

# Load to specific device
pipeline = bentoml.diffusers.load_model(
    "stable_diffusion_v15:latest",
    device="cuda"
)
```

## Service Integration

### Text-to-Image Generation Service

```python
import bentoml
import torch
from PIL import Image
from typing import Optional, List
import io
import base64

@bentoml.service(
    resources={"gpu": "1", "memory": "12Gi"},
    traffic={"timeout": 300}  # Extended timeout for generation
)
class StableDiffusionService:
    def __init__(self):
        # Load Stable Diffusion pipeline
        self.pipeline = bentoml.diffusers.load_model(
            "stable_diffusion_v15:latest",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load model metadata
        model_ref = bentoml.diffusers.get("stable_diffusion_v15:latest")
        self.default_steps = model_ref.metadata.get("max_steps", 20)
        self.default_guidance = model_ref.metadata.get("guidance_scale", 7.5)
        self.default_resolution = model_ref.metadata.get("resolution", 512)
    
    @bentoml.api
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None
    ) -> dict:
        # Use provided parameters or defaults
        steps = num_inference_steps or self.default_steps
        guidance = guidance_scale or self.default_guidance
        width = width or self.default_resolution
        height = height or self.default_resolution
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image
        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height
                ).images[0]
            
            # Convert image to base64 for JSON response
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": image_b64,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "parameters": {
                    "steps": steps,
                    "guidance_scale": guidance,
                    "width": width,
                    "height": height,
                    "seed": seed
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
    @bentoml.api
    def generate_batch(
        self,
        prompts: List[str],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> List[dict]:
        results = []
        
        for prompt in prompts:
            result = self.generate_image(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            results.append(result)
        
        return results
```

### Image-to-Image Generation Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "12Gi"}
)
class ImageToImageService:
    def __init__(self):
        # Load Image-to-Image pipeline
        self.pipeline = bentoml.diffusers.load_model("img2img_pipeline:latest")
    
    @bentoml.api
    def transform_image(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> dict:
        try:
            # Ensure image is RGB and correct size
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if needed (maintain aspect ratio)
            max_size = 512
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Generate transformed image
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result_image = self.pipeline(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            result_image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "transformed_image": image_b64,
                "original_size": image.size,
                "result_size": result_image.size,
                "parameters": {
                    "prompt": prompt,
                    "strength": strength,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Inpainting Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "12Gi"}
)
class InpaintingService:
    def __init__(self):
        self.pipeline = bentoml.diffusers.load_model("inpainting_pipeline:latest")
    
    @bentoml.api
    def inpaint_image(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> dict:
        try:
            # Ensure images are RGB and same size
            image = image.convert('RGB')
            mask = mask.convert('RGB')
            
            # Resize to match
            size = (512, 512)
            image = image.resize(size)
            mask = mask.resize(size)
            
            # Perform inpainting
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = self.pipeline(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Convert result to base64
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            result_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "inpainted_image": result_b64,
                "prompt": prompt,
                "image_size": size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### ControlNet Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "16Gi"}
)
class ControlNetService:
    def __init__(self):
        # Load ControlNet pipeline
        self.pipeline = bentoml.diffusers.load_model("controlnet_pipeline:latest")
        
        # Load preprocessors if saved with model
        model_ref = bentoml.diffusers.get("controlnet_pipeline:latest")
        self.preprocessors = model_ref.custom_objects.get("preprocessors", {})
    
    @bentoml.api
    def generate_with_control(
        self,
        image: Image.Image,
        prompt: str,
        control_type: str = "canny",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0
    ) -> dict:
        try:
            # Preprocess control image based on type
            if control_type == "canny":
                control_image = self.apply_canny_edge_detection(image)
            elif control_type == "depth":
                control_image = self.apply_depth_estimation(image)
            else:
                control_image = image
            
            # Generate image with ControlNet
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = self.pipeline(
                    prompt=prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale
                ).images[0]
            
            # Convert images to base64
            control_buffer = io.BytesIO()
            control_image.save(control_buffer, format="PNG")
            control_b64 = base64.b64encode(control_buffer.getvalue()).decode()
            
            result_buffer = io.BytesIO()
            result.save(result_buffer, format="PNG")
            result_b64 = base64.b64encode(result_buffer.getvalue()).decode()
            
            return {
                "success": True,
                "generated_image": result_b64,
                "control_image": control_b64,
                "control_type": control_type,
                "prompt": prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def apply_canny_edge_detection(self, image: Image.Image) -> Image.Image:
        # Simple Canny edge detection (in practice, use cv2 or similar)
        import numpy as np
        from PIL import ImageFilter
        
        # Convert to grayscale and apply edge detection
        gray_image = image.convert('L')
        edge_image = gray_image.filter(ImageFilter.FIND_EDGES)
        
        # Convert back to RGB
        return edge_image.convert('RGB')
    
    def apply_depth_estimation(self, image: Image.Image) -> Image.Image:
        # Placeholder for depth estimation
        # In practice, you'd use a depth estimation model
        return image.convert('L').convert('RGB')
```

## Advanced Features

### Multi-Pipeline Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "20Gi"}
)
class MultiDiffusionService:
    def __init__(self):
        # Load multiple pipelines
        self.txt2img = bentoml.diffusers.load_model("txt2img_pipeline:latest")
        self.img2img = bentoml.diffusers.load_model("img2img_pipeline:latest")
        self.inpainting = bentoml.diffusers.load_model("inpainting_pipeline:latest")
        
        # Optimization: Share components between pipelines if possible
        # self.txt2img.vae = self.img2img.vae  # Share VAE
    
    @bentoml.api
    def generate_text_to_image(self, prompt: str, **kwargs) -> dict:
        return self._generate_with_pipeline(self.txt2img, prompt, **kwargs)
    
    @bentoml.api
    def generate_image_to_image(self, image: Image.Image, prompt: str, **kwargs) -> dict:
        return self._generate_with_pipeline(self.img2img, prompt, image=image, **kwargs)
    
    @bentoml.api
    def generate_inpainting(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        prompt: str, 
        **kwargs
    ) -> dict:
        return self._generate_with_pipeline(
            self.inpainting, 
            prompt, 
            image=image, 
            mask_image=mask, 
            **kwargs
        )
    
    def _generate_with_pipeline(self, pipeline, prompt: str, **kwargs) -> dict:
        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = pipeline(prompt=prompt, **kwargs)
                
            if hasattr(result, 'images'):
                image = result.images[0]
            else:
                image = result
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": image_b64,
                "prompt": prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Custom Scheduler Integration

```python
@bentoml.service
class CustomSchedulerService:
    def __init__(self):
        self.pipeline = bentoml.diffusers.load_model("flexible_pipeline:latest")
        
        # Import different schedulers
        from diffusers import (
            DDPMScheduler, DDIMScheduler, PNDMScheduler, 
            LMSDiscreteScheduler, EulerDiscreteScheduler
        )
        
        self.schedulers = {
            "ddpm": DDPMScheduler.from_config(self.pipeline.scheduler.config),
            "ddim": DDIMScheduler.from_config(self.pipeline.scheduler.config),
            "pndm": PNDMScheduler.from_config(self.pipeline.scheduler.config),
            "lms": LMSDiscreteScheduler.from_config(self.pipeline.scheduler.config),
            "euler": EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        }
    
    @bentoml.api
    def generate_with_scheduler(
        self,
        prompt: str,
        scheduler_name: str = "ddim",
        num_inference_steps: int = 20
    ) -> dict:
        try:
            # Set the scheduler
            if scheduler_name in self.schedulers:
                self.pipeline.scheduler = self.schedulers[scheduler_name]
            
            # Generate image
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps
                ).images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": image_b64,
                "scheduler_used": scheduler_name,
                "available_schedulers": list(self.schedulers.keys())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

## Performance Optimization

### Memory Optimization

```python
@bentoml.service(resources={"gpu": "1", "memory": "8Gi"})
class OptimizedDiffusionService:
    def __init__(self):
        self.pipeline = bentoml.diffusers.load_model("optimized_pipeline:latest")
        
        # Apply memory optimizations
        if torch.cuda.is_available():
            # Enable memory efficient attention
            self.pipeline.enable_memory_efficient_attention()
            
            # Enable VAE slicing for lower memory usage
            self.pipeline.enable_vae_slicing()
            
            # Enable CPU offloading if needed
            # self.pipeline.enable_sequential_cpu_offload()
            
            # Enable attention slicing
            self.pipeline.enable_attention_slicing()
    
    @bentoml.api
    def generate_optimized(self, prompt: str) -> dict:
        try:
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.autocast("cuda"):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=20
                ).images[0]
            
            # Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": image_b64,
                "optimizations_enabled": [
                    "memory_efficient_attention",
                    "vae_slicing", 
                    "attention_slicing"
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Batch Processing

```python
@bentoml.service
class BatchDiffusionService:
    def __init__(self):
        self.pipeline = bentoml.diffusers.load_model("batch_pipeline:latest")
        self.max_batch_size = 4
    
    @bentoml.api
    def generate_batch(
        self, 
        prompts: List[str],
        num_inference_steps: int = 20
    ) -> List[dict]:
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i + self.max_batch_size]
            
            try:
                with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    images = self.pipeline(
                        prompt=batch_prompts,
                        num_inference_steps=num_inference_steps
                    ).images
                
                # Process batch results
                for j, (prompt, image) in enumerate(zip(batch_prompts, images)):
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    results.append({
                        "success": True,
                        "image": image_b64,
                        "prompt": prompt,
                        "batch_index": i + j
                    })
                    
            except Exception as e:
                # Handle batch errors
                for prompt in batch_prompts:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "prompt": prompt
                    })
        
        return results
```

## Best Practices

### 1. Resource Management

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "16Gi"},
    traffic={"timeout": 600}  # Longer timeout for image generation
)
class ResourceManagedService:
    def __init__(self):
        # Load with specific device and memory settings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipeline = bentoml.diffusers.load_model(
            "managed_pipeline:latest",
            device=device
        )
        
        # Configure based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 8 * 1024**3:  # Less than 8GB
                self.pipeline.enable_sequential_cpu_offload()
    
    @bentoml.api
    def generate_managed(self, prompt: str) -> dict:
        try:
            # Monitor GPU memory if available
            memory_before = None
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
            
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = self.pipeline(prompt=prompt).images[0]
            
            # Memory cleanup
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                
                memory_info = {
                    "memory_before_mb": memory_before / 1024**2 if memory_before else None,
                    "memory_after_mb": memory_after / 1024**2,
                    "memory_freed_mb": (memory_after - memory_before) / 1024**2 if memory_before else None
                }
            else:
                memory_info = {}
            
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": image_b64,
                "memory_info": memory_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### 2. Error Handling and Validation

```python
@bentoml.service
class ValidatedDiffusionService:
    def __init__(self):
        try:
            self.pipeline = bentoml.diffusers.load_model("validated_pipeline:latest")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}")
            self.model_loaded = False
    
    def validate_parameters(self, **params) -> dict:
        """Validate generation parameters"""
        errors = []
        
        # Validate steps
        if "num_inference_steps" in params:
            steps = params["num_inference_steps"]
            if not isinstance(steps, int) or steps < 1 or steps > 100:
                errors.append("num_inference_steps must be between 1 and 100")
        
        # Validate guidance scale
        if "guidance_scale" in params:
            guidance = params["guidance_scale"]
            if not isinstance(guidance, (int, float)) or guidance < 0 or guidance > 30:
                errors.append("guidance_scale must be between 0 and 30")
        
        # Validate dimensions
        for dim in ["width", "height"]:
            if dim in params:
                value = params[dim]
                if not isinstance(value, int) or value % 8 != 0 or value < 128 or value > 1024:
                    errors.append(f"{dim} must be divisible by 8 and between 128 and 1024")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    @bentoml.api
    def generate_validated(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512
    ) -> dict:
        # Check if model is loaded
        if not self.model_loaded:
            return {"success": False, "error": "Diffusion model not available"}
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            return {"success": False, "error": "Prompt cannot be empty"}
        
        if len(prompt) > 1000:
            return {"success": False, "error": "Prompt too long (max 1000 characters)"}
        
        # Validate parameters
        validation = self.validate_parameters(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
        
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Invalid parameters",
                "details": validation["errors"]
            }
        
        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                ).images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": image_b64,
                "parameters": {
                    "prompt": prompt,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "dimensions": f"{width}x{height}"
                }
            }
            
        except Exception as e:
            logger.error(f"Diffusion generation error: {e}")
            return {
                "success": False,
                "error": "Generation failed",
                "details": str(e)
            }
```

For more Diffusers examples and advanced techniques, visit the [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/) and the [BentoML examples repository](https://github.com/bentoml/BentoML/tree/main/examples/).