"""
Stable Diffusion Image Generation Service
This service takes a text prompt and generates an image using Stable Diffusion
"""

import io
import os
import base64
import bentoml
from pydantic import BaseModel, Field
from typing import Dict, Any
from PIL import Image
import torch


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field("", description="Negative prompt to avoid certain elements")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    num_inference_steps: int = Field(20, ge=1, le=50, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    seed: int = Field(-1, description="Random seed (-1 for random)")


@bentoml.service(
    resources={"gpu": 1, "memory": "8Gi"},
    traffic={"timeout": 300}
)
class StableDiffusionService:
    """Stable Diffusion image generation service using Hugging Face Diffusers"""
    
    def __init__(self):
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Ensure HF_HOME is set for custom cache directory
        # This respects the user's .zprofile setting: HF_HOME="/Volumes/Second/huggingface"
        if 'HF_HOME' not in os.environ:
            # Default to user's cache if not set
            default_hf_home = os.path.expanduser("~/.cache/huggingface")
            os.environ['HF_HOME'] = default_hf_home
            print(f"HF_HOME not set, using default: {default_hf_home}")
        else:
            print(f"Using HF_HOME: {os.environ['HF_HOME']}")
        
        # Also set related cache variables to ensure consistency
        if 'TRANSFORMERS_CACHE' not in os.environ:
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'hub')
        
        # Check device availability: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Load the Stable Diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        if self.device == "cuda":
            # CUDA: Use float16 for GPU acceleration
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            
        elif self.device == "mps":
            # Apple Silicon MPS: Use float32 to avoid black images
            # float16 is unstable on MPS and causes black outputs
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            
            # Enable attention slicing for better memory usage on Apple Silicon
            self.pipe.enable_attention_slicing("max")
            
            # Ensure VAE also uses float32 to prevent black images
            if hasattr(self.pipe, 'vae'):
                self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
                
        else:
            # CPU version
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
    
    @bentoml.api
    def generate_image(self, request: ImageGenerationRequest) -> Dict[str, Any]:
        """Generate an image from a text prompt"""
        
        # Set random seed if specified
        if request.seed >= 0:
            torch.manual_seed(request.seed)
        
        # Generate image
        with torch.inference_mode():
            result = self.pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt if request.negative_prompt else None,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
        
        # Get the generated image
        image = result.images[0]
        
        # Convert image to base64 for JSON response
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "success": True,
            "image": image_base64,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "device_used": self.device
        }
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "StableDiffusionService",
            "version": "1.0.0",
            "device": self.device,
            "model": "runwayml/stable-diffusion-v1-5"
        }