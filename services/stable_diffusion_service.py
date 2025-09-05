"""
Stable Diffusion Image Generation Service
This service takes a text prompt and generates an image using Stable Diffusion
"""

import bentoml
from pydantic import BaseModel, Field
from typing import Dict, Any
from diffusers import StableDiffusionPipeline

from utils.stable_diffusion import (
    BasePipelineManager, 
    pil_to_base64,
    validate_dimensions
)


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
        # Initialize pipeline manager with Stable Diffusion v1.5
        self.pipeline_manager = BasePipelineManager(
            model_id="runwayml/stable-diffusion-v1-5",
            pipeline_class=StableDiffusionPipeline
        )
        
        # Store device info for API responses
        self.device = self.pipeline_manager.device
    
    @bentoml.api
    def generate_image(self, request: ImageGenerationRequest) -> Dict[str, Any]:
        """Generate an image from a text prompt"""
        
        # Validate and adjust dimensions 
        width, height = validate_dimensions(request.width, request.height)
        
        # Generate image using pipeline manager
        result = self.pipeline_manager.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt if request.negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed if request.seed >= 0 else None
        )
        
        # Get the generated image and convert to base64
        image = result.images[0]
        image_base64 = pil_to_base64(image)
        
        return {
            "success": True,
            "image": image_base64,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "device_used": self.device
        }
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        health_info = self.pipeline_manager.get_health_status()
        health_info.update({
            "service": "StableDiffusionService",
            "version": "1.0.0"
        })
        return health_info