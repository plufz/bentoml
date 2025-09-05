"""
Base pipeline manager for Stable Diffusion models.
Provides common functionality for loading and configuring different SD pipelines.
"""

import torch
from typing import Dict, Any, Optional, Union
from diffusers import DiffusionPipeline
from .device import detect_device, get_optimal_dtype, DeviceType
from .hf_env import setup_hf_environment, print_hf_setup


class BasePipelineManager:
    """
    Base class for managing Stable Diffusion pipelines.
    
    Handles device detection, model loading, and device-specific optimizations.
    """
    
    def __init__(
        self, 
        model_id: str,
        pipeline_class: type = DiffusionPipeline,
        custom_hf_home: Optional[str] = None,
        device: Optional[DeviceType] = None
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            model_id: HuggingFace model identifier
            pipeline_class: Diffusion pipeline class to use
            custom_hf_home: Custom HuggingFace cache directory
            device: Specific device to use (auto-detected if None)
        """
        self.model_id = model_id
        self.pipeline_class = pipeline_class
        
        # Setup HuggingFace environment
        hf_home = setup_hf_environment(custom_hf_home)
        print_hf_setup(hf_home)
        
        # Detect and configure device
        self.device = device or detect_device()
        self.dtype = get_optimal_dtype(self.device)
        
        # Load pipeline
        self.pipe = self._load_pipeline()
        
    def _load_pipeline(self) -> DiffusionPipeline:
        """
        Load and configure the diffusion pipeline based on device capabilities.
        
        Returns:
            DiffusionPipeline: Configured pipeline ready for inference
        """
        print(f"Loading {self.model_id} on {self.device} with {self.dtype}")
        
        # Load pipeline with appropriate settings
        pipe = self.pipeline_class.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        pipe = pipe.to(self.device)
        
        # Apply device-specific optimizations
        self._apply_device_optimizations(pipe)
        
        return pipe
    
    def _apply_device_optimizations(self, pipe: DiffusionPipeline) -> None:
        """
        Apply device-specific optimizations to the pipeline.
        
        Args:
            pipe: Diffusion pipeline to optimize
        """
        if self.device == "cuda":
            # CUDA optimizations
            pipe.enable_attention_slicing()
            
        elif self.device == "mps":
            # Apple Silicon MPS optimizations
            pipe.enable_attention_slicing("max")
            
            # Ensure VAE uses float32 to prevent black images
            if hasattr(pipe, 'vae'):
                pipe.vae = pipe.vae.to(dtype=torch.float32)
                
        # CPU doesn't need special optimizations
    
    def generate(
        self, 
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Generate content using the pipeline.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to avoid certain elements
            width: Output width
            height: Output height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducible generation
            **kwargs: Additional pipeline-specific arguments
            
        Returns:
            Pipeline output (format depends on pipeline type)
        """
        # Set random seed if specified
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
        
        # Generate with inference mode for efficiency
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            dict: Pipeline information including model, device, etc.
        """
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "pipeline_class": self.pipeline_class.__name__,
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status information for the pipeline.
        
        Returns:
            dict: Health status information
        """
        return {
            "status": "healthy",
            "device": self.device,
            "model": self.model_id,
            "dtype": str(self.dtype)
        }