"""
Stable Diffusion utilities for building multiple SD endpoints.

This package provides common functionality for building multiple Stable Diffusion
endpoints, including device detection, pipeline management, and image processing.
"""

from .device import detect_device, get_optimal_dtype, get_device_info, DeviceType
from .hf_env import setup_hf_environment, get_hf_cache_info, print_hf_setup
from .image_processing import pil_to_base64, base64_to_pil, save_image, set_random_seed, validate_dimensions
from .pipeline_manager import BasePipelineManager

__all__ = [
    # Device utilities
    "detect_device",
    "get_optimal_dtype", 
    "get_device_info",
    "DeviceType",
    
    # HuggingFace environment
    "setup_hf_environment",
    "get_hf_cache_info",
    "print_hf_setup",
    
    # Image processing
    "pil_to_base64",
    "base64_to_pil", 
    "save_image",
    "set_random_seed",
    "validate_dimensions",
    
    # Pipeline management
    "BasePipelineManager",
]