"""
Device detection and configuration utilities for ML models.
Handles device selection and optimization across CUDA, MPS, and CPU.
"""

import torch
from typing import Literal

DeviceType = Literal["cuda", "mps", "cpu"]


def detect_device() -> DeviceType:
    """
    Automatically detect the best available device for ML inference.
    
    Priority order:
    1. MPS (Apple Silicon) 
    2. CUDA (NVIDIA GPU)
    3. CPU (fallback)
    
    Returns:
        str: Device type ("mps", "cuda", or "cpu")
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_optimal_dtype(device: DeviceType) -> torch.dtype:
    """
    Get the optimal torch dtype for the given device.
    
    Args:
        device: Device type
        
    Returns:
        torch.dtype: Optimal dtype for the device
    """
    if device == "cuda":
        # CUDA supports float16 for faster inference
        return torch.float16
    elif device == "mps":
        # MPS requires float32 to avoid black image generation
        return torch.float32
    else:
        # CPU uses float32
        return torch.float32


def get_device_info(device: DeviceType) -> dict:
    """
    Get detailed information about the device.
    
    Args:
        device: Device type
        
    Returns:
        dict: Device information including name, memory, etc.
    """
    info = {"device": device}
    
    if device == "cuda" and torch.cuda.is_available():
        info.update({
            "name": torch.cuda.get_device_name(),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "compute_capability": torch.cuda.get_device_capability()
        })
    elif device == "mps":
        info.update({
            "name": "Apple Silicon MPS",
            "optimization": "Memory-efficient with float32 precision"
        })
    else:
        info.update({
            "name": "CPU",
            "warning": "Inference will be slower on CPU"
        })
    
    return info