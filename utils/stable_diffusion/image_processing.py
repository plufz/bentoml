"""
Image processing utilities for Stable Diffusion services.
Handles image format conversion, base64 encoding, and PIL operations.
"""

import io
import base64
from typing import Union, Optional
from PIL import Image
import torch


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        str: Base64 encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(base64_string: str) -> Image.Image:
    """
    Convert base64 encoded string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image.Image: PIL Image object
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def save_image(image: Image.Image, filepath: str, format: Optional[str] = None) -> None:
    """
    Save PIL Image to file.
    
    Args:
        image: PIL Image object
        filepath: Path to save the image
        format: Image format (inferred from filepath if not provided)
    """
    image.save(filepath, format=format)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducible generation.
    
    Args:
        seed: Random seed value (ignored if < 0)
    """
    if seed >= 0:
        torch.manual_seed(seed)


def validate_dimensions(width: int, height: int, min_size: int = 256, max_size: int = 1024) -> tuple[int, int]:
    """
    Validate and clamp image dimensions to acceptable range.
    
    Args:
        width: Image width
        height: Image height  
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
        
    Returns:
        tuple: (validated_width, validated_height)
    """
    width = max(min_size, min(max_size, width))
    height = max(min_size, min(max_size, height))
    
    # Ensure dimensions are multiples of 8 for Stable Diffusion
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    return width, height