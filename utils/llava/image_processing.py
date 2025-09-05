"""
Image processing utilities for LLaVA services.
Handles image input validation, format conversion, and preprocessing.
"""

import io
import base64
from typing import Union, Optional
from PIL import Image
import requests


def validate_image_format(image: Union[str, bytes, Image.Image]) -> bool:
    """
    Validate if the input is a valid image format.
    
    Args:
        image: Image data in various formats
        
    Returns:
        bool: True if valid image format
    """
    try:
        if isinstance(image, Image.Image):
            return True
        elif isinstance(image, str):
            # Could be base64 or URL
            if image.startswith('data:image/'):
                # Base64 image
                return True
            elif image.startswith(('http://', 'https://')):
                # URL - try to fetch and validate
                response = requests.head(image, timeout=5)
                return response.headers.get('content-type', '').startswith('image/')
            else:
                # Try to decode as base64
                try:
                    image_data = base64.b64decode(image)
                    Image.open(io.BytesIO(image_data))
                    return True
                except Exception:
                    return False
        elif isinstance(image, bytes):
            Image.open(io.BytesIO(image))
            return True
        else:
            return False
    except Exception:
        return False


def process_image_input(image: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    Process various image input formats into PIL Image.
    
    Args:
        image: Image data as URL, base64 string, bytes, or PIL Image
        
    Returns:
        Image.Image: Processed PIL Image
        
    Raises:
        ValueError: If image format is invalid or processing fails
    """
    try:
        if isinstance(image, Image.Image):
            return image
        
        elif isinstance(image, str):
            if image.startswith('data:image/'):
                # Data URL format: data:image/png;base64,iVBORw0KGgoA...
                header, data = image.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data))
            
            elif image.startswith(('http://', 'https://')):
                # URL - fetch the image
                response = requests.get(image, timeout=10)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            
            else:
                # Assume base64 encoded image
                image_data = base64.b64decode(image)
                return Image.open(io.BytesIO(image_data))
        
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
            
    except Exception as e:
        raise ValueError(f"Failed to process image input: {str(e)}")


def prepare_image_for_llava(image: Image.Image, max_size: int = 1344) -> Image.Image:
    """
    Prepare image for LLaVA processing by resizing if necessary.
    
    LLaVA-1.6 supports up to 672x672, 336x1344, 1344x336 aspect ratios.
    
    Args:
        image: PIL Image to prepare
        max_size: Maximum dimension size
        
    Returns:
        Image.Image: Prepared image
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Check if resizing is needed
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    # Resize image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)