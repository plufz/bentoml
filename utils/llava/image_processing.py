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
                img = Image.open(io.BytesIO(image_data))
            
            elif image.startswith(('http://', 'https://')):
                # URL - fetch the image
                response = requests.get(image, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            
            else:
                # Assume base64 encoded image
                try:
                    image_data = base64.b64decode(image)
                    img = Image.open(io.BytesIO(image_data))
                except Exception as decode_error:
                    raise ValueError(f"Invalid base64 image data: {decode_error}")
        
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
        
        # Ensure image is valid and has reasonable size
        if img.size[0] == 0 or img.size[1] == 0:
            raise ValueError("Image has zero width or height")
        
        # Convert single-channel or palette mode images
        if img.mode in ('P', 'L'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            # Convert RGBA to RGB by compositing with white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background
            
        return img
            
    except Exception as e:
        raise ValueError(f"Failed to process image input: {str(e)}")


