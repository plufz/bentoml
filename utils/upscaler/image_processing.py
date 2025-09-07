"""
Image processing utilities for the upscaler service
"""

import io
import base64
import requests
import tempfile
from PIL import Image
from pathlib import Path
from typing import Dict, Any


def validate_image_format(image_input) -> bool:
    """Validate if input is a valid image format"""
    try:
        if isinstance(image_input, (str, Path)):
            # File path
            Image.open(image_input)
        elif hasattr(image_input, 'read'):
            # File-like object
            Image.open(image_input)
        else:
            return False
        return True
    except Exception:
        return False


def download_image(url: str) -> bytes:
    """Download image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; PhotoUpscaler/1.0)'
        }
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError(f"URL does not point to an image (content-type: {content_type})")
        
        return response.content
        
    except Exception as e:
        raise ValueError(f"Failed to download image: {str(e)}")


def validate_output_format(format_str: str) -> str:
    """Validate and normalize output format"""
    format_upper = format_str.upper()
    allowed_formats = {'PNG', 'JPEG', 'WEBP'}
    
    if format_upper not in allowed_formats:
        raise ValueError(f"Invalid output format. Allowed: {', '.join(allowed_formats)}")
    
    return format_upper


def image_to_base64(image: Image.Image, output_format: str = "PNG", quality: int = 95) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    
    save_kwargs = {"format": output_format}
    if output_format == "JPEG":
        save_kwargs.update({"quality": quality, "optimize": True})
    elif output_format == "PNG":
        save_kwargs["optimize"] = True
    
    image.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode()


def process_image_file(
    image_file: Path,
    scale_factor: float,
    face_enhance: bool,
    output_format: str,
    quality: int,
    pipeline_manager
) -> Dict[str, Any]:
    """Process an uploaded image file for upscaling"""
    try:
        # Validate output format
        output_format = validate_output_format(output_format)
        
        # Validate quality
        if not (50 <= quality <= 100):
            quality = 95
        
        # Load image
        try:
            image = Image.open(image_file)
            original_size = image.size
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid image file: {str(e)}"
            }
        
        # Upscale image
        upscaled_image = pipeline_manager.upscale_image(
            image, scale_factor, face_enhance
        )
        
        # Convert to base64
        image_base64 = image_to_base64(upscaled_image, output_format, quality)
        
        return {
            "success": True,
            "image_base64": image_base64,
            "info": {
                "original_size": original_size,
                "upscaled_size": upscaled_image.size,
                "scale_factor": scale_factor,
                "face_enhance": face_enhance,
                "output_format": output_format,
                "quality": quality if output_format == "JPEG" else None,
                "device": pipeline_manager.device,
                "model": pipeline_manager.model_name
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def process_image_url(
    url: str,
    scale_factor: float,
    face_enhance: bool,
    output_format: str,
    quality: int,
    pipeline_manager
) -> Dict[str, Any]:
    """Process an image from URL for upscaling"""
    try:
        # Validate output format
        output_format = validate_output_format(output_format)
        
        # Validate quality
        if not (50 <= quality <= 100):
            quality = 95
        
        # Download image
        try:
            image_data = download_image(url)
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image from URL: {str(e)}"
            }
        
        # Upscale image
        upscaled_image = pipeline_manager.upscale_image(
            image, scale_factor, face_enhance
        )
        
        # Convert to base64
        image_base64 = image_to_base64(upscaled_image, output_format, quality)
        
        return {
            "success": True,
            "image_base64": image_base64,
            "info": {
                "original_size": original_size,
                "upscaled_size": upscaled_image.size,
                "scale_factor": scale_factor,
                "face_enhance": face_enhance,
                "output_format": output_format,
                "quality": quality if output_format == "JPEG" else None,
                "device": pipeline_manager.device,
                "model": pipeline_manager.model_name,
                "source_url": url
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }