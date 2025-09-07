"""
Photo Upscaling Service using Real-ESRGAN
This service upscales images using AI-powered super-resolution models
"""

import bentoml
from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, Optional
from pathlib import Path

from utils.upscaler import (
    UpscalerPipelineManager,
    process_image_file,
    process_image_url
)


class UpscaleUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the image to upscale")
    scale_factor: float = Field(2.0, ge=1.0, le=4.0, description="Upscaling factor (1.0-4.0)")
    face_enhance: bool = Field(False, description="Enable face enhancement for portraits")
    output_format: str = Field("PNG", description="Output format: PNG, JPEG, WEBP")
    quality: int = Field(95, ge=50, le=100, description="JPEG quality (50-100)")


@bentoml.service(
    resources={"memory": "4Gi", "gpu": 1},
    traffic={"timeout": 120}
)
class PhotoUpscalerService:
    """AI-powered photo upscaling service using Real-ESRGAN"""
    
    def __init__(self):
        self.pipeline_manager = UpscalerPipelineManager()
        self.device = self.pipeline_manager.device
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "PhotoUpscalerService",
            "version": "1.0.0",
            "device": self.device,
            "model": self.pipeline_manager.model_name,
            "capabilities": [
                "photo_upscaling",
                "face_enhancement",
                "batch_processing",
                "multiple_formats"
            ]
        }
    
    @bentoml.api
    def upscale_file(
        self, 
        image_file: Path,
        scale_factor: float = 2.0,
        face_enhance: bool = False,
        output_format: str = "PNG",
        quality: int = 95
    ) -> Dict[str, Any]:
        """
        Upscale an uploaded image file.
        
        Args:
            image_file: Image file to upscale
            scale_factor: Upscaling factor (1.0-4.0)
            face_enhance: Enable face enhancement
            output_format: Output format (PNG, JPEG, WEBP)
            quality: JPEG quality (50-100)
        """
        try:
            result = process_image_file(
                image_file=image_file,
                scale_factor=scale_factor,
                face_enhance=face_enhance,
                output_format=output_format,
                quality=quality,
                pipeline_manager=self.pipeline_manager
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "image_base64": result["image_base64"],
                    "upscaling_info": result["info"]
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Upscaling failed: {str(e)}"
            }
    
    @bentoml.api 
    def upscale_url(self, request: UpscaleUrlRequest) -> Dict[str, Any]:
        """
        Upscale an image from a URL.
        
        Args:
            request: Upscaling request with URL and parameters
        """
        try:
            result = process_image_url(
                url=str(request.url),
                scale_factor=request.scale_factor,
                face_enhance=request.face_enhance,
                output_format=request.output_format,
                quality=request.quality,
                pipeline_manager=self.pipeline_manager
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "image_base64": result["image_base64"], 
                    "upscaling_info": result["info"]
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Upscaling failed: {str(e)}"
            }