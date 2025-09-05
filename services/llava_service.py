"""
LLaVA Vision-Language Service using llama-cpp-python
This service takes an image, text prompt, and optional JSON schema to generate structured responses
"""

import bentoml
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union
from PIL import Image

from utils.llava import (
    validate_json_schema,
    validate_image_format
)
from utils.llava.llamacpp_pipeline_manager import LLaVALlamaCppPipelineManager


class VisionLanguageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt or question about the image")
    image: Union[str, bytes] = Field(..., description="Image as URL, base64 string, or bytes")
    json_schema: Optional[Dict[str, Any]] = Field(None, description="Optional JSON schema for structured output")
    include_raw_response: bool = Field(False, description="Include raw model response in output")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Generation temperature (0.0-2.0)")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=2048, description="Maximum tokens to generate")


@bentoml.service(
    resources={"memory": "4Gi"},
    traffic={"timeout": 120}
)
class LLaVAService:
    """LLaVA vision-language service using llama-cpp-python for faster GGUF model inference"""
    
    def __init__(self):
        # Initialize the llama-cpp pipeline manager
        self.pipeline_manager = LLaVALlamaCppPipelineManager(
            max_new_tokens=512,
            temperature=0.1,
        )
        
        # Store device info for API responses
        self.device = self.pipeline_manager.device
    
    @bentoml.api
    def analyze_image(self, request: VisionLanguageRequest) -> Dict[str, Any]:
        """
        Analyze image with text prompt and optional structured JSON output
        
        Args:
            request: Vision-language request with image, prompt, and optional JSON schema
            
        Returns:
            Dict containing analysis results, either as structured JSON or raw text
        """
        
        # Validate inputs
        if not validate_image_format(request.image):
            return {
                "success": False,
                "error": "Invalid image format. Supported: URL, base64, or image bytes",
                "format": "error"
            }
        
        if request.json_schema and not validate_json_schema(request.json_schema):
            return {
                "success": False,
                "error": "Invalid JSON schema provided",
                "format": "error"
            }
        
        # Update pipeline parameters if specified
        original_max_tokens = self.pipeline_manager.max_new_tokens
        original_temperature = self.pipeline_manager.temperature
        
        if request.max_new_tokens:
            self.pipeline_manager.max_new_tokens = request.max_new_tokens
        if request.temperature is not None:
            self.pipeline_manager.temperature = request.temperature
        
        try:
            # Generate structured response
            result = self.pipeline_manager.generate_structured_response(
                image=request.image,
                prompt=request.prompt,
                json_schema=request.json_schema,
                include_raw_response=request.include_raw_response
            )
            
            # Add request metadata
            result.update({
                "input_prompt": request.prompt,
                "has_json_schema": request.json_schema is not None,
                "device_used": self.device,
            })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process request: {str(e)}",
                "format": "error",
                "device_used": self.device,
            }
        
        finally:
            # Restore original parameters
            self.pipeline_manager.max_new_tokens = original_max_tokens
            self.pipeline_manager.temperature = original_temperature
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        health_info = self.pipeline_manager.get_health_status()
        health_info.update({
            "service": "LLaVAService",
            "version": "1.0.0",
            "capabilities": [
                "image_analysis",
                "visual_question_answering", 
                "structured_json_output",
                "multimodal_chat"
            ]
        })
        return health_info
    
    @bentoml.api 
    def get_example_schemas(self) -> Dict[str, Any]:
        """Get example JSON schemas for common use cases"""
        return {
            "image_description": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Detailed image description"},
                    "objects": {"type": "array", "items": {"type": "string"}, "description": "List of objects in the image"},
                    "scene": {"type": "string", "description": "Type of scene (indoor, outdoor, etc.)"},
                    "mood": {"type": "string", "description": "Overall mood or atmosphere"}
                },
                "required": ["description", "objects", "scene"]
            },
            "object_detection": {
                "type": "object", 
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "confidence"]
                        }
                    },
                    "total_objects": {"type": "integer", "minimum": 0}
                },
                "required": ["objects", "total_objects"]
            },
            "image_qa": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Direct answer to the question"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string", "description": "Explanation of the reasoning"},
                    "additional_details": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["answer", "confidence"]
            },
            "text_extraction": {
                "type": "object",
                "properties": {
                    "text_found": {"type": "boolean"},
                    "extracted_text": {"type": "string"},
                    "text_locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "region": {"type": "string", "description": "Approximate location in image"}
                            }
                        }
                    }
                },
                "required": ["text_found", "extracted_text"]
            }
        }